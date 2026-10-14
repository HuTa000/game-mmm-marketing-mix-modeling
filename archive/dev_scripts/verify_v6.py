"""
验证 v6：用底层 numpy 重算 Hill，彻底定位虚高根源
==================================================
已确认公式（pymc-marketing 1.2.0）：
  Hill(x) = 1 - kappa^slope / (kappa^slope + x^slope)
  其中 kappa = 半饱和点，slope = 陡峭度
  HillSaturation.function = beta × Hill(x)   ← beta 是渐近线高度

本脚本用 numpy 直接实现（绕过 pytensor 的 XTensor 限制），
对比：内置 channel_contribution / 手工重算 / 预测法 / 官方 get_contributions
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series


def geom_adstock_np(x, decay, l_max):
    """numpy 版几何 adstock"""
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        acc = 0.0
        for lag in range(min(l_max, t) + 1):
            acc += decay ** lag * x[t - lag]
        out[t] = acc
    return out


def hill_np(x, slope, kappa, beta):
    return beta * (1.0 - kappa ** slope / (kappa ** slope + x ** slope))


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))

    print("=" * 100)
    print(f"target_scale = {ts:,.2f}   真实 y 合计 = {float(df['revenue'].sum())/1e4:,.0f} 万元")
    print("=" * 100)

    print("\n【参数回收】")
    print(f"{'渠道':10s} {'decay':>7s} {'真d':>6s} {'slope':>7s} {'kappa':>9s} "
          f"{'beta':>9s} {'真β':>6s} {'z均值':>7s}")
    print("-" * 78)
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        be = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"{c:10s} {d:>7.3f} {TRUE_DECAY[c]:>6.2f} {sl:>7.3f} {ka:>9.3f} "
              f"{be:>9.2f} {TRUE_BETA[c]:>6.2f} {X[c].mean():>7.3f}")

    # ---- 手工重算各渠道贡献（不乘 scale）----
    print("\n【手工重算 Hill 输出】Hill(x)=beta×(1-kappa^s/(kappa^s+x^s))，输入=标准化渠道")
    rows = []
    ad_cache = {}
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        be = float(pt["saturation_beta"].values[:, :, i].mean())
        ad = geom_adstock_np(X[c].values, d, cfg.adstock_l_max)
        ad_cache[c] = ad
        sat_out = hill_np(ad, sl, ka, be)
        rows.append({
            "渠道": c,
            "ad均值": ad.mean(),
            "饱和输出均值": sat_out.mean(),
            "输出总和": sat_out.sum(),
            "×scale万": sat_out.sum() * ts / 1e4,
        })
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    manual_sum = sum(r["×scale万"] for r in rows) * 1e4
    true_y = float(df["revenue"].sum())
    print(f"\n手工(×scale)媒体合计 = {manual_sum/1e4:,.0f} 万元 ({manual_sum/true_y:.1%})")
    print(f"真实媒体贡献          = 11,178 万元 (46.7%)")

    # ---- 内置 channel_contribution ----
    print("\n【内置 channel_contribution（后验已存）】")
    cc = pt["channel_contribution"].values  # (chain,draw,date,channel)
    print(f"{'渠道':10s} {'内置×scale万':>14s} {'手工×scale万':>14s} {'比值':>8s}")
    print("-" * 52)
    for i, c in enumerate(cfg.channel_cols):
        builtin = cc[:, :, :, i].sum() * ts / 1e4
        manual = rows[i]["×scale万"]
        print(f"{c:10s} {builtin:>14,.0f} {manual:>14,.0f} {builtin/manual:>8.3f}")

    # ---- ROI 四方对照 ----
    print("\n" + "=" * 100)
    print("ROI 四方对照")
    print("=" * 100)
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    print(f"{'渠道':10s} {'花费万':>9s} {'真ROI':>7s} {'预测法':>8s} "
          f"{'内置CC':>8s} {'手工Hill':>9s} {'官方A':>8s}")
    print("-" * 68)
    for i, c in enumerate(cfg.channel_cols):
        s = float(get_spend_series(df, c).sum())
        Xstd = X.copy()
        pb = np.asarray(mmm.predict(Xstd)).ravel() * ts
        Xcf = Xstd.copy(); Xcf[c] = 0
        r_pred = (pb - np.asarray(mmm.predict(Xcf)).ravel() * ts).sum() / s
        r_cc = cc[:, :, :, i].sum() * ts / s
        r_manual = rows[i]["×scale万"] * 1e4 / s
        r_off = float(np.asarray(chA.sel(channel=c).values).sum()) / s
        print(f"{c:10s} {s/1e4:>9,.0f} {TRUE_BETA[c]:>7.2f} {r_pred:>8.3f} "
              f"{r_cc:>8.3f} {r_manual:>9.3f} {r_off:>8.3f}")

    # ---- 关键：adstock+hill 的"无饱和"检验 ----
    print("\n" + "=" * 100)
    print("饱和检验：标准化输入的 adstock 是否远小于 kappa（若远小于→几乎无饱和→贡献虚高）")
    print("=" * 100)
    for i, c in enumerate(cfg.channel_cols):
        ad = ad_cache[c]
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        ratio = ad.mean() / ka if ka else np.nan
        print(f"  {c:10s} ad均值={ad.mean():.3f}  kappa={ka:.3f}  "
              f"比值={ratio:.3f}  {'★远未饱和' if ratio < 0.5 else ''}")


if __name__ == "__main__":
    main()
