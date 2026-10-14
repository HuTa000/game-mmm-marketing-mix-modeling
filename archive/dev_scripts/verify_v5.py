"""
验证 v5：用正确参数名重算 Hill，破解虚高根源
=============================================
【重大发现】pymc-marketing 1.x 的 HillSaturation 参数名与数学符号错位：
  · saturation_slope = Hill 曲线的陡峭度 κ（公式里的指数）
  · saturation_kappa = 半饱和点 γ（公式里的分母常数）
  · saturation_beta  = 渐近线高度（不是"每单位花费的贡献"！）

公式：Hill(x) = beta × x^slope / (x^slope + kappa^slope)

【这解释了 ROOT CAUSE】
  如果 saturation_beta 是"渐近线高度"，而渐近线是**绝对收入量级**，
  那么在标准化输入（均值=1）下，模型需要 beta ≈ 贡献总量 / 周数
  才能拟合，但这个 beta 与"花费效率"无直接关系 →
  ROI = beta × Hill / spend，量级就不对了。

本脚本：用正确公式手工重算，验证这个假设。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series
from pymc_marketing.mmm import GeometricAdstock, HillSaturation


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))

    print("=" * 90)
    print("后验参数完整清单")
    print("=" * 90)
    for v in pt.data_vars:
        print(f"  {v:35s} {tuple(pt[v].shape)}")

    adstock = GeometricAdstock(l_max=cfg.adstock_l_max)
    sat = HillSaturation()

    print("\n" + "=" * 90)
    print("用正确公式手工重算：Hill(x) = beta × x^slope / (x^slope + kappa^slope)")
    print("=" * 90)
    rows = []
    for i, c in enumerate(cfg.channel_cols):
        x_std = X[c].values
        d_est = float(pt["adstock_alpha"].values[:, :, i].mean())
        ad = adstock.function(x=x_std, alpha=d_est, dim="date")
        slope = float(pt["saturation_slope"].values[:, :, i].mean())
        kappa = float(pt["saturation_kappa"].values[:, :, i].mean())
        beta = float(pt["saturation_beta"].values[:, :, i].mean())
        sat_out = beta * (ad ** slope) / (ad ** slope + kappa ** slope)
        rows.append({
            "渠道": c,
            "decay": d_est, "真d": TRUE_DECAY[c],
            "slopeκ": slope,
            "kappaγ": kappa,
            "beta": beta,
            "ad范围": f"[{ad.min():.2f},{ad.max():.2f}]",
            "饱和输出范围": f"[{sat_out.min():.2f},{sat_out.max():.2f}]",
            "总贡献万": sat_out.sum() / 1e4,
            "×scale万": sat_out.sum() * ts / 1e4,
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    true_y = float(df["revenue"].sum())
    raw_sum = sum(r["总贡献万"] for r in rows) * 1e4
    scaled_sum = sum(r["×scale万"] for r in rows) * 1e4
    print(f"\n手工贡献(不乘scale) = {raw_sum/1e4:,.0f} 万元 ({raw_sum/true_y:.1%})")
    print(f"手工贡献(乘scale)   = {scaled_sum/1e4:,.0f} 万元 ({scaled_sum/true_y:.1%})")
    print(f"真实媒体贡献        = 11,178 万元 (46.7%)")

    # ---- 关键：对比后验内置 channel_contribution ----
    print("\n" + "=" * 90)
    print("对比：后验内置 channel_contribution vs 手工重算（判定模型内部公式）")
    print("=" * 90)
    if "channel_contribution" in pt:
        cc = pt["channel_contribution"].values
        for i, c in enumerate(cfg.channel_cols):
            builtin = cc[:, :, :, i].mean(axis=(0, 1))
            # 手工（不含beta）
            x_std = X[c].values
            d_est = float(pt["adstock_alpha"].values[:, :, i].mean())
            ad = adstock.function(x=x_std, alpha=d_est, dim="date")
            slope = float(pt["saturation_slope"].values[:, :, i].mean())
            kappa = float(pt["saturation_kappa"].values[:, :, i].mean())
            beta = float(pt["saturation_beta"].values[:, :, i].mean())
            manual = beta * (ad ** slope) / (ad ** slope + kappa ** slope)
            print(f"  {c:10s} 内置均值={builtin.mean():.4f} "
                  f"手工均值={manual.mean():.4f} 差={builtin.mean()-manual.mean():+.4f}")

    # ---- ROI 对照 ----
    print("\n" + "=" * 90)
    print("ROI 对照（三种还原方式）")
    print("=" * 90)
    print(f"{'渠道':10s} {'真ROI':>7s} {'预测法':>8s} {'内置CC×scale':>13s} {'官方get_contrib':>15s}")
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    for i, c in enumerate(cfg.channel_cols):
        s = float(get_spend_series(df, c).sum())
        r_true = TRUE_BETA[c]
        # 预测法
        Xstd = X.copy()
        pb = np.asarray(mmm.predict(Xstd)).ravel() * ts
        Xcf = Xstd.copy(); Xcf[c] = 0
        r_pred = (pb - np.asarray(mmm.predict(Xcf)).ravel() * ts).sum() / s
        # 内置
        r_cc = (pt["channel_contribution"].values[:, :, :, i].sum() * ts) / s
        # 官方
        r_off = float(np.asarray(chA.sel(channel=c).values).sum()) / s
        print(f"{c:10s} {r_true:>7.2f} {r_pred:>8.3f} {r_cc:>13.3f} {r_off:>15.3f}")


if __name__ == "__main__":
    main()
