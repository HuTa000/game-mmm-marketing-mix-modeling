"""
验证 v14：最终确认 —— 量级正确 + 参数回收 + ROI 排序
=======================================================
v13 结论：模型完全正确！
  · get_contributions 返回的就是正确的贡献（单位与 y 相同）
  · beta 回收漂亮：tiktok 2.137 vs 2.30、kol 1.228 vs 1.20
  · "ROI=0.000" 只是数据生成量级问题（y 是"7"，花费是"1428万"）

本脚本把数据生成量级改到"元"级别（与真实业务一致），
最终确认 ROI 排序与量级。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def geom_adstock(x, decay, l_max=6):
    out = np.zeros_like(x, dtype=float); carry = 0.0
    for i in range(len(x)):
        carry = x[i] + decay * carry; out[i] = carry
    return out


def hill(x, slope, kappa):
    return (x ** slope) / (x ** slope + kappa ** slope)


def build_model(cfg, overrides):
    kwargs = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
                  target_column=cfg.target_col,
                  adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
                  saturation=HillSaturation(),
                  yearly_seasonality=cfg.yearly_seasonality)
    base = MMM(**kwargs)
    mc = dict(base.default_model_config); mc.update(overrides)
    kwargs["model_config"] = mc
    return MMM(**kwargs)


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    N = len(df)

    # ---- 生成"元级"收入，与真实的周均 150 万元一致 ----
    # 真实花费周均约 49 万元；ROI 1.2~2.3 → 媒体贡献周均约 70 万元
    spend_total_weekly = float(sum(df[c].mean() for c in cfg.channel_cols))
    print(f"周均总花费 = {spend_total_weekly:,.0f} 元")

    media_series = {}
    for ch in cfg.channel_cols:
        x = df[ch].values / df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        h = hill(ad, TRUE_KAPPA[ch], 1.2)
        # 让该渠道总贡献 = TRUE_BETA × 总花费
        scale = TRUE_BETA[ch] * df[ch].sum() / h.sum()
        media_series[ch] = scale * h
        print(f"  {ch:10s} 花费={df[ch].sum():>14,.0f}  贡献={media_series[ch].sum():>14,.0f}  "
              f"ROI={media_series[ch].sum()/df[ch].sum():.3f} (设计 {TRUE_BETA[ch]})")

    media = sum(media_series.values())
    media_weekly = media.mean()
    # 基线占比 55%
    baseline_weekly = media_weekly * (0.55 / 0.45)
    baseline = baseline_weekly * (0.85 + 0.15 * np.sin(2*np.pi*np.arange(N)/52 - 0.8))
    baseline += np.linspace(0, baseline_weekly * 0.15, N)
    rng = np.random.default_rng(11)
    df["revenue"] = baseline + media + rng.normal(0, 0.04 * media_weekly, N)

    true_y = float(df["revenue"].sum())
    print(f"\n收入: 周均={df['revenue'].mean():,.0f} 元  总和={true_y:,.0f}")
    print(f"真实媒体占比 = {media.sum()/true_y:.1%}  (设计 45%)")
    print(f"真实整体 ROI = {true_y / spend_total_weekly / (N/N):.3f}")

    X, y, meta, _ = prepare_data(df, cfg)

    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    mmm = build_model(cfg, ov)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = mmm.idata.posterior

    # ---- 正确的贡献（get_contributions 已含 original_scale）----
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))

    print("\n" + "=" * 90)
    print("最终结果")
    print("=" * 90)
    rows = []
    for c in cfg.channel_cols:
        v = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(df[c].sum())
        t = float(media_series[c].sum())
        rows.append({"渠道": c, "估贡献万": v / 1e4, "真贡献万": t / 1e4,
                     "花费万": s / 1e4, "估ROI": v / s, "真ROI": t / s})
    roi = pd.DataFrame(rows)
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic
    print(roi.round(3).to_string(index=False))
    print(f"\nSpearman ρ = {rho:.3f}  {'✓ 排序正确' if rho > 0.7 else '✗ 排序错误'}")

    ch_sum = float(np.asarray(chA.values).sum())
    print(f"估媒体占比 = {ch_sum/true_y:.1%}   (真实 {media.sum()/true_y:.1%})")

    # 参数回收
    print("\n参数回收:")
    print(f"{'渠道':10s} {'估decay':>9s} {'真d':>5s} {'估beta':>9s} {'真β':>6s} {'β偏差':>8s}")
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"{c:10s} {d:>9.3f} {TRUE_DECAY[c]:>5.2f} {b:>9.3f} {TRUE_BETA[c]:>6.2f} "
              f"{b/TRUE_BETA[c]-1:>+8.1%}")

    # 诊断
    import arviz as az
    s = az.summary(idata, var_names=["adstock_alpha", "saturation_beta",
                                     "saturation_kappa", "adstock_alpha"],
                   ci_prob=0.94, round_to=3)
    n_bad = int((s["r_hat"] > 1.01).sum())
    ndiv = int(idata.sample_stats["diverging"].sum().values)
    ntot = int(idata.sample_stats["diverging"].size)
    print(f"\n诊断: R-hat>1.01 = {n_bad}/{len(s)}, "
          f"divergences = {ndiv}/{ntot} ({100*ndiv/ntot:.2f}%), "
          f"min ESS = {s['ess_bulk'].min():.0f}")


if __name__ == "__main__":
    main()
