"""
验证 v12：固定 slope，只估 kappa+beta（提升可识别性）
=======================================================
v11 发现：即使数据严格按模型结构生成，参数回收依然失败：
  · decay 回收完美（说明 Adstock 结构可识别）
  · 但 beta 与 kappa 联合不可识别（google kappa→0.07, tiktok kappa→0.09）

原因：Hill(ad; slope, kappa, beta) 中三个参数只在特定组合下影响曲线，
      存在"等价曲面"（slope↑ + kappa↓ 与 slope↓ + kappa↑ 可能给近似的曲线）。

解法（MMM 业界常见做法）：
  1. 固定 slope = 1.5（先验集中，几乎不参与估计）
  2. 用更紧的 kappa 先验（围绕 1.0）
  3. beta 用 HalfNormal，但给足够空间

另外：本次同时修正基线量级（v11 的基线设太小）。
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
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm, get_spend_series
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


def evaluate(tag, mmm, cfg, df, y):
    pt = mmm.idata.posterior
    print(f"\n{'='*88}\n{tag}\n{'='*88}")
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    true_y = float(df["revenue"].sum())
    ch_sum = float(np.asarray(chA.values).sum())
    print(f"媒体占比 {ch_sum/true_y:.1%}  (真实 {float(df['_true_media_total'].sum()/true_y):.1%})")
    print(f"{'渠道':10s} {'估decay':>9s} {'真d':>5s} {'估slope':>9s} {'估kappa':>9s} "
          f"{'估beta':>9s} {'真β':>6s} {'估ROI':>8s} {'真ROI':>7s}")
    rows = []
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(get_spend_series(df, c).sum())
        rows.append({"渠道": c, "估ROI": g / s, "真ROI": TRUE_BETA[c]})
        print(f"{c:10s} {d:>9.3f} {TRUE_DECAY[c]:>5.2f} {sl:>9.3f} {ka:>9.3f} "
              f"{b:>9.3f} {TRUE_BETA[c]:>6.2f} {g/s:>8.3f} {TRUE_BETA[c]:>7.2f}")
    roi = pd.DataFrame(rows)
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic
    print(f"Spearman ρ = {rho:.3f}  "
          f"{'✓' if rho > 0.7 else '✗'}")
    return rho


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)

    # ---- 生成符合模型结构的 y，基线量级修正 ----
    N = len(df)
    media = np.zeros(N)
    for ch in cfg.channel_cols:
        x = df[ch].values / df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        media += TRUE_BETA[ch] * hill(ad, TRUE_KAPPA[ch], 1.2)

    # 基线：让媒体占比 ≈ 45%（贴近真实设计）
    media_level = np.median(media)
    baseline_lvl = media_level * (0.55 / 0.45)
    baseline = baseline_lvl * (0.75 + 0.25 * np.sin(2 * np.pi * np.arange(N) / 52 - 0.8))
    baseline += np.linspace(0, baseline_lvl * 0.2, N)
    rng = np.random.default_rng(11)
    df["revenue"] = baseline + media + rng.normal(0, 0.05 * media_level, N)
    print(f"生成数据: 媒体占比 ≈ {media.mean()/df['revenue'].mean():.1%}")

    X, y, meta, _ = prepare_data(df, cfg)

    # ---- 实验1：固定 slope=1.5，kappa Gamma(4,4) ----
    ov1 = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    m1 = build_model(cfg, ov1)
    _, _ = fit_mmm(m1, X, y, cfg, progressbar=False)
    rho1 = evaluate("实验1：slope≈0（HN 0.1）, kappa Gamma(4,4) 均值1", m1, cfg, df, y)

    # ---- 实验2：全固定 slope，kappa 紧 ----
    ov2 = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.05, dims="channel"),
        "saturation_kappa": Prior("HalfNormal", sigma=0.8, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    m2 = build_model(cfg, ov2)
    _, _ = fit_mmm(m2, X, y, cfg, progressbar=False)
    rho2 = evaluate("实验2：slope更紧 + kappa HN(0.8)", m2, cfg, df, y)

    # ---- 实验3：全部使用默认先验（对照）----
    m3 = build_model(cfg, {})
    _, _ = fit_mmm(m3, X, y, cfg, progressbar=False)
    rho3 = evaluate("实验3：默认先验（对照）", m3, cfg, df, y)

    print(f"\n{'='*88}\n汇总\n{'='*88}")
    print(f"实验1 (slope固定, kappa松): ρ = {rho1:.3f}")
    print(f"实验2 (slope固定, kappa紧): ρ = {rho2:.3f}")
    print(f"实验3 (默认先验)         : ρ = {rho3:.3f}")


if __name__ == "__main__":
    main()
