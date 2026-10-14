"""
验证 v18：确认"负截距"是 ROOT CAUSE，并给出修正方案
======================================================
v17 决定性发现：
  mu.sum()              = 100.09
  intercept × 156       = -391.6   ← 截距是"每周"值，负得离谱
  Σchannel_contribution = 491.64
  验证：-391.6 + 491.64 = 100.04 ✓ 可加性成立！

【ROOT CAUSE 确认】
  截距被估成 -2.51/周（= -391.6 总计），
  迫使媒体贡献膨胀到 491.6 才能凑出总量 100.1。
  于是"媒体占比"从真实的 47% 虚高到 400%（约 8.5 倍）。

【为什么截距会为负】
  渠道量级差异 + 先验不当。虽然我们已把渠道 ÷ 自身均值，
  但模型的 intercept 先验是 Normal(0, 2)（wided），
  在 identity link + 大 target_scale 下，截距需要承担"基线"角色，
  却被渠道共线性推成负数。

【修正方案】
  1. 提高 intercept 先验均值/收紧 —— 让基线由截距承担
  2. 关键：把"基线"从渠道里分离出来 —— 用 control 或让 intercept 有正先验
  3. 最佳解：给 intercept 一个**基于数据的正先验**（如 Normal(y_mean/ts, ...)）

本脚本测试 3 种 intercept 先验设置。
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
        carry = x[i] + decay*carry; out[i] = carry
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


def run(tag, mmm, df, cfg, media_series, y):
    pt = mmm.idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    true_y = float(df["revenue"].sum())
    true_media = float(sum(s.sum() for s in media_series.values()))

    ic = float(pt["intercept_contribution"].values.mean())
    n = len(df)
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    ch_sum = float(np.asarray(chA.values).sum())

    print(f"\n{'='*88}\n{tag}\n{'='*88}")
    print(f"  intercept = {ic:.3f}/周  → 总计 {ic*n:,.1f}")
    print(f"  Σcc = {float(pt['channel_contribution'].values.mean(axis=(0,1)).sum()):.1f}")
    print(f"  官方 get_contributions 媒体占比 = {ch_sum/true_y:.1%}  (真实 {true_media/true_y:.1%})")
    print(f"  {'渠道':10s} {'估ROI':>9s} {'真ROI':>8s}")
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(df[c].sum())
        rows.append({"渠道": c, "估ROI": g/s, "真ROI": TRUE_BETA[c]})
        print(f"  {c:10s} {g/s:>9.3f} {TRUE_BETA[c]:>8.2f}")
    rho = spearmanr(pd.DataFrame(rows)["估ROI"], pd.DataFrame(rows)["真ROI"]).statistic
    print(f"  Spearman ρ = {rho:.3f} {'✓' if rho > 0.7 else '✗'}")
    return rho, ch_sum / true_y


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    N = len(df)

    media_series = {}
    for ch in cfg.channel_cols:
        x = df[ch].values / df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        h = hill(ad, TRUE_KAPPA[ch], 1.2)
        media_series[ch] = TRUE_BETA[ch] * df[ch].sum() / h.sum() * h
    media = sum(media_series.values())
    baseline_weekly = media.mean() * (0.55/0.45)
    baseline = baseline_weekly * (0.85 + 0.15*np.sin(2*np.pi*np.arange(N)/52-0.8))
    baseline += np.linspace(0, baseline_weekly*0.15, N)
    df["revenue"] = baseline + media + np.random.default_rng(11).normal(0, 0.04*media.mean(), N)
    y = df["revenue"].astype(float)

    X, _, _, _ = prepare_data(df, cfg)
    ts_est = y.mean() / 1.0     # 粗估

    base_ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }

    # ---- 方案1：默认 intercept 先验（对照）----
    m1 = build_model(cfg, dict(base_ov))
    fit_mmm(m1, X, y, cfg, progressbar=False)
    rho1, s1 = run("方案1：默认 intercept 先验", m1, df, cfg, media_series, y)

    # ---- 方案2：给 intercept 一个基于数据的正先验 ----
    # 内部 y 被除以 target_scale(~1.38×y_mean)，基线占比约55%
    # 所以 intercept 在内部空间 ≈ 0.55 / 1.384 ≈ 0.40
    ov2 = dict(base_ov)
    ov2["intercept"] = Prior("Normal", mu=0.40, sigma=0.15)
    m2 = build_model(cfg, ov2)
    fit_mmm(m2, X, y, cfg, progressbar=False)
    rho2, s2 = run("方案2：intercept ~ Normal(0.40, 0.15)", m2, df, cfg, media_series, y)

    # ---- 方案3：更紧的正先验 ----
    ov3 = dict(base_ov)
    ov3["intercept"] = Prior("Normal", mu=0.45, sigma=0.08)
    m3 = build_model(cfg, ov3)
    fit_mmm(m3, X, y, cfg, progressbar=False)
    rho3, s3 = run("方案3：intercept ~ Normal(0.45, 0.08)", m3, df, cfg, media_series, y)

    print(f"\n{'='*88}\n汇总\n{'='*88}")
    print(f"{'方案':30s} {'媒体占比':>12s} {'Spearman':>10s}")
    print(f"{'真实':30s} {0.469:>12.1%} {1.0:>10.3f}")
    print(f"{'1 默认intercept':30s} {s1:>12.1%} {rho1:>10.3f}")
    print(f"{'2 intercept Normal(0.4,0.15)':30s} {s2:>12.1%} {rho2:>10.3f}")
    print(f"{'3 intercept Normal(0.45,0.08)':30s} {s3:>12.1%} {rho3:>10.3f}")


if __name__ == "__main__":
    main()
