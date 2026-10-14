"""
验证 v15：最终确定贡献还原公式（一次性搞清）
==============================================
v14 现象：
  · saturation_beta 回收近乎完美（tiktok -0.4%）
  · 但 get_contributions 的贡献虚高 ~7 倍
  · 媒体占比 400.6% vs 真实 46.9%

关键线索：v13（y 量级=7）时 get_contributions sum=920 ≈ y sum=1092，
          v14（y 量级=1.5e6）时 get_contributions sum 虚高 7 倍。

→ 说明 get_contributions 的 original_scale 乘 target_scale 是**正确的**，
  但我们的数据有两层缩放，导致重复。

本脚本对同一模型，直接对比 4 种候选公式，找出哪个媒体占比 = 46.9%：
  F1: channel_contribution (标准化空间, 不还原)
  F2: channel_contribution × target_scale
  F3: get_contributions["channels"]
  F4: channel_contribution × y_mean / target_scale
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

    media_series = {}
    for ch in cfg.channel_cols:
        x = df[ch].values / df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        h = hill(ad, TRUE_KAPPA[ch], 1.2)
        media_series[ch] = TRUE_BETA[ch] * df[ch].sum() / h.sum() * h
    media = sum(media_series.values())
    media_weekly = media.mean()
    baseline_weekly = media_weekly * (0.55 / 0.45)
    baseline = baseline_weekly * (0.85 + 0.15 * np.sin(2*np.pi*np.arange(N)/52 - 0.8))
    baseline += np.linspace(0, baseline_weekly * 0.15, N)
    df["revenue"] = baseline + media + np.random.default_rng(11).normal(0, 0.04*media_weekly, N)

    true_y = float(df["revenue"].sum())
    true_media = float(media.sum())
    print(f"真实: y_sum={true_y:,.0f}  media_sum={true_media:,.0f}  占比={true_media/true_y:.1%}")

    X, y, meta, _ = prepare_data(df, cfg)
    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    mmm = build_model(cfg, ov)
    fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = mmm.idata.posterior

    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    cc = pt["channel_contribution"].values  # (chain,draw,date,channel)
    y_mean = float(y.mean())
    print(f"\ntarget_scale = {ts:,.2f}   y_mean = {y_mean:,.2f}   ts/y_mean = {ts/y_mean:.4f}")
    print(f"channel_scale = {np.asarray(mmm.idata.constant_data['channel_scale'].values)}")

    cc_m = cc.mean(axis=(0, 1))     # (date, channel)
    print(f"\nchannel_contribution 原始: sum={cc_m.sum():,.2f}  mean={cc_m.mean():.4f}")
    print(f"predict(X) sum = {np.asarray(mmm.predict(X)).ravel().sum():,.4f}")

    print("\n" + "=" * 92)
    print("候选还原公式对比（目标：媒体占比 ≈ 46.9%）")
    print("=" * 92)
    cands = {
        "F1 不还原 cc": cc_m.sum(),
        "F2 cc × ts": cc_m.sum() * ts,
        "F3 cc × y_mean": cc_m.sum() * y_mean,
        "F4 cc × ts（含channel_scale除法）": (cc_m / np.asarray(
            mmm.idata.constant_data['channel_scale'].values)).sum() * ts,
    }
    for name, total in cands.items():
        print(f"  {name:40s} sum={total:>18,.0f}  占比={total/true_y:>8.1%}  "
              f"{'★正确' if abs(total/true_y - true_media/true_y) < 0.15 else ''}")

    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    print(f"\n  F5 官方 get_contributions: sum={np.asarray(chA.values).sum():>18,.0f}  "
          f"占比={np.asarray(chA.values).sum()/true_y:.1%}")

    # ---- 如果 F2 正确，算 ROI ----
    print("\n" + "=" * 92)
    print("用 F2（cc × target_scale）算各渠道 ROI")
    print("=" * 92)
    for c in cfg.channel_cols:
        i = cfg.channel_cols.index(c)
        g = cc_m[:, i].sum() * ts
        s = float(df[c].sum())
        print(f"  {c:10s} 贡献={g/1e4:>12,.0f} 万元  估ROI={g/s:>7.3f}  真ROI={TRUE_BETA[c]:.2f}")

    # ---- F5 官方 ----
    print("\n用 F5（官方 get_contributions）算各渠道 ROI")
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(df[c].sum())
        print(f"  {c:10s} 贡献={g/1e4:>12,.0f} 万元  估ROI={g/s:>7.3f}  真ROI={TRUE_BETA[c]:.2f}")


if __name__ == "__main__":
    main()
