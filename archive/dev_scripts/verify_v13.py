"""
验证 v13：修正评估代码 + 确认实验1 的成功
==========================================
v12 实验1的 beta 回收 RESULTS 非常漂亮：
  facebook 1.238 vs 1.10
  google   1.301 vs 1.45
  tiktok   2.137 vs 2.30
  youtube  1.001 vs 0.65
  kol      1.228 vs 1.20
但估ROI 显示 0.000 → 评估代码 bug（get_contributions 的取值方式）。

本脚本：
  1. 修正评估（用正确的 sel + 检查 chA 结构）
  2. 复现实验1，确认 ROI 排序
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


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    N = len(df)

    media = np.zeros(N)
    for ch in cfg.channel_cols:
        x = df[ch].values / df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        media += TRUE_BETA[ch] * hill(ad, TRUE_KAPPA[ch], 1.2)

    media_level = np.median(media)
    baseline_lvl = media_level * (0.55 / 0.45)
    baseline = baseline_lvl * (0.75 + 0.25 * np.sin(2 * np.pi * np.arange(N) / 52 - 0.8))
    baseline += np.linspace(0, baseline_lvl * 0.2, N)
    rng = np.random.default_rng(11)
    df["revenue"] = baseline + media + rng.normal(0, 0.05 * media_level, N)

    # 真实媒体贡献（元）
    # media 是"标准化空间"的贡献，需换算到元：真实贡献 = media × 1（因为生成时直接用 media）
    true_media = media.copy()
    true_y = float(df["revenue"].sum())
    print(f"生成数据: media均值={media.mean():.3f}  revenue均值={df['revenue'].mean():,.0f}")
    print(f"真实媒体占比 = {true_media.sum()/true_y:.1%}")

    X, y, meta, _ = prepare_data(df, cfg)

    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    mmm = build_model(cfg, ov)
    _, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = mmm.idata.posterior

    # ---- 检查 get_contributions 结构 ----
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    print(f"\nget_contributions['channels'] shape={chA.shape} dims={chA.dims}")
    print(f"  coords: {list(chA.coords)}")
    print(f"  各渠道总量:")
    spend_sum_map = {}
    for c in cfg.channel_cols:
        ser = df[c]  # 裸名（normalize_columns 已改）
        spend_sum_map[c] = float(ser.sum())
        v = np.asarray(chA.sel(channel=c).values)
        print(f"    {c:10s} 贡献sum={v.sum():>16,.0f}  花费sum={spend_sum_map[c]:>14,.0f}  "
              f"ROI={v.sum()/spend_sum_map[c]:.3f}")

    # ---- ROI 表 ----
    rows = []
    for c in cfg.channel_cols:
        v = float(np.asarray(chA.sel(channel=c).values).sum())
        rows.append({"渠道": c, "估贡献万": v / 1e4, "花费万": spend_sum_map[c] / 1e4,
                     "估ROI": v / spend_sum_map[c], "真ROI": TRUE_BETA[c]})
    roi = pd.DataFrame(rows)
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic
    print("\n" + "=" * 88)
    print(roi.round(3).to_string(index=False))
    print(f"Spearman ρ = {rho:.3f}  {'✓ 排序正确' if rho > 0.7 else '✗'}")

    ch_sum = float(np.asarray(chA.values).sum())
    print(f"\n媒体占比 = {ch_sum/true_y:.1%}  (真实 {true_media.sum()/true_y:.1%})")

    # ---- 参数回收 ----
    print("\n参数回收:")
    print(f"{'渠道':10s} {'估decay':>9s} {'真d':>5s} {'估slope':>9s} {'估kappa':>9s} "
          f"{'估beta':>9s} {'真β':>6s} {'β偏差':>8s}")
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"{c:10s} {d:>9.3f} {TRUE_DECAY[c]:>5.2f} {sl:>9.3f} {ka:>9.3f} "
              f"{b:>9.3f} {TRUE_BETA[c]:>6.2f} {b/TRUE_BETA[c]-1:>+8.1%}")


if __name__ == "__main__":
    main()
