"""
验证 v16：终极实验 —— 不预先标准化渠道，看官方公式是否自洽
==============================================================
v15 关键数据：
  · channel_contribution sum = 451.51
  · predict(X) sum = 112.71
  · 比值 = 4.005
  · target_scale = 2,113,874 （y 均值 1,527,233）
  · saturation_beta 均值 = 0.7225   ← 恰好 = y_mean/ts = 0.7225

【重大推断】
  PyMC-Marketing 1.2.0 内部：
    y_internal = y / target_scale        ← target_scale ≈ y_mean × 1.384
    channel_contribution 是 y_internal 空间的量

  而我们把渠道先除以了自身均值（标准化到均值1），
  这一步让模型"看不到"渠道的绝对量级，
  但 pymc-marketing 内部又计算了 channel_scale（= ?）再除一次。

  结果：贡献被两次"相对化"，乘 target_scale 还原就多了。

【终极测试】不预先标准化渠道，直接用原始花费输入模型，
          让 pymc-marketing 自己处理尺度。看 get_contributions 是否自洽。
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
from mmm_core import MMMConfig, normalize_columns, fit_mmm
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


def run(tag, X, y, df, cfg, true_media_series, overrides):
    mmm = build_model(cfg, overrides)
    fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = mmm.idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    true_y = float(df["revenue"].sum())
    true_media = float(sum(s.sum() for s in true_media_series.values()))

    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    ch_sum = float(np.asarray(chA.values).sum())

    print(f"\n{'='*88}\n{tag}\n{'='*88}")
    print(f"target_scale={ts:,.0f}  y_mean={float(y.mean()):,.0f}")
    print(f"官方 get_contributions 媒体占比 = {ch_sum/true_y:.1%}  (真实 {true_media/true_y:.1%})")
    print(f"\n{'渠道':10s} {'估ROI':>9s} {'真ROI':>8s}")
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(df[c].sum())
        rows.append({"渠道": c, "估ROI": g / s, "真ROI": TRUE_BETA[c]})
        print(f"{c:10s} {g/s:>9.3f} {TRUE_BETA[c]:>8.2f}")
    rho = spearmanr(pd.DataFrame(rows)["估ROI"], pd.DataFrame(rows)["真ROI"]).statistic
    print(f"Spearman ρ = {rho:.3f} {'✓' if rho > 0.7 else '✗'}")
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
    media_weekly = media.mean()
    baseline_weekly = media_weekly * (0.55 / 0.45)
    baseline = baseline_weekly * (0.85 + 0.15*np.sin(2*np.pi*np.arange(N)/52-0.8))
    baseline += np.linspace(0, baseline_weekly * 0.15, N)
    df["revenue"] = baseline + media + np.random.default_rng(11).normal(0, 0.04*media_weekly, N)
    y = df["revenue"].astype(float)

    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }

    # ---- 方案 A：渠道标准化（×均值1），y 原始 ----
    XA = df[["date"] + cfg.channel_cols].copy()
    for c in cfg.channel_cols:
        XA[c] = XA[c] / XA[c].mean()
    for c in cfg.control_cols:
        if c in df.columns:
            XA[c] = (df[c] - df[c].mean()) / (df[c].std() or 1)
    run("方案A：渠道标准化(÷均值)，y 原始", XA, y, df, cfg, media_series, ov)

    # ---- 方案 B：渠道 ÷ 1e6（保留相对量级，缩小到 ~1 附近）----
    XB = df[["date"] + cfg.channel_cols].copy()
    for c in cfg.channel_cols:
        XB[c] = XB[c] / 1e6
    for c in cfg.control_cols:
        if c in df.columns:
            XB[c] = (df[c] - df[c].mean()) / (df[c].std() or 1)
    print(f"\n[方案B] 渠道值域: {XB[cfg.channel_cols].min().min():.3f} ~ {XB[cfg.channel_cols].max().max():.3f}")
    run("方案B：渠道÷1e6（保留渠道间相对量级）", XB, y, df, cfg, media_series, ov)

    # ---- 方案 C：y 也标准化 ----
    yC = y / y.mean()
    run("方案C：渠道标准化 + y 标准化", XA, yC, df, cfg, media_series, ov)


if __name__ == "__main__":
    main()
