"""
验证 v17：最后一击 —— 为什么 Σchannel_contribution ≠ mu？
============================================================
已知（方案A，y 为元级）：
  · channel_contribution.sum() = 451.51
  · mu.sum() (=predict sum)    = 112.71
  · 比值 = 4.005 ≈ 4.0

而 v8（y 元级、宽先验）中：
  · channel_contribution.sum() = 150.07
  · mu.sum()                   = 100.10
  · 比值 = 1.499 ≈ 1.5

两个比值不同（4.0 vs 1.5）→ 取决于先验！
说明这不是固定的缩放 bug，而是……

【新假设】channel_contribution 的维度理解错误！
  shape = (chain, draw, date, channel)
  我做了 .mean(axis=(0,1)).sum()  → 先对链/抽平均，再对 date & channel 求和
  这是对的。

  但 mu 的 shape = (chain, draw, date)
  .mean(axis=(0,1)).sum()  → 对 date 求和

  Σcc 应该 = mu（identity link 可加）。

【最可能的解释】channel_contribution 用的是**标准化后的 channel_data**，
          而 mu 用的是……同样。

本脚本逐项拆解 mu，找出缺失的项。
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
        carry = x[i] + decay*carry; out[i] = carry
    return out


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    N = len(df)
    X, y, meta, _ = prepare_data(df, cfg)   # 渠道÷均值, y 原始

    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    kwargs = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
                  target_column=cfg.target_col,
                  adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
                  saturation=HillSaturation(),
                  yearly_seasonality=cfg.yearly_seasonality)
    base = MMM(**kwargs); mc = dict(base.default_model_config); mc.update(ov)
    kwargs["model_config"] = mc
    mmm = MMM(**kwargs)
    fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = mmm.idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))

    print("=" * 90)
    print("mu 的完整拆解（标准化空间）")
    print("=" * 90)
    mu = pt["mu"].values.mean(axis=(0, 1))                  # (date,)
    cc = pt["channel_contribution"].values.mean(axis=(0, 1))  # (date, channel)
    ic = float(pt["intercept_contribution"].values.mean())
    ys = pt["yearly_seasonality_contribution"].values.mean(axis=(0, 1))
    fc = pt["fourier_contribution"].values.mean(axis=(0, 1))  # (date, 4)

    print(f"  mu.sum()                   = {mu.sum():.4f}")
    print(f"  intercept_contribution     = {ic:.4f}")
    print(f"  Σchannel_contribution.sum()= {cc.sum():.4f}   (date×channel)")
    print(f"  yearly_seasonality.sum()   = {ys.sum():.4f}")
    print(f"  fourier.sum()              = {fc.sum():.4f}")

    total_parts = ic * len(mu) + cc.sum() + ys.sum() + fc.sum()
    print(f"\n  各项之和 = {total_parts:.4f}")
    print(f"  mu.sum() = {mu.sum():.4f}")
    print(f"  差 = {mu.sum() - total_parts:.4f}")

    # 关键：channel_scale 的作用
    print("\n" + "=" * 90)
    print("channel_scale 的作用")
    print("=" * 90)
    cs = np.asarray(mmm.idata.constant_data["channel_scale"].values)
    print(f"  channel_scale = {cs}")
    print(f"  channel_data 均值（应全=1）= "
          f"{np.asarray(mmm.idata.constant_data['channel_data'].values).mean(axis=0)}")

    # 用 channel_scale 修正
    cc_fixed = cc / cs[None, :]
    print(f"\n  Σcc.sum()          = {cc.sum():.4f}")
    print(f"  Σ(cc/cs).sum()     = {cc_fixed.sum():.4f}")
    print(f"  Σ(cc×cs).sum()     = {(cc*cs[None,:]).sum():.4f}")
    print(f"  mu - intercept×n   = {mu.sum() - ic*len(mu):.4f}")

    # adstock+hill 手工重算（用 channel_data）
    print("\n" + "=" * 90)
    print("手工 adstock+hill vs 内置 cc（单渠道 facebook）")
    print("=" * 90)
    cdata = np.asarray(mmm.idata.constant_data["channel_data"].values)  # (date, 5)
    for i, c in enumerate(cfg.channel_cols[:2]):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        be = float(pt["saturation_beta"].values[:, :, i].mean())
        ad = np.array([sum(d**lag * cdata[max(0,t-lag), i]
                           for lag in range(min(6, t)+1)) for t in range(N)])
        h = hill(ad, sl, ka)
        manual = be * h
        print(f"  {c}: ad均值={ad.mean():.4f} hill均值={h.mean():.4f} "
              f"beta={be:.3f}")
        print(f"    手工 beta×hill 均值 = {manual.mean():.4f}")
        print(f"    内置 cc 均值        = {cc[:, i].mean():.4f}")
        print(f"    比值 = {cc[:, i].mean()/manual.mean():.4f}")
        print(f"    × channel_scale[{i}]={cs[i]:.4f} → "
              f"{cc[:,i].mean()/cs[i]:.4f}")

    print(f"\n  [对照] 全部渠道 cc 均值 = {cc.mean():.4f}")
    print(f"         mu 均值 = {mu.mean():.4f}")
    print(f"         真实 y 均值(标准化) = {y.mean()/ts:.4f}")


if __name__ == "__main__":
    main()
