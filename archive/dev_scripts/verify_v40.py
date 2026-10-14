"""
验证 v40：模型内部的 channel_contribution 是怎么算出来的？
========================================================
前面所有脚本都在用 get_contributions 的**输出**。
现在直接进模型内部，验证:
  channel_contribution == beta × Hill(adstock(x), kappa, slope) ?

如果不相等，说明我对模型结构的理解有偏差 —— 那就是 38 个脚本都没找到的根因。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC
from mmm_core import (MMMConfig, normalize_columns, prepare_data, fit_mmm, get_target_scale, build_mmm_safe)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

mmm, note = build_mmm_safe(cfg, control_cols=uc)
idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)
pt = mmm.idata.posterior
ts = get_target_scale(mmm)

print("=" * 84)
print("[1] 模型内部的中间量是否都存在")
print("=" * 84)
print(f"  posterior 变量: {list(pt.data_vars)}")
cd = mmm.idata.constant_data
print(f"  constant_data: {list(cd.data_vars)}")

print("\n" + "=" * 84)
print("[2] 手工复算 channel_contribution")
print("=" * 84)
# 取后验均值
beta = pt["saturation_beta"].values.mean(axis=(0, 1))       # (channel,)
kappa = pt["saturation_kappa"].values.mean(axis=(0, 1))
slope = pt["saturation_slope"].values.mean(axis=(0, 1))
decay = pt["adstock_alpha"].values.mean(axis=(0, 1))

# 渠道数据（标准化后）
ch_raw = np.asarray(cd["channel_data"].values)   # (date, channel)
n = ch_raw.shape[0]
print(f"  channel_data shape={ch_raw.shape}, 各列均值={ch_raw.mean(axis=0).round(4)}")

# 手工 adstock（几何）
def geo_ad(x, d, l_max=6):
    out = np.zeros(len(x))
    for i in range(len(x)):
        s = 0.0
        for lag in range(l_max):
            if i - lag >= 0:
                s += (d ** lag) * x[i - lag]
        out[i] = s
    return out

# 手工 Hill
def hill(x, k, s):
    return (x ** s) / (x ** s + k ** s)

cc_model = pt["channel_contribution"].values.mean(axis=(0, 1))  # (date, channel)
print(f"  channel_contribution shape={cc_model.shape}")

print(f"\n  {'渠道':10s} {'beta':>8s} {'kappa':>8s} {'slope':>8s} {'decay':>8s} "
      f"{'手工均值':>11s} {'模型均值':>11s} {'比值':>8s}")
for i, c in enumerate(CHANNELS):
    ad = geo_ad(ch_raw[:, i], decay[i])
    h = hill(ad, kappa[i], slope[i])
    manual = beta[i] * h
    print(f"  {c:10s} {beta[i]:>8.3f} {kappa[i]:>8.3f} {slope[i]:>8.3f} {decay[i]:>8.3f} "
          f"{manual.mean():>11.5f} {cc_model[:, i].mean():>11.5f} "
          f"{manual.mean()/cc_model[:,i].mean() if cc_model[:,i].mean()!=0 else 0:>8.3f}")

print("\n" + "=" * 84)
print("[3] 关键：adstock 后的 x 均值 vs 原始 x 均值")
print("=" * 84)
print(f"  {'渠道':10s} {'原始x均值':>12s} {'adstock后均值':>14s} {'放大倍数':>10s} {'Hill(ad)':>10s} {'Hill(1)':>10s}")
for i, c in enumerate(CHANNELS):
    xm = ch_raw[:, i].mean()
    ad = geo_ad(ch_raw[:, i], decay[i])
    adm = ad.mean()
    h_ad = hill(ad, kappa[i], slope[i]).mean()
    h_1 = (1.0 ** slope[i]) / (1.0 ** slope[i] + kappa[i] ** slope[i])
    print(f"  {c:10s} {xm:>12.4f} {adm:>14.4f} {adm/xm:>10.3f} {h_ad:>10.4f} {h_1:>10.4f}")

print("\n" + "=" * 84)
print("[4] 真实生成式 vs 模型结构 对比")
print("=" * 84)
print("  真实（data_generator）:")
print("    贡献_c(t) = TRUE_BETA_c × mean(x_c) × [ad_t·Hill_t] / mean(ad·Hill)")
print("    → 归一化使 sum(贡献_c) = TRUE_BETA_c × sum(x_c)")
print()
print("  模型（pymc-marketing）:")
print("    channel_contribution_c(t) = beta_c × Hill(adstock(x_c,t))")
print("    → **没有归一化**！")
print()
print("  【关键差异】模型的 channel_contribution 直接 = beta × Hill(ad)，")
print("  而 Hill(ad) 的典型值约 0.2~0.5，β 约 0.1~3。")
print("  所以 channel_contribution 的均值 ≈ beta × 0.3。")
print()
print(f"  实测各渠道 beta × Hill(ad) 均值:")
for i, c in enumerate(CHANNELS):
    ad = geo_ad(ch_raw[:, i], decay[i])
    h = hill(ad, kappa[i], slope[i]).mean()
    print(f"    {c:10s} {beta[i]:.3f} × {h:.4f} = {beta[i]*h:.5f}")
print(f"\n  Σ(模型贡献) 标准化空间 = {sum(beta[i]*hill(geo_ad(ch_raw[:,i],decay[i]),kappa[i],slope[i]).mean() for i in range(5)):.4f}")
print(f"  真实媒体占比 × (y_mean/y_max) = 0.469 × {y.mean()/ts:.4f} = {0.469*y.mean()/ts:.4f}")
print(f"  模型 Σ贡献 × ts = {sum(beta[i]*hill(geo_ad(ch_raw[:,i],decay[i]),kappa[i],slope[i]).mean() for i in range(5))*n*ts:,.0f}")
print(f"  真实媒体总贡献 = {df['_true_media_total'].sum():,.0f}")
