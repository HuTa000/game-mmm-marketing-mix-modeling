"""
验证 v28：检查是不是评估代码的 bug（channel 顺序错位）
========================================================
ρ=-0.9 在 12 种配置下完全一致 → 强烈怀疑是评估代码问题，
而非模型问题（模型表现不可能这么"稳定地错"）。

重点检查：
  1. get_contributions 的 channel 坐标顺序 vs cfg.channel_cols
  2. sel(channel=c) 是否真的取到该渠道
  3. 有没有 broadcast 错位
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_GAMMA_FRAC, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

ov = {
    "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
    "saturation_slope": Prior("HalfNormal", sigma=0.15, dims="channel"),
    "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
    "saturation_beta": Prior("HalfNormal", sigma=2.0, dims="channel"),
    "intercept": Prior("Normal", mu=0.10, sigma=0.05, dims=()),
}
kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
          target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
          saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
          control_columns=uc)
b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
mmm = MMM(**kw)
fit_mmm(mmm, X, y, cfg, progressbar=False)

dsA = mmm.data.get_contributions(original_scale=True)
chA = dsA["channels"].mean(("chain", "draw"))
print("get_contributions['channels']:")
print("  dims:", chA.dims)
print("  channel coords:", list(chA.coords["channel"].values))
print("  cfg.channel_cols:", cfg.channel_cols)
print()

print("=== 各渠道贡献总量（用 sel）===")
for c in cfg.channel_cols:
    v = float(np.asarray(chA.sel(channel=c).values).sum())
    s = float(df[c].sum())
    print(f"  {c:10s} 贡献={v/1e4:>10,.0f}万  花费={s/1e4:>9,.0f}万  ROI={v/s:>6.3f}  真ROI={TRUE_BETA[c]:.2f}")

print("\n=== 各渠道贡献总量（用位置索引）===")
for i, c in enumerate(cfg.channel_cols):
    v = float(np.asarray(chA.values)[:, i].sum())
    print(f"  {c:10s} 位置{i} 贡献={v/1e4:>10,.0f}万")

print("\n=== 检查 x 的位置 vs channel 位置 ===")
cd = np.asarray(mmm.idata.constant_data["channel_data"].values)
print("  constant_data channel_data 各列均值:", cd.mean(axis=0))
print("  我们传入的 X 各列均值:", X[cfg.channel_cols].mean().values)

# 关键检验：predict 对每个渠道做反事实
print("\n=== predict 反事实（逐个把渠道置0）===")
pb = np.asarray(mmm.predict(X)).ravel()
for c in cfg.channel_cols:
    Xc = X.copy(); Xc[c] = 0.0
    pc = np.asarray(mmm.predict(Xc)).ravel()
    print(f"  {c:10s} Δpred={ (pb-pc).sum():.4f}")

print("\n=== 后验 saturation_beta / kappa / slope 各渠道 ===")
pt = mmm.idata.posterior
for i, c in enumerate(cfg.channel_cols):
    b_ = float(pt["saturation_beta"].values[:, :, i].mean())
    k_ = float(pt["saturation_kappa"].values[:, :, i].mean())
    s_ = float(pt["saturation_slope"].values[:, :, i].mean())
    d_ = float(pt["adstock_alpha"].values[:, :, i].mean())
    print(f"  {c:10s} beta={b_:>7.3f} kappa={k_:>7.3f} slope={s_:>6.3f} decay={d_:.3f}  "
          f"(真 beta={TRUE_BETA[c]:.2f} decay={TRUE_DECAY[c]:.2f} k={TRUE_KAPPA[c]:.1f} frac={TRUE_GAMMA_FRAC[c]:.1f})")
