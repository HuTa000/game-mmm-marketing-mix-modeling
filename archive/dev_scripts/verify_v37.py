"""
验证 v37：intercept_contribution 到底是什么？为什么要为负就错？
==============================================================
前面 36 个脚本我一直假设"截距必须为正"，但从没验证过这个假设。
现在直接看源码：intercept_contribution 是如何从 intercept 算出来的。
"""
import os, sys, warnings, inspect
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

print("=" * 78)
print("[1] 源码：intercept_contribution 怎么算的")
print("=" * 78)
from pymc_marketing.mmm import base
src = inspect.getsource(base)
lines = src.split("\n")
for i, l in enumerate(lines):
    if "intercept_contribution" in l or ("intercept" in l and "def " in l):
        print(f"  L{i}: {l}")

print("\n" + "=" * 78)
print("[2] 找 intercept 相关的属性/方法定义")
print("=" * 78)
for name in dir(MMM):
    if "intercept" in name.lower():
        print(f"  {name}")

print("\n" + "=" * 78)
print("[3] 打印 get_contributions 的源码")
print("=" * 78)
try:
    from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
    s2 = inspect.getsource(MMMIDataWrapper)
    for i, l in enumerate(s2.split("\n")):
        if "def get_contributions" in l or "def _get_contributions" in l:
            print(f"  L{i}: {l}")
except Exception as e:
    print("  ", e)

print("\n" + "=" * 78)
print("[4] 实测：intercept_contribution 与 intercept × target_scale 的关系")
print("=" * 78)
from data_generator import generate_game_data
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm, get_target_scale

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
          target_column=cfg.target_col,
          adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
          saturation=HillSaturation(), yearly_seasonality=2, control_columns=uc)
b = MMM(**kw); m = dict(b.default_model_config)
m.update({
    "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
    "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
    "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
    "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    "intercept": Prior("Normal", mu=cfg.intercept_mu, sigma=cfg.intercept_sigma, dims=()),
    "gamma_control": Prior("Normal", mu=0.0, sigma=0.2, dims="control"),
})
kw["model_config"] = m
mmm = MMM(**kw)
fit_mmm(mmm, X, y, cfg, progressbar=False)

pt = mmm.idata.posterior
ts = get_target_scale(mmm)
print(f"  target_scale = {ts:,.0f}")
print(f"  posterior 变量: {list(pt.data_vars)}")

ic = pt["intercept_contribution"].values
print(f"\n  intercept_contribution: mean={ic.mean():+.4f} std={ic.std():.4f}")
print(f"  × target_scale = {ic.mean()*ts:+,.0f} 元/周")

# constant_data 里有没有原始 intercept
cd = mmm.idata.constant_data
print(f"\n  constant_data 变量: {list(cd.data_vars)}")
for v in cd.data_vars:
    arr = np.asarray(cd[v].values)
    print(f"    {v}: shape={arr.shape} mean={arr.mean():.4f}"
          + (f"  → {arr.ravel()[:3]}" if arr.size <= 6 else ""))

print("\n" + "=" * 78)
print("[5] 关键：直接看 y 的标准化方式")
print("=" * 78)
print(f"  y.mean()={y.mean():,.0f}  y.max()={y.max():,.0f}  y.min()={y.min():,.0f}")
print(f"  若 y_std = y / target_scale，则 y_std ∈ [{y.min()/ts:.4f}, {y.max()/ts:.4f}]")
print(f"  intercept_contribution 后验 = {ic.mean():+.4f}")
print("  → 若 intercept 代表每周基线在标准化空间的水平")
print(f"    那 y_std 的均值是 {y.mean()/ts:.4f}，截距应当在 0~{y.mean()/ts:.4f} 之间")
print(f"    实测 {ic.mean():+.4f} 为负 → 确实异常（除非定义不同）")

# 检查 mu 的构成
print("\n" + "=" * 78)
print("[6] 直接验证 mu 的分解")
print("=" * 78)
n = len(y)
mu = pt["mu"].values.mean(axis=(0, 1))
cc = pt["channel_contribution"].values.mean(axis=(0, 1))   # (date, channel)
icc = pt["intercept_contribution"].values.mean()
ys = pt["yearly_seasonality_contribution"].values.mean(axis=(0, 1))
fs = pt["fourier_contribution"].values.mean(axis=(0, 1))
print(f"  mu      前5期: {mu[:5]}")
print(f"  icc     常数 : {icc:+.4f}")
print(f"  cc合计  前5期: {cc[:5].sum(axis=1)}")
print(f"  ys      前5期: {ys[:5]}")
print(f"  fs      前5期: {fs[:5]}")
print(f"\n  icc + cc + ys + fs 前5期 = {icc + cc[:5].sum(axis=1) + ys[:5] + fs[:5]}")
print(f"  mu          前5期 = {mu[:5]}")
print(f"  → 是否一致: {np.allclose(icc + cc[:5].sum(axis=1) + ys[:5] + fs[:5], mu[:5], atol=1e-4)}")

print("\n  ** 若 icc 是每期都加的常数，则 icc + cc + ys + fs = mu 应成立 **")
print(f"  icc 出现在 mu 里的次数 = 1（每期加一次）")
print(f"  那 icc={icc:+.4f} 表示每期基线，ytd 累计 = {icc*n:+.2f}")
print(f"  y_std 总和 = {y.sum()/ts:.2f}")
print(f"  icc*n = {icc*n:+.2f} 占 y_std 总和 {icc*n/(y.sum()/ts):.1%}")
print(f"  真实基线占比 = {df['_true_baseline'].sum()/y.sum():.1%}")
