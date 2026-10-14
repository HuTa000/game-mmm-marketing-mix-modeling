"""
验证 v30：intercept 先验到底有没有生效？
=========================================
上一步发现：设了 model_config["intercept"]，但后验里只有
intercept_contribution，没有 intercept。需要确认二者关系。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

print("=" * 70)
print("[1] default_model_config 的全部键")
print("=" * 70)
mmm0 = MMM(date_column="date", channel_columns=["a", "b"],
           target_column="y", adstock=GeometricAdstock(l_max=6),
           saturation=HillSaturation(), yearly_seasonality=2)
dc = mmm0.default_model_config
for k, v in dc.items():
    print(f"  {k:24s} = {v}")

print("\n" + "=" * 70)
print("[2] 设 intercept 先验后，模型里实际有哪些自由变量")
print("=" * 70)
mc = dict(dc)
mc.update({
    "intercept": Prior("Normal", mu=0.10, sigma=0.05, dims=()),
})
mmm = MMM(date_column="date", channel_columns=["a", "b"],
          target_column="y", adstock=GeometricAdstock(l_max=6),
          saturation=HillSaturation(), yearly_seasonality=2,
          model_config=mc)
print("  模型自由变量:", list(mmm.model.free_RVs))
print()
for rv in mmm.model.free_RVs:
    print(f"    {rv.name:26s} dist={rv.owner.opclass}")

print("\n" + "=" * 70)
print("[3] 如果 intercept 不是自由变量 → 说明它改的是别的")
print("=" * 70)
has_i = any("intercept" in rv.name for rv in mmm.model.free_RVs)
print(f"  含 intercept 的自由变量: {has_i}")

print("\n" + "=" * 70)
print("[4] 检查 MMM 内部是否用 target_scale 缩放 y（关键）")
print("=" * 70)
d = pd.DataFrame({
    "date": pd.date_range("2023-01-02", periods=20, freq="W"),
    "a": np.linspace(1, 3, 20), "b": np.linspace(2, 4, 20),
    "y": np.linspace(500, 1500, 20) * 1000,
})
try:
    mmm2.prior_predictive  # noqa
except Exception:
    pass
mm_fit = MMM(date_column="date", channel_columns=["a", "b"], target_column="y",
             adstock=GeometricAdstock(l_max=6), saturation=HillSaturation(),
             yearly_seasonality=2, model_config=mc)
print("  定义 y 的观测节点的 mu 是否被缩放？看 mu 的量级")
print("  用最小 MCMC 采样 20 步做先验检查...")
try:
    pp = mm_fit.sample_prior_predictive(X=d, y=d["y"], samples=20, random_seed=0)
    mu_prior = pp.prior["mu"].values
    print(f"    mu 先验范围: [{np.nanmin(mu_prior):,.2f}, {np.nanmax(mu_prior):,.2f}]")
    print(f"    y 的量级: mean={d['y'].mean():,.0f}")
    print(f"    → 若 mu 比 y 小 1000 倍，说明 y 被 target_scale 缩放了")
except Exception as e:
    print(f"    先验采样失败: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("[5] MMM 内部的 target_scale 取值逻辑")
print("=" * 70)
import inspect
try:
    src = inspect.getsource(MMM.__init__)
    key = [l.strip() for l in src.split("\n") if "target_scale" in l or "scale" in l.lower()]
    for l in key[:20]:
        print("   ", l)
except Exception as e:
    print("  ", e)

# 找出 target_scale 相关的方法
for name in dir(mmm0):
    if "scale" in name.lower() or "data" == name:
        print(f"  method: {name}")
