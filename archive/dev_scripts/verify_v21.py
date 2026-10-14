"""
验证 v21：一次搞清 —— 控制变量/intercept 是否真的生效
========================================================
v20 发现两次运行结果完全一致 → 怀疑参数没生效。

本脚本用最小实验直接检查：
  1. 建模后 posterior 里有没有 gamma_control（控制变量系数）
  2. intercept 先验换了以后，intercept_contribution 的后验均值有没有变
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

def build(cfg, ov, use_controls):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality)
    if use_controls:
        kw["control_columns"] = cfg.control_cols
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)
print("prepare_data 返回的 used_controls:", uc)
print("X 的列:", list(X.columns))

# --- A: 不带控制变量，intercept = N(0, 2)（默认）---
ovA = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
       "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
       "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
       "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel")}
mA = build(cfg, ovA, use_controls=False)
fit_mmm(mA, X, y, cfg, progressbar=False)
postA = list(mA.idata.posterior.data_vars)
print("\n[A] 无控制变量:")
print("   posterior vars:", postA)
print("   gamma_control 存在?", "gamma_control" in postA)
print("   intercept_contribution 均值 =",
      float(mA.idata.posterior["intercept_contribution"].values.mean()))

# --- B: 不带控制变量，intercept = N(0.45, 0.02) ---
ovB = dict(ovA); ovB["intercept"] = Prior("Normal", mu=0.45, sigma=0.02)
mB = build(cfg, ovB, use_controls=False)
fit_mmm(mB, X, y, cfg, progressbar=False)
print("\n[B] 无控制变量, intercept N(0.45,0.02):")
print("   intercept_contribution 均值 =",
      float(mB.idata.posterior["intercept_contribution"].values.mean()))
print("   => 若与 [A] 不同，说明 intercept 先验生效")

# --- C: 带控制变量 ---
mC = build(cfg, ovB, use_controls=True)
fit_mmm(mC, X, y, cfg, progressbar=False)
postC = list(mC.idata.posterior.data_vars)
print("\n[C] 带控制变量:")
print("   gamma_control 存在?", "gamma_control" in postC)
if "gamma_control" in postC:
    print("   gamma_control 均值:", mC.idata.posterior["gamma_control"].values.mean(axis=(0,1)))
print("   intercept_contribution 均值 =",
      float(mC.idata.posterior["intercept_contribution"].values.mean()))
