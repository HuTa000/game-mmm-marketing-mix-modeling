"""
验证 v29：截距为何仍为负？控制变量 gammas 是否过大？
=====================================================
线索：
  1. 截距 -0.108（标准化空间），乘 target_scale 后是负收入 → 不可能
  2. 媒体贡献 133% → 说明有分量在"倒扣"
  3. 数据升级后 OLS 已能识别（ρ=0.9），模型却还错

怀疑：控制变量（特别是 trend）把基线吃掉了，
      导致 intercept 被夹到负值。
      trend 是 0~1 线性增长，与 y 高度共线！
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      build_mmm_safe, fit_mmm, diagnose, get_target_scale)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

print("=" * 70)
print("[1] y 的基本情况")
print("=" * 70)
print(f"  y_mean = {y.mean():,.0f}   y_max = {y.max():,.0f}   y_min = {y.min():,.0f}")
print(f"  y_mean / y_max = {y.mean()/y.max():.4f}")

print("\n" + "=" * 70)
print("[2] 控制变量的量级与共线性（关键怀疑点）")
print("=" * 70)
ctrl_raw = df[["version_update", "competitor_activity", "holiday", "trend"]]
print(ctrl_raw.describe().round(3).to_string())
print("\n  与控制变量的相关系数（原始）：")
for c in uc:
    r = np.corrcoef(df[c].astype(float), y.astype(float))[0, 1]
    print(f"    {c:22s} corr(y) = {r:+.3f}")

print("\n  标准化后的控制变量（进 X 的实际值）：")
print(X[uc].describe().round(3).to_string())

print("\n" + "=" * 70)
print("[3] 关键诊断：OLS 把 y 用【纯控制变量】回归，看能解释多少")
print("=" * 70)
import statsmodels.api as sm
Xc = sm.add_constant(X[uc].astype(float))
r_ctrl = sm.OLS(y.astype(float), Xc).fit()
print(f"  仅控制变量 R² = {r_ctrl.rsquared:.4f}")
print(f"  截距 = {r_ctrl.params['const']:,.0f}  (= {r_ctrl.params['const']/y.max():.4f} 标准化)")

Xall = sm.add_constant(X[uc + cfg.channel_cols].astype(float))
r_all = sm.OLS(y.astype(float), Xall).fit()
print(f"\n  控制+渠道 R² = {r_all.rsquared:.4f}")
print(f"  截距 = {r_all.params['const']:,.0f}  (= {r_all.params['const']/y.max():.4f} 标准化)")
print("\n  各控制变量系数（对 y 的元/单位）：")
for c in uc:
    print(f"    {c:22s} {r_all.params[c]:>14,.0f}")

print("\n" + "=" * 70)
print("[4] 建模")
print("=" * 70)
mmm, note = build_mmm_safe(cfg, control_cols=uc)
print(f"  {note}")
idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)
print(f"  采样 {el:.1f}s")

ts = get_target_scale(mmm)
print(f"  target_scale = {ts:,.0f}   (= y.max()? {abs(ts - y.max())/y.max() < 0.01})")

pt = mmm.idata.posterior
print("\n" + "=" * 70)
print("[5] 后验：截距 + 控制变量 gammas（重点看是否异常大）")
print("=" * 70)
ic = pt["intercept_contribution"].values.mean()
print(f"  intercept_contribution = {ic:+.4f}  (×ts = {ic*ts:+,.0f} 元/周)")
print(f"    业务含义：每周基线应为正数（老玩家自然付费），实测为负 → 异常")

print("\n  控制变量 gamma（标准化空间）：")
for c in uc:
    for v in [f"gamma_control[{c}]", f"gamma_control_{c}"]:
        if v in pt:
            g = pt[v].values.mean()
            print(f"    {c:22s} gamma = {g:+.4f}  (×ts = {g*ts:+,.0f} 元)")
            break
    else:
        pass
print("  posterior 中所有变量:", [v for v in pt.data_vars if "gamma" in v or "intercept" in v])

# 检查 trend 是否吃掉了基线
print("\n" + "=" * 70)
print("[6] 结论性诊断：trend 与基线的关系")
print("=" * 70)
true_bl = df["_true_baseline"].values
tr = df["trend"].values
print(f"  corr(trend, 真实基线) = {np.corrcoef(tr, true_bl)[0,1]:+.4f}")
print(f"  真实基线: 均值={true_bl.mean():,.0f}  范围[{true_bl.min():,.0f}, {true_bl.max():,.0f}]")
print(f"  真实基线的趋势幅度占比 = {(true_bl.max()-true_bl.min())/true_bl.mean():.1%}")
print(f"  → trend 变量几乎就是基线的线性部分，两者高度共线")
