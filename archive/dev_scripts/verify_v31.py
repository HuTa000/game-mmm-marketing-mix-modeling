"""
验证 v31：intercept 先验的中心值应该是多少？
=============================================
已知：
  · target_scale = y.max() = 10,465,056
  · y_mean / y_max = 0.3275
  · 真实基线 / y_mean ≈ 53%（设计值）
  · OLS 截距（标准化）= 0.1356

模型内部：y_std = y / target_scale ∈ [0.142, 1.0]
截距要先承担什么？
  intercept_contribution（标准化空间）= 基线中【不随时间变化】的部分 / target_scale

我们需要算出这个值到底是多少。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import generate_game_data, CHANNELS, TRUE_BETA

df = generate_game_data(n_weeks=156, seed=2024)
y = df["revenue"].values
ts = y.max()

print("=" * 70)
print("[1] 各分量的绝对量级（元）")
print("=" * 70)
baseline = df["_true_baseline"].values
media = df["_true_media_total"].values
ctrl_vu = df["version_update"].values * 1.0     # 已经是"元"级别效果
ctrl_ca = df["competitor_activity"].values
ctrl_hd = df["holiday"].values
trend_var = df["trend"].values
noise = y - baseline - media - (df["version_update"].values
                                + df["competitor_activity"].values
                                + df["holiday"].values)
print(f"  y              均值={y.mean():>12,.0f}  范围[{y.min():>11,.0f}, {y.max():>11,.0f}]")
print(f"  真实基线        均值={baseline.mean():>12,.0f}  范围[{baseline.min():>11,.0f}, {baseline.max():>11,.0f}]")
print(f"  媒体总贡献      均值={media.mean():>12,.0f}")
print(f"  version_update  均值={df['version_update'].mean():>12,.0f}")
print(f"  competitor      均值={df['competitor_activity'].mean():>12,.0f}")
print(f"  holiday         均值={df['holiday'].mean():>12,.0f}")
print(f"  噪声            均值={noise.mean():>12,.0f}  std={noise.std():>11,.0f}")

print("\n" + "=" * 70)
print("[2] 真实基线里，有多少是【不随时间变化】的常数部分？")
print("=" * 70)
# 基线 = intercept_const + trend + season
# 用 OLS 把基线拆成 常数 + trend + sin/cos
import statsmodels.api as sm
t = np.arange(len(baseline))
X = np.column_stack([
    np.ones(len(t)),
    df["trend"].values,
    np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52),
    np.sin(4 * np.pi * t / 52), np.cos(4 * np.pi * t / 52),
])
names = ["const", "trend", "sin1", "cos1", "sin2", "cos2"]
r = sm.OLS(baseline, X).fit()
print(f"  R² = {r.rsquared:.4f}")
for n_, v in zip(names, r.params):
    print(f"    {n_:8s} {v:>14,.0f}")

const_part = r.params[0]
print(f"\n  基线常数部分 = {const_part:,.0f} 元/周")
print(f"  标准化后（÷ y_max）= {const_part / ts:.4f}   ← 这才是 intercept 先验该设的中心值！")

print("\n" + "=" * 70)
print("[3] 对比：我当前设的 0.10 偏差多少")
print("=" * 70)
target = const_part / ts
print(f"  正确中心值 : {target:.4f}")
print(f"  当前设定   : 0.1000")
print(f"  相对偏差   : {(0.10 - target) / target:+.1%}")

print("\n" + "=" * 70)
print("[4] 直接做个小实验：把先验中心改成正确值会怎样")
print("=" * 70)
print(f"  建议: intercept_mu = {target:.3f}   sigma = {target*0.3:.3f}")
print(f"  （sigma 取中心值的 30%，给足空间但不至于乱跑）")

# 更严格：考虑模型内部可能对 y 做了额外缩放
print("\n" + "=" * 70)
print("[5] 验证：OLS 直接对【标准化 y】回归，看截距是否 = 上面的值")
print("=" * 70)
y_std = y / ts
ctrl = df[["version_update", "competitor_activity", "holiday", "trend"]].values
ctrl_std = (ctrl - ctrl.mean(0)) / ctrl.std(0)
ch = df[[f"{c}_spend" for c in CHANNELS]].values
ch_std = ch / ch.mean(0)
Xa = sm.add_constant(np.column_stack([ctrl_std, ch_std]))
ra = sm.OLS(y_std, Xa).fit()
print(f"  R² = {ra.rsquared:.4f}")
print(f"  截距 = {ra.params[0]:.4f}")
print(f"  → 与 [2] 的推算 {target:.4f} 对比")
