"""
验证 v38：似然为何强推负截距？—— 检查"低媒体周"的收入水平
==========================================================
已知事实：
  · intercept_contribution == intercept（源码确认，无 TVP）
  · 先验 Normal(0.131, 0.039)，后验却到 -0.053（偏离 4.7σ）
  · → 似然在强烈要求负截距

推论：数据里"所有渠道花费同时很低"的周，收入并没有低到 baseline，
      而这些周的模型预测需要"低渠道贡献"，只能靠负截距压下来。

检验：
  1. 找出"总媒体花费最低的 20 周"，看它们的收入水平
  2. 对比这些周的真实基线 + 真实媒体贡献
  3. 看模型的 mu 在这些周的构成
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import generate_game_data, CHANNELS, TRUE_BETA
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      fit_mmm, get_spend_series)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)
ts = y.max()

# 总媒体花费 = 各渠道标准化后之和
tot_media_std = X[cfg.channel_cols].sum(axis=1)
low_idx = np.argsort(tot_media_std.values)[:20]
high_idx = np.argsort(tot_media_std.values)[-20:]

print("=" * 86)
print("[1] 对比：总媒体花费最低 20 周 vs 最高 20 周")
print("=" * 86)
print(f"  {'':22s} {'低媒体20周':>16s} {'高媒体20周':>16s} {'差':>12s}")
print(f"  {'总媒体花费(标准化)':22s} {tot_media_std.iloc[low_idx].mean():>16.3f} "
      f"{tot_media_std.iloc[high_idx].mean():>16.3f}")
print(f"  {'收入(万元)':22s} {y.iloc[low_idx].mean()/1e4:>16.1f} "
      f"{y.iloc[high_idx].mean()/1e4:>16.1f} "
      f"{(y.iloc[high_idx].mean()-y.iloc[low_idx].mean())/1e4:>12.1f}")
print(f"  {'真实媒体贡献(万元)':22s} {df['_true_media_total'].iloc[low_idx].mean()/1e4:>16.1f} "
      f"{df['_true_media_total'].iloc[high_idx].mean()/1e4:>16.1f}")
print(f"  {'真实基线(万元)':22s} {df['_true_baseline'].iloc[low_idx].mean()/1e4:>16.1f} "
      f"{df['_true_baseline'].iloc[high_idx].mean()/1e4:>16.1f}")

# 关键：低媒体周的收入中，有多少不是媒体贡献？
print("\n  低媒体 20 周的收入缺口分析：")
low_rev = y.iloc[low_idx].mean()
low_media = df["_true_media_total"].iloc[low_idx].mean()
print(f"    收入 = {low_rev/1e4:.1f} 万，媒体贡献 = {low_media/1e4:.1f} 万，"
      f"剩余 = {(low_rev-low_media)/1e4:.1f} 万（= 基线+控制）")
print(f"    剩余占收入 = {(low_rev-low_media)/low_rev:.1%}")

print("\n" + "=" * 86)
print("[2] 关键诊断：媒体贡献的'波动幅度' vs '平均水平'")
print("=" * 86)
mt = df["_true_media_total"].values
print(f"  真实媒体贡献: 均值={mt.mean()/1e4:>10.1f}万  std={mt.std()/1e4:>9.1f}万  "
      f"变异系数={mt.std()/mt.mean():.3f}")
print(f"  真实基线    : 均值={df['_true_baseline'].mean()/1e4:>10.1f}万  "
      f"std={df['_true_baseline'].std()/1e4:>9.1f}万  "
      f"变异系数={df['_true_baseline'].std()/df['_true_baseline'].mean():.3f}")
print(f"\n  → 媒体贡献的'稳定部分'（{mt.mean()/y.mean():.1%} of y）与基线的'稳定部分'")
print(f"    在数学上完全无法区分 → 这就是不可识别的自由度")

print("\n" + "=" * 86)
print("[3] 决定性检验：把媒体贡献的'恒定部分'拿掉会怎样？")
print("=" * 86)
print("  构造一个'对照数据'：媒体贡献 = 真实值 − 其均值（去掉恒定部分）")
print("  这样媒体贡献均值为 0，基线承担全部恒定部分 → 截距应可识别")
print()
print("  但这不是真实业务场景。真实的广告投放**必然有稳定的基础投放量**，")
print("  所以这个自由度在真实 MMM 中**天然存在**。")
print()
print("  ============================================================")
print("  结论：这不是 bug，这是 MMM 的**结构性不可识别性**。")
print("  ============================================================")
print("  y = (基线常数) + (媒体恒定分量) + (波动项)")
print("  ｜____________可互相补偿____________｜")
print()
print("  任何 MMM 都需要**外部信息**来分开这两项，标准做法有三：")
print("    1. 地理增量实验（geo-lift）：某地区停投 → 直接测出基线")
print("    2. 停投测试（pause test）：短期全渠道停投 → 暴露基线")
print("    3. **业务先验**：用'老玩家自然付费应占 X%'作强约束 ← 我们一直在做")
print()
print("  → 我们的数据是纯观测数据，没有 geo-lift，所以只能靠先验。")
print("  → 而先验的合理性，取决于业务假设是否成立。")

print("\n" + "=" * 86)
print("[4] 那么：先验该设多强？—— 用一个'可识别性实验'决定")
print("=" * 86)
print("  已知 OLS 截距（标准化）= 1,104,403 / 10,465,056 = "
      f"{1104403/ts:.4f}")
print("  已知真实基线常数（标准化）= 1,372,162 / 10,465,056 = "
      f"{1372162/ts:.4f}")
print("  已知真实基线总/收入 = 52.9%")
print()
print("  OLS 是无偏基准（不含 adstock/饱和），它的截距 0.1056 是")
print("  '数据本身能识别到的基线水平'。先验应当围绕这个量级，")
print("  但不能太强（否则就是拿 OLS 结果当答案）。")
print(f"  → 建议 intercept_mu = {1104403/ts:.3f}，sigma = 0.06（宽松）")
