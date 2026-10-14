"""
验证 v34：get_contributions(original_scale=True) 真的还原到原尺度了吗？
====================================================================
怀疑点：
  · 数据生成器：kol 总花费 1,578 万元
  · v32/v33 的 ROI：kol=4.49 → 意味着贡献 = 4.49 × 1578 = 7,085 万元
  · 而历史总收入才 53,459 万元，5 渠道合计 69.9% = 37,368 万元
  · 真值：媒体应占 46.9% = 25,000 万元

关键检查：
  1. get_contributions 各分量之和 = predict 的 y 吗？（守恒性）
  2. 各分量之和 / 真实 y.sum() = ?
  3. 分渠道贡献 / 该渠道花费 = ? 与真值比
  4. 用 predict 做反事实的 Δy 与 get_contributions 对比
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import generate_game_data, CHANNELS, TRUE_BETA
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      fit_mmm, get_spend_series, build_mmm_safe, get_target_scale)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

mmm, note = build_mmm_safe(cfg, control_cols=uc)
idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)
ts = get_target_scale(mmm)

print("=" * 74)
print("[1] 尺度基准")
print("=" * 74)
print(f"  y.sum()        = {y.sum():>16,.0f}  (真实总收入)")
print(f"  y.mean()       = {y.mean():>16,.0f}")
print(f"  y.max()        = {y.max():>16,.0f}  = target_scale? {abs(ts-y.max())/y.max()<1e-6}")
print(f"  真实媒体总贡献  = {df['_true_media_total'].sum():>16,.0f}  "
      f"({df['_true_media_total'].sum()/y.sum():.1%})")
print(f"  真实基线总      = {df['_true_baseline'].sum():>16,.0f}  "
      f"({df['_true_baseline'].sum()/y.sum():.1%})")

print("\n" + "=" * 74)
print("[2] get_contributions(original_scale=True) 的分量")
print("=" * 74)
ds = mmm.data.get_contributions(original_scale=True)
print(f"  data_vars: {list(ds.data_vars)}")
total_all = 0.0
for v in ds.data_vars:
    arr = ds[v].mean(("chain", "draw")).values
    s = float(np.asarray(arr).sum())
    print(f"    {v:16s} sum = {s:>16,.0f}   ({s/y.sum():>7.1%} of y.sum)  shape={np.asarray(arr).shape}")
    if v not in ("channels",):
        total_all += s
ch_sum = float(np.asarray(ds["channels"].mean(("chain","draw")).values).sum())
print(f"\n  channels 合计      = {ch_sum:>16,.0f}  ({ch_sum/y.sum():.1%})")
print(f"  真实媒体总贡献      = {df['_true_media_total'].sum():>16,.0f}  ({df['_true_media_total'].sum()/y.sum():.1%})")
print(f"  → 比值 = {ch_sum/df['_true_media_total'].sum():.3f}")

print("\n" + "=" * 74)
print("[3] 分渠道：贡献 vs 花费 vs 真值")
print("=" * 74)
chd = ds["channels"].mean(("chain", "draw"))
print(f"  {'渠道':10s} {'估计贡献':>14s} {'真实贡献':>14s} {'估计花费':>14s} "
      f"{'估ROI':>8s} {'真ROI':>8s}")
tot_est = tot_true = 0.0
for c in CHANNELS:
    e = float(np.asarray(chd.sel(channel=c).values).sum())
    t = float(df[f"_true_contrib_{c}"].sum())
    sp = float(get_spend_series(df, c).sum())
    tot_est += e; tot_true += t
    print(f"  {c:10s} {e:>14,.0f} {t:>14,.0f} {sp:>14,.0f} "
          f"{e/sp:>8.2f} {TRUE_BETA[c]:>8.2f}")
print(f"  {'合计':10s} {tot_est:>14,.0f} {tot_true:>14,.0f}")

print("\n" + "=" * 74)
print("[4] 关键：predict 的 mu 之和 vs channels 之和")
print("=" * 74)
pt = mmm.idata.posterior
mu_sum = float(pt["mu"].values.mean(axis=(0,1)).sum())
print(f"  Σmu (标准化空间)         = {mu_sum:>14,.4f}")
print(f"  Σmu × target_scale       = {mu_sum*ts:>14,.0f}")
print(f"  真实 y.sum()              = {y.sum():>14,.0f}")
print(f"  → 比值 = {mu_sum*ts/y.sum():.4f}")

print("\n" + "=" * 74)
print("[5] 各分量之和 是否 = Σmu")
print("=" * 74)
cc = float(pt["channel_contribution"].values.mean(axis=(0,1)).sum())
ic = float(pt["intercept_contribution"].values.mean())
n = len(y)
ys = float(pt["yearly_seasonality_contribution"].values.mean(axis=(0,1)).sum())
fs = float(pt["fourier_contribution"].values.mean(axis=(0,1)).sum())
gc = 0.0
if "gamma_control" in pt:
    gcv = pt["gamma_control"].values  # (chain, draw, control)
    cd = mmm.idata.constant_data
    for i, c in enumerate(uc):
        g = gcv.mean(axis=(0,1))[i]
        # 控制变量值是标准化的，在 constant_data 里
        if c in cd:
            cval = np.asarray(cd[c].values)
            gc += float(g * cval.sum())
parts = ic*n + cc + ys + fs + gc
print(f"  intercept×n = {ic*n:>14,.4f}")
print(f"  channel_contrib 合计 = {cc:>14,.4f}")
print(f"  seasonality  = {ys:>14,.4f}")
print(f"  fourier      = {fs:>14,.4f}")
print(f"  control      = {gc:>14,.4f}")
print(f"  ---- 合计 = {parts:>14,.4f}")
print(f"  Σmu       = {mu_sum:>14,.4f}")
print(f"  偏差 = {(parts-mu_sum)/mu_sum:+.6f}")

print("\n" + "=" * 74)
print("[6] channel_contribution 合计 × ts  对比 真实媒体贡献")
print("=" * 74)
print(f"  channel_contribution × ts = {cc*ts:>14,.0f}")
print(f"  真实媒体总贡献             = {df['_true_media_total'].sum():>14,.0f}")
print(f"  → 比值 = {cc*ts/df['_true_media_total'].sum():.4f}")
print(f"\n  channels(original_scale) 合计 = {ch_sum:>14,.0f}")
print(f"  channel_contribution × ts    = {cc*ts:>14,.0f}")
print(f"  → 二者是否一致？ {abs(ch_sum-cc*ts)/max(ch_sum,1)<0.01}")
