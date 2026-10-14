"""
验证 v36：直接看后验分布的形状 —— 找出"平坡"参数
=================================================
前 35 个脚本都在看"结果对不对"，现在改为看"参数本身"。
如果某参数后验是平坡（无峰、区间极宽），它就是不可识别的。

输出：
  1. 每个参数的后验均值/标准差/94%CI/区间宽度比
  2. beta 与 kappa 的散点（看是否呈双曲线式的互补关系）
  3. 把它画出来存图
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from data_generator import generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      fit_mmm, get_spend_series, build_mmm_safe)

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

mmm, note = build_mmm_safe(cfg, control_cols=uc)
idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)
pt = mmm.idata.posterior

print("=" * 88)
print("[1] 参数后验的可识别性体检")
print("=" * 88)
print(f"  {'参数':28s} {'均值':>10s} {'std':>9s} {'94%CI 下':>10s} {'上':>10s} "
      f"{'CI宽/均值':>10s}  判定")
print("-" * 88)


def check(name, arr, true_val=None):
    a = np.asarray(arr).ravel()
    m, s = float(a.mean()), float(a.std())
    lo, hi = float(np.percentile(a, 3)), float(np.percentile(a, 97))
    ratio = (hi - lo) / abs(m) if abs(m) > 1e-9 else np.inf
    if ratio > 3:
        verdict = "✗ 平坡(不可识别)"
    elif ratio > 1.5:
        verdict = "△ 较宽"
    else:
        verdict = "✓ 清晰"
    tstr = f" (真{true_val})" if true_val is not None else ""
    print(f"  {name:28s} {m:>10.4f} {s:>9.4f} {lo:>10.4f} {hi:>10.4f} {ratio:>10.2f}  {verdict}{tstr}")
    return dict(name=name, mean=m, sd=s, lo=lo, hi=hi, ratio=ratio, true=true_val)


rows = []
for i, c in enumerate(CHANNELS):
    rows.append(check(f"beta[{c}]", pt["saturation_beta"].values[:, :, i], TRUE_BETA[c]))
for i, c in enumerate(CHANNELS):
    rows.append(check(f"kappa[{c}]", pt["saturation_kappa"].values[:, :, i], None))
for i, c in enumerate(CHANNELS):
    rows.append(check(f"slope[{c}]", pt["saturation_slope"].values[:, :, i], TRUE_KAPPA[c]))
for i, c in enumerate(CHANNELS):
    rows.append(check(f"decay[{c}]", pt["adstock_alpha"].values[:, :, i], TRUE_DECAY[c]))
rows.append(check("intercept_contribution", pt["intercept_contribution"].values, None))
rows.append(check("y_sigma", pt["y_sigma"].values, None))

print("\n" + "=" * 88)
print("[2] beta 与 kappa 的联合分布（看是否互补）")
print("=" * 88)
print(f"  {'渠道':10s} {'corr(beta,kappa)':>18s}  含义")
for i, c in enumerate(CHANNELS):
    b = pt["saturation_beta"].values[:, :, i].ravel()
    k = pt["saturation_kappa"].values[:, :, i].ravel()
    r = float(np.corrcoef(b, k)[0, 1])
    mean = ("强互补(不可识别)" if abs(r) > 0.7
            else ("中互补" if abs(r) > 0.4 else "弱"))
    print(f"  {c:10s} {r:>18.3f}  {mean}")

print("\n" + "=" * 88)
print("[3] 关键检验：beta × Hill(mean(x)) 是否被锁定？")
print("=" * 88)
# 渠道标准化后 mean(x)=1，Hill 形状依赖于 kappa 与 slope
slope = pt["saturation_slope"].values
kappa = pt["saturation_kappa"].values
beta = pt["saturation_beta"].values
# Hill(1) = 1^s / (1^s + kappa^s) = 1 / (1 + kappa^s)
hill1 = 1.0 / (1.0 + kappa ** slope)     # (chain,draw,channel)
comb = beta * hill1                       # 这是"标准化空间的贡献水平"
print(f"  {'渠道':10s} {'beta均值':>9s} {'Hill(1)均值':>12s} {'beta×Hill(1)':>14s} {'std':>9s}")
for i, c in enumerate(CHANNELS):
    print(f"  {c:10s} {beta[:,:,i].mean():>9.3f} {hill1[:,:,i].mean():>12.4f} "
          f"{comb[:,:,i].mean():>14.5f} {comb[:,:,i].std():>9.5f}")
print("\n  真实媒体贡献水平（标准化）= 1,603,826 / y_max / 5 渠道 = "
      f"{df['_true_media_total'].mean()/y.max()/5:.5f}")

# ---- 画图 ----
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "outputs", "figures")
os.makedirs(outdir, exist_ok=True)

fig, axes = plt.subplots(3, 5, figsize=(20, 11))
for i, c in enumerate(CHANNELS):
    # beta 后验
    axes[0, i].hist(pt["saturation_beta"].values[:, :, i].ravel(), bins=40,
                    color="#2E86AB", alpha=.8)
    axes[0, i].axvline(TRUE_BETA[c], color="#C0392B", ls="--", lw=2,
                       label=f"真值 {TRUE_BETA[c]}")
    axes[0, i].set_title(f"beta[{c}]", fontsize=11)
    axes[0, i].legend(fontsize=8)
    # kappa 后验
    axes[1, i].hist(pt["saturation_kappa"].values[:, :, i].ravel(), bins=40,
                    color="#E67E22", alpha=.8)
    axes[1, i].set_title(f"kappa[{c}]  CI={rows[len(CHANNELS)+i]['ratio']:.1f}x", fontsize=11)
    # decay 后验
    axes[2, i].hist(pt["adstock_alpha"].values[:, :, i].ravel(), bins=40,
                    color="#27AE60", alpha=.8)
    axes[2, i].axvline(TRUE_DECAY[c], color="#C0392B", ls="--", lw=2)
    axes[2, i].set_title(f"decay[{c}]", fontsize=11)
plt.suptitle("后验分布体检：beta / kappa / decay（红线=真值）", fontsize=14)
plt.tight_layout()
p1 = os.path.join(outdir, "diag_v36_posteriors.png")
plt.savefig(p1, dpi=110, bbox_inches="tight")
plt.close()

# beta-kappa 散点
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, c in enumerate(CHANNELS):
    b = pt["saturation_beta"].values[:, :, i].ravel()[::5]
    k = pt["saturation_kappa"].values[:, :, i].ravel()[::5]
    axes[i].scatter(k, b, s=4, alpha=.3, color="#8E44AD")
    axes[i].axhline(TRUE_BETA[c], color="#C0392B", ls="--", lw=1.5)
    axes[i].set_xlabel("kappa")
    axes[i].set_title(f"{c}  corr={np.corrcoef(b,k)[0,1]:+.2f}", fontsize=11)
axes[0].set_ylabel("beta")
plt.suptitle("beta 与 kappa 的联合后验（若呈斜线带 → 互补不可识别）", fontsize=13)
plt.tight_layout()
p2 = os.path.join(outdir, "diag_v36_beta_kappa.png")
plt.savefig(p2, dpi=110, bbox_inches="tight")
plt.close()

print(f"\n已保存图: \n  {p1}\n  {p2}")
