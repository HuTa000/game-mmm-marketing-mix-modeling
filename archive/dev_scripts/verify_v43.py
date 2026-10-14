"""
验证 v43：mean 缩放后，系统校准所有先验
=======================================
背景：改成 mean 缩放后 y_std 均值 = 1.0（原来是 0.27），
      所有参数的"合理量级"都变了，需要重新校准。

真实参数（数据生成器设计值）：
  decay:  0.35 0.40 0.25 0.20 0.45
  kappa(半饱和点) = TRUE_GAMMA_FRAC × mean(spend) / channel_scale
          其中 channel_scale = max(spend)，所以标准化后的半饱和点 =
          TRUE_GAMMA_FRAC × mean / max   ← 这个才该是先验中心！
  slope:  1.6 1.8 2.1 1.4 1.5
  beta:   1.10 1.45 2.30 0.65 1.20（真实 ROI）

先算清楚"模型内部看到的 x"是什么量级。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

from data_generator import (generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY,
                            TRUE_KAPPA, TRUE_GAMMA_FRAC)

raw = generate_game_data(n_weeks=156, seed=2024)
raw = raw.rename(columns={f"{c}_spend": c for c in CHANNELS})

print("=" * 90)
print("[1] 模型内部看到的 x 是什么？（外面÷均值，模型内÷max）")
print("=" * 90)
print(f"  {'渠道':10s} {'mean':>12s} {'max':>12s} {'÷mean后max':>12s} "
      f"{'内部x范围':>18s} {'内部x均值':>10s}")
inner = {}
for c in CHANNELS:
    x = raw[c].values
    x_div_mean = x / x.mean()          # 我们传给模型的
    x_inner = x_div_mean / x_div_mean.max()   # 模型内部再÷max
    inner[c] = {"mean": x_inner.mean(), "max": x_inner.max()}
    print(f"  {c:10s} {x.mean():>12,.0f} {x.max():>12,.0f} {x_div_mean.max():>12.4f} "
          f"[{x_inner.min():>7.4f},{x_inner.max():>7.4f}] {x_inner.mean():>10.4f}")

print("\n" + "=" * 90)
print("[2] 模型内部的'半饱和点'应该是多少？")
print("=" * 90)
print("  真实数据里: 贡献 = beta × Hill(adstock(x) ; slope, kappa_abs)")
print("              kappa_abs = TRUE_GAMMA_FRAC × mean(x)   （绝对元）")
print("  但模型内部 x_inner = x / mean(x) / max(x/mean(x)) = x / max(x)")
print("  所以换算到模型内部尺度:")
print("      kappa_inner = kappa_abs / max(x) = TRUE_GAMMA_FRAC × mean(x)/max(x)")
print()
print(f"  {'渠道':10s} {'FRAC':>6s} {'mean/max':>10s} {'kappa_inner':>12s} {'slope真值':>10s}")
kappas = {}
for c in CHANNELS:
    x = raw[c].values
    ratio = x.mean() / x.max()
    ki = TRUE_GAMMA_FRAC[c] * ratio
    kappas[c] = ki
    print(f"  {c:10s} {TRUE_GAMMA_FRAC[c]:>6.1f} {ratio:>10.4f} {ki:>12.4f} {TRUE_KAPPA[c]:>10.1f}")
print(f"\n  → kappa_inner 均值 = {np.mean(list(kappas.values())):.4f}")
print(f"     这是 saturation_kappa 先验该围绕的中心！")
print(f"     （Gamma 先验中心 = alpha/beta，所以要设成 alpha/beta ≈ {np.mean(list(kappas.values())):.3f}）")

print("\n" + "=" * 90)
print("[3] beta 在模型内部的合理水平")
print("=" * 90)
print("  真实: Σ贡献_c = TRUE_BETA_c × Σspend_c")
print("  模型: 贡献_c(t) = beta_c × Hill(adstock(x_inner))  ")
print("  在 mean 缩放下，模型贡献还原到原尺度 = channel_contribution × target_scale")
print("  而 target_scale = y.mean() = ")
print(f"    {raw['revenue'].mean():,.0f}")
print()
y_mean = raw["revenue"].mean()
print(f"  真实媒体总贡献均值 = {raw['_true_media_total'].mean():,.0f}")
print(f"  归一化到 y_mean:  Σ贡献 / y_mean = {raw['_true_media_total'].mean()/y_mean:.4f}")
print(f"  分到 5 渠道平均 = {raw['_true_media_total'].mean()/y_mean/5:.4f}")
print()
print("  各渠道真实贡献均值 / y_mean:")
for c in CHANNELS:
    r = raw[f"_true_contrib_{c}"].mean() / y_mean
    print(f"    {c:10s} {r:.4f}")
print()
print("  模型贡献 = beta × Hill(ad) ；Hill(ad_inner) 典型值约 0.2~0.5")
print("  所以 beta 合理水平 ≈ 贡献水平 / Hill ≈ 0.05/0.3 ~ 0.3/0.3")
for c in CHANNELS:
    h = 0.35  # 粗估
    r = raw[f"_true_contrib_{c}"].mean() / y_mean
    print(f"    {c:10s} beta ≈ {r:.4f} / {h} = {r/h:.3f}")
print()
print("  → 各渠道 beta 差异很大（0.2~2.0），单一先验中心无法覆盖")
print("  → 但 beta 与数据中的渠道特征相关：")
for c in CHANNELS:
    r = raw[f"_true_contrib_{c}"].mean() / y_mean
    sp = raw[c].mean()
    print(f"    {c:10s} 真beta(r尺度)={r:.4f} 花费均值={sp:>10,.0f}")
