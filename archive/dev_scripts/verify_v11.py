"""
验证 v11：验证"数据生成器 vs 模型结构不一致"的假设
====================================================
假设：数据生成器用 `shape/shape_sum*target` 校准，
      导致生成数据的隐含 beta ≠ TRUE_BETA。
      而模型假设的是 `贡献 = beta × Hill(adstock(x))`。
      两者结构不同 → 参数回收必然失败。

检验方法：用 numpy 按【模型结构】重新生成一份数据
  y = intercept + Σ beta_c × Hill(adstock(spend_c)) + 噪声
然后看模型能否回收 beta。

如果这份新数据能回收，就证明：
  · 模型实现是正确的
  · 之前的失败是数据生成器的问题（校准方式与模型结构不匹配）
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series


def geom_adstock(x, decay, l_max=6):
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(len(x)):
        carry = x[i] + decay * carry
        out[i] = carry
    return out


def hill(x, slope, kappa):
    """标准 Hill: x^slope / (x^slope + kappa^slope)，输出 0~1"""
    return (x ** slope) / (x ** slope + kappa ** slope)


def main():
    # ---------- 1. 用真实投放数据（但按模型结构生成 y）----------
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    N = len(df)

    # 关键：按【模型结构】重算媒体贡献
    # 用标准化后的花费（均值=1），这样 kappa 就是"半饱和点=1倍平均投放"
    media = np.zeros(N)
    for ch in cfg.channel_cols:
        raw = df[ch].values
        avg = raw.mean()
        x = raw / avg                                  # 标准化到均值 1
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max)
        # 用真实 kappa（此处语义=陡度）与真实半饱和点（设在 1.2，即略高于平均）
        h = hill(ad, TRUE_KAPPA[ch], 1.2)
        media += TRUE_BETA[ch] * h                     # ← 直接是 beta × Hill
        globals().setdefault("_dbg", []).append((ch, ad.mean(), h.mean()))

    print("各渠道（标准化输入）：")
    for ch, adm, hm in globals()["_dbg"]:
        print(f"  {ch:10s} ad均值={adm:.3f}  Hill均值={hm:.4f}  "
              f"beta={TRUE_BETA[ch]:.2f}  → 贡献均值={TRUE_BETA[ch]*hm:.4f}")

    # 基线（用真实数据的基线量级）
    baseline = df["_true_baseline"].values if "_true_baseline" in df.columns else None
    if baseline is None:
        # 退化：用真实 revenue 减去真实媒体，再按比例
        base_level = df["revenue"].values.mean() * 0.5
        baseline = np.linspace(base_level * 0.7, base_level * 1.3, N)
    rng = np.random.default_rng(7)
    y_new = baseline + media * 1e6 + rng.normal(0, 2.5e4, N)

    print(f"\n新 y: mean={y_new.mean():,.0f}  sum={y_new.sum():,.0f}")
    print(f"新 y 中媒体占比 ≈ {media.mean()*1e6/y_new.mean():.1%}")

    df2 = df.copy()
    df2["revenue"] = y_new

    X, y, meta, _ = prepare_data(df2, cfg)
    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior

    print("\n" + "=" * 84)
    print("参数回收（按模型结构生成的数据）")
    print("=" * 84)
    print(f"{'渠道':10s} {'估decay':>9s} {'真decay':>9s} {'估beta':>9s} {'真beta':>8s} "
          f"{'估slope':>9s} {'真k':>6s} {'估kappa':>9s}")
    print("-" * 84)
    for i, c in enumerate(cfg.channel_cols):
        d = float(pt["adstock_alpha"].values[:, :, i].mean())
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        sl = float(pt["saturation_slope"].values[:, :, i].mean())
        ka = float(pt["saturation_kappa"].values[:, :, i].mean())
        print(f"{c:10s} {d:>9.3f} {TRUE_DECAY[c]:>9.2f} {b:>9.3f} {TRUE_BETA[c]:>8.2f} "
              f"{sl:>9.3f} {TRUE_KAPPA[c]:>6.2f} {ka:>9.3f}")

    # ROI
    print("\nROI 检验:")
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(get_spend_series(df2, c).sum())
        rows.append({"渠道": c, "估ROI": g / s, "真ROI": TRUE_BETA[c]})
    roi = pd.DataFrame(rows)
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic
    print(roi.round(3).to_string(index=False))
    print(f"Spearman ρ = {rho:.3f}")


if __name__ == "__main__":
    main()
