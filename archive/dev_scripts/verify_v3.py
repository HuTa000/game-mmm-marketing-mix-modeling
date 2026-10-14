"""
验证 v3：定位 ROI 偏差的真正根源
==================================
已确认事实：
  1. 尺度转换是对的 —— predict(标准化输入)×target_scale 总量精度 99.96%
  2. 但各渠道贡献占比严重失真（媒体占 150% vs 真实 46.7%）
  3. beta 被系统性低估，kappa 被乱估（tiktok 0.27 vs 真实~2.0）

怀疑：**把渠道除以自身均值**这个标准化动作，
      抹平了渠道之间的绝对量级差异，导致模型无法识别
      "1美元 tiktok" 与 "1美元 facebook" 的效率差异 → ROI 失序。

本脚本做 3 组对照实验：
  A. 渠道÷自身均值（当前做法）
  B. 渠道÷统一常数（1e4，即单位改成"万元"）→ 保留渠道间相对量级
  C. 原始尺度（不缩放）
比较三者：总量精度 / 媒体占比 / ROI 排序正确性。

为省时间用 quick_mode，并且只对 B/C 重跑（A 的结果直接引用）。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, TRUE_BETA
from mmm_core import MMMConfig, normalize_columns, build_mmm_safe, fit_mmm, get_spend_series


def run_case(name, X, y, df, cfg):
    mmm, note = build_mmm_safe(cfg)
    idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    try:
        scale = float(np.squeeze(np.asarray(
            mmm.idata.constant_data["target_scale"].values)))
    except Exception:
        scale = 1.0

    true_y = float(df["revenue"].sum())
    pred = np.asarray(mmm.predict(X)).ravel() * scale

    contrib = {}
    for c in cfg.channel_cols:
        Xcf = X.copy()
        Xcf[c] = 0.0
        contrib[c] = pred - np.asarray(mmm.predict(Xcf)).ravel() * scale
    cdf = pd.DataFrame(contrib)
    media_sum = float(cdf.sum().sum())

    rows = []
    for c in cfg.channel_cols:
        g = float(cdf[c].sum()); s = float(get_spend_series(df, c).sum())
        rows.append({"渠道": c, "花费万": s / 1e4, "贡献万": g / 1e4,
                     "估ROI": g / s, "真ROI": TRUE_BETA[c],
                     "倍差": (g / s) / TRUE_BETA[c]})
    roi = pd.DataFrame(rows)

    # 排序正确性
    est_order = list(roi.sort_values("估ROI", ascending=False)["渠道"])
    true_order = list(roi.sort_values("真ROI", ascending=False)["渠道"])
    # Spearman
    from scipy.stats import spearmanr
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic

    print(f"\n{'='*78}\n实验 {name}\n{'='*78}")
    print(f"采样 {elapsed:.0f}s | 总量精度 {pred.sum()/true_y-1:+.1%} | "
          f"媒体占比 {media_sum/true_y:.1%} (真实 46.7%)")
    print(f"估ROI排序: {est_order}")
    print(f"真ROI排序: {true_order}")
    print(f"Spearman秩相关 = {rho:.3f}  {'✓ 排序正确' if rho > 0.7 else '✗ 排序错误'}")
    print(roi.round(3).to_string(index=False))

    # beta 回收
    pt = idata.posterior
    print("beta 回收:", {c: round(float(pt['saturation_beta'].values[:,:,i].mean()),2)
                        for i,c in enumerate(cfg.channel_cols)})
    print("真 beta  :", {c: TRUE_BETA[c] for c in cfg.channel_cols})
    return roi, rho


def main():
    df0 = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df0, cfg)
    y = df["revenue"].astype(float)

    # ---- 实验 B：渠道 ÷ 1e4（单位改万元，保留渠道间相对量级）----
    XB = df[["date"] + cfg.channel_cols].copy()
    for c in cfg.channel_cols:
        XB[c] = XB[c] / 1e4
    for c in cfg.control_cols:
        if c in df.columns:
            XB[c] = (df[c] - df[c].mean()) / (df[c].std() or 1)
    print(f"[实验B] 渠道值域: min={XB[cfg.channel_cols].min().min():.1f} "
          f"max={XB[cfg.channel_cols].max().max():.1f}")
    run_case("B: 渠道÷1e4（保留渠道间相对量级）", XB, y, df, cfg)

    # ---- 实验 C：原始尺度 ----
    XC = df[["date"] + cfg.channel_cols].copy()
    for c in cfg.control_cols:
        if c in df.columns:
            XC[c] = (df[c] - df[c].mean()) / (df[c].std() or 1)
    print(f"\n[实验C] 渠道值域: min={XC[cfg.channel_cols].min().min():,.0f} "
          f"max={XC[cfg.channel_cols].max().max():,.0f}")
    run_case("C: 原始尺度（不缩放渠道）", XC, y, df, cfg)


if __name__ == "__main__":
    main()
