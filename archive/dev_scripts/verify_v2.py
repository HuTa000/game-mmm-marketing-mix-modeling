"""
验证 v2：用「标准化输入」做反事实，检查 ROI 是否与真实 beta 对齐
==============================================================
v1 发现：predict(标准化输入)×scale 总量正确（99.96%），
         但 predict(原始输入)×scale 虚高 4.87 倍。
v1 还发现：官方守恒分解总量虚高 1.5 倍 → 说明 saturation_beta 被高估。

本脚本检验：
  1. 用标准化输入做反事实，ROI 排序/量级是否正确
  2. 官方 get_contributions 的 channels 分量（dict 结构）怎么取
  3. beta 估计值 vs 真实 beta 的差距（定位虚高根源）
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, get_true_contribution, TRUE_BETA, TRUE_DECAY
from mmm_core import (
    MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm,
    get_target_scale, get_spend_series,
)


def main():
    print("=" * 78)
    print("验证 v2：标准化输入反事实 + 参数回收诊断")
    print("=" * 78)

    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, used_ctrl = prepare_data(df, cfg)

    mmm, note = build_mmm_safe(cfg)
    idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    scale = get_target_scale(mmm)
    print(f"采样完成 {elapsed:.1f}s, target_scale={scale:,.0f}, note={note}")

    true_y_sum = float(df["revenue"].sum())

    # ---------- 反事实（标准化输入） ----------
    Xstd = X.copy()
    pred_base = np.asarray(mmm.predict(Xstd)).ravel() * scale
    print(f"\n[predict(标准化)×scale] 合计 = {pred_base.sum()/1e4:,.0f} 万元 "
          f"(真实 {true_y_sum/1e4:,.0f} 万元, 偏差 {pred_base.sum()/true_y_sum-1:+.1%})")

    contrib = {}
    for c in cfg.channel_cols:
        Xcf = Xstd.copy()
        Xcf[c] = 0.0
        pred_cf = np.asarray(mmm.predict(Xcf)).ravel() * scale
        contrib[c] = pred_base - pred_cf
    contrib_df = pd.DataFrame(contrib, index=df.index)
    media_sum = float(contrib_df.sum().sum())
    print(f"\n[反事实·标准化] 媒体贡献合计 = {media_sum/1e4:,.0f} 万元 "
          f"(占比 {media_sum/true_y_sum:.1%}, 真实 46.7%)")

    print("\n渠道         贡献(万元)   花费(万元)   估ROI    真ROI   偏差")
    print("-" * 68)
    for c in cfg.channel_cols:
        g = float(contrib_df[c].sum())
        s = float(get_spend_series(df, c).sum())
        roi_est, roi_true = g / s, TRUE_BETA[c]
        print(f"{c:10s} {g/1e4:>10,.0f} {s/1e4:>11,.0f} {roi_est:>8.3f} "
              f"{roi_true:>8.2f} {roi_est/roi_true-1:>+7.0%}")

    # ---------- 参数回收：beta & decay & kappa ----------
    s_tbl = idata.posterior
    print("\n" + "=" * 78)
    print("参数回收诊断（定位虚高根源）")
    print("=" * 78)
    print(f"{'渠道':10s} {'估beta':>10s} {'真beta':>8s} | "
          f"{'估decay':>9s} {'真decay':>8s} | {'估kappa':>9s}")
    for i, c in enumerate(cfg.channel_cols):
        b_est = float(s_tbl["saturation_beta"].values[:, :, i].mean())
        d_est = float(s_tbl["adstock_alpha"].values[:, :, i].mean())
        k_est = float(s_tbl["saturation_kappa"].values[:, :, i].mean())
        print(f"{c:10s} {b_est:>10.3f} {TRUE_BETA[c]:>8.2f} | "
              f"{d_est:>9.3f} {TRUE_DECAY[c]:>8.2f} | {k_est:>9.3f}")

    # ---------- get_contributions 的 channels 结构 ----------
    print("\n" + "=" * 78)
    print("官方 get_contributions 的 channels 分量结构")
    print("=" * 78)
    dsA = mmm.data.get_contributions(original_scale=True)
    ch_var = dsA["channels"]
    print("  channels dims:", ch_var.dims, " shape:", ch_var.shape)
    print("  channels coords:", {k: list(v.values)[:8] for k, v in ch_var.coords.items()})
    ch_mean = ch_var.mean(("chain", "draw"))
    print("  channels 各渠道合计:")
    for c in cfg.channel_cols:
        try:
            arr = np.asarray(ch_mean.sel(channel=c).values)
            print(f"    {c:10s} {arr.sum()/1e4:>12,.0f} 万元")
        except Exception as e:
            print(f"    {c:10s} sel失败: {e}")


if __name__ == "__main__":
    main()
