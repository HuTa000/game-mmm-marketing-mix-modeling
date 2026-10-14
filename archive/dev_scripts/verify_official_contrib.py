"""
验证脚本：确认官方推荐的贡献分解 API 是否正确还原尺度
========================================================
目的：用官方 `mmm.data.get_contributions(original_scale=True)` 与
      `mmm.compute_counterfactual_contributions_dataset()` 两条路线，
      对比真实贡献，确定哪条是正确解。

背景：阶段 2 用 mmm.predict() 反事实法得到的结果系统性偏高约 3.7 倍，
      怀疑是尺度还原方式不对（predict 返回标准化空间 + 自己乘 target_scale
      可能漏了某些量，或者 predict 内部对 channel 又做了一次缩放）。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, get_true_contribution, TRUE_BETA
from mmm_core import (
    MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm,
    get_target_scale, get_spend_series,
)


def main():
    print("=" * 78)
    print("验证：官方贡献分解 API 的尺度正确性")
    print("=" * 78)

    # ---------- 1. 数据 ----------
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)

    X, y, meta, used_ctrl = prepare_data(df, cfg)
    print(f"\n数据: X={X.shape}, y={y.shape}")
    print(f"y 均值(元) = {y.mean():,.0f}   总和 = {y.sum():,.0f}")
    print(f"y 均值(万元) = {y.mean()/1e4:,.1f}")

    # ---------- 2. 建模 ----------
    mmm, note = build_mmm_safe(cfg)
    print(f"模型: {note}")
    idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    print(f"采样完成 {elapsed:.1f}s")

    scale = get_target_scale(mmm)
    print(f"\ntarget_scale = {scale:,.2f}")

    # ---------- 3. 真实贡献 ----------
    true_c = get_true_contribution(df)
    true_cols = [c for c in cfg.channel_cols if c in true_c.columns]
    true_media_sum = float(true_c[true_cols].sum().sum())
    print(f"\n真实媒体贡献合计 = {true_media_sum/1e4:,.0f} 万元")
    print(f"真实总收入        = {float(df['revenue'].sum())/1e4:,.0f} 万元")
    print(f"真实媒体占比      = {true_media_sum/df['revenue'].sum():.1%}")

    # ---------- 4. 路线 A：官方 mmm.data.get_contributions ----------
    print("\n" + "=" * 78)
    print("路线 A：mmm.data.get_contributions(original_scale=True)  [官方推荐·守恒分解]")
    print("=" * 78)
    try:
        dsA = mmm.data.get_contributions(original_scale=True)
        print("  返回 dataset 变量:", list(dsA.data_vars))
        dsA_mean = dsA.mean(("chain", "draw"))
        # 打印各分量总量
        for v in dsA.data_vars:
            arr = np.asarray(dsA_mean[v].values)
            print(f"    {v:35s} 合计 = {arr.sum():>16,.0f} 元 "
                  f"({arr.sum()/1e4:>10,.1f} 万元)")
        A_total = sum(float(np.asarray(dsA_mean[v].values).sum())
                      for v in dsA.data_vars)
        print(f"  >>> 所有分量合计 = {A_total/1e4:,.1f} 万元 "
              f"(真实 y = {float(df['revenue'].sum())/1e4:,.1f} 万元)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        dsA = None

    # ---------- 5. 路线 B：官方反事实分解 ----------
    print("\n" + "=" * 78)
    print("路线 B：mmm.compute_counterfactual_contributions_dataset()  [反事实·不守恒]")
    print("=" * 78)
    try:
        dsB = mmm.compute_counterfactual_contributions_dataset(central_tendency="median")
        dsB_mean = dsB.mean(("chain", "draw"))
        print("  返回 dataset 变量:", list(dsB.data_vars))
        for v in dsB.data_vars:
            arr = np.asarray(dsB_mean[v].values)
            print(f"    {v:35s} 合计 = {arr.sum():>16,.0f} 元")
    except Exception as e:
        import traceback
        traceback.print_exc()
        dsB = None

    # ---------- 6. 对比 ROI ----------
    print("\n" + "=" * 78)
    print("ROI 对比（目标：模型估的 ROI 应接近真实 beta）")
    print("=" * 78)
    # 真实 ROI
    print("\n真实值:")
    for c in cfg.channel_cols:
        print(f"    {c:10s} 真 ROI = {TRUE_BETA[c]:.2f}")

    if dsA is not None:
        print("\n路线 A（官方守恒分解）:")
        for c in cfg.channel_cols:
            if c in dsA_mean.data_vars:
                gain = float(np.asarray(dsA_mean[c].values).sum())
                spend = float(get_spend_series(df, c).sum())
                print(f"    {c:10s} 估 ROI = {gain/spend:.3f}   "
                      f"(贡献 {gain/1e4:,.0f} 万元 / 花费 {spend/1e4:,.0f} 万元)")

    if dsB is not None:
        print("\n路线 B（官方反事实分解）:")
        for c in cfg.channel_cols:
            if c in dsB_mean.data_vars:
                gain = float(np.asarray(dsB_mean[c].values).sum())
                spend = float(get_spend_series(df, c).sum())
                print(f"    {c:10s} 估 ROI = {gain/spend:.3f}   "
                      f"(贡献 {gain/1e4:,.0f} 万元 / 花费 {spend/1e4:,.0f} 万元)")

    # ---------- 7. 路线 C：predict 原始尺度（对照） ----------
    print("\n" + "=" * 78)
    print("路线 C：mmm.predict(原始输入) × target_scale  [当前实现，对照]")
    print("=" * 78)
    X_raw = df[["date"] + cfg.channel_cols].copy()
    pred_raw = np.asarray(mmm.predict(X_raw)).ravel() * scale
    print(f"  predict×scale 合计 = {pred_raw.sum()/1e4:,.1f} 万元 "
          f"(真实 y = {float(df['revenue'].sum())/1e4:,.1f} 万元)")
    X_std = X.copy()
    pred_std = np.asarray(mmm.predict(X_std)).ravel() * scale
    print(f"  predict(标准化输入)×scale 合计 = {pred_std.sum()/1e4:,.1f} 万元")


if __name__ == "__main__":
    main()
