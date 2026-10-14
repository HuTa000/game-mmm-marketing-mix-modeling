"""
验证 v7：确定 predict 的返回空间 + 校准因子
============================================
核心疑问：predict 到底返回什么空间？

已知：
  · 真实 y 合计 = 23,943 万元 = 239,432,850 元
  · target_scale = 2,392,544.58
  · 239,432,850 / 2,392,545 = 100.06   ← 巧了！接近 100
  · 事实上 y 均值 = 1,534,826，y均值/scale = 0.641

  之前的结论"predict(标准化)×scale = 真实"要重新验证：
  23,947 万元 vs 23,943 万元 —— 这是对的。

但内置 channel_contribution × scale = 4,261,683 万元 —— 差 329 倍。

  → 说明内置 channel_contribution 的坐标轴（dim）和我的手工重算不同！
     内置 shape = (chain, draw, date, channel)，
     如果我先 .sum() 把 chain/draw/date/channel 全加了，
     得到的是 sum over 5渠道 × 156周，量级自然大。

  等等——那为什么 × ts 后是 4,261,683 万元？
  手工 = 12,941 万元，差 329 倍。
  329 ≈ ...  让我直接查 channel_contribution 的绝对值。

本脚本精确诊断：
  1. predict 返回值 vs 真实 y（原始空间？标准化空间？）
  2. channel_contribution 的真实数值范围
  3. 用 predict 做反事实，寻找正确的还原因子
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
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior

    ts = mmm.idata.constant_data["target_scale"]
    print("target_scale:", np.asarray(ts.values), "shape", np.asarray(ts.values).shape)
    ts_val = float(np.squeeze(np.asarray(ts.values)))

    true_y_sum = float(df["revenue"].sum())
    print(f"\n真实 y: sum={true_y_sum:,.0f}  mean={y.mean():,.0f}")
    print(f"y_sum / ts = {true_y_sum/ts_val:.2f}")
    print(f"y_mean / ts = {y.mean()/ts_val:.4f}")

    # ---------- 1. predict 的空间 ----------
    print("\n" + "=" * 78)
    print("1. mmm.predict(X) 返回值诊断")
    print("=" * 78)
    p = np.asarray(mmm.predict(X)).ravel()
    print(f"  predict(X)          : sum={p.sum():.3f}  mean={p.mean():.4f}  max={p.max():.3f}")
    print(f"  predict(X) × ts     : sum={p.sum()*ts_val:,.0f}  (真实 {true_y_sum:,.0f})")
    print(f"  比值 sum×ts/true    : {p.sum()*ts_val/true_y_sum:.6f}")
    print(f"  比值 sum/true       : {p.sum()/true_y_sum:.6f}")
    print(f"  → 结论: predict 返回" + ("标准化空间（需×ts）" if abs(p.sum()*ts_val/true_y_sum-1) < 0.05 else "原始空间"))

    # ---------- 2. channel_contribution 数值范围 ----------
    print("\n" + "=" * 78)
    print("2. 内置 channel_contribution 数值诊断")
    print("=" * 78)
    cc = pt["channel_contribution"].values
    print(f"  shape = {cc.shape}  整体: mean={cc.mean():.6f} max={cc.max():.4f} sum={cc.sum():.4f}")
    print(f"  单渠道(facebook) 单日: mean={cc[:,:,:,0].mean():.6f} "
          f"max={cc[:,:,:,0].max():.4f} sum={cc[:,:,:,0].sum():.4f}")
    print(f"\n  各渠道 sum（不乘 ts）:")
    for i, c in enumerate(cfg.channel_cols):
        s = cc[:, :, :, i].sum()
        print(f"    {c:10s} sum={s:>14.4f}   ×ts={s*ts_val:>18,.0f}  "
              f"×ts/1e4={s*ts_val/1e4:>14,.0f} 万元")
    print(f"\n  全部渠道 sum×ts = {cc.sum()*ts_val/1e4:,.0f} 万元")
    print(f"  但 predict 空间 sum×ts = {p.sum()*ts_val/1e4:,.0f} 万元")
    print(f"  → 若两者相等，则内置 CC 与 predict 同空间")

    # 对比：predict 的内部值
    print(f"\n  【对照】predict mean={p.mean():.6f}")
    print(f"  【对照】channel_contribution 全渠道全时间 mean = {cc.mean():.6f}")

    # ---------- 3. 反事实 + 正确还原 ----------
    print("\n" + "=" * 78)
    print("3. 反事实贡献（标准化输入，逐步寻找正确因子）")
    print("=" * 78)
    Xstd = X.copy()
    pb = np.asarray(mmm.predict(Xstd)).ravel()
    print(f"  predict(Xstd) sum = {pb.sum():.4f}")
    contrib_raw = {}
    for c in cfg.channel_cols:
        Xcf = Xstd.copy(); Xcf[c] = 0.0
        contrib_raw[c] = pb - np.asarray(mmm.predict(Xcf)).ravel()
    raw_sum = sum(v.sum() for v in contrib_raw.values())
    print(f"  反事实贡献 raw sum = {raw_sum:.4f}")
    print(f"  ×ts = {raw_sum*ts_val:,.0f}  (真实媒体 11,178 万元 = {11178e4:,.0f})")

    # 各种因子尝试
    print(f"\n  尝试不同还原因子:")
    for name, factor in [("ts", ts_val), ("1.0", 1.0), ("y_mean", y.mean()),
                         ("y_mean/ts*ts", y.mean()), ("ts*0.0143", ts_val*0.0143)]:
        m = raw_sum * factor
        print(f"    ×{name:14s} = {m/1e4:>14,.0f} 万元  "
              f"(媒体占比 {m/true_y_sum:.1%})")

    # ---------- 4. 直接看 mmm 的原始预测分量 ----------
    print("\n" + "=" * 78)
    print("4. 用官方 API 拿正确的渠道贡献（推荐方案）")
    print("=" * 78)
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    print(f"  get_contributions 各渠道 ×ts? 先看原始值:")
    for i, c in enumerate(cfg.channel_cols):
        v = float(np.asarray(chA.sel(channel=c).values).sum())
        s_spend = float(get_spend_series(df, c).sum())
        print(f"    {c:10s} 原始sum={v:>14,.0f}  花费={s_spend:>12,.0f}  ROI={v/s_spend:.3f}")


if __name__ == "__main__":
    main()
