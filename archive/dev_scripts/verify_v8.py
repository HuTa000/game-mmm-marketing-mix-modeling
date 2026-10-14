"""
验证 v8：拆解 mu 的构成，找出 1.5 倍差异
==========================================
已确认：
  · predict(X).sum() = 100.08   （模型最终预测，标准化空间）
  · channel_contribution.sum() = 150.06  （5渠道 × 156周，标准化空间）
  · 差 1.5 倍

identity link 下模型是：y = intercept + Σ channel_contribution + Σ fourier + controls
理论上各项之和应 = predict 输出。

本脚本把每一项列出来，看差在哪里。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, used_ctrl = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior

    print("=" * 88)
    print("后验 posterior 数据变量清单（带 shape）")
    print("=" * 88)
    for v in pt.data_vars:
        print(f"  {v:38s} {tuple(pt[v].shape)}")

    print("\n" + "=" * 88)
    print("各分量求和（标准化空间，取后验均值）")
    print("=" * 88)

    n_chain, n_draw = pt.sizes["chain"], pt.sizes["draw"]
    nd = n_chain * n_draw
    items = []
    for v in pt.data_vars:
        arr = pt[v].values
        # 只保留有 date 维或标量的
        if "date" in pt[v].dims:
            val = arr.mean(axis=(0, 1)).sum()  # 对 date 求和，链/抽平均
            items.append((v, val, arr.mean(axis=(0, 1)))) 
        else:
            val = arr.mean()
            items.append((v, val, val))

    print(f"{'变量':38s} {'时间维和(标准化)':>18s} {'×ts(万元)':>16s}")
    print("-" * 76)
    for v, val, _ in items:
        if np.isscalar(val):
            print(f"{v:38s} {val:>18.4f} (无date维)")
        else:
            print(f"{v:38s} {val:>18.4f} {val*2392544.58/1e4:>16,.0f}")

    # 关键：channel_contribution 沿 channel 维求和
    print("\n" + "=" * 88)
    print("关键：channel_contribution 沿 channel 维求和后的时间序列 vs 其它项")
    print("=" * 88)
    cc = pt["channel_contribution"].values  # (chain,draw,date,channel)
    cc_total = cc.sum(axis=3).mean(axis=(0, 1))  # 对渠道求和 → (date,)
    print(f"  channel_contribution 沿渠道求和: sum={cc_total.sum():.4f}  "
          f"mean={cc_total.mean():.4f}")

    # 各其它项
    for v in pt.data_vars:
        if v == "channel_contribution":
            continue
        if "date" in pt[v].dims:
            arr = pt[v].values
            # 若有 channel 维也求和
            axes = tuple(i for i, d in enumerate(pt[v].dims) if d in ("chain", "draw"))
            s = arr.mean(axis=axes)
            remaining = tuple(i for i, d in enumerate(pt[v].dims) if d not in ("chain", "draw"))
            if remaining:
                s = s.sum(axis=tuple(range(len(remaining) - 1)))
            print(f"  {v:38s} sum={np.asarray(s).sum():.4f}")

    # predict
    p = np.asarray(mmm.predict(X)).ravel()
    print(f"\n  mmm.predict(X)  sum={p.sum():.4f}")

    # 手工加总
    print("\n" + "=" * 88)
    print("手工加总检验（identity link 应守恒）")
    print("=" * 88)
    total = cc_total.sum()
    print(f"  Σ channel_contribution   = {total:.4f}")
    print(f"  predict(X)               = {p.sum():.4f}")
    print(f"  差 = {p.sum() - total:.4f}")
    print(f"  比值 predict/Σcc = {p.sum()/total:.4f}")

    # 检查是否 channel_scale 的影响
    print("\n  检查 constant_data:")
    cd = mmm.idata.constant_data
    for v in cd.data_vars:
        arr = np.asarray(cd[v].values)
        print(f"    {v:24s} shape={arr.shape}  mean={arr.mean():.4f} "
              f"sum={arr.sum():.4f}")


if __name__ == "__main__":
    main()
