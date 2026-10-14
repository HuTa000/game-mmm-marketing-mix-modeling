# -*- coding: utf-8 -*-
"""
阶段 3 探针 v2：精确读优化窗口内的数据节点值
重点回答：
  A) control_data 在优化窗口内是否被归零？
  B) target_data / channel_data / channel_scale 是什么？
  C) 优化窗口的花费底座 = 历史值 还是 0？
  D) 决策期的 channel_data 是不是"待优化变量"（即会被 optimize_allocations 改）？
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

from data_generator import generate_game_data
from mmm_core import MMMConfig, prepare_data, build_mmm_safe, get_spend_series

SEP = "=" * 78
def sec(t): print("\n" + SEP + "\n" + t + "\n" + SEP)


def main():
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    cfg = MMMConfig()
    X, y, meta, used_ctrl = prepare_data(df, cfg)

    with open(os.path.join(ROOT, "outputs/reports/stage2_mmm_idata.pkl"), "rb") as f:
        idata = pickle.load(f)["idata"]

    mmm, _ = build_mmm_safe(cfg, used_ctrl)
    mmm.build_model(X, y); mmm.idata = idata

    start, end = "2025-06-08", "2025-08-31"
    opt = mmm.budget_optimizer(start, end)
    m = opt.model

    def getval(name):
        if name not in m.named_vars:
            return None
        node = m.named_vars[name]
        try:
            return np.asarray(node.get_value())
        except Exception:
            return np.asarray(node.eval()) if hasattr(node, "eval") else None

    sec("A. control_data 在优化窗口内的取值")
    cd = getval("control_data")
    if cd is not None:
        print(f"control_data shape = {cd.shape}  (date, control)")
        print(f"整体: min={cd.min():.6f} max={cd.max():.6f} mean={cd.mean():.6f} nonzero={np.count_nonzero(cd)}")
        arr = cd if cd.ndim == 2 else cd.reshape(1, -1)
        ctrl_names = list(np.asarray(m.coords["control"]))
        for j, cn in enumerate(ctrl_names):
            col = arr[:, j]
            print(f"  {cn:24s} min={col.min():+.6f} max={col.max():+.6f} mean={col.mean():+.6f} nonzero={np.count_nonzero(col)}/{len(col)}")
        # 训练期该控制变量的均值（标准化后应为 ~0）
        print("\n  [对照] 训练期标准化后的控制变量统计（应 mean≈0, std≈1）:")
        for cn in ctrl_names:
            if cn in X.columns:
                print(f"    {cn:24s} mean={X[cn].mean():+.4f} std={X[cn].std():.4f}")
    else:
        print("  control_data 不存在")

    sec("B. target_data / channel_scale / target_scale")
    for nm in ("target_data", "target_scale", "channel_scale"):
        v = getval(nm)
        print(f"  {nm}: shape={None if v is None else np.asarray(v).shape} "
              f"value={None if v is None else np.asarray(v).ravel()[:6]}")

    sec("C. channel_data（优化窗口花费底座）")
    ch = getval("channel_data")
    if ch is not None:
        print(f"shape={ch.shape} (date, channel)")
        names = list(np.asarray(m.coords["channel"]))
        # 这是缩放后的值；换回原始：× channel_scale × 模型内 channel 均值
        cs = getval("channel_scale")
        for j, cn in enumerate(names):
            col = ch[:, j]
            print(f"  {cn:10s} min={col.min():.6f} max={col.max():.6f} mean={col.mean():.6f} nonzero={np.count_nonzero(col)}/{len(col)}")
        print(f"\n  channel_scale = {cs}")
        # 与训练期 X 对比
        print("  [对照] 训练期 X 里各渠道（已÷均值）统计:")
        for cn in names:
            if cn in X.columns:
                print(f"    {cn:10s} mean={X[cn].mean():.6f} max={X[cn].max():.6f}")

    sec("D. optimize_allocations / 决策变量结构")
    print(f"  opt.num_periods = {opt.num_periods}")
    print(f"  opt.adstock_periods = {opt.adstock_periods}")
    print(f"  opt.carry_in_periods = {opt.carry_in_periods}")
    print(f"  opt.date_dim = {getattr(opt, 'date_dim', None)}")
    print(f"  opt.channel_data_var = {getattr(opt, 'channel_data_var', None)}")
    print(f"  opt.channel_contribution_var = {getattr(opt, 'channel_contribution_var', None)}")
    # 决策期在 date 轴上的位置
    dts = pd.to_datetime(np.asarray(m.coords["date"]))
    ci, cp = opt.carry_in_periods, opt.adstock_periods
    print(f"\n  日期轴: {dts[0].date()} ~ {dts[-1].date()} 共 {len(dts)} 期")
    print(f"    carry_in 区: idx 0..{ci-1}  ({dts[0].date()} ~ {dts[ci-1].date()})")
    print(f"    决策区:      idx {ci}..{len(dts)-cp-1}  ({dts[ci].date()} ~ {dts[len(dts)-cp-1].date()})")
    print(f"    carry_over 区: idx {len(dts)-cp}..{len(dts)-1}  ({dts[len(dts)-cp].date()} ~ {dts[-1].date()})")

    sec("E. 控制变量在优化里的实际贡献")
    cc = getval("control_contribution")
    if cc is not None:
        print(f"  control_contribution shape={cc.shape} sum={np.asarray(cc).sum():.6f}")
    print("  说明：若 control_data 全为 0，则 control_contribution 恒为 0")


if __name__ == "__main__":
    main()
