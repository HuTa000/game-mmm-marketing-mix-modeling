# -*- coding: utf-8 -*-
"""
阶段 3 探针 v10：终极标定 —— pack 到底做不做 channel_scales 换算？
设计一个"可证伪"的实验：
  模型是确定性的（给定后验均值）。若 pack 做除法：
    pack(货币 v) == pack(决策向量 v/scale)
  反之则不相等。
用一个已知响应单调的渠道，扫描不同货币量，看响应饱和点落在哪。
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, xarray as xr
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
    mmm, _ = build_mmm_safe(cfg, cfg.control_cols)
    mmm.build_model(X, y); mmm.idata = idata
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    spend_mean = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")

    sec("A. channel_scales 的实际作用")
    print(f"  opt.channel_scales = {np.asarray(opt.channel_scales)}")
    print(f"  spend_max          = {spend_max}")

    # 决策向量的形状/量级探测
    sec("B. 决策向量空间探测（用内部 API）")
    try:
        x0 = opt._variables.x0(hist_total.sum())
        print(f"  _variables.x0(total) shape={np.asarray(x0).shape}")
        print(f"    values = {np.round(np.asarray(x0),4)}")
        print(f"    (均匀分配时各渠道相等)")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")

    sec("C. bounds 的单位：内部如何解读 bounds")
    # 给货币 bounds，看 bounds 数组
    hist_total = spend_mean * 13
    try:
        bl = {c: (float(hist_total[i]) * 0.6, float(hist_total[i]) * 1.4) for i, c in enumerate(cfg.channel_cols)}
        b_arr = np.broadcast_to([np.asarray(bl[c]) for c in cfg.channel_cols], (5, 2))
        print(f"  货币 bounds 数组:\n    {b_arr}")
        bnds = opt._variables.bounds(hist_total.sum(), overrides={opt.channel_data_var: [(lo, hi) for lo, hi in b_arr[opt.budgets_to_optimize.values]]})
        print(f"  内部 bounds 对象: {bnds}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:200]}")

    sec("D. 直接扫描：pack(货币) vs pack(货币/scale)")
    from pytensor.graph import rewrite_graph
    from pytensor import function as pt_function
    m = opt._pymc_model; flat = opt._budgets_flat
    tgt = rewrite_graph(m["total_media_contribution_original_scale"].values,
                        include=("lower_xtensor", "canonicalize", "stabilize"))
    fn = pt_function([flat], tgt)

    def resp_from_decision(xvec):
        return float(np.asarray(fn(np.asarray(xvec, dtype=float))).sum())

    def pack_da(vals):
        return opt._pack_decision_vector(xr.DataArray(np.asarray(vals, dtype=float),
                                                      coords={"channel": cfg.channel_cols}, dims=("channel",)))

    idx = cfg.channel_cols.index("tiktok")
    print(f"  只投 tiktok，扫描 plan 值 v（直接当决策向量）：")
    print(f"    {'v':>16} {'response':>18}")
    for v in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        vv = np.zeros(5); vv[idx] = v
        print(f"    {v:>16.4f} {resp_from_decision(pack_da(vv)):>18,.0f}")

    print(f"\n  只投 tiktok，扫描货币 M（若自动÷scale，等价于 v=M/scale）：")
    print(f"    scale[tiktok] = {spend_max[idx]:,.0f}")
    print(f"    {'M(元)':>16} {'M/scale':>12} {'response':>18}")
    for Mv in (spend_max[idx]*0.1, spend_max[idx]*0.5, spend_max[idx], spend_max[idx]*5):
        vv = np.zeros(5); vv[idx] = Mv
        print(f"    {Mv:>16,.0f} {Mv/spend_max[idx]:>12.4f} {resp_from_decision(pack_da(vv)):>18,.0f}")

    sec("E. 与 D 对照：把货币换算成决策向量再传")
    print(f"    {'M(元)':>16} {'response (M/scale)':>20}")
    for Mv in (spend_max[idx]*0.1, spend_max[idx]*0.5, spend_max[idx], spend_max[idx]*5):
        vv = np.zeros(5); vv[idx] = Mv / spend_max[idx]
        print(f"    {Mv:>16,.0f} {resp_from_decision(pack_da(vv)):>20,.0f}")
    print("\n  ==> 若 D 与 E 的 response 相同，说明 pack 自动÷scale（bounds 也应按货币给）")
    print("  ==> 若 D 与 E 不同，说明 pack 不换算（bounds 应按决策向量给）")

    sec("F. allocate_budget 实际用的是哪种：对比两种 bounds 传法")
    # 传货币 bounds
    bl_money = {c: (float(hist_total[i]) * 0.9, float(hist_total[i]) * 1.1) for i, c in enumerate(cfg.channel_cols)}
    r1 = opt.allocate_budget(total_budget=hist_total.sum(), budget_bounds=bl_money)
    print(f"  传货币 bounds(0.9~1.1x): {np.round(np.asarray(r1.budgets),0)}")
    print(f"    sum = {np.asarray(r1.budgets).sum():,.0f}  (target {hist_total.sum():,.0f})")

    # 传决策向量 bounds
    bl_vec = {c: (float(hist_total[i]/spend_max[i]) * 0.9, float(hist_total[i]/spend_max[i]) * 1.1)
              for i, c in enumerate(cfg.channel_cols)}
    r2 = opt.allocate_budget(total_budget=hist_total.sum(), budget_bounds=bl_vec)
    print(f"  传决策向量bounds(0.9~1.1x): {np.round(np.asarray(r2.budgets),4)}")
    print(f"    sum = {np.asarray(r2.budgets).sum():,.4f}  (target {hist_total.sum():,.0f})")


if __name__ == "__main__":
    main()
