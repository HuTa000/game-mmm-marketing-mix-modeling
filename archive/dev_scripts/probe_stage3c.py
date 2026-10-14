# -*- coding: utf-8 -*-
"""
阶段 3 探针 v3：验证单位换算的正确性（这是数字对不对的命门）
问题：
  Q1) target_scale = 2,923,358 是 y.max() 还是 y.mean()？要与我 cfg.target_scaling="mean" 一致
  Q2) BudgetOptimizer.channel_scales = 1.0 是什么？和模型内 channel_scale 什么关系？
  Q3) allocate_budget 返回的 budgets 单位是"元"吗？
  Q4) 用返回的 budgets 反推响应，与 evaluate_plan 是否一致？
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

    sec("Q1. target_scale 对照")
    ts = float(np.asarray(mmm.model.named_vars["target_scale"].get_value()))
    print(f"  模型内 target_scale       = {ts:,.2f}")
    print(f"  y.mean() (原始尺度)       = {y.mean():,.2f}")
    print(f"  y.max()  (原始尺度)       = {y.max():,.2f}")
    print(f"  y.mean()/y.max()          = {y.mean()/y.max():.6f}")
    which = "mean" if abs(ts - y.mean()) < abs(ts - y.max()) else "max"
    print(f"  ==> target_scale 用的是  : {which}")
    print(f"  ==> 与 cfg.target_scaling 一致？ {which == cfg.target_scaling}")

    sec("Q2. channel_scales 是什么")
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    print(f"  BudgetOptimizer.channel_scales = {np.asarray(opt.channel_scales)}")
    print(f"  模型内 channel_scale           = {np.asarray(mmm.model.named_vars['channel_scale'].get_value())}")
    print(f"  opt.cost_per_unit = {opt.cost_per_unit}")
    print(f"  opt.budgets_to_optimize = {np.asarray(opt.budgets_to_optimize)}")
    print("  说明：channel_data 在模型里已是 [花费/渠道均值]/channel_scale")
    print("        channel_scales 若是 1.0，说明优化器认为'花费输入已是货币单位以外的一致尺度'")

    sec("Q3. allocate_budget 输出单位")
    # 决策区 13 周，历史周均总花费
    weekly_total = {c: float(get_spend_series(df, c).mean()) for c in cfg.channel_cols}
    hist_sum = sum(weekly_total.values())
    print(f"  历史周均总花费 = {hist_sum:,.0f} 元/周")
    print(f"  13 周历史总花费 = {hist_sum*13:,.0f} 元")
    print(f"  各渠道周均: " + ", ".join(f"{c}={v:,.0f}" for c, v in weekly_total.items()))

    total_budget = hist_sum * 13
    res = opt.allocate_budget(total_budget=total_budget)
    budgets = np.asarray(res.budgets)
    print(f"\n  total_budget 传入 = {total_budget:,.0f}")
    print(f"  res.budgets = {budgets}")
    print(f"  res.budgets.sum() = {budgets.sum():,.2f}")
    print(f"  ==> sum == total_budget ? {np.isclose(budgets.sum(), total_budget)}")
    print(f"  res.budgets dtype/shape = {budgets.dtype} {budgets.shape}")
    if hasattr(res, "spend_var_names"):
        print(f"  res.spend_var_names = {res.spend_var_names}")
    if hasattr(res, "optimized_vars"):
        ov = res.optimized_vars
        print(f"  res.optimized_vars keys = {list(ov.keys()) if hasattr(ov,'keys') else type(ov)}")
        try:
            for k, v in ov.items():
                print(f"     {k}: shape={np.asarray(v).shape} head={np.asarray(v).ravel()[:4]}")
        except Exception as e:
            print(f"     展开失败: {e}")
    if hasattr(res, "scipy_result"):
        sr = res.scipy_result
        print(f"  scipy_result.success = {getattr(sr,'success',None)}")
        print(f"  scipy_result.fun     = {getattr(sr,'fun',None)}")
        print(f"  scipy_result.message = {getattr(sr,'message',None)}")
        print(f"  scipy_result.nit     = {getattr(sr,'nit',None)}")

    sec("Q4. evaluate_plan 与 budgets 一致性")
    try:
        er = opt.evaluate_plan(budgets, total_budget=total_budget) if "total_budget" in (opt.evaluate_plan.__code__.co_varnames) else opt.evaluate_plan(budgets)
        print(f"  evaluate_plan 返回类型: {type(er)}")
        print(f"  {er if not hasattr(er,'shape') else np.asarray(er).shape}")
    except TypeError as e:
        print(f"  evaluate_plan(budgets) 签名不符: {e}")
        try:
            er = opt.evaluate_plan(budgets)
            print(f"  重试成功: {type(er)}")
        except Exception as e2:
            print(f"  仍失败: {type(e2).__name__}: {e2}")
    except Exception as e:
        print(f"  evaluate_plan 失败: {type(e).__name__}: {e}")

    sec("Q5. 历史分配下（不优化）的响应，作为 baseline")
    hist_alloc = np.array([weekly_total[c] * 13 for c in cfg.channel_cols])
    print(f"  历史分配 13 周: {hist_alloc}")
    print(f"  历史分配 sum  : {hist_alloc.sum():,.0f}")
    try:
        plan_hist = hist_alloc / hist_alloc.sum()
        # evaluate_plan 可能接受"分配向量"（按渠道）
        er2 = opt.evaluate_plan(hist_alloc)
        er2v = np.asarray(er2)
        print(f"  evaluate_plan(历史分配) -> shape={er2v.shape} mean={er2v.mean():,.2f}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
