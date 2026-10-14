# -*- coding: utf-8 -*-
"""
阶段 3 探针脚本（一次性验证，跑通后可删）
目的：
  1) 确认 BudgetOptimizer 在优化窗口内把控制变量填成了什么值（怀疑被归零）
  2) 确认 mmm.budget_optimizer(start_date, end_date) 推荐入口能否跑通
  3) 摸清 allocate_budget 返回结构
  4) 确认 create_optimization_model 的窗口参数推断

跑法： .venv/Scripts/python.exe src/probe_stage3.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from data_generator import generate_game_data
from mmm_core import MMMConfig, prepare_data, get_spend_series

SEP = "=" * 78


def sec(t):
    print("\n" + SEP)
    print(t)
    print(SEP)


def main():
    # ---------- 0. 数据 ----------
    sec("0. 生成数据 + 读取阶段 2 模型")
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    cfg = MMMConfig()
    X, y, meta, used_ctrl = prepare_data(df, cfg)
    print(f"数据: {df.shape}, 日期 {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"控制变量: {used_ctrl}")
    print(f"渠道: {cfg.channel_cols}")

    pkl = os.path.join(ROOT, "outputs", "reports", "stage2_mmm_idata.pkl")
    if not os.path.exists(pkl):
        print("[!] 没找到 stage2_mmm_idata.pkl，请先跑 stage2_model.py")
        return
    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    idata = blob["idata"]
    print(f"读入 idata: {list(idata.posterior.data_vars)[:8]} ...")
    print(f"控制变量在 posterior 里？ 'gamma_control' in posterior -> {'gamma_control' in idata.posterior}")

    # ---------- 1. 重建 MMM 对象（build_model 建结构，注入 idata，不重采样） ----------
    sec("1. 从存档重建 MMM 对象（build_model + 注入 idata，不重新采样）")
    from mmm_core import build_mmm_safe
    mmm, note = build_mmm_safe(cfg, used_ctrl)
    print(f"build_mmm_safe note: {note}")
    # 【关键】build_model 会设置 xarray_dataset，create_optimization_model 依赖它
    #        但 build_model 只建结构不采样，很快
    mmm.build_model(X, y)
    print(f"xarray_dataset 已建立: {type(mmm.xarray_dataset)}")
    print(f"   dims: {dict(mmm.xarray_dataset.sizes)}")
    # 注入 idata（不重新采样）
    mmm.idata = idata
    print("已注入 posterior。")

    # ---------- 2. 探测窗口 ----------
    sec("2. 探测优化窗口 / carryover 结构")
    try:
        eff = mmm.effective_carryover_lags()
        print(f"effective_carryover_lags() = {eff}")
    except Exception as e:
        print(f"[!] effective_carryover_lags 失败: {type(e).__name__}: {e}")

    dates = pd.to_datetime(df["date"])
    print(f"adstock l_max = {cfg.adstock_l_max}")
    print(f"数据日期数 = {len(dates)}")

    # 取一个 13 周的优化窗口（最后 13 周之前留出 carryover）
    # 为了看清楚，取中间一段
    opt_start = dates.iloc[-30].date()
    opt_end = dates.iloc[-18].date()
    print(f"试探优化窗口: {opt_start} ~ {opt_end}")

    # ---------- 3. create_optimization_model 全貌 ----------
    sec("3. create_optimization_model 返回的 pm.Model 里有什么")
    try:
        opt_model = mmm.create_optimization_model(opt_start, opt_end)
        print(f"返回类型: {type(opt_model)}")
        print(f"自由变量 (free_RVs): {[v.name for v in opt_model.free_RVs]}")
        print(f"观测变量 (observed_RVs): {[v.name for v in opt_model.observed_RVs]}")
        print(f"固定数据 (Data vars): {[v.name for v in opt_model.data_vars]}")
        print(f"坐标: {list(opt_model.coords.keys())}")
        for k, v in opt_model.coords.items():
            arr = np.asarray(v)
            print(f"   coords[{k}]: len={len(arr)} 前5={list(arr[:5])}")
        # 找渠道数据节点
        for name in ("channel_data", "channel_spend"):
            if name in opt_model.named_vars:
                node = opt_model.named_vars[name]
                print(f"\n[{name}] type={type(node)}")
                try:
                    val = node.get_value() if hasattr(node, "get_value") else None
                    if val is not None:
                        print(f"   shape={np.asarray(val).shape}")
                        print(f"   sum={np.asarray(val).sum():.4f}  mean={np.asarray(val).mean():.4f}")
                except Exception as e2:
                    print(f"   取值失败: {e2}")
        # 控制变量数据节点
        print("\n--- 控制变量节点检查 ---")
        for cv in used_ctrl:
            found = [n for n in opt_model.named_vars if cv in n.lower()]
            print(f"  控制变量 '{cv}': 相关节点 = {found}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] create_optimization_model 失败: {type(e).__name__}: {e}")

    # ---------- 4. budget_optimizer 推荐入口 ----------
    sec("4. mmm.budget_optimizer() 推荐入口")
    try:
        opt = mmm.budget_optimizer(
            opt_start, opt_end,
            response_variable="total_media_contribution_original_scale",
        )
        print(f"返回类型: {type(opt)}")
        for attr in ("num_periods", "adstock_periods", "carry_in_periods",
                     "channel_scales", "budgets_to_optimize", "cost_per_unit",
                     "response_variable", "utility_function"):
            if hasattr(opt, attr):
                v = getattr(opt, attr)
                try:
                    vs = np.asarray(v)
                    print(f"   .{attr} = {v if vs.ndim == 0 else vs[:8]}")
                except Exception:
                    print(f"   .{attr} = {v}")

        # 看优化窗口内的 channel_data（这就是花不花钱的决策变量底座？）
        print("\n--- 优化窗口模型里的数据节点 ---")
        m = opt.model
        for n in m.named_vars:
            if any(k in n.lower() for k in ("channel", "control", "spend", "data")):
                print(f"   {n}")

        # ---------- 5. allocate_budget ----------
        sec("5. allocate_budget 试跑")
        total = float(get_spend_series(df, "facebook").mean())  # 单渠道均值作参照
        current_total = sum(float(get_spend_series(df, c).mean()) for c in cfg.channel_cols)
        # 窗口内当前花费（用决策期数 × 周均），先粗算
        print(f"全渠道周均总花费 ≈ {current_total:,.0f} 元")

        # 先不含约束试跑
        result = opt.allocate_budget(total_budget=current_total * 4)  # 4 周窗口的预算
        print(f"allocate_budget 返回类型: {type(result)}")
        print(f"   dir: {[a for a in dir(result) if not a.startswith('_')]}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] budget_optimizer / allocate_budget 失败: {type(e).__name__}: {e}")

    sec("探针结束")


if __name__ == "__main__":
    main()
