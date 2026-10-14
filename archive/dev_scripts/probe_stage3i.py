# -*- coding: utf-8 -*-
"""
阶段 3 探针 v9：验证"加边界约束"后优化是否稳定、可解释
关键认知（v8 已证）：
  utility 含一个 ~697万的"carry-in 常数底"（决策期花费=0时仍存在）
  真正可优化的是"增量部分"
设计：
  1) 加 budget_bounds（每渠道 0.5x~1.5x 当前值），看结果是否稳定、合理
  2) 对比不同窗口长度，看 carry-in 占比
  3) 用"历史分配"作为 x0（warm start），对比 x0=None（均匀起点）
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


def build(df, cfg, X, y, idata):
    mmm, _ = build_mmm_safe(cfg, cfg.control_cols)
    mmm.build_model(X, y); mmm.idata = idata
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    return mmm


def main():
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    cfg = MMMConfig()
    X, y, meta, used_ctrl = prepare_data(df, cfg)
    with open(os.path.join(ROOT, "outputs/reports/stage2_mmm_idata.pkl"), "rb") as f:
        idata = pickle.load(f)["idata"]
    mmm = build(df, cfg, X, y, idata)

    spend_mean = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    hist_weekly = spend_mean.copy()
    N_WEEKS = 13
    hist_total = hist_weekly * N_WEEKS

    sec("1. 对比不同窗口长度：carry-in 常数底占比")
    from pytensor.graph import rewrite_graph
    from pytensor import function as pt_function
    for (s, e, nw) in [("2025-03-02", "2025-05-25", 13),
                       ("2025-06-08", "2025-08-31", 13),
                       ("2025-02-02", "2025-09-28", 35)]:
        try:
            o = mmm.budget_optimizer(s, e)
            m = o._pymc_model
            flat = o._budgets_flat
            tgt = rewrite_graph(m["total_media_contribution_original_scale"].values,
                                include=("lower_xtensor", "canonicalize", "stabilize"))
            fn = pt_function([flat], tgt)
            plan = xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",))
            x = o._pack_decision_vector(plan)
            v_hist = float(np.asarray(fn(x)).sum())
            v_zero = float(np.asarray(fn(np.zeros_like(x))).sum())
            print(f"  窗口 {s}~{e} ({o.num_periods}期): hist={v_hist:>14,.0f} zero={v_zero:>14,.0f} "
                  f"底占比={v_zero/v_hist*100:>6.1f}%  增量={v_hist-v_zero:>12,.0f}")
        except Exception as ex:
            print(f"  窗口 {s}~{e} 失败: {type(ex).__name__}: {str(ex)[:100]}")

    sec("2. 加边界约束（0.6x ~ 1.4x 当前）")
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    bounds = {c: (float(hist_total[i]) * 0.6, float(hist_total[i]) * 1.4)
              for i, c in enumerate(cfg.channel_cols)}
    print("  bounds:")
    for c, (lo, hi) in bounds.items():
        print(f"    {c:10s} [{lo:>14,.0f}, {hi:>14,.0f}]  当前 {hist_total[cfg.channel_cols.index(c)]:>14,.0f}")

    res = opt.allocate_budget(total_budget=hist_total.sum(), budget_bounds=bounds, callback=True)
    b = np.asarray(res.budgets)
    print(f"\n  优化结果: {np.round(b,0)}")
    print(f"  scipy nit={res.scipy_result.nit} fun={res.scipy_result.fun:,.2f} success={res.scipy_result.success}")
    print(f"\n  {'渠道':10s} {'当前':>14} {'优化':>14} {'变化':>10} {'上限':>14}")
    for i, c in enumerate(cfg.channel_cols):
        hi = bounds[c][1]
        print(f"  {c:10s} {hist_total[i]:>14,.0f} {b[i]:>14,.0f} {(b[i]/hist_total[i]-1)*100:>9.1f}% {hi:>14,.0f}")

    sec("3. x0 敏感性（均匀起点 vs 历史起点）")
    for tag, x0 in (("None(均匀)", None), ("历史分配", xr.DataArray(hist_total, coords={"channel": cfg.channel_cols}, dims=("channel",)))):
        r = opt.allocate_budget(total_budget=hist_total.sum(), budget_bounds=bounds, x0=x0)
        print(f"  x0={tag:10s} -> {np.round(np.asarray(r.budgets),0)}  fun={r.scipy_result.fun:,.2f} nit={r.scipy_result.nit}")

    sec("4. 用增量口径评估（剔除常数底）")
    from pytensor.graph import rewrite_graph
    from pytensor import function as pt_function
    m = opt._pymc_model; flat = opt._budgets_flat
    tgt = rewrite_graph(m["total_media_contribution_original_scale"].values,
                        include=("lower_xtensor", "canonicalize", "stabilize"))
    fn = pt_function([flat], tgt)
    v0 = float(np.asarray(fn(np.zeros_like(opt._pack_decision_vector(xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",)))))).sum())

    def val(vals):
        xi = opt._pack_decision_vector(xr.DataArray(np.asarray(vals), coords={"channel": cfg.channel_cols}, dims=("channel",)))
        return float(np.asarray(fn(xi)).sum())

    v_hist = val(hist_total)
    v_opt = val(b)
    print(f"  常数底(zero-spend)     = {v0:>16,.0f}")
    print(f"  历史分配 总量          = {v_hist:>16,.0f}   增量={v_hist-v0:>14,.0f}")
    print(f"  优化分配 总量          = {v_opt:>16,.0f}   增量={v_opt-v0:>14,.0f}")
    print(f"  => 增量提升 = {((v_opt-v0)/(v_hist-v0)-1)*100:+.2f}%")
    print(f"  => 总量提升 = {(v_opt/v_hist-1)*100:+.2f}%")

    sec("5. 收敛性：多次运行是否稳定（不同 total_budget）")
    for mult in (0.8, 1.0, 1.2):
        tb = hist_total.sum() * mult
        bb = {c: (float(hist_total[i] / hist_total.sum() * tb) * 0.6,
                  float(hist_total[i] / hist_total.sum() * tb) * 1.4)
              for i, c in enumerate(cfg.channel_cols)}
        r = opt.allocate_budget(total_budget=tb, budget_bounds=bb)
        rb = np.asarray(r.budgets)
        print(f"  total×{mult}: {np.round(rb,0)}")


if __name__ == "__main__":
    main()
