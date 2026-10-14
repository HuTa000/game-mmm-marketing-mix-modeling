# -*- coding: utf-8 -*-
"""
阶段 3 探针 v11：用【官方 API 自身的口径】做最终结算
规则（v10 已证）：
  - allocate_budget(total_budget=货币, budget_bounds=货币) => 返回 budgets 也是货币
  - evaluate_plan(plan) 的 plan 要用【货币 + evaluate_plan 内部会换算?】—— 待测
最终问题只有一个：
  evaluate_plan(货币 plan) 与 allocate_budget 的 objective 是否一致？
  => 用 allocate_budget 返回的 budgets 直接喂 evaluate_plan，若 utility == -scipy_result.fun，则口径统一
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
    hist_total = spend_mean * 13
    TB = hist_total.sum()

    def da(vals):
        return xr.DataArray(np.asarray(vals, float), coords={"channel": cfg.channel_cols}, dims=("channel",))

    sec("A. allocate_budget 的 budgets vs scipy_result.fun")
    r = opt.allocate_budget(total_budget=TB)
    b = np.asarray(r.budgets)
    print(f"  budgets = {np.round(b,0)}")
    print(f"  scipy fun = {r.scipy_result.fun:,.4f}   => objective = {r.scipy_result.fun:,.4f}")
    print(f"  scipy nit = {r.scipy_result.nit}")

    sec("B. evaluate_plan(货币 budgets) 是否复现同一个 objective")
    ev = opt.evaluate_plan(da(b))
    print(f"  evaluate_plan(budgets).objective = {ev.objective:,.4f}")
    print(f"  scipy_result.fun                 = {r.scipy_result.fun:,.4f}")
    print(f"  一致？ {np.isclose(ev.objective, r.scipy_result.fun, rtol=1e-6)}")

    sec("C. 用 evaluate_plan 比较 历史 vs 优化（同一个正确的尺子）")
    ev_h = opt.evaluate_plan(da(hist_total))
    ev_o = opt.evaluate_plan(da(b))
    print(f"  历史分配 utility = {ev_h.utility:,.4f}")
    print(f"  优化分配 utility = {ev_o.utility:,.4f}")
    print(f"  提升 = {(ev_o.utility/ev_h.utility-1)*100:+.2f}%")

    sec("D. 加合理边界约束（货币）后再比较")
    for lo_m, hi_m in ((0.8, 1.2), (0.6, 1.4), (0.5, 2.0)):
        bl = {c: (float(hist_total[i]*lo_m), float(hist_total[i]*hi_m)) for i, c in enumerate(cfg.channel_cols)}
        try:
            rr = opt.allocate_budget(total_budget=TB, budget_bounds=bl)
        except Exception as e:
            print(f"  bounds[{lo_m},{hi_m}]: 优化失败 ({type(e).__name__}: {str(e)[:60]})，尝试 return_if_fail=True")
            try:
                rr = opt.allocate_budget(total_budget=TB, budget_bounds=bl, return_if_fail=True)
                print(f"     return_if_fail 成功，success={rr.scipy_result.success}")
            except Exception as e2:
                print(f"     仍失败: {type(e2).__name__}: {str(e2)[:80]}")
                continue
        bb = np.asarray(rr.budgets)
        eo = opt.evaluate_plan(da(bb))
        eh = opt.evaluate_plan(da(hist_total))
        print(f"  bounds[{lo_m},{hi_m}]: {np.round(bb,0)}")
        print(f"      提升 = {(eo.utility/eh.utility-1)*100:+.2f}%   nit={rr.scipy_result.nit}  success={rr.scipy_result.success}")

    sec("E. 用增量口径（扣除 zero-spend 常数底）")
    ev_z = opt.evaluate_plan(da(np.zeros(5)))
    print(f"  zero-spend utility(常数底) = {ev_z.utility:,.4f}")
    ev_h2 = opt.evaluate_plan(da(hist_total))
    bl = {c: (float(hist_total[i]*0.6), float(hist_total[i]*1.4)) for i, c in enumerate(cfg.channel_cols)}
    rr = opt.allocate_budget(total_budget=TB, budget_bounds=bl)
    ev_o2 = opt.evaluate_plan(da(np.asarray(rr.budgets)))
    d_base = ev_z.utility
    print(f"  历史 utility = {ev_h2.utility:,.4f}")
    print(f"  优化 utility = {ev_o2.utility:,.4f}")
    print(f"  历史增量 = {ev_h2.utility - d_base:,.4f}")
    print(f"  优化增量 = {ev_o2.utility - d_base:,.4f}")
    inc_h = ev_h2.utility - d_base
    inc_o = ev_o2.utility - d_base
    print(f"  增量提升 = {(inc_o/inc_h-1)*100:+.2f}%" if inc_h > 0 else "  增量提升 = N/A(历史增量<=0)")
    print(f"  总量提升 = {(ev_o2.utility/ev_h2.utility-1)*100:+.2f}%")
    print(f"  常数底占比 = {d_base/ev_h2.utility*100:.1f}%")

    sec("F. 后验稳健性：evaluate_response_distribution")
    for tag, v in (("历史", hist_total), ("优化", np.asarray(rr.budgets)), ("零花费", np.zeros(5))):
        d = np.asarray(opt.evaluate_response_distribution(da(v)))
        print(f"  {tag:8s} mean={np.nanmean(d):>16,.0f}  std={np.nanstd(d):>12,.0f} "
              f"p5={np.nanpercentile(d,5):>16,.0f} p95={np.nanpercentile(d,95):>16,.0f}")

    sec("G. 全押检验：不设约束时的极限行为")
    r2 = opt.allocate_budget(total_budget=TB)
    print(f"  无约束 budgets = {np.round(np.asarray(r2.budgets),0)}  (应全押某渠道)")


if __name__ == "__main__":
    main()
