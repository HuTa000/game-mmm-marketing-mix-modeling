# -*- coding: utf-8 -*-
"""
阶段 3 探针 v4：验证 channel_scales 修复是否解决"均匀分配 + nit=1"

根因假设：
  channel_data 节点存的是 spend / max(spend)（双重缩放后的值）
  但 BudgetOptimizer 默认 channel_scales=1.0，以为节点存的是原始货币
  => 写进去的预算被当成 1/max 缩放值 => 量级爆炸 => 全渠道饱和 => 梯度消失

修复：channel_scales[c] = max(spend_c)  (训练期原始元)
      因为 节点值 = spend / max(spend)  =>  spend = 节点值 × max(spend)
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

    sec("STEP 1. 反推 channel_data 节点的真实含义")
    model_ch_scale = np.asarray(mmm.model.named_vars["channel_scale"].get_value())
    ch_mean = {c: float(X[c].mean()) for c in cfg.channel_cols}     # prepare_data 里已÷均值 => 应为 1.0
    spend_max = {c: float(get_spend_series(df, c).max()) for c in cfg.channel_cols}
    spend_mean = {c: float(get_spend_series(df, c).mean()) for c in cfg.channel_cols}
    print(f"  模型 channel_scale       = {model_ch_scale}")
    print(f"  X 各渠道均值(应=1.0)     = {[round(ch_mean[c],4) for c in cfg.channel_cols]}")
    print(f"  原始花费 max (元)        = {[round(spend_max[c],1) for c in cfg.channel_cols]}")
    print(f"  原始花费 mean(元)        = {[round(spend_mean[c],1) for c in cfg.channel_cols]}")

    # 验证：节点值 = spend / spend_max ？
    # 【修复】MMM.budget_optimizer 内部用 getattr(self,"_channel_scales",1.0)，
    #        MMM 无此属性 => 回退 1.0 => 单位错。手动挂上正确的换算系数。
    ch_scale_fix = np.array([spend_max[c] for c in cfg.channel_cols])
    mmm._channel_scales = ch_scale_fix
    print(f"\n  手动挂载 mmm._channel_scales = {ch_scale_fix}")

    # 闭环验证：channel_scale × 渠道均值 是否 = spend_max
    for i, c in enumerate(cfg.channel_cols):
        lhs = model_ch_scale[i] * spend_mean[c]
        print(f"    {c:9s} channel_scale×mean = {lhs:>14,.1f}  vs  spend_max = {spend_max[c]:>14,.1f}  "
              f"{'OK' if abs(lhs-spend_max[c])<1 else 'MISMATCH'}")

    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    print(f"\n  BudgetOptimizer.channel_scales = {np.asarray(opt.channel_scales)}")

    # 用训练期某一周验证换算
    dts = pd.to_datetime(np.asarray(opt.model.coords["date"]))
    # 取训练期最后一个窗口外的一周做对照不易，这里直接验证公式：
    # 节点值 × channel_scales 应该 ≈ 货币花费
    idx0 = np.where(dts == pd.Timestamp("2025-06-08"))[0]
    print(f"  决策起点的 date idx = {idx0}")
    cd = np.asarray(opt.model.named_vars["channel_data"].get_value())
    print(f"  决策起点 channel_data（应为0，因为待优化）= {cd[idx0[0]] if len(idx0) else 'N/A'}")

    sec("STEP 2. 用修复后的 channel_scales 重新优化")
    weekly_total = {c: spend_mean[c] for c in cfg.channel_cols}
    hist_sum = sum(weekly_total.values())
    total_budget = hist_sum * 13
    print(f"  13周总预算 = {total_budget:,.0f} 元")

    res = opt.allocate_budget(total_budget=total_budget, callback=True)
    budgets = np.asarray(res.budgets)
    print(f"\n  res.budgets = {budgets}")
    print(f"  sum = {budgets.sum():,.2f}  (target {total_budget:,.2f})")
    print(f"  scipy: success={res.scipy_result.success} nit={res.scipy_result.nit} fun={res.scipy_result.fun:,.2f}")
    print(f"  message={res.scipy_result.message}")

    print("\n  --- 优化轨迹（每步 fun / ||jac||）---")
    if res.callback_info:
        for i, it in enumerate(res.callback_info[:15]):
            jn = np.linalg.norm(np.asarray(it["jac"]))
            print(f"    iter {i}: fun={it['fun']:,.2f}  ||jac||={jn:.6e}")
        print(f"    总迭代数 = {len(res.callback_info)}")
    else:
        print("    callback_info 为空")

    print("\n  --- 与历史分配对比 ---")
    for c, b in zip(cfg.channel_cols, budgets):
        cur = weekly_total[c] * 13
        print(f"    {c:9s} 当前={cur:>12,.0f}  优化={b:>12,.0f}  变化={(b/cur-1)*100:+7.1f}%")

    sec("STEP 3. 尝试 evaluate_plan（用正确的 DataArray plan）")
    import xarray as xr
    try:
        # plan 需要是 {channel_data: DataArray} 或 DataArray
        plan_da = xr.DataArray(
            budgets / budgets.sum(),   # 或 budgets 本身？
            coords={"channel": cfg.channel_cols},
            dims=("channel",),
        )
        out = opt.evaluate_plan(plan_da)
        print(f"  evaluate_plan(归一化 plan) -> {out}")
    except Exception as e:
        print(f"  [1] 归一化 plan 失败: {type(e).__name__}: {str(e)[:200]}")
        try:
            plan_da = xr.DataArray(budgets, coords={"channel": cfg.channel_cols}, dims=("channel",))
            out = opt.evaluate_plan(plan_da)
            print(f"  evaluate_plan(货币 plan) -> {out}")
        except Exception as e2:
            print(f"  [2] 货币 plan 失败: {type(e2).__name__}: {str(e2)[:200]}")

    sec("STEP 4. evaluate_response_distribution")
    try:
        dist = opt.evaluate_response_distribution(plan_da)
        d = np.asarray(dist)
        print(f"  shape={d.shape}  mean={np.nanmean(d):,.2f}  std={np.nanstd(d):,.2f}")
        print(f"  若 shape=(chains,draws) 或 (samples,)，均值即期望总收入")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:250]}")


if __name__ == "__main__":
    main()
