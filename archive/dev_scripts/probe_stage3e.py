# -*- coding: utf-8 -*-
"""
阶段 3 探针 v5：标定 evaluate_plan / allocate_budget 的货币单位语义
这是数字能否讲通的关键！

实验设计：
  固定一个"只投 tiktok"的 plan，令其货币总量从 100万 递增到 1亿，
  看 evaluate_plan 的 utility 如何变化。
  - 若 plan 语义是"每周花费"，则总量翻倍 => 响应翻倍左右（次线性因饱和）
  - 通过形状判断饱和点，反推量纲
再对照：历史分配下的响应 vs 优化分配下的响应
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import xarray as xr

from data_generator import generate_game_data
from mmm_core import MMMConfig, prepare_data, build_mmm_safe, get_spend_series

SEP = "=" * 78
def sec(t): print("\n" + SEP + "\n" + t + "\n" + SEP)
CH = None; CFG = None

def plan_da(vals):
    return xr.DataArray(np.array(vals, dtype=float), coords={"channel": CFG.channel_cols}, dims=("channel",))


def main():
    global CH, CFG
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    cfg = MMMConfig(); CFG = cfg
    X, y, meta, used_ctrl = prepare_data(df, cfg)
    with open(os.path.join(ROOT, "outputs/reports/stage2_mmm_idata.pkl"), "rb") as f:
        idata = pickle.load(f)["idata"]
    mmm, _ = build_mmm_safe(cfg, used_ctrl)
    mmm.build_model(X, y); mmm.idata = idata

    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    spend_mean = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")

    hist_weekly = spend_mean.copy()
    hist_total = hist_weekly * 13
    print(f"历史周均: {dict(zip(cfg.channel_cols, np.round(hist_weekly,0)))}")
    print(f"13周历史总: {hist_total.sum():,.0f}")

    sec("实验 A. 固定 total_budget，用 allocate_budget 看 budgets 与响应的量纲关系")
    for mult in (0.5, 1.0, 2.0):
        tb = hist_total.sum() * mult
        res = opt.allocate_budget(total_budget=tb)
        b = np.asarray(res.budgets)
        # 用返回的 budgets 评估（当作"每期"花费）
        ev = opt.evaluate_plan(plan_da(b), total_budget=tb)
        ev13 = opt.evaluate_plan(plan_da(b / 13), total_budget=tb)
        print(f"\n  budget×{mult}: total={tb:,.0f}")
        print(f"    budgets   = {np.round(b,0)}")
        print(f"    utility(传 budgets 原值)      = {ev.utility:,.2f}")
        print(f"    utility(传 budgets/13)        = {ev13.utility:,.2f}")
        print(f"    => 若两者相差 ~13 倍，则 plan 语义 = 每期花费")

    sec("实验 B. 只投 tiktok，货币总量递增（看饱和形状）")
    idx_t = cfg.channel_cols.index("tiktok")
    print(f"{'总量(元)':>16}  {'utility':>16}  {'平均ROAS*':>12}")
    for tot in (1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8):
        v = np.zeros(5); v[idx_t] = tot
        ev = opt.evaluate_plan(plan_da(v), total_budget=hist_total.sum())
        print(f"{tot:>16,.0f}  {ev.utility:>16,.2f}  {ev.utility/tot:>12.4f}")
    print("  * 平均ROAS = utility / 传入总量；若 plan 是'每期'语义，则单位不匹配会体现在数值上")

    sec("实验 C. 用训练期真实花费口径重建：每周 tiktok 花费 → 响应")
    print(f"{'每周花费(元)':>16}  {'utility':>16}")
    for w in (spend_mean[idx_t]*0.5, spend_mean[idx_t], spend_mean[idx_t]*2, spend_mean[idx_t]*5):
        v = np.zeros(5); v[idx_t] = w
        ev = opt.evaluate_plan(plan_da(v), total_budget=hist_total.sum())
        print(f"{w:>16,.0f}  {ev.utility:>16,.2f}")

    sec("实验 D. 历史分配 vs 优化分配（同一把尺子）")
    ev_hist = opt.evaluate_plan(plan_da(hist_total), total_budget=hist_total.sum())
    res_opt = opt.allocate_budget(total_budget=hist_total.sum())
    ev_opt = opt.evaluate_plan(plan_da(np.asarray(res_opt.budgets)), total_budget=hist_total.sum())
    print(f"  历史分配 utility = {ev_hist.utility:,.2f}")
    print(f"  优化分配 utility = {ev_opt.utility:,.2f}")
    print(f"  提升 = {(ev_opt.utility/ev_hist.utility-1)*100:+.2f}%")

    sec("实验 E. evaluate_response_distribution（带后验分布）")
    for tag, v in (("历史", hist_total), ("优化", np.asarray(res_opt.budgets))):
        d = np.asarray(opt.evaluate_response_distribution(plan_da(v)))
        print(f"  {tag}: samples={d.size} mean={np.nanmean(d):,.2f} "
              f"p5={np.nanpercentile(d,5):,.2f} p95={np.nanpercentile(d,95):,.2f}")

    sec("实验 F. 训练期预测口径核对：用阶段 2 贡献分解的媒体总量做锚")
    cdf = pd.read_csv(os.path.join(ROOT, "outputs/reports/stage2_contributions.csv"))
    print(f"  stage2_contributions.csv 列: {list(cdf.columns)[:8]}")
    print(f"  head:\n{cdf.head(3)}")
    tot_col = [c for c in cdf.columns if "总贡献" in c or "total" in c.lower()]
    print(f"  可能的合计列: {tot_col}")
    if tot_col:
        print(f"  训练期媒体总贡献 = {cdf[tot_col[0]].sum():,.0f}")
    print(f"  训练期 y.sum() = {y.sum():,.0f}  (156周)")
    print(f"  训练期 y.sum()/156 = {y.sum()/156:,.0f} (周均总收入)")


if __name__ == "__main__":
    main()
