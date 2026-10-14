# -*- coding: utf-8 -*-
"""
阶段 3 探针 v6：查清"媒体贡献绝对量级"的 2.2 倍偏差

已知锚点：
  训练期周均总收入 y.mean() = 2,923,358 元
  阶段2 分解: 媒体占 57% => 周均媒体贡献应 ≈ 1,666,314 元
  优化器 utility(历史分配)/13 = 47,606,769/13 = 3,662,059 元/周
  => 偏差 = 3,662,059 / 1,666,314 = 2.198

假设：优化模型的 total_media_contribution_original_scale 多乘了一次 target_scale，
      或者 channel_contribution 在优化模型里用了不同的 target_scale。

验证方法：
  1) 用同一优化的 plan（历史分配）与
  2) 用阶段 2 的贡献分解函数 compute_contributions_library，
     在相同的 13 周窗口上计算媒体贡献，直接对比
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import xarray as xr

from data_generator import generate_game_data
from mmm_core import (MMMConfig, prepare_data, build_mmm_safe, get_spend_series,
                      compute_contributions_library, get_target_scale)

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

    sec("0. target_scale 双重检查")
    ts_opt_model = None
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    m = opt.model
    v = np.asarray(m.named_vars["target_scale"].get_value())
    print(f"  优化模型 target_scale     = {v}")
    print(f"  训练期 y.mean()           = {y.mean():,.2f}")
    print(f"  训练期 y.max()            = {y.max():,.2f}")

    sec("1. 优化模型内部：单渠道贡献量级")
    # 拿 channel_contribution 在历史 plan 下的值
    hist_weekly = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    plan = xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",))
    # 用 evaluate_plan 的 utility（已含 target_scale）
    ev = opt.evaluate_plan(plan, total_budget=hist_weekly.sum() * 13)
    u = ev.utility
    print(f"  utility(历史周均 plan) = {u:,.2f}")
    print(f"  => 若 utility 是 13 周总和: 周均 = {u/13:,.2f}")
    print(f"  => 阶段2 口径周均媒体贡献  = 1,666,314  (预估值)")
    print(f"  => 比值 = {(u/13)/1666314:.4f}")

    sec("2. 阶段 2 的贡献分解在同一窗口的值（对照基准）")
    contrib, info = compute_contributions_library(mmm, df, cfg, meta)
    contrib = contrib.copy()
    contrib["date"] = pd.to_datetime(contrib["date"])
    ch_cols = [c for c in cfg.channel_cols if c in contrib.columns]
    contrib["_media_total"] = contrib[ch_cols].sum(axis=1)
    print(f"  info: media_sum={info.get('media_sum'):,.0f}  true_y_sum={info.get('true_y_sum'):,.0f}  "
          f"media_share={info.get('media_share'):.4f}  target_scale={info.get('target_scale'):,.1f}")
    win = contrib[(contrib["date"] >= "2025-06-08") & (contrib["date"] <= "2025-08-31")]
    print(f"  窗口 {win['date'].min().date()} ~ {win['date'].max().date()}, {len(win)} 周")
    print(f"  窗口媒体贡献总和(阶段2口径) = {win['_media_total'].sum():,.2f}")
    print(f"  窗口媒体贡献周均            = {win['_media_total'].mean():,.2f}")
    print(f"  vs 优化器 utility/13        = {u/13:,.2f}")
    print(f"  => 比值 = {(u/13)/win['_media_total'].mean():.4f}")

    sec("3. 偏差来源定位：单渠道拆解")
    print(f"  {'渠道':10s} {'阶段2窗口和':>16} {'优化器(拆解)':>16}")
    # 逐渠道投历史周均，看单渠道 utility
    for i, c in enumerate(cfg.channel_cols):
        plan_one = np.zeros(5); plan_one[i] = hist_weekly[i]
        ev1 = opt.evaluate_plan(xr.DataArray(plan_one, coords={"channel": cfg.channel_cols}, dims=("channel",)),
                                total_budget=hist_weekly.sum()*13)
        print(f"  {c:10s} {win[c].sum():>16,.2f} {ev1.utility:>16,.2f}")

    sec("4. 用完整历史 plan 拆解总响应（线性可加性检查）")
    ev_full = opt.evaluate_plan(xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",)),
                                total_budget=hist_weekly.sum()*13)
    print(f"  全渠道 utility = {ev_full.utility:,.2f}")
    print(f"  阶段2 窗口总和 = {win['_media_total'].sum():,.2f}")
    print(f"  比值 = {ev_full.utility/win['_media_total'].sum():.4f}")

    sec("5. 结论：偏差是否等于 target_scale / 某常数？")
    ratio = ev_full.utility / win["_media_total"].sum()
    print(f"  比值 = {ratio:.6f}")
    print(f"  target_scale = {np.asarray(m.named_vars['target_scale'].get_value()).ravel()[0]:,.2f}")
    print(f"  y.mean() = {y.mean():,.2f}")
    print(f"  1/ratio = {1/ratio:.6f}")

    sec("6. 是否存在需要除以 num_periods 的口径？")
    print(f"  opt.num_periods = {opt.num_periods}, 窗口周数 = {len(win)}")


if __name__ == "__main__":
    main()
