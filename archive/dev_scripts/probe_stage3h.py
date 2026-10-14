# -*- coding: utf-8 -*-
"""
阶段 3 探针 v8：在优化器的 _pymc_model 上编译诊断函数
直接读 channel_contribution / total_media_contribution_original_scale
这是最终定论：响应量级到底对不对
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
    mmm, _ = build_mmm_safe(cfg, used_ctrl)
    mmm.build_model(X, y); mmm.idata = idata
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    m = opt._pymc_model   # 优化器真正使用的模型
    flat = opt._budgets_flat

    sec("A. _pymc_model 中的关键变量")
    for n in ("channel_contribution", "total_media_contribution_original_scale",
              "control_contribution", "intercept_contribution", "mu", "channel_data"):
        print(f"  {n:50s} in model? {n in m.named_vars}")

    sec("B. 编译诊断函数")
    from pytensor import function
    from pytensor.graph import rewrite_graph

    targets = {}
    for n in ("channel_contribution", "total_media_contribution_original_scale",
              "control_contribution", "intercept_contribution", "mu", "channel_data"):
        if n in m.named_vars:
            node = m[n]
            try:
                targets[n] = rewrite_graph(node.values if hasattr(node, "values") else node,
                                           include=("lower_xtensor", "canonicalize", "stabilize"))
            except Exception as e:
                print(f"  {n} rewrite 失败: {e}")

    fn = function([flat], list(targets.values()))

    # 历史周均 plan
    hist_weekly = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    plan = xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",))
    x = opt._pack_decision_vector(plan)

    sec("C. 历史周均 plan 下各量的值")
    outs = fn(x)
    for n, v in zip(targets.keys(), outs):
        arr = np.asarray(v)
        print(f"  {n:48s} shape={str(arr.shape):16s} sum={arr.sum():>18,.4f}")

    sec("D. 全零 plan（纯基线）")
    x0 = np.zeros_like(x)
    outs0 = fn(x0)
    base = {}
    for n, v in zip(targets.keys(), outs0):
        arr = np.asarray(v)
        base[n] = arr.sum()
        print(f"  {n:48s} sum={arr.sum():>18,.4f}")

    sec("E. 差分：历史 plan − 全零 plan = 纯媒体增量")
    for n, v in zip(targets.keys(), outs):
        arr = np.asarray(v)
        print(f"  {n:48s} delta={arr.sum()-base[n]:>18,.4f}")

    sec("F. 与阶段 2 窗口口径对照")
    contrib = pd.read_csv(os.path.join(ROOT, "outputs/reports/stage2_contributions.csv"))
    contrib["date"] = pd.to_datetime(contrib["date"])
    win = contrib[(contrib["date"] >= "2025-06-08") & (contrib["date"] <= "2025-08-31")]
    ch_cols = [c for c in cfg.channel_cols if c in contrib.columns]
    print(f"  阶段2 窗口媒体贡献和 = {win[ch_cols].sum().sum():,.2f}")

    sec("G. 逐渠道单独投（诊断 channel_contribution）")
    print(f"  {'渠道':10s} {'plan(周)':>14} {'ch_contrib.sum':>18} {'total_orig':>18}")
    for i, c in enumerate(cfg.channel_cols):
        vv = np.zeros(5); vv[i] = hist_weekly[i]
        xi = opt._pack_decision_vector(xr.DataArray(vv, coords={"channel": cfg.channel_cols}, dims=("channel",)))
        oi = fn(xi)
        dd = dict(zip(targets.keys(), oi))
        cc = float(np.asarray(dd["channel_contribution"]).sum())
        tt = float(np.asarray(dd["total_media_contribution_original_scale"]).sum())
        print(f"  {c:10s} {vv[i]:>14,.0f} {cc:>18,.2f} {tt:>18,.2f}")

    sec("H. 关键比值")
    o = dict(zip(targets.keys(), outs))
    cc = float(np.asarray(o["channel_contribution"]).sum())
    tt = float(np.asarray(o["total_media_contribution_original_scale"]).sum())
    print(f"  total_orig / channel_contribution = {tt/cc if cc else float('nan'):,.6f}")
    print(f"  target_scale                      = {np.asarray(m.named_vars['target_scale'].get_value()).ravel()[0]:,.2f}")
    print(f"  channel_scale                     = {np.asarray(m.named_vars['channel_scale'].get_value())}")


if __name__ == "__main__":
    main()
