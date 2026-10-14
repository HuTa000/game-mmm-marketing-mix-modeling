# -*- coding: utf-8 -*-
"""
验证 response_curve.py：
  1) 响应曲线是否单调递增、边际递减（凸性）
  2) 曲线在"当前花费"点的平均ROI 是否与阶段2的 ROI 排序一致
  3) 边际回报排序是否符合预期（tiktok 最高）
  4) 曲线数值与官方 evaluate_plan 在"单渠道投当前花费"时的响应是否同量级
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, xarray as xr
from data_generator import generate_game_data
from mmm_core import MMMConfig, prepare_data, build_mmm_safe, get_spend_series
import response_curve as rc

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

    sec("1. 抽取参数")
    p = rc.extract_channel_params(mmm, cfg, idata)
    print(f"  channels = {p['channels']}")
    print(f"  n_samples = {p['n_samples']} (chain×draw = {p['sample_coords']})")
    print(f"  target_scale = {p['target_scale']:,.2f}")
    print(f"  channel_scale = {np.round(p['channel_scale'],4)}")
    print(f"\n  {'渠道':10s} {'beta均值':>12} {'kappa均值':>12} {'slope均值':>12} {'alpha均值':>12}")
    for j, c in enumerate(p["channels"]):
        print(f"  {c:10s} {p['beta'][:,j].mean():>12.4f} {p['kappa'][:,j].mean():>12.4f} "
              f"{p['slope'][:,j].mean():>12.4f} {p['alpha'][:,j].mean():>12.4f}")

    sec("2. 渠道均值")
    means = rc.compute_channel_means(df, cfg)
    for c, v in means.items():
        print(f"  {c:10s} 周均 = {v:>14,.0f} 元")

    sec("3. 响应曲线：单调性与形状检验")
    curves = rc.compute_all_curves(mmm, cfg, df, multiple=3.0, n_grid=60, idata=idata)
    for c, cu in curves.items():
        r = cu.response_mean
        mono = np.all(np.diff(r) >= -1e-9)
        d1 = np.diff(r)
        # Hill 是 S 形（sigmoid）：前段凸(斜率递增)、后段凹(斜率递减)
        # 正确判据：(a) 单调递增  (b) 尾部斜率 < 头部最大斜率（最终饱和）
        d1_max = d1.max()
        tail_flat = d1[-1] < d1_max * 0.3        # 尾部斜率明显小于峰值 => 已趋饱和
        print(f"  {c:10s} 单调递增={mono}  尾部饱和={tail_flat}  "
              f"R(0)={r[0]:>12,.0f}  R(当前)={cu.current_response:>12,.0f}  R(3x)={r[-1]:>12,.0f}  "
              f"饱和度@当前={np.interp(cu.current_spend, cu.spend_grid, cu.saturation_pct):.3f}")

    sec("4. 边际回报（优化依据）")
    mr = rc.compute_marginal_roi(mmm, cfg, df, idata=idata)
    print(mr.to_string(index=False))
    print(f"\n  边际回报排序: {list(mr['渠道'])}")

    sec("5. 与阶段2 ROI 对照")
    roi2 = pd.read_csv(os.path.join(ROOT, "outputs/reports/stage2_roi_ci.csv"))
    print(roi2.to_string(index=False))
    print(f"\n  阶段2 ROI 排序: {list(roi2.sort_values('mean', ascending=False)['channel'])}")
    print(f"  边际回报排序:   {list(mr['渠道'])}")

    sec("6. 交叉验证：曲线预测 vs 官方 evaluate_plan（单渠道投当前花费）")
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    hist_weekly = np.array([means[c] for c in cfg.channel_cols])

    def da(v):
        return xr.DataArray(np.asarray(v, float), coords={"channel": cfg.channel_cols}, dims=("channel",))

    print(f"  {'渠道':10s} {'曲线预测(周)':>16} {'官方(13周)/13':>16} {'比值':>8}")
    for i, c in enumerate(cfg.channel_cols):
        v = np.zeros(5); v[i] = hist_weekly[i]
        ev = opt.evaluate_plan(da(v))
        official_weekly = ev.utility / 13.0
        curve_pred = curves[c].current_response
        ratio = curve_pred / official_weekly if official_weekly else np.nan
        print(f"  {c:10s} {curve_pred:>16,.0f} {official_weekly:>16,.0f} {ratio:>8.3f}")
    print("\n  注：官方 utility 含 carry-in 余温（常数底），曲线是纯当期响应，")
    print("      所以比值 <1 属正常。重点看【排序一致性】。")

    sec("7. 真实 ROI 对照（数据生成器的真值）")
    print(f"  真值 ROI: {df.attrs.get('true_beta')}")


if __name__ == "__main__":
    main()
