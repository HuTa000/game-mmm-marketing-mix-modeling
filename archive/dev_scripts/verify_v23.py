"""
验证 v23：数据量级修正后，重新标定 intercept 先验
====================================================
数据生成器已修正，现在：
  y_mean = 2,118,221   基线占比 52.9%
  若 target_scale = y_mean → intercept 内部水平应 ≈ 0.53

测多种 intercept 先验，找到能同时满足"截距>0 + beta回收 + ROI排序"的设置。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_generator import generate_game_data, TRUE_BETA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def build(cfg, ov):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=cfg.control_cols)
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)


def run(tag, mmm, df, cfg, tms):
    pt = mmm.idata.posterior
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    ty = float(df["revenue"].sum()); tm = float(sum(s.sum() for s in tms.values()))
    ic = float(pt["intercept_contribution"].values.mean())
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    cs = float(np.asarray(chA.values).sum())
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum()); s = float(df[c].sum())
        rows.append({"c": c, "est": g/s, "true": TRUE_BETA[c]})
    rho = spearmanr([r["est"] for r in rows], [r["true"] for r in rows]).statistic
    print(f"\n{'='*86}\n{tag}  (target_scale={ts:,.0f})\n{'='*86}")
    print(f"  intercept={ic:+.3f}/周 | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%}) | ρ={rho:.3f}")
    print(f"  {'渠道':10s} {'估beta':>9s} {'真beta':>8s} {'估ROI':>9s} {'真ROI':>8s}")
    for i, r in enumerate(rows):
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"  {r['c']:10s} {b:>9.3f} {r['true']:>8.2f} {r['est']:>9.3f} {r['true']:>8.2f}")
    return rho, cs/ty, ic


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    tshare = df['_true_media_total'].sum()/df['revenue'].sum()
    print(f"真实媒体占比 = {tshare:.1%}")
    X, y, meta, uc = prepare_data(df, cfg)

    common = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
              "saturation_slope": Prior("HalfNormal", sigma=0.15, dims="channel"),
              "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
              "saturation_beta": Prior("HalfNormal", sigma=1.5, dims="channel")}
    cfgs = [
        ("N(0.095, 0.03)", Prior("Normal", mu=0.095, sigma=0.03, dims=())),
        ("N(0.095, 0.05)", Prior("Normal", mu=0.095, sigma=0.05, dims=())),
        ("HalfNormal(0.15)", Prior("HalfNormal", sigma=0.15, dims=())),
        ("N(0.10, 0.05)", Prior("Normal", mu=0.10, sigma=0.05, dims=())),
    ]
    res = []
    for tag, ip in cfgs:
        ov = dict(common); ov["intercept"] = ip
        m = build(cfg, ov); fit_mmm(m, X, y, cfg, progressbar=False)
        rho, share, ic = run(f"intercept ~ {tag}", m, df, cfg, tms)
        res.append((tag, share, rho, ic))
    print(f"\n{'='*86}\n汇总\n{'='*86}")
    print(f"{'配置':20s} {'媒体占比':>10s} {'Spearman':>10s} {'截距':>9s}")
    print(f"{'真实':20s} {tshare:>10.1%} {1.0:>10.3f}")
    for t, s, r, i in res: print(f"{t:20s} {s:>10.1%} {r:>10.3f} {i:>+9.3f}")


if __name__ == "__main__":
    main()
