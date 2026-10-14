"""
验证 v22：最终调参 —— 让 intercept 自由但为正，靠 trend 承载趋势
====================================================================
v21 后状态：
  · 截距 +0.401 ✓，可加性 ✓，decay 全命中 ✓
  · 但 beta 全被压到 0.2（真实 1.1~2.3），ROI 整体偏小 5 倍
  · 原因：intercept N(0.45, 0.15) 把基线水平锁死，模型只好压 beta

修正方向：
  · intercept 放宽（sigma 大一些，mu 小一些），允许模型自己找基线水平
  · 但要求为正 → 用 HalfNormal 或带截断的 Normal
  · 同时：数据生成器已改为"贡献 = TRUE_BETA × Hill(标准化输入)"
    → 标准化输入下 Hill ≈ 0.3~0.6，所以 beta 应该能回收

测试多种 intercept 先验：
  A. HalfNormal(0.5)      —— 允许大范围正值
  B. Normal(0.4, 0.3)     —— 宽的正态
  C. Normal(0.4, 0.5)     —— 更宽
  D. HalfNormal(1.0)      —— 很宽
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm, get_target_scale
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def build(cfg, ov):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=cfg.control_cols)
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)


def run(tag, mmm, df, cfg, true_media_series):
    pt = mmm.idata.posterior
    ty = float(df["revenue"].sum()); tm = float(sum(s.sum() for s in true_media_series.values()))
    ic = float(pt["intercept_contribution"].values.mean())
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    cs = float(np.asarray(chA.values).sum())
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum()); s = float(df[c].sum())
        rows.append({"c": c, "est": g/s, "true": TRUE_BETA[c]})
    rho = spearmanr([r["est"] for r in rows], [r["true"] for r in rows]).statistic
    print(f"\n{'='*84}\n{tag}\n{'='*84}")
    print(f"  intercept={ic:.3f}/周 | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%}) | ρ={rho:.3f}")
    print(f"  {'渠道':10s} {'估beta':>9s} {'真beta':>8s} {'估ROI':>9s} {'真ROI':>8s}")
    for i, r in enumerate(rows):
        c = r["c"]
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"  {c:10s} {b:>9.3f} {TRUE_BETA[c]:>8.2f} {r['est']:>9.3f} {r['true']:>8.2f}")
    return rho, cs/ty


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    print(f"真实媒体占比 = {df['_true_media_total'].sum()/df['revenue'].sum():.1%}")

    X, y, meta, uc = prepare_data(df, cfg)
    print("控制变量:", uc)

    common = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
              "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
              "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
              "saturation_beta": Prior("HalfNormal", sigma=1.5, dims="channel")}
    cfgs = [
        ("A  HalfNormal(0.5)",  Prior("HalfNormal", sigma=0.5,  dims=())),
        ("B  Normal(0.4, 0.3)", Prior("Normal", mu=0.4, sigma=0.3, dims=())),
        ("C  Normal(0.4, 0.5)", Prior("Normal", mu=0.4, sigma=0.5, dims=())),
        ("D  HalfNormal(1.0)",  Prior("HalfNormal", sigma=1.0,  dims=())),
    ]
    res = []
    for tag, ip in cfgs:
        ov = dict(common); ov["intercept"] = ip
        m = build(cfg, ov); fit_mmm(m, X, y, cfg, progressbar=False)
        rho, share = run(tag, m, df, cfg, tms)
        res.append((tag, share, rho))
    print(f"\n{'='*84}\n汇总\n{'='*84}")
    print(f"{'配置':24s} {'媒体占比':>12s} {'Spearman':>10s}")
    print(f"{'真实':24s} {df['_true_media_total'].sum()/df['revenue'].sum():>12.1%} {1.0:>10.3f}")
    for t, s, r in res: print(f"{t:24s} {s:>12.1%} {r:>10.3f}")


if __name__ == "__main__":
    main()
