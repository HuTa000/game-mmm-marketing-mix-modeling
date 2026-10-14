"""
验证 v19：微调 intercept 先验，把媒体占比校准到接近真实
=========================================================
v18 突破：intercept Normal(0.45, 0.08) 让 Spearman ρ=1.000！
但媒体占比仍 95.6%（真实 46.9%），说明 intercept 还偏小。

逻辑：内部空间 y_internal = y / target_scale，target_scale ≈ 1.384 × y_mean
      基线占比 55% → intercept_internal ≈ 0.55 / 1.384 = 0.397
      但 v18 方案3 估出的 intercept = 0.032，远小于 0.397 → 仍被渠道抢走

测试更紧、更中心的 intercept 先验。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

def geom_adstock(x, decay, l_max=6):
    out = np.zeros_like(x, dtype=float); c = 0.0
    for i in range(len(x)): c = x[i] + decay*c; out[i] = c
    return out

def hill(x, s, k): return (x**s)/(x**s + k**s)

def build(cfg, ov):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality)
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)

def run(tag, mmm, df, cfg, media_series):
    pt = mmm.idata.posterior; ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    ty = float(df["revenue"].sum()); tm = float(sum(s.sum() for s in media_series.values()))
    ic = float(pt["intercept_contribution"].values.mean())
    dsA = mmm.data.get_contributions(original_scale=True); chA = dsA["channels"].mean(("chain","draw"))
    cs = float(np.asarray(chA.values).sum())
    print(f"\n{'='*80}\n{tag}\n{'='*80}")
    print(f"  intercept={ic:.3f}/周 (总计 {ic*len(df):.1f}) | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%})")
    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum()); s = float(df[c].sum())
        rows.append({"c": c, "est": g/s, "true": TRUE_BETA[c]})
    rho = spearmanr([r["est"] for r in rows], [r["true"] for r in rows]).statistic
    for r in rows: print(f"    {r['c']:10s} 估ROI={r['est']:>7.3f}  真ROI={r['true']:.2f}")
    print(f"  Spearman ρ = {rho:.3f} {'✓' if rho>0.7 else '✗'}")
    return rho, cs/ty

def main():
    df = generate_game_data(n_weeks=156, seed=2024); cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg); N = len(df)
    ms = {}
    for ch in cfg.channel_cols:
        x = df[ch].values/df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max); h = hill(ad, TRUE_KAPPA[ch], 1.2)
        ms[ch] = TRUE_BETA[ch]*df[ch].sum()/h.sum()*h
    media = sum(ms.values()); bw = media.mean()*(0.55/0.45)
    base = bw*(0.85+0.15*np.sin(2*np.pi*np.arange(N)/52-0.8)) + np.linspace(0, bw*0.15, N)
    df["revenue"] = base + media + np.random.default_rng(11).normal(0, 0.04*media.mean(), N)
    y = df["revenue"].astype(float); X, _, _, _ = prepare_data(df, cfg)
    common = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
              "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
              "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
              "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel")}
    res = []
    for mu, sg in [(0.40, 0.05), (0.40, 0.03), (0.42, 0.02), (0.45, 0.02)]:
        ov = dict(common); ov["intercept"] = Prior("Normal", mu=mu, sigma=sg)
        m = build(cfg, ov); fit_mmm(m, X, y, cfg, progressbar=False)
        rho, share = run(f"intercept ~ Normal({mu}, {sg})", m, df, cfg, ms)
        res.append((f"N({mu},{sg})", share, rho))
    print(f"\n{'='*80}\n汇总\n{'='*80}")
    print(f"{'配置':22s} {'媒体占比':>12s} {'Spearman':>10s}")
    print(f"{'真实':22s} {0.469:>12.1%} {1.0:>10.3f}")
    for n, s, r in res: print(f"{n:22s} {s:>12.1%} {r:>10.3f}")

if __name__ == "__main__": main()
