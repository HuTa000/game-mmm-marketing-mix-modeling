"""
验证 v20：加趋势项后人 不靠"手调 intercept"能否自动恢复
==========================================================
v19 发现：媒体占比可通过 intercept 先验手动调节，
         但这是"凑数"，不是统计上的正当做法。

【更本质的诊断】
数据生成器的基线 = intercept + trend(线性增长) + season_rev(正弦)
但我给模型的只有：
  · intercept（常数）
  · yearly_seasonality（Fourier）
  · 3 个控制变量
→ **线性趋势没有被建模！** 模型只能用 intercept 或渠道去"硬扛"趋势，
  导致 intercept 被推成负值（趋势增长被渠道吸收了）。

【修正】给模型加线性趋势控制变量 t。
这是 MMM 标准做法：baseline = intercept + trend + seasonality。
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

def build(cfg, ov, control_cols=None):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality)
    if control_cols is not None:
        kw["control_columns"] = control_cols
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)

def run(tag, mmm, df, cfg, ms):
    pt = mmm.idata.posterior
    ty = float(df["revenue"].sum()); tm = float(sum(s.sum() for s in ms.values()))
    ic = float(pt["intercept_contribution"].values.mean())
    dsA = mmm.data.get_contributions(original_scale=True); chA = dsA["channels"].mean(("chain","draw"))
    cs = float(np.asarray(chA.values).sum())
    print(f"\n{'='*82}\n{tag}\n{'='*82}")
    print(f"  intercept={ic:.3f}/周 | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%})")
    rows=[]
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum()); s = float(df[c].sum())
        rows.append({"c":c,"est":g/s,"true":TRUE_BETA[c]})
    rho = spearmanr([r["est"] for r in rows],[r["true"] for r in rows]).statistic
    for r in rows: print(f"    {r['c']:10s} 估ROI={r['est']:>7.3f}  真ROI={r['true']:.2f}")
    print(f"  Spearman ρ = {rho:.3f} {'✓' if rho>0.7 else '✗'}")
    return rho, cs/ty

def main():
    df = generate_game_data(n_weeks=156, seed=2024); cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg); N = len(df)
    ms={}
    for ch in cfg.channel_cols:
        x = df[ch].values/df[ch].values.mean()
        ad = geom_adstock(x, TRUE_DECAY[ch], cfg.adstock_l_max); h = hill(ad, TRUE_KAPPA[ch], 1.2)
        ms[ch] = TRUE_BETA[ch]*df[ch].sum()/h.sum()*h
    media = sum(ms.values()); bw = media.mean()*(0.55/0.45)
    base = bw*(0.85+0.15*np.sin(2*np.pi*np.arange(N)/52-0.8)) + np.linspace(0, bw*0.15, N)
    df["revenue"] = base + media + np.random.default_rng(11).normal(0, 0.04*media.mean(), N)
    y = df["revenue"].astype(float)

    # ---- 关键：加入线性趋势控制变量 ----
    cfg2 = MMMConfig(quick_mode=True)
    cfg2.control_cols = list(cfg.control_cols) + ["trend"]
    cfg2 = cfg2.apply_quick_mode()
    df["trend"] = np.arange(N) / N     # 0~1 线性趋势

    common = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
              "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
              "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
              "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
              "intercept": Prior("Normal", mu=0.35, sigma=0.15)}
    X, _, _, _ = prepare_data(df, cfg2)
    print("控制变量:", cfg2.control_cols)
    m = build(cfg2, common)
    fit_mmm(m, X, y, cfg2, progressbar=False)
    rho, share = run("加线性趋势控制变量 + intercept N(0.35,0.15)", m, df, cfg2, ms)

    # 对照：不加趋势
    common2 = dict(common); common2["intercept"] = Prior("Normal", mu=0.35, sigma=0.15)
    X0, _, _, _ = prepare_data(df, cfg)
    m0 = build(cfg, common2)
    fit_mmm(m0, X0, y, cfg, progressbar=False)
    rho0, share0 = run("对照：不加趋势", m0, df, cfg, ms)
    print(f"\n{'='*82}\n汇总\n{'='*82}")
    print(f"{'真实':34s} {0.469:>12.1%} {1.0:>10.3f}")
    print(f"{'不带趋势':34s} {share0:>12.1%} {rho0:>10.3f}")
    print(f"{'带趋势':34s} {share:>12.1%} {rho:>10.3f}")

if __name__ == "__main__": main()
