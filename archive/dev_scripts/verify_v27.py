"""
验证 v27：找到数据修正后的最佳配置（快速网格搜索）
====================================================
已确认：
  · 数据生成器已修正（真实 ROI = 设计值，媒体占比 46.7%）
  · 求解 ROOT CAUSE 是 beta↔kappa 不可识别
  · v18 用 intercept N(0.45,0.08)（当时数据是"元级"）达到 ρ=1.000

现在数据量级变了（y 均值 153 万，target_scale = y.max()）。
做一个小网格搜索，找最佳 intercept 先验 + kappa 先验组合。
"""
import os, sys, warnings, itertools
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def build(cfg, ov, uc):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=uc)
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)


def evaluate(mmm, df, cfg, tms):
    pt = mmm.idata.posterior
    ty = float(df["revenue"].sum()); tm = float(sum(s.sum() for s in tms.values()))
    ic = float(pt["intercept_contribution"].values.mean())
    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    cs = float(np.asarray(chA.values).sum())
    est, tru = [], []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum()); s = float(df[c].sum())
        est.append(g/s); tru.append(TRUE_BETA[c])
    rho = spearmanr(est, tru).statistic
    return rho, cs/ty, ic, est


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    tshare = df['_true_media_total'].sum()/df['revenue'].sum()
    X, y, meta, uc = prepare_data(df, cfg)
    print(f"真实媒体占比={tshare:.1%}  y_mean={y.mean():,.0f}")

    results = []
    # 网格：intercept mu × kappa 先验
    for ic_mu in [0.05, 0.08, 0.10, 0.12]:
        for kappa_a, kappa_b in [(4,4), (2,1), (1,1)]:
            ov = {
                "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
                "saturation_slope": Prior("HalfNormal", sigma=0.15, dims="channel"),
                "saturation_kappa": Prior("Gamma", alpha=kappa_a, beta=kappa_b, dims="channel"),
                "saturation_beta": Prior("HalfNormal", sigma=2.0, dims="channel"),
                "intercept": Prior("Normal", mu=ic_mu, sigma=0.05, dims=()),
            }
            try:
                m = build(cfg, ov, uc)
                fit_mmm(m, X, y, cfg, progressbar=False)
                rho, share, ic, est = evaluate(m, df, cfg, tms)
                results.append({"ic_mu": ic_mu, "kappa": f"G({kappa_a},{kappa_b})",
                                "rho": rho, "share": share, "ic": ic})
                print(f"  ic_mu={ic_mu} kappa=G({kappa_a},{kappa_b}) → "
                      f"ρ={rho:+.3f} share={share:.1%} intercept={ic:+.4f}")
            except Exception as e:
                print(f"  ic_mu={ic_mu} kappa=G({kappa_a},{kappa_b}) → 失败 {e}")

    r = pd.DataFrame(results).sort_values("rho", ascending=False)
    print("\n" + "="*70)
    print("按 ρ 排序（前 5）:")
    print(r.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
