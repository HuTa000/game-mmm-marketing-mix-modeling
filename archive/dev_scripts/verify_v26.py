"""
验证 v26：不标准化渠道 —— 让 kappa 固定为确定值（用 pm.Deterministic 覆盖）
============================================================================
v25 失败原因：用 HalfNormal(0.01) 近似固定 kappa，实际把 kappa 压到 0，
导致 Hill≡0、beta 全部坍缩到 0.075。

正确做法：用 pymc-marketing 的 `Prior("Deterministic", ...)` 或
在 model_config 里用**常数**指定 kappa。
我们直接用**绝对量级**的渠道输入 + 固定 kappa=mean(spend)×2。

其实最直接的路径：**不标准化渠道**，此时
    contribution = beta × Hill(adstock(spend) ; slope, kappa_abs)
beta 量纲 = 收入/（无量纲）= 收入量级
此时 beta 与 ROI 的关系更直接。

本脚本对比：
  A. kappa 用 pm.Deterministic 固定为常数（通过 Prior("Deterministic")）
  B. 完全不标准化渠道 + 宽先验
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA, CHANNELS
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def build_with_fixed_vals(cfg, kappa_vals, slope_val, control_cols, beta_sigma=2.0):
    """用 Deterministic 固定 kappa/slope 为常数"""
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=control_cols)
    b = MMM(**kw); mc = dict(b.default_model_config)
    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_kappa": Prior("Deterministic", sigma=kappa_vals, dims="channel"),
        "saturation_slope": Prior("Deterministic", sigma=slope_val, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel"),
        "intercept": Prior("Normal", mu=0.10, sigma=0.05, dims=()),
    }
    mc.update(ov); kw["model_config"] = mc
    return MMM(**kw)


def run(tag, mmm, df, cfg, tms):
    pt = mmm.idata.posterior
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
    print(f"\n{'='*84}\n{tag}\n{'='*84}")
    print(f"  intercept={ic:+.4f} | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%}) | ρ={rho:+.3f}")
    print(f"  {'渠道':10s} {'估beta':>9s} {'估ROI':>9s} {'真ROI':>7s}")
    for i, r in enumerate(rows):
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"  {r['c']:10s} {b:>9.3f} {r['est']:>9.3f} {r['true']:>7.2f}")
    return rho, cs/ty


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    print(f"真实媒体占比 = {df['_true_media_total'].sum()/df['revenue'].sum():.1%}")
    X, y, meta, uc = prepare_data(df, cfg)

    # 标准化空间下，kappa 是半饱和点；实测各渠道 adstock(x_std) 均值约 1.4~2.0
    # 所以半饱和点应设在 1.5 附近
    for kappa in [1.5, 2.5, 4.0]:
        kv = np.full(5, float(kappa))
        try:
            m = build_with_fixed_vals(cfg, kv, 1.5, uc)
            fit_mmm(m, X, y, cfg, progressbar=False)
            run(f"Deterministic kappa={kappa}, slope=1.5", m, df, cfg, tms)
        except Exception as e:
            print(f"\n[kappa={kappa}] 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
