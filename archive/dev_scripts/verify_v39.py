"""
验证 v39：渠道标准化是否抹掉了识别基线所需的信息？
==================================================
怀疑：
  prepare_data 把渠道除以自身均值（→均值=1），
  这一步把"渠道之间的绝对量级差异"抹掉了。
  而量级差异可能是识别"基线水平"的关键信息。

对比 3 种缩放方案：
  S1) 渠道 ÷ 自身均值（当前，均值=1）
  S2) 渠道 ÷ 1e6      （保留量级差异，但数值适中）
  S3) 渠道不缩放（原始元）   —— 数值巨大，看模型能否处理
  S4) 渠道 ÷ 10000（万元）   —— 数值友好 + 保留量级

同时 y 的缩放交给 PyMC-Marketing 内部（target_scale = y.max()）。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY
from mmm_core import (MMMConfig, normalize_columns, fit_mmm, get_target_scale)

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

raw = generate_game_data(n_weeks=156, seed=2024)


def build_X(df, scale_mode):
    X = df[["date"] + CHANNELS].copy()
    if scale_mode == "mean":
        for c in CHANNELS:
            X[c] = X[c] / X[c].mean()
    elif scale_mode == "1e6":
        for c in CHANNELS:
            X[c] = X[c] / 1e6
    elif scale_mode == "1e4":
        for c in CHANNELS:
            X[c] = X[c] / 1e4
    elif scale_mode == "raw":
        pass
    for c in ["version_update", "competitor_activity", "holiday", "trend"]:
        mu, sd = df[c].mean(), df[c].std()
        X[c] = (df[c] - mu) / (sd if sd > 0 else 1)
    return X


def run(name, scale_mode, intercept_mu, beta_mu, beta_sigma=1.0):
    df = normalize_columns(raw.copy(), MMMConfig())
    X = build_X(df, scale_mode)
    y = df["revenue"].astype(float)
    uc = ["version_update", "competitor_activity", "holiday", "trend"]

    kw = dict(date_column="date", channel_columns=CHANNELS, target_column="revenue",
              adstock=GeometricAdstock(l_max=6), saturation=HillSaturation(),
              yearly_seasonality=2, control_columns=uc)
    b = MMM(**kw); m = dict(b.default_model_config)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("LogNormal", mu=beta_mu, sigma=beta_sigma, dims="channel"),
        "intercept": Prior("Normal", mu=intercept_mu, sigma=0.06, dims=()),
        "gamma_control": Prior("Normal", mu=0.0, sigma=0.2, dims="control"),
    })
    kw["model_config"] = m
    mmm = MMM(**kw)
    idata, el = fit_mmm(mmm, X, y, MMMConfig(quick_mode=True).apply_quick_mode(),
                        progressbar=False)

    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    est = {c: float(np.asarray(chd.sel(channel=c).values).sum()) for c in CHANNELS}
    roi = {c: est[c] / raw[f"{c}_spend"].sum() for c in CHANNELS}
    rho, p = spearmanr([roi[c] for c in CHANNELS], [TRUE_BETA[c] for c in CHANNELS])
    ic = float(idata.posterior["intercept_contribution"].values.mean())
    mshare = sum(est.values()) / y.sum()

    print(f"--- {name} ({el:.0f}s) ---")
    print(f"  截距={ic:+.4f} {'✓正' if ic > 0 else '✗负'}  媒体占比={mshare:.1%} (真46.9%)")
    print(f"  ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in CHANNELS))
    print(f"  真 : " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in CHANNELS))
    print(f"  ρ={rho:+.3f} p={p:.3f}  排序: {' > '.join(sorted(roi, key=lambda k:-roi[k]))}")
    print()
    return dict(name=name, ic=ic, mshare=mshare, rho=rho, p=p)


# 每个缩放方案下，intercept 先验的合理中心不同：
#   S1 mean:   真实基线常数/ y.max = 0.131
#   S2 1e6:    intercept 单位变成"百万花费对应的收入"，先验要重估
#   S4 1e4:    同理
res = []
res.append(run("S1 渠道÷均值 (当前)", "mean", 0.098, -0.7))
res.append(run("S2 渠道÷1e6", "1e6", 0.098, -0.7))
res.append(run("S4 渠道÷1e4(万元)", "1e4", 0.098, -0.7))
res.append(run("S4b 渠道÷1e4 + 宽beta先验", "1e4", 0.098, 0.0, 1.5))

print("=" * 80)
print(pd.DataFrame(res).round(3).to_string(index=False))
