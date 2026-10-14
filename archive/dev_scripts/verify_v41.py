"""
验证 v41：正确做法 —— 外面不缩放渠道，让模型自己用 max 缩放
==========================================================
根因（v40 发现）：
  PyMC-Marketing 1.2.0 默认 scaling 策略是 method="max"：
      channel_scale_c = max(x_c)          # 模型内部自动算
      channel_data_scaled = x_c / max(x_c)
  而我在 prepare_data 里又做了 x_c / mean(x_c) → **双重缩放**。
  结果 Hill() 看到的输入水平被严重扭曲，
  beta 被迫补偿 → ROI 排序错乱、截距被推负。

正确做法：外面**保持原始花费**，让模型自己缩放。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, CHANNELS, TRUE_BETA
from mmm_core import fit_mmm, get_target_scale

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.mmm.scaling import FixedScaling, Scaling, DataDerivedScaling
from pymc_extras.prior import Prior

raw = generate_game_data(n_weeks=156, seed=2024)
raw = raw.rename(columns={f"{c}_spend": c for c in CHANNELS})
uc = ["version_update", "competitor_activity", "holiday", "trend"]


def build_X(mode):
    X = raw[["date"] + CHANNELS].copy()
    if mode == "div_mean":       # 当前做法（双重缩放）
        for c in CHANNELS:
            X[c] = X[c] / X[c].mean()
    elif mode == "raw":          # 正确做法：不缩放
        pass
    elif mode == "div_1e4":      # 缩放到万元
        for c in CHANNELS:
            X[c] = X[c] / 1e4
    for c in uc:
        mu, sd = raw[c].mean(), raw[c].std()
        X[c] = (raw[c] - mu) / (sd if sd > 0 else 1)
    return X


def run(name, mode, scaling=None, intercept_mu=0.098, beta_mu=None):
    X = build_X(mode)
    y = raw["revenue"].astype(float)
    kw = dict(date_column="date", channel_columns=CHANNELS, target_column="revenue",
              adstock=GeometricAdstock(l_max=6), saturation=HillSaturation(),
              yearly_seasonality=2, control_columns=uc)
    if scaling is not None:
        kw["scaling"] = scaling
    b = MMM(**kw); m = dict(b.default_model_config)
    bm = beta_mu if beta_mu is not None else np.log(0.5)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("LogNormal", mu=bm, sigma=1.0, dims="channel"),
        "intercept": Prior("Normal", mu=intercept_mu, sigma=0.06, dims=()),
        "gamma_control": Prior("Normal", mu=0.0, sigma=0.2, dims="control"),
    })
    kw["model_config"] = m
    mmm = MMM(**kw)
    if hasattr(mmm, "_channel_scales"):
        pass

    from mmm_core import MMMConfig
    idata, el = fit_mmm(mmm, X, y, MMMConfig(quick_mode=True).apply_quick_mode(),
                        progressbar=False)

    # 看模型实际用的 channel_scale
    try:
        cs = mmm.data.get_channel_scale()
        cs_str = "  ".join(f"{c}={float(np.asarray(cs.sel(channel=c).values)):.4g}"
                           for c in CHANNELS if c in cs.coords.get("channel", []))
    except Exception as e:
        cs_str = f"(取不到: {e})"

    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    est = {c: float(np.asarray(chd.sel(channel=c).values).sum()) for c in CHANNELS}
    roi = {c: est[c] / raw[c].sum() for c in CHANNELS}
    rho, p = spearmanr([roi[c] for c in CHANNELS], [TRUE_BETA[c] for c in CHANNELS])
    ic = float(idata.posterior["intercept_contribution"].values.mean())
    mshare = sum(est.values()) / y.sum()

    print(f"--- {name} ({el:.0f}s) ---")
    print(f"  channel_scale: {cs_str}")
    print(f"  截距={ic:+.4f} {'✓正' if ic>0 else '✗负'}  媒体占比={mshare:.1%} (真46.9%)")
    print(f"  ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in CHANNELS))
    print(f"  真 : " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in CHANNELS))
    print(f"  ρ={rho:+.3f} p={p:.3f}  排序: {' > '.join(sorted(roi, key=lambda k:-roi[k]))}")
    print()
    return dict(name=name, ic=ic, mshare=mshare, rho=rho, p=p)


res = []
res.append(run("A) 渠道÷均值(当前,双重缩放)", "div_mean"))
res.append(run("B) 渠道不缩放(原始元)", "raw"))
res.append(run("C) 渠道÷1e4(万元)", "div_1e4"))

print("=" * 78)
print(pd.DataFrame(res).round(3).to_string(index=False))
