"""
验证 v42：target_scale = y.max() 是否导致截距必然为负？
======================================================
推论链：
  · target_scale = y.max() = 10,837,440
  · y_std = y / y.max() ∈ [0.105, 1.0]，均值仅 0.2697
  · 模型要拟合 y_std；而渠道贡献 = beta × Hill(...)
  · 若 Σ渠道贡献 的典型水平 > 0.27，截距只能为负来倒扣
  · → 负截距是"scale 选择"的必然结果，不是 bug！

验证：显式控制 scaling.target.method
  T1) max（默认）→ y_std 均值 0.27
  T2) mean         → y_std 均值 1.0
  T3) 固定值 = y.max() 的某个倍数
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, CHANNELS, TRUE_BETA
from mmm_core import fit_mmm, MMMConfig

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.mmm.scaling import FixedScaling, DataDerivedScaling, Scaling
from pymc_extras.prior import Prior

raw = generate_game_data(n_weeks=156, seed=2024)
raw = raw.rename(columns={f"{c}_spend": c for c in CHANNELS})
uc = ["version_update", "competitor_activity", "holiday", "trend"]


def build_X():
    X = raw[["date"] + CHANNELS].copy()
    for c in CHANNELS:
        X[c] = X[c] / X[c].mean()
    for c in uc:
        mu, sd = raw[c].mean(), raw[c].std()
        X[c] = (raw[c] - mu) / (sd if sd > 0 else 1)
    return X


X = build_X()
y = raw["revenue"].astype(float)
print(f"y.max()={y.max():,.0f}  y.mean()={y.mean():,.0f}  y.mean()/y.max()={y.mean()/y.max():.4f}")
print()


def run(name, scaling, intercept_mu):
    kw = dict(date_column="date", channel_columns=CHANNELS, target_column="revenue",
              adstock=GeometricAdstock(l_max=6), saturation=HillSaturation(),
              yearly_seasonality=2, control_columns=uc, scaling=scaling)
    b = MMM(**kw); m = dict(b.default_model_config)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("LogNormal", mu=np.log(0.5), sigma=1.0, dims="channel"),
        "intercept": Prior("Normal", mu=intercept_mu, sigma=0.05, dims=()),
        "gamma_control": Prior("Normal", mu=0.0, sigma=0.2, dims="control"),
    })
    kw["model_config"] = m
    mmm = MMM(**kw)
    idata, el = fit_mmm(mmm, X, y, MMMConfig(quick_mode=True).apply_quick_mode(),
                        progressbar=False)
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    est = {c: float(np.asarray(chd.sel(channel=c).values).sum()) for c in CHANNELS}
    roi = {c: est[c] / raw[c].sum() for c in CHANNELS}
    rho, p = spearmanr([roi[c] for c in CHANNELS], [TRUE_BETA[c] for c in CHANNELS])
    ic = float(idata.posterior["intercept_contribution"].values.mean())
    mshare = sum(est.values()) / y.sum()

    print(f"--- {name} ({el:.0f}s) ---")
    print(f"  target_scale = {ts:,.0f}   截距={ic:+.4f} {'✓正' if ic>0 else '✗负'}")
    print(f"  媒体占比={mshare:.1%} (真53.9%)")
    print(f"  ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in CHANNELS))
    print(f"  真 : " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in CHANNELS))
    print(f"  ρ={rho:+.3f} p={p:.3f}  {' > '.join(sorted(roi, key=lambda k:-roi[k]))}")
    print()
    return dict(name=name, ts=ts, ic=ic, mshare=mshare, rho=rho, p=p)


res = []
res.append(run("T1 max缩放(默认)", None, 0.069))
res.append(run("T2 mean缩放", Scaling(target=DataDerivedScaling(method="mean", dims=()),
                                   channel=DataDerivedScaling(method="max", dims=())), 0.60))
res.append(run("T3 固定缩放=y.mean()", Scaling(target=FixedScaling(value=float(y.mean())),
                                            channel=DataDerivedScaling(method="max")), 0.60))

print("=" * 82)
print(pd.DataFrame(res).round(4).to_string(index=False))
