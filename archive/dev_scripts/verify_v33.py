"""
验证 v33：beta 先验 HalfNormal(1.0) 是不是元凶？
================================================
证据链：
  · OLS（纯线性）              → ρ = +0.900 ✓
  · M1 贝叶斯纯线性(同模型)     → ρ = -0.600 ✗  ← 荒谬！只差先验
  · 唯一区别：贝叶斯给了 beta ~ HalfNormal(1.0) 等先验

HalfNormal 的**众数是 0** → 先验在系统性地把 beta 推向 0。
5 个渠道一起被推，推力大小取决于各自的"后验陡度"，
结果排序完全乱掉。

本验证扫 4 种 beta 先验：
  a) HalfNormal(1.0)      当前（众数0，推0）
  b) HalfNormal(3.0)      放宽
  c) Normal(1.0, 1.0)     对称，众数1.0
  d) LogNormal(0, 0.7)    正值，众数~0.5，但不过度推0
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, CHANNELS, TRUE_BETA
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      fit_mmm, get_spend_series)

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)

# 用 M1 配置（纯线性：饱和压平，无 adstock）隔离 beta 先验的影响
def run(name, beta_prior_factory):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col,
              adstock=GeometricAdstock(l_max=1),
              saturation=HillSaturation(),
              yearly_seasonality=cfg.yearly_seasonality, control_columns=uc)
    b = MMM(**kw); m = dict(b.default_model_config)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=1.0, beta=50.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.01, dims="channel"),
        "saturation_kappa": Prior("HalfNormal", sigma=1.0, dims="channel"),
        "saturation_beta": beta_prior_factory(),
        "intercept": Prior("Normal", mu=cfg.intercept_mu, sigma=cfg.intercept_sigma, dims=()),
        "gamma_control": Prior("Normal", mu=0.0, sigma=cfg.control_sigma, dims="control"),
    })
    kw["model_config"] = m
    mmm = MMM(**kw)
    idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)

    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    est = {c: float(np.asarray(chd.sel(channel=c).values).sum()) for c in cfg.channel_cols}
    roi = {c: est[c] / get_spend_series(df, c).sum() for c in cfg.channel_cols}
    rho, p = spearmanr([roi[c] for c in cfg.channel_cols],
                       [TRUE_BETA[c] for c in cfg.channel_cols])
    ic = float(idata.posterior["intercept_contribution"].values.mean())
    bpost = {c: float(idata.posterior["saturation_beta"].values[:, :, i].mean())
             for i, c in enumerate(cfg.channel_cols)}
    print(f"--- {name} ({el:.0f}s) ---")
    print(f"  截距={ic:+.4f}  媒体占比={sum(est.values())/y.sum():.1%}")
    print(f"  估计ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in cfg.channel_cols))
    print(f"  真  ROI: " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in cfg.channel_cols))
    print(f"  后验beta: " + "  ".join(f"{c}={bpost[c]:>6.3f}" for c in cfg.channel_cols))
    print(f"  ρ = {rho:+.3f}  p={p:.3f}")
    print()
    return dict(name=name, rho=rho, p=p, intercept=ic,
                media_share=sum(est.values()) / y.sum(), **roi)


res = []
res.append(run("a) HalfNormal(1.0) 当前", lambda: Prior("HalfNormal", sigma=1.0, dims="channel")))
res.append(run("b) HalfNormal(3.0) 放宽", lambda: Prior("HalfNormal", sigma=3.0, dims="channel")))
res.append(run("c) Normal(1.0,1.0)", lambda: Prior("Normal", mu=1.0, sigma=1.0, dims="channel")))
res.append(run("d) LogNormal(0,0.7)", lambda: Prior("LogNormal", mu=0.0, sigma=0.7, dims="channel")))

print("=" * 78)
print(pd.DataFrame(res)[["name", "rho", "p", "intercept", "media_share"]].round(3).to_string(index=False))
