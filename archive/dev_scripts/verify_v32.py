"""
验证 v32：分层剥离不可识别性来源
=================================
假设：数据已可识别（OLS ρ=0.9），但 MMM 引入了 adstock+饱和，
      额外 15 个自由度（5 decay + 5 kappa + 5 slope），
      把可识别性又吃掉了。

实验：分 4 组，逐步增加模型复杂度，看 ROI 排序何时崩
  M1: 纯线性（无 adstock 无饱和）     —— 等价 OLS，基准
  M2: 加 Adstock（饱和用线性）        —— 5 个新参数
  M3: 加饱和（adstock 固定 l_max=1）  —— 5+5 个新参数
  M4: 全模型（本项目当前配置）        —— 15 个新参数

每组都用同一套先验，看 Spearman ρ 与 β 回收。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY
from mmm_core import (MMMConfig, normalize_columns, prepare_data,
                      fit_mmm, get_target_scale, get_spend_series)

from pymc_marketing.mmm import (
    MMM, GeometricAdstock, HillSaturation,
    LogisticSaturation, MichaelisMentenSaturation)
from pymc_extras.prior import Prior

df = generate_game_data(n_weeks=156, seed=2024)
cfg = MMMConfig(quick_mode=True).apply_quick_mode()
df = normalize_columns(df, cfg)
X, y, meta, uc = prepare_data(df, cfg)
ts = y.max()
print(f"y.max() = {ts:,.0f},  y_mean/y_max = {y.mean()/ts:.4f}")
print(f"真实 ROI: " + "  ".join(f"{c}={TRUE_BETA[c]:.2f}" for c in CHANNELS))
print()


def run(name, adstock, saturation, extra=None):
    mc = {}
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=adstock, saturation=saturation,
              yearly_seasonality=cfg.yearly_seasonality, control_columns=uc)
    b = MMM(**kw); m = dict(b.default_model_config)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "intercept": Prior("Normal", mu=cfg.intercept_mu, sigma=cfg.intercept_sigma, dims=()),
        "gamma_control": Prior("Normal", mu=0.0, sigma=cfg.control_sigma, dims="control"),
    })
    if extra:
        m.update(extra)
    kw["model_config"] = m
    mmm = MMM(**kw)
    idata, el = fit_mmm(mmm, X, y, cfg, progressbar=False)

    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    est = {}
    for c in cfg.channel_cols:
        est[c] = float(np.asarray(chd.sel(channel=c).values).sum())
    roi = {c: est[c] / get_spend_series(df, c).sum() for c in cfg.channel_cols}
    rho, p = spearmanr([roi[c] for c in cfg.channel_cols],
                       [TRUE_BETA[c] for c in cfg.channel_cols])
    media_share = sum(est.values()) / y.sum()

    ic = float(idata.posterior["intercept_contribution"].values.mean())
    print(f"--- {name} (耗时{el:.0f}s) ---")
    print(f"  intercept = {ic:+.4f}  媒体占比 = {media_share:.1%}")
    print(f"  ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in cfg.channel_cols))
    print(f"  真值: " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in cfg.channel_cols))
    print(f"  排序: {' > '.join(sorted(roi, key=lambda k: -roi[k]))}")
    print(f"  Spearman ρ = {rho:+.3f}  (p={p:.3f})")
    print()
    return dict(name=name, rho=rho, p=p, intercept=ic, media_share=media_share, **roi)


res = []
# M1: 纯线性 —— l_max=1 且 alpha 压到 ~0（等价于无 adstock），饱和斜率也压平
res.append(run("M1 纯线性(无adstock无饱和)", GeometricAdstock(l_max=1),
               HillSaturation(),
               extra={"saturation_slope": Prior("HalfNormal", sigma=0.01, dims="channel"),
                      "adstock_alpha": Prior("Beta", alpha=1.0, beta=50.0, dims="channel")}))

# M2: 只有 adstock（饱和压平）
res.append(run("M2 只加Adstock", GeometricAdstock(l_max=6), HillSaturation(),
               extra={"saturation_slope": Prior("HalfNormal", sigma=0.01, dims="channel")}))

# M3: adstock + 饱和
res.append(run("M3 Adstock+Hill饱和(全模型)", GeometricAdstock(l_max=6), HillSaturation(),
               extra={"saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel")}))

print("=" * 78)
print("汇总")
print("=" * 78)
print(pd.DataFrame(res)[["name", "rho", "p", "intercept", "media_share"]].round(3).to_string(index=False))
