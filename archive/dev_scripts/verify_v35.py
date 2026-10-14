"""
验证 v35：基线该由谁承担？—— 4 组对照
======================================
假设：数据里基线 = 常数 + 线性趋势 + 季节正弦，
      而我又额外放了一个 `trend` 控制变量（就是那个线性趋势归一化）。
      → 趋势被"表达"了两次（基线里的 + trend 变量），完全共线。
      → 同时 fourier 季节项与基线正弦也共线。
      → 结果：intercept 与 trend、fourier 无法分离，截距被挤成负数。

4 组对照（同一份数据，只改"给模型哪些控制变量"）：
  A) 全部控制变量（trend + version + competitor + holiday）   —— 当前配置
  B) 去掉 trend（只留 version/competitor/holiday）             —— 验证 trend 是元凶
  C) 只留 trend（去掉其他三个）
  D) 无控制变量（全去掉）                                       —— 纯靠截距+渠道+季节

判据：
  · intercept 是否 > 0
  · 媒体占比是否 ≈ 46.8%
  · Spearman ρ 是否恢复
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

true_media_share = df["_true_media_total"].sum() / df["revenue"].sum()
print(f"真实媒体占比 = {true_media_share:.1%}   真实基线占比 = "
      f"{df['_true_baseline'].sum()/df['revenue'].sum():.1%}")
print(f"真实 ROI: " + "  ".join(f"{c}={TRUE_BETA[c]:.2f}" for c in CHANNELS))
print(f"trend 与真实基线的相关 = "
      f"{np.corrcoef(df['trend'].values, df['_true_baseline'].values)[0,1]:+.4f}")
print()


def run(name, controls):
    # 每个配置单独 prepare（控制变量集合不同）
    cfg2 = MMMConfig(quick_mode=True)
    cfg2.control_cols = controls
    cfg2.apply_quick_mode()
    X, y, meta, uc = prepare_data(df, cfg2)

    kw = dict(date_column=cfg2.date_col, channel_columns=cfg2.channel_cols,
              target_column=cfg2.target_col,
              adstock=GeometricAdstock(l_max=cfg2.adstock_l_max),
              saturation=HillSaturation(),
              yearly_seasonality=cfg2.yearly_seasonality)
    if uc:
        kw["control_columns"] = uc
    b = MMM(**kw); m = dict(b.default_model_config)
    m.update({
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.1, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
        "intercept": Prior("Normal", mu=cfg2.intercept_mu,
                           sigma=cfg2.intercept_sigma, dims=()),
    })
    if uc:
        m["gamma_control"] = Prior("Normal", mu=0.0, sigma=0.2, dims="control")
    kw["model_config"] = m
    mmm = MMM(**kw)
    idata, el = fit_mmm(mmm, X, y, cfg2, progressbar=False)

    ds = mmm.data.get_contributions(original_scale=True)
    chd = ds["channels"].mean(("chain", "draw"))
    bl = ds["baseline"].mean(("chain", "draw"))
    est = {c: float(np.asarray(chd.sel(channel=c).values).sum()) for c in cfg2.channel_cols}
    roi = {c: est[c]/get_spend_series(df, c).sum() for c in cfg2.channel_cols}
    rho, p = spearmanr([roi[c] for c in cfg2.channel_cols],
                       [TRUE_BETA[c] for c in cfg2.channel_cols])
    ic = float(idata.posterior["intercept_contribution"].values.mean())
    mshare = sum(est.values()) / y.sum()

    print(f"--- {name} ({el:.0f}s) ---")
    print(f"  截距={ic:+.4f} {'✓' if ic > 0 else '✗ 负'}   媒体占比={mshare:.1%} "
          f"(真 {true_media_share:.1%}, 误差 {(mshare/true_media_share-1):+.0%})")
    print(f"  估计ROI: " + "  ".join(f"{c}={roi[c]:>6.2f}" for c in cfg2.channel_cols))
    print(f"  真  ROI: " + "  ".join(f"{c}={TRUE_BETA[c]:>6.2f}" for c in cfg2.channel_cols))
    print(f"  排序: {' > '.join(sorted(roi, key=lambda k: -roi[k]))}")
    print(f"  ρ = {rho:+.3f}  p={p:.3f}")
    print()
    return dict(name=name, intercept=ic, media_share=mshare, rho=rho, p=p,
                n_ctrl=len(uc), **{f"roi_{c}": roi[c] for c in cfg2.channel_cols})


res = []
res.append(run("A) 全部控制变量(含trend)", ["version_update", "competitor_activity", "holiday", "trend"]))
res.append(run("B) 去掉trend", ["version_update", "competitor_activity", "holiday"]))
res.append(run("C) 只留trend", ["trend"]))
res.append(run("D) 无控制变量", []))

print("=" * 84)
print("汇总")
print("=" * 84)
print(pd.DataFrame(res)[["name", "n_ctrl", "intercept", "media_share", "rho", "p"]]
      .round(3).to_string(index=False))
