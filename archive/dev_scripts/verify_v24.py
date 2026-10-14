"""
验证 v24：定位 beta 与 ROI 的映射关系（最后一击）
====================================================
观察到的关键事实（v23）：
  · google 估 beta = 2.34（真 1.45）→ 但估 ROI 只有 3.0（最低档）
  · tiktok 估 beta = 2.83（真 2.30）→ 估 ROI 3.7
  · facebook 估 beta = 0.30（真 1.10）→ 估 ROI 7.65（最高！）

→ beta 与 ROI 是**负相关**！这说明模型在用"低 beta + 高饱和"或"高 beta + 低饱和"
  的**等价组合**拟合同一份数据（不可识别曲面）。

关键：模型里 contribution = beta × Hill(adstock(x_std); slope, kappa)
  由于 x_std 已除以均值，adstock(x_std) 的均值 ≈ 1/(1-decay) ∈ [1.1, 2]
  此时 beta 的**真实含义**是"当 adstock 达到半饱和点时的贡献水平"，
  它与"花费绝对量级"无关 → 所以 beta ≠ ROI。

【结论】在这套建模框架下：
  · beta 反映的是"标准化空间的效率"
  · ROI = 贡献/花费，需要贡献还原到原始尺度
  · 由于 kappa/slope 不可识别，单渠道 ROI 的**绝对水平**必然有偏
  · 但**排序**应该在数据足够时恢复

本脚本测试：加大独立脉冲数量（提供更多识别信息）+ 使用更合理的 kappa 先验。
"""
import os, sys, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx="); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from data_generator import (
    generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC, CHANNELS)
from mmm_core import MMMConfig, normalize_columns, prepare_data, fit_mmm
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior


def build(cfg, ov, control_cols):
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=control_cols)
    b = MMM(**kw); mc = dict(b.default_model_config); mc.update(ov); kw["model_config"] = mc
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
        rows.append({"c": c, "est": g/s, "true": TRUE_BETA[c], "g": g})
    rho = spearmanr([r["est"] for r in rows], [r["true"] for r in rows]).statistic
    print(f"\n{'='*86}\n{tag}\n{'='*86}")
    print(f"  intercept={ic:+.4f} | 媒体占比 {cs/ty:.1%} (真实 {tm/ty:.1%}) | ρ={rho:+.3f}")
    print(f"  {'渠道':10s} {'估beta':>8s} {'估ROI':>9s} {'真ROI':>7s}")
    for i, r in enumerate(rows):
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"  {r['c']:10s} {b:>8.3f} {r['est']:>9.3f} {r['true']:>7.2f}")
    return rho, cs/ty, ic


def main():
    # 加大识别信息：更多独立脉冲
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=False)      # 用完整采样
    cfg.draws, cfg.tune, cfg.chains = 600, 600, 2
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    tshare = df['_true_media_total'].sum()/df['revenue'].sum()
    print(f"真实媒体占比 = {tshare:.1%}")

    X, y, meta, uc = prepare_data(df, cfg)

    common = {"adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
              "saturation_slope": Prior("HalfNormal", sigma=0.15, dims="channel"),
              "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),
              "saturation_beta": Prior("HalfNormal", sigma=2.0, dims="channel"),
              "intercept": Prior("Normal", mu=0.10, sigma=0.05, dims=())}

    # 关键：加更多控制变量（trend 已在），并降低 kappa 先验强度让数据说话
    m = build(cfg, common, uc)
    fit_mmm(m, X, y, cfg, progressbar=False)
    run("完整采样 + 标准先验", m, df, cfg, tms)

    # 变体：让 kappa 更宽（Gamma(2,1)，均值 2）
    ov2 = dict(common)
    ov2["saturation_kappa"] = Prior("Gamma", alpha=2.0, beta=1.0, dims="channel")
    m2 = build(cfg, ov2, uc)
    fit_mmm(m2, X, y, cfg, progressbar=False)
    run("kappa ~ Gamma(2,1) 更宽", m2, df, cfg, tms)


if __name__ == "__main__":
    main()
