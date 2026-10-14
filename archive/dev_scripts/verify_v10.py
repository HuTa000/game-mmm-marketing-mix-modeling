"""
验证 v10：修正先验，恢复 beta 可识别性
========================================
诊断结论（v1-v9）：
  · 尺度转换正确：predict(标准化输入)×target_scale 精度 99.99%
  · 但渠道贡献虚高 3.2 倍，beta 被系统性低估
  · 根因：**先验太宽 + 渠道共线性 → beta/kappa 联合不可识别**

修正策略（每条都有业务依据）：
  1. saturation_kappa（半饱和点）：标准化输入均值为 1，
     半饱和点应围绕 1 附近（即"平均投放量附近开始饱和"）
     → 用 Gamma(4, 4)（均值 1.0，较窄）
  2. saturation_slope（陡峭度）：业务上 S 型，取值 0.5~2
     → 用 Gamma(2, 1.5)（均值 1.33）
  3. saturation_beta（渐近线）：既然输入标准化到均值 1，
     单渠道单周贡献不应超过总收入的某个比例
     → 用 HalfNormal(0.5) 收窄

本脚本对比：宽先验 vs 紧先验 的 beta 回收与 ROI 排序。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA
from mmm_core import (
    MMMConfig, normalize_columns, prepare_data, fit_mmm, get_spend_series,
)


def build_mmm_with_priors(cfg, prior_overrides):
    from pymc_marketing.mmm import (
        MMM, GeometricAdstock, HillSaturation,
    )
    try:
        from pymc_extras.prior import Prior
    except ImportError:
        from pymc_marketing.model_config import Prior

    kwargs = dict(
        date_column=cfg.date_col,
        channel_columns=cfg.channel_cols,
        target_column=cfg.target_col,
        adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
        saturation=HillSaturation(),
        yearly_seasonality=cfg.yearly_seasonality,
    )
    base = MMM(**kwargs)
    mc = dict(base.default_model_config)
    mc.update(prior_overrides)
    kwargs["model_config"] = mc
    return MMM(**kwargs)


def evaluate(tag, mmm, X, y, df, cfg, idata):
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    true_y = float(df["revenue"].sum())

    dsA = mmm.data.get_contributions(original_scale=True)
    chA = dsA["channels"].mean(("chain", "draw"))
    ch_sum = float(np.asarray(chA.values).sum())

    rows = []
    for c in cfg.channel_cols:
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(get_spend_series(df, c).sum())
        rows.append({"渠道": c, "贡献万": g / 1e4, "估ROI": g / s, "真ROI": TRUE_BETA[c]})
    roi = pd.DataFrame(rows)
    rho = spearmanr(roi["估ROI"], roi["真ROI"]).statistic

    pt = idata.posterior
    print(f"\n{'='*84}\n{tag}\n{'='*84}")
    print(f"媒体占比 {ch_sum/true_y:.1%} (真实 46.7%) | Spearman ρ={rho:.3f}")
    print(roi.round(3).to_string(index=False))
    print("beta回收:", {c: round(float(pt['saturation_beta'].values[:,:,i].mean()), 2)
                        for i, c in enumerate(cfg.channel_cols)})
    print("真beta  :", {c: TRUE_BETA[c] for c in cfg.channel_cols})
    print("kappa回收:", {c: round(float(pt['saturation_kappa'].values[:,:,i].mean()), 2)
                         for i, c in enumerate(cfg.channel_cols)})
    return rho, ch_sum / true_y


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    from pymc_extras.prior import Prior

    # --- 宽先验（当前）---
    wide = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=3.0, beta=2.0, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=1.0, dims="channel"),
    }
    mmm1 = build_mmm_with_priors(cfg, wide)
    idata1, e1 = fit_mmm(mmm1, X, y, cfg, progressbar=False)
    rho1, share1 = evaluate("【对照】宽先验（当前配置）", mmm1, X, y, df, cfg, idata1)

    # --- 紧先验（半饱和点 ≈ 1）---
    tight = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_kappa": Prior("Gamma", alpha=4.0, beta=4.0, dims="channel"),  # 均值1
        "saturation_slope": Prior("Gamma", alpha=2.0, beta=1.5, dims="channel"),  # 均值1.33
        "saturation_beta": Prior("HalfNormal", sigma=0.5, dims="channel"),
    }
    mmm2 = build_mmm_with_priors(cfg, tight)
    idata2, e2 = fit_mmm(mmm2, X, y, cfg, progressbar=False)
    rho2, share2 = evaluate("【实验1】紧先验（kappa均值=1, beta HalfNormal(0.5)）",
                            mmm2, X, y, df, cfg, idata2)

    # --- 极紧：kappa 用 HalfNormal 且 beta 用更小 ---
    tight2 = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        "saturation_kappa": Prior("HalfNormal", sigma=0.5, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=0.3, dims="channel"),
    }
    mmm3 = build_mmm_with_priors(cfg, tight2)
    idata3, e3 = fit_mmm(mmm3, X, y, cfg, progressbar=False)
    rho3, share3 = evaluate("【实验2】极紧先验（kappa HN(0.5), beta HN(0.3)）",
                            mmm3, X, y, df, cfg, idata3)

    print(f"\n{'='*84}\n汇总\n{'='*84}")
    print(f"{'配置':40s} {'媒体占比':>10s} {'Spearman':>10s}")
    print(f"{'宽先验(当前)':40s} {share1:>10.1%} {rho1:>10.3f}")
    print(f"{'紧先验':40s} {share2:>10.1%} {rho2:>10.3f}")
    print(f"{'极紧先验':40s} {share3:>10.1%} {rho3:>10.3f}")
    print(f"{'真实':40s} {0.467:>10.1%} {1.0:>10.3f}")


if __name__ == "__main__":
    main()
