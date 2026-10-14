"""
验证 v25：固定 kappa + slope，只估 beta —— 打破不可识别曲面
==============================================================
v24 结论：ρ=-0.9，beta 与真实 ROI 系统性反向。
病因：kappa（半饱和点）与 beta（渐近线）在数学上互相补偿：
    Hill(x;k,b) 中"k 小"≡"x 大"，"beta 大"可以补偿"饱和早"。
    数据中只有 ~4 个独立脉冲/渠道 → 无法同时识别 k 和 beta。

【解法】MMM 业界标准做法（Google Meridian / Robyn 都用）：
    把 kappa 固定在业务先验值（不估计），只估 beta 与 decay。
    这样 beta 就**唯一对应**"效率"，可识别。

固定值取法：
    kappa_abs(c) = TRUE_GAMMA_FRAC(c) × mean(spend_c)  ← 但我们不该"偷看"真值
    更正当：用 EDA 得到的"花费中位数"作为半饱和点的业务估计，
    或用"平均花费的 2 倍"作为通用假设（保守：认为通常未饱和）。

本脚本测试：kappa 固定为 mean(spend)×{1.0, 2.0, 3.0}
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


def build_with_fixed_kappa(cfg, beta_sigma, kappa_mult, control_cols):
    """固定 kappa/slope：用 Deterministic 或直接删掉这两个先验"""
    kw = dict(date_column=cfg.date_col, channel_columns=cfg.channel_cols,
              target_column=cfg.target_col, adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
              saturation=HillSaturation(), yearly_seasonality=cfg.yearly_seasonality,
              control_columns=control_cols)
    b = MMM(**kw); mc = dict(b.default_model_config)
    # 注意：无法直接"固定"一个 PyMC 变量，但可以用极窄的先验近似固定
    ov = {
        "adstock_alpha": Prior("Beta", alpha=2.0, beta=4.0, dims="channel"),
        # kappa = 半饱和点，用**绝对量级**的确定值 → 但模型内部 x 是标准化的
        # 所以传入的 kappa 是"标准化空间"的值
        "saturation_kappa": Prior("HalfNormal", sigma=0.01 * kappa_mult, dims="channel"),
        "saturation_slope": Prior("HalfNormal", sigma=0.02, dims="channel"),
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
    print(f"  {'渠道':10s} {'估beta':>8s} {'估ROI':>9s} {'真ROI':>7s}")
    for i, r in enumerate(rows):
        b = float(pt["saturation_beta"].values[:, :, i].mean())
        print(f"  {r['c']:10s} {b:>8.3f} {r['est']:>9.3f} {r['true']:>7.2f}")
    return rho, cs/ty


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    tms = {c: df[f"_true_contrib_{c}"].values for c in cfg.channel_cols}
    print(f"真实媒体占比 = {df['_true_media_total'].sum()/df['revenue'].sum():.1%}")
    X, y, meta, uc = prepare_data(df, cfg)

    for km in [1.0, 3.0, 8.0]:
        m = build_with_fixed_kappa(cfg, 2.0, km, uc)
        fit_mmm(m, X, y, cfg, progressbar=False)
        run(f"固定 kappa(sigma={0.01*km}), slope(sigma=0.02), 只估 beta", m, df, cfg, tms)


if __name__ == "__main__":
    main()
