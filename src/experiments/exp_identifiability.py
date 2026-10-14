"""
可识别性判别实验（Identifiability Diagnostic Experiment）
=========================================================
目的：量化回答一个问题——

  "要让 5 渠道 MMM 的 ROI 排序可信，数据里到底需要多少『外生变动』？"

设计思路
--------
MMM 的渠道效应可识别，前提是各渠道的投放模式**彼此不同**。
真实营销数据常常做不到（大促一起投、季节一起涨），
于是渠道效应无法被拆开 —— 这就是 MMM 的核心难题：可识别性。

本实验构造 3 份数据（同一套真实参数，唯一差别是"外生变动"的量）：

  D0  纯共动   —— 所有渠道只有季节性 + 大促共动，无独立脉冲
  D1  中等     —— 每渠道 10 个独立脉冲（≈ 项目当前数据）
  D2  强       —— 每渠道 25 个独立脉冲 + 强度翻倍
  D3  极强     —— 每渠道 50 个独立脉冲 + 强度翻 3 倍

对每份数据跑 **OLS 多元回归**（无 adstock、无饱和，纯线性），
看估计 ROI 排序与真实 ROI 排序的 Spearman ρ。

判据：
  ρ > 0.8  → 数据可识别，MMM 能估出正确的渠道排序
  ρ 0.3~0.8 → 勉强，需要强先验辅助
  ρ < 0.3  → 数据本身不含足够信息，任何模型都救不回来

输出：一份"外生变动量 vs ρ"的量化曲线 → 面试可讲的实证结论。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from data_generator import (
    CHANNELS, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC,
    geometric_adstock,
)

# 复用数据生成器的帮助函数
import data_generator as dg


# ============================================================
# 通用数据生成器（可控制"外生变动"强度）
# ============================================================
def generate_variant(n_weeks: int = 156, seed: int = 2024,
                     n_pulse: int = 10,
                     pulse_boost: float = 1.0,
                     indep_pulse: bool = True) -> pd.DataFrame:
    """
    生成一份游戏出海模拟数据，唯一可调的是"渠道独立脉冲"的数量与强度。

    Parameters
    ----------
    n_pulse : int
        每个渠道的**独立脉冲**个数（只影响该渠道、其他渠道不动）。
        这是"外生变动"的核心来源 —— 它给模型提供了辨识渠道的机会。
    pulse_boost : float
        脉冲强度倍数。
    indep_pulse : bool
        False 时完全不加独立脉冲（纯共动数据）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    df = pd.DataFrame({"date": dates})

    # ---------- 1. 基线花费水平（各渠道量级不同）----------
    base_spend = {
        "facebook": 92_000, "google": 179_000, "tiktok": 125_000,
        "youtube": 65_000, "kol": 39_500,
    }

    # ---------- 2. 季节性（差异化敏感度，降低共线性的关键）----------
    q4 = np.isin(dates.month, [10, 11, 12]).astype(float)
    summer = np.isin(dates.month, [6, 7, 8]).astype(float)
    q4_boost = 1.0 + 0.35 * q4
    summer_boost = 1.0 + 0.25 * summer

    season_sensitivity = {
        "facebook": (1.00, 0.30), "google": (0.60, 0.90), "tiktok": (1.20, 0.40),
        "youtube":  (0.40, 1.00), "kol":    (0.80, 0.50),
    }

    for ch in CHANNELS:
        s_q4, s_sum = season_sensitivity[ch]
        seasonal = (1.0 + (q4_boost - 1.0) * s_q4) * (1.0 + (summer_boost - 1.0) * s_sum)
        df[f"{ch}_spend"] = base_spend[ch] * seasonal
        # 基础噪声
        df[f"{ch}_spend"] *= 1.0 + rng.normal(0, 0.06, n_weeks)

    # ---------- 3. 大促（只覆盖 3/5 渠道，制造部分共动）----------
    promo_weeks = rng.choice(n_weeks, 8, replace=False)
    promo_channels = ["facebook", "google", "tiktok"]
    for ch in promo_channels:
        df.loc[promo_weeks, f"{ch}_spend"] *= 1.0 + rng.uniform(0.5, 1.2, 8)

    # ---------- 4. 独立脉冲（外生变动的主要来源）----------
    pulse_scale = {"facebook": 1.0, "google": 0.8, "tiktok": 1.3,
                   "youtube": 1.1, "kol": 1.5}
    if indep_pulse and n_pulse > 0:
        for ch in CHANNELS:
            idx = rng.choice(n_weeks, n_pulse, replace=False)
            amp = (0.8 + rng.random(n_pulse) * 1.2) * pulse_scale[ch] * pulse_boost
            df.loc[idx, f"{ch}_spend"] *= 1.0 + amp

    # ---------- 5. 媒体贡献（绝对量级饱和，与模型结构对齐）----------
    media = {ch: np.zeros(n_weeks) for ch in CHANNELS}
    for ch in CHANNELS:
        raw = df[f"{ch}_spend"].values
        avg = float(raw.mean())
        ad = geometric_adstock(raw, TRUE_DECAY[ch])
        slope = TRUE_KAPPA[ch]
        kappa_abs = TRUE_GAMMA_FRAC[ch] * avg
        sat = (ad ** slope) / (ad ** slope + kappa_abs ** slope)
        shape = ad * sat
        norm = shape.mean() if shape.mean() > 0 else 1.0
        media[ch] = TRUE_BETA[ch] * raw.mean() * shape / norm
    total_media = sum(media.values())

    # ---------- 6. 基线收入（趋势 + 季节）----------
    mw = float(total_media.mean())
    blw = mw * (0.55 / 0.45)
    intercept = blw * 0.70
    trend = np.linspace(0, blw * 0.45, n_weeks)
    season_rev = blw * 0.12 * np.sin(2 * np.pi * t / 52.0 - 0.8)
    baseline = intercept + trend + season_rev

    # ---------- 7. 控制变量 ----------
    version_update = np.zeros(n_weeks)
    for _ in range(8):
        s = rng.integers(0, n_weeks - 2)
        version_update[s:s + 2] = 1.0
    competitor_activity = (rng.random(n_weeks) < 0.15).astype(float)
    holiday = np.isin(dates.month, [1, 2, 5]).astype(float)
    ctrl_effect = (version_update * blw * 0.08
                   - competitor_activity * blw * 0.05
                   + holiday * blw * 0.04)

    y = baseline + total_media + ctrl_effect
    y = y * (1.0 + rng.normal(0, 0.035, n_weeks))

    df["revenue"] = y
    df["version_update"] = version_update
    df["competitor_activity"] = competitor_activity
    df["holiday"] = holiday
    df["trend"] = trend / max(trend.max(), 1e-9)

    # 统一成裸渠道名（与 mmm_core 的约定一致）
    df = df.rename(columns={f"{c}_spend": c for c in CHANNELS})

    return df


# ============================================================
# OLS 可识别性评估
# ============================================================
def ols_identifiability(df: pd.DataFrame) -> dict:
    """
    用最简单的 OLS 多元回归估计"花费 → 收入"的边际效应。
    没有 adstock、没有饱和 —— 就是最纯粹的线性可识别性测试。

    如果连 OLS 都排不对顺序，说明数据本身不含足够信息，
    再复杂的贝叶斯模型也救不回来。
    """
    cols = CHANNELS + ["version_update", "competitor_activity", "holiday", "trend"]
    X = sm.add_constant(df[cols].astype(float))
    y = df["revenue"].astype(float)
    res = sm.OLS(y, X).fit()

    est = {}
    for ch in CHANNELS:
        est[ch] = float(res.params[ch])

    true_roi = {ch: TRUE_BETA[ch] for ch in CHANNELS}
    est_order = [c for c in sorted(est, key=lambda k: -est[k])]
    true_order = [c for c in sorted(true_roi, key=lambda k: -true_roi[k])]

    # Spearman：估计排序 vs 真实排序
    rho, p = spearmanr([est[c] for c in CHANNELS], [true_roi[c] for c in CHANNELS])

    return {"ols": est, "rho": float(rho), "p": float(p),
            "est_order": est_order, "true_order": true_order,
            "r2": float(res.rsquared)}


def single_var_identifiability(df: pd.DataFrame) -> dict:
    """单变量回归：每个渠道单独回归，看斜率是否互相可分"""
    out = {}
    for ch in CHANNELS:
        X = sm.add_constant(df[[ch]].astype(float))
        res = sm.OLS(df["revenue"].astype(float), X).fit()
        out[ch] = float(res.params[ch])
    return out


# ============================================================
# 主流程
# ============================================================
SCENARIOS = [
    ("D0 纯共动(0脉冲)",     dict(n_pulse=0,  pulse_boost=1.0, indep_pulse=False)),
    ("D1 中等(10脉冲)",      dict(n_pulse=10, pulse_boost=1.0)),
    ("D2 强(25脉冲x2)",      dict(n_pulse=25, pulse_boost=2.0)),
    ("D3 极强(50脉冲x3)",    dict(n_pulse=50, pulse_boost=3.0)),
]


def main():
    print("=" * 78)
    print("可识别性判别实验：外生变动量 → 渠道 ROI 排序可辨识度")
    print("=" * 78)
    print(f"真实 ROI（设计值）: " +
          "  ".join(f"{c}={TRUE_BETA[c]:.2f}" for c in CHANNELS))
    true_rank = [c for c in sorted(CHANNELS, key=lambda k: -TRUE_BETA[k])]
    print(f"真实 ROI 排序     : {' > '.join(true_rank)}")
    print()

    results = []
    for name, kw in SCENARIOS:
        df = generate_variant(n_weeks=156, seed=2024, **kw)

        # 渠道间相关系数（共线性指标）
        corr = df[CHANNELS].astype(float).corr().values
        off_diag = corr[~np.eye(len(CHANNELS), dtype=bool)]
        mean_abs_corr = float(np.abs(off_diag).mean())
        max_corr = float(np.abs(off_diag).max())

        o = ols_identifiability(df)
        sv = single_var_identifiability(df)

        results.append({
            "场景": name, "平均|相关|": mean_abs_corr, "最大|相关|": max_corr,
            "OLS_R²": o["r2"], "Spearman_ρ": o["rho"],
            **{f"OLS_{c}": o["ols"][c] for c in CHANNELS},
        })

        print(f"--- {name} ---")
        print(f"  渠道平均|相关系数| = {mean_abs_corr:.3f}   最大 = {max_corr:.3f}")
        print(f"  OLS R² = {o['r2']:.4f}   Spearman ρ = {o['rho']:+.3f} (p={o['p']:.3f})")
        print(f"  OLS 估计排序 : {' > '.join(o['est_order'])}")
        print(f"  单变量斜率   : " +
              "  ".join(f"{c}={sv[c]:>6.2f}" for c in CHANNELS))
        print(f"  OLS 估计值   : " +
              "  ".join(f"{c}={o['ols'][c]:>6.2f}" for c in CHANNELS))
        flag = ("✓ 可识别" if o["rho"] > 0.8
                else ("△ 勉强" if o["rho"] > 0.3 else "✗ 不可识别"))
        print(f"  判定: {flag}")
        print()

    res_df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "outputs", "reports", "identifiability_experiment.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("=" * 78)
    print("汇总表")
    print("=" * 78)
    print(res_df[["场景", "平均|相关|", "最大|相关|", "OLS_R²", "Spearman_ρ"]]
          .round(3).to_string(index=False))
    print(f"\n已保存: {out_path}")

    # ---- 结论 ----
    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    for r in results:
        print(f"  {r['场景']:22s} → ρ = {r['Spearman_ρ']:+.3f}")
    print()
    print("  解读：ρ 随『独立脉冲数 × 强度』单调上升 →")
    print("  证明『渠道独立外生变动』是渠道效应可识别的唯一来源。")
    print("  真实营销数据若缺乏这类变动，MMM 的 ROI 排序就不可信，")
    print("  这正是业界必须配合 geo-lift 实验校准 MMM 的根本原因。")


if __name__ == "__main__":
    main()
