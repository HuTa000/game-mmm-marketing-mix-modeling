"""
阶段 0：环境自检 + PyMC-Marketing 官方 MMM 示例跑通
=====================================================
运行：
    .venv/Scripts/python.exe src/stage0_smoke_test.py

【重要】Windows 上跑 PyMC 必须遵守两条铁律：
  1. 所有建模代码必须放在 `if __name__ == "__main__":` 里面！
     原因：PyMC 多链采样会开子进程（Windows 用 spawn），
     子进程会重新 import 本脚本。如果没有 main 保护，
     子进程又会去开子进程 → RuntimeError 死循环。
  2. 建议显式设定轮数较少 + chains/cores 对齐。

本脚本做三件事：
  1. 打印关键库版本（确认环境正确）
  2. 造一份"迷你投放数据"跑通完整 MMM 流程
  3. 输出后验 + 诊断指标，让你看到贝叶斯 MMM 吐出什么
"""
import os
import sys
import warnings
import time

# 必须在 import pymc 之前设置：禁用 C 编译器尝试，避免 Windows 上的报错/卡顿
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")


def step1_env_check():
    """第 1 步：环境自检"""
    print("=" * 64)
    print("第 1 步：环境自检")
    print("=" * 64)
    import importlib
    core = ["pymc", "pymc_marketing", "arviz", "numpy", "pandas",
            "matplotlib", "scipy", "statsmodels", "streamlit", "plotly"]
    for m in core:
        try:
            mod = importlib.import_module(m)
            print(f"  [OK]  {m:18s} {getattr(mod, '__version__', 'n/a')}")
        except ImportError:
            print(f"  [!!]  {m:18s} 未安装")
    print(f"  Python {sys.version.split()[0]}")
    print(f"  PYTENSOR_FLAGS = {os.environ.get('PYTENSOR_FLAGS')}")


def make_toy_data(n_weeks=104, seed=42):
    """
    第 2 步：构造玩具数据（5 渠道 × 104 周）
    关键：我们用已知的"真实参数"造数据，之后看模型能否把它们猜回来。
    """
    print()
    print("=" * 64)
    print("第 2 步：构造玩具数据（5 渠道 × %d 周）" % n_weeks)
    print("=" * 64)

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    channels = ["facebook", "google", "tiktok", "youtube", "kol"]
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")

    # --- 各渠道每周花费（万元）---
    spend = pd.DataFrame({
        "date": dates,
        "facebook": rng.gamma(2.0, 3.0, n_weeks) + 2,
        "google":   rng.gamma(2.5, 4.0, n_weeks) + 4,
        "tiktok":   rng.gamma(1.8, 5.0, n_weeks) + 1,
        "youtube":  rng.gamma(2.0, 2.5, n_weeks) + 1,
        "kol":      rng.gamma(1.5, 2.0, n_weeks),
    })
    # 大促周加投（模拟真实投放节奏）
    promo_weeks = [10, 11, 48, 49, 90, 91]
    spend.loc[promo_weeks, channels] *= 2.2

    # --- 真实参数（待模型回收）---
    true_beta = {"facebook": 1.8, "google": 2.4, "tiktok": 3.1,
                 "youtube": 1.2, "kol": 0.9}
    true_decay = {"facebook": 0.5, "google": 0.6, "tiktok": 0.3,
                  "youtube": 0.4, "kol": 0.7}

    def geometric_adstock(x, decay):
        """A_t = x_t + decay * A_{t-1}  （几何衰减）"""
        out = np.zeros_like(x, dtype=float)
        carry = 0.0
        for i in range(len(x)):
            carry = x[i] + decay * carry
            out[i] = carry
        return out

    adstocked = {c: geometric_adstock(spend[c].values, true_decay[c])
                 for c in channels}
    media_effect = sum(true_beta[c] * adstocked[c] for c in channels)

    trend = np.linspace(30, 45, n_weeks)
    season = 6 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    y = trend + season + media_effect + rng.normal(0, 3.0, n_weeks)
    y = np.maximum(y, 1.0)

    df = spend.copy()
    df["y"] = y
    X = df[["date"] + channels]
    y_series = df["y"]

    print(f"  数据形状: {df.shape}")
    print(f"  目标 y 范围: {y.min():.1f} ~ {y.max():.1f} (均值 {y.mean():.1f})")
    print("  各渠道平均周花费(万元) / 真实 beta:")
    for c in channels:
        print(f"    {c:10s} 花费 {spend[c].mean():6.2f}   true_beta={true_beta[c]}")
    return X, y_series, channels, true_beta


def step3_build_model(channels):
    """第 3 步：构建 MMM"""
    print()
    print("=" * 64)
    print("第 3 步：构建贝叶斯 MMM（GeometricAdstock + HillSaturation）")
    print("=" * 64)
    from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
    mmm = MMM(
        date_column="date",
        channel_columns=channels,
        target_column="y",
        adstock=GeometricAdstock(l_max=8),
        saturation=HillSaturation(),
        yearly_seasonality=2,
    )
    print("  模型构建完成（构建本身极快，耗时在采样）。")
    return mmm


def step4_sample(mmm, X, y_series, quick=True):
    """
    第 4 步：NUTS 采样
    quick=True → 演示模式（快，但后验略粗糙）
    quick=False → 正式模式（慢，结果更稳）
    """
    print()
    print("=" * 64)
    print("第 4 步：MCMC 采样（numpyro 后端）")
    print("=" * 64)

    if quick:
        draws, tune, chains = 300, 300, 2
    else:
        draws, tune, chains = 1000, 1000, 4

    print(f"  配置: draws={draws} tune={tune} chains={chains} 后端=numpyro")
    t0 = time.time()
    # 【关键 API 细节】PyMC-Marketing 1.x 中，采样参数要作为顶层 **kwargs 直接传入，
    # 而不是放进 sample_kwargs（sample_kwargs 只给 advi/map 等变分方法用）。
    # 若误放进 sample_kwargs，chains/draws 会被忽略，悄悄用默认值（4链×1000步）→ 巨慢。
    idata = mmm.fit(
        X, y_series,
        method="mcmc",
        chains=chains,
        cores=chains,             # cores 应与 chains 对齐
        draws=draws,
        tune=tune,
        random_seed=42,
        progressbar=True,
        nuts_sampler="numpyro",
        target_accept=0.95,       # 默认 0.8；调高可显著减少 divergences（代价：略慢）
    )
    print(f"  采样完成，耗时 {time.time() - t0:.1f}s")
    return idata


def step5_diagnostics(mmm, idata):
    """第 5 步：拟合诊断"""
    print()
    print("=" * 64)
    print("第 5 步：拟合诊断（R-hat / ESS）")
    print("=" * 64)
    import arviz as az

    post = idata.posterior
    want = ["intercept", "y_sigma", "saturation_beta",
            "adstock_alpha", "saturation_kappa", "saturation_slope"]
    var_names = [v for v in post.data_vars if v in want]
    if not var_names:
        var_names = list(post.data_vars)[:8]

    # 【API 变更】ArviZ 1.x 把 hdi_prob 改名为 ci_prob
    try:
        summary = az.summary(idata, var_names=var_names, ci_prob=0.94, round_to=3)
    except TypeError:
        summary = az.summary(idata, var_names=var_names, hdi_prob=0.94, round_to=3)
    print(summary.to_string())

    # --- divergences 检查（采样器"撞墙"次数）---
    try:
        n_div = int(idata.sample_stats["diverging"].sum().values)
        total = idata.sample_stats["diverging"].size
        print(f"\n  [divergences] {n_div} / {total} 次"
              f" ({100.0 * n_div / max(total, 1):.2f}%)")
        if n_div == 0:
            print("    -> 无发散，采样健康 ✓")
        else:
            print("    -> 出现发散：后验地形有难点。修法：")
            print("       1) 提高 target_accept（如 0.9 -> 0.95）")
            print("       2) 增加 tune 步数")
            print("       3) 检查先验是否过窄 / 数据是否标准化")
    except Exception as e:
        print(f"  [提示] divergences 检查失败: {e}")

    if "r_hat" in summary.columns:
        bad = summary[summary["r_hat"] > 1.01]
        print()
        if len(bad) == 0:
            print("  [结论] 所有参数 R-hat < 1.01 -> 链已收敛，结果可信 ✓")
        else:
            print(f"  [警告] {len(bad)}/{len(summary)} 个参数 R-hat > 1.01")
            print("         demo 模式(2链×300)属正常；正式分析请用 quick=False(4链×1000)")
    return summary


def step6_contributions(mmm):
    """第 6 步：渠道贡献分解"""
    print()
    print("=" * 64)
    print("第 6 步：渠道贡献分解（各渠道贡献了多少收入）")
    print("=" * 64)
    try:
        contrib = mmm.compute_mean_contributions_over_time(central_tendency="median")
        print(f"  返回类型: {type(contrib).__name__}, 形状: {getattr(contrib, 'shape', 'n/a')}")
        print(contrib.head(3).to_string())
        return contrib
    except Exception as e:
        print(f"  [提示] 调用异常: {type(e).__name__}: {e}")
        return None


def step7_save(mmm, idata):
    """第 7 步：保存"""
    print()
    print("=" * 64)
    print("第 7 步：保存结果")
    print("=" * 64)
    os.makedirs("outputs", exist_ok=True)
    try:
        idata.to_netcdf("outputs/stage0_idata.nc")
        print("  后验已保存: outputs/stage0_idata.nc")
    except Exception as e:
        print(f"  [提示] 保存失败: {e}")


def main():
    step1_env_check()
    X, y_series, channels, true_beta = make_toy_data()
    mmm = step3_build_model(channels)
    idata = step4_sample(mmm, X, y_series, quick=True)
    summary = step5_diagnostics(mmm, idata)
    step6_contributions(mmm)
    step7_save(mmm, idata)

    print()
    print("=" * 64)
    print("  阶段 0 完成：流程已跑通 ✓")
    print("=" * 64)
    # 提示：对比真实 beta 与估出的 beta
    try:
        b = summary.loc["saturation_beta", "mean"]
        print("\n  [对照] 模型估出的平均 saturation_beta =", b,
              "| 真实 beta 的均值 =",
              sum(true_beta.values()) / len(true_beta))
    except Exception:
        pass


if __name__ == "__main__":          # ← Windows 上跑 PyMC 必须有这行！
    main()
