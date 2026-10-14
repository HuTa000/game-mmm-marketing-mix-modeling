"""
诊断：为什么渠道贡献分解会出现负数截距、量级失衡？
=====================================================
结论预判：因为 5 个渠道花费量级差异大（kol ~3万 vs google ~16万），
         而 MMM 默认使用 identity link + 未标准化，
         模型只能靠"把截距推到负数"来抵消媒体贡献的超额部分。
"""
import os, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def make_data(n=104, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    sp = pd.DataFrame({"date": dates,
        "facebook": rng.gamma(2.0, 3.0, n) + 2,
        "google":   rng.gamma(2.5, 4.0, n) + 4,
        "tiktok":   rng.gamma(1.8, 5.0, n) + 1,
        "youtube":  rng.gamma(2.0, 2.5, n) + 1,
        "kol":      rng.gamma(1.5, 2.0, n)})
    return sp

if __name__ == "__main__":
    from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation

    sp = make_data()
    CH = ["facebook", "google", "tiktok", "youtube", "kol"]
    print("各渠道花费统计：")
    print(sp[CH].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    print()
    print("量级比 (max/mean):")
    for c in CH:
        print(f"  {c:10s} {sp[c].max()/sp[c].mean():.2f}")

    # --- 诊断 1：渠道间相关性（共线性）---
    print("\n渠道间相关系数矩阵：")
    print(sp[CH].corr().round(2).to_string())

    # --- 诊断 2：VIF ---
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        Xv = add_constant(sp[CH].values)
        print("\nVIF（方差膨胀因子，>5 提示中度共线，>10 严重）：")
        for i, c in enumerate(CH, start=1):
            v = variance_inflation_factor(Xv, i)
            flag = " <-- 注意" if v > 5 else ""
            print(f"  {c:10s} {v:7.2f}{flag}")
    except Exception as e:
        print("VIF 计算失败:", e)

    # --- 诊断 3：对比"未标准化" vs "标准化"建模 ---
    print("\n" + "=" * 60)
    print("对比实验：同样的数据，未标准化 vs 标准化")
    print("=" * 60)

    # 方式 A：直接用原始量级（当前做法）
    print("\n[A] 原始量级建模 ...")
    mmm_a = MMM(date_column="date", channel_columns=CH, target_column="y",
                adstock=GeometricAdstock(l_max=8), saturation=HillSaturation(),
                yearly_seasonality=2)
    # 造 y
    from stage0_smoke_test import make_toy_data
    X, y, _, _ = make_toy_data()
    ida = mmm_a.fit(X, y, method="mcmc", chains=2, cores=2, draws=300, tune=300,
                    random_seed=42, progressbar=False, nuts_sampler="numpyro",
                    target_accept=0.95)
    c_a = mmm_a.compute_mean_contributions_over_time(central_tendency="median")
    print("  intercept 均值:", round(c_a["intercept"].mean(), 2))
    print("  各渠道平均贡献：")
    for c in CH:
        if c in c_a.columns:
            print(f"    {c:10s} {c_a[c].mean():8.2f}")
