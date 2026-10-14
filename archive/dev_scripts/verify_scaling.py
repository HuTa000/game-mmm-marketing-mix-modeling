"""
验证：加数据标准化后，贡献分解是否恢复正常？
"""
import os, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def make_toy_data(n_weeks=104, seed=42):
    rng = np.random.default_rng(seed)
    channels = ["facebook", "google", "tiktok", "youtube", "kol"]
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")
    spend = pd.DataFrame({"date": dates,
        "facebook": rng.gamma(2.0, 3.0, n_weeks) + 2,
        "google":   rng.gamma(2.5, 4.0, n_weeks) + 4,
        "tiktok":   rng.gamma(1.8, 5.0, n_weeks) + 1,
        "youtube":  rng.gamma(2.0, 2.5, n_weeks) + 1,
        "kol":      rng.gamma(1.5, 2.0, n_weeks)})
    spend.loc[[10, 11, 48, 49, 90, 91], channels] *= 2.2

    true_beta = {"facebook": 1.8, "google": 2.4, "tiktok": 3.1, "youtube": 1.2, "kol": 0.9}
    true_decay = {"facebook": 0.5, "google": 0.6, "tiktok": 0.3, "youtube": 0.4, "kol": 0.7}

    def geo(x, d):
        out = np.zeros_like(x, dtype=float); c = 0.0
        for i in range(len(x)):
            c = x[i] + d * c; out[i] = c
        return out

    eff = sum(true_beta[c] * geo(spend[c].values, true_decay[c]) for c in channels)
    y = (np.linspace(30, 45, n_weeks)
         + 6 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
         + eff + rng.normal(0, 3.0, n_weeks))
    df = spend.copy(); df["y"] = np.maximum(y, 1.0)
    return df, channels, true_beta


if __name__ == "__main__":
    from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation

    df, CH, true_beta = make_toy_data()

    # ---------- 标准化：每个渠道除以自己的均值（保留"相对量级"）----------
    X = df[["date"] + CH].copy()
    y = df["y"].copy()
    means = X[CH].mean()
    X[CH] = X[CH] / means          # 除以均值 → 变成"花费相当于几倍平均"
    y_mean = y.mean()
    y_scaled = y / y_mean          # 目标也除以均值

    print("标准化后各渠道均值（应全为 1.0）：")
    print(X[CH].mean().round(4).to_string())
    print(f"\ny_scaled 均值 = {y_scaled.mean():.4f}")

    N = len(df)
    mmm = MMM(date_column="date", channel_columns=CH, target_column="y",
              adstock=GeometricAdstock(l_max=8),
              saturation=HillSaturation(),
              yearly_seasonality=2)

    print("\n开始采样（标准化数据）...")
    idata = mmm.fit(X, y_scaled, method="mcmc", chains=2, cores=2,
                    draws=400, tune=400, random_seed=42, progressbar=False,
                    nuts_sampler="numpyro", target_accept=0.95)

    import arviz as az
    # 注意：1.x 里截距参数名是 intercept_contribution，不是 intercept
    avail = list(idata.posterior.data_vars)
    pick = [v for v in ["intercept_contribution", "saturation_beta",
                        "adstock_alpha", "saturation_kappa"] if v in avail]
    s = az.summary(idata, var_names=pick, ci_prob=0.94)
    print("\n关键参数：")
    print(s.to_string())

    n_div = int(idata.sample_stats["diverging"].sum().values)
    print(f"\ndivergences = {n_div}")

    # ---------- 贡献分解（把标准化还原成原始单位）----------
    c = mmm.compute_mean_contributions_over_time(central_tendency="median")
    print("\n标准化空间的平均贡献：")
    print(f"  intercept : {c['intercept'].mean():8.3f}")
    for ch in CH:
        if ch in c.columns:
            print(f"  {ch:10s}: {c[ch].mean():8.3f}")

    # 还原：乘以 y_mean
    print(f"\n还原到原始单位（× y_mean={y_mean:.2f}）：")
    print(f"  intercept : {c['intercept'].mean() * y_mean:8.2f} 万元")
    for ch in CH:
        if ch in c.columns:
            print(f"  {ch:10s}: {c[ch].mean() * y_mean:8.2f} 万元")

    # ---------- 对比真实 ROI ----------
    print("\n对比：标准化后的 beta vs 真实 beta（需换算）")
    print("  （标准化后 beta 含义改变，这里只看相对排序是否合理）")
    for ch in CH:
        try:
            v = s.loc[f"saturation_beta[{ch}]", "mean"]
            print(f"  {ch:10s} 估计 beta={v:6.3f}   真实 beta={true_beta[ch]}")
        except Exception:
            pass
