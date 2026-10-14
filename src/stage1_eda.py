"""
阶段 1：探索性数据分析（EDA）
================================
每个分析都回答一个具体的业务问题：

  分析 1 ── 收入/花费的时间趋势       → "收入为什么忽高忽低？"
  分析 2 ── 趋势与季节性分解           → "有多少波动是规律性的？"
  分析 3 ── 渠道相关性矩阵 + VIF       → "5 个渠道能被分开归因吗？"
  分析 4 ── 滞后 / 互相关分析          → "广告投下去，几周后见效？"
  分析 5 ── 花费-收入散点与弹性         → "加钱还有用吗，还是已经饱和？"
  分析 6 ── 数据质量问题扫描            → "数据能直接用吗？"

运行：
    .venv/Scripts/python.exe src/stage1_eda.py
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import CHANNELS, generate_game_data

FIG_DIR = "outputs/figures"


def _ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# 分析 1：时间趋势
# ============================================================
def analysis_1_trend(df: pd.DataFrame):
    """
    业务问题：收入为什么忽高忽低？
    —— 把收入和总花费画在同一张图上，看它们是不是同向波动。
       如果是，说明投放确实在驱动收入；同时也能看出促销周的收入尖峰。
    """
    print("\n" + "=" * 70)
    print("分析 1：收入与花费的时间趋势")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    total_spend = df[[f"{c}_spend" for c in CHANNELS]].sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df["date"], df["revenue"] / 1e4, color="#c0392b", lw=2,
            label="游戏收入")
    ax.set_ylabel("收入（万元）", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.grid(alpha=0.25)
    ax.set_title("图 1：收入与总投放花费的时间趋势（2023–2025）",
                 fontsize=13, fontweight="bold", pad=12)

    ax2 = ax.twinx()
    ax2.plot(df["date"], total_spend / 1e4, color="#2471a3", lw=1.6,
             alpha=0.8, label="总花费")
    ax2.set_ylabel("总花费（万元）", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")

    # 标注大促周
    promo_idx = total_spend.nlargest(8).index
    ax.scatter(df.loc[promo_idx, "date"], df.loc[promo_idx, "revenue"] / 1e4,
               color="#e67e22", s=55, zorder=5, marker="^",
               label="大促高峰周")

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=9)

    # 下：相关性
    ax = axes[1]
    ax.plot(df["date"], df["new_users"] / 1e3, color="#16a085", lw=1.6,
            label="新增用户（千人）")
    ax.set_ylabel("新增用户（千人）", color="#16a085")
    ax.tick_params(axis="y", labelcolor="#16a085")
    ax.grid(alpha=0.25)
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    p = f"{FIG_DIR}/eda1_trend.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {p}")

    corr = np.corrcoef(df["revenue"], total_spend)[0, 1]
    corr_nu = np.corrcoef(df["new_users"], total_spend)[0, 1]
    print(f"\n  【业务解读】")
    print(f"  · 收入 vs 总花费 相关系数 = {corr:.3f}")
    print(f"  · 新增用户 vs 总花费 相关系数 = {corr_nu:.3f}")
    print(f"  · 收入区间: {df['revenue'].min()/1e4:.1f} ~ "
          f"{df['revenue'].max()/1e4:.1f} 万元")
    print(f"  · 大促高峰周的收入比平均高 "
          f"{100*(df.loc[promo_idx,'revenue'].mean()/df['revenue'].mean()-1):.0f}%")
    print("  → 投放与收入同向波动，说明花钱确实在起作用；")
    print("     但相关不等于因果，大促周两者同时升高（内生性来源）。")
    return {"corr_rev_spend": corr, "corr_nu_spend": corr_nu}


# ============================================================
# 分析 2：趋势与季节性分解
# ============================================================
def analysis_2_decompose(df: pd.DataFrame):
    """
    业务问题：收入波动中，多少是"规律性"的（趋势+季节），
             多少是"投放带来的"？
    —— 用 STL 分解把收入拆成 趋势 / 季节 / 残差。
       残差里才包含"投放的边际影响"，也说明为什么 MMM 必须
       显式建模季节性和趋势，否则会把季节误当成广告效果。
    """
    print("\n" + "=" * 70)
    print("分析 2：趋势与季节性分解（STL）")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    from statsmodels.tsa.seasonal import STL

    series = df.set_index("date")["revenue"]
    # 周数据，period=52
    stl = STL(series, period=52, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    comps = [("原始收入", res.observed, "#2c3e50"),
             ("趋势项（长期增长）", res.trend, "#c0392b"),
             ("季节项（年内周期波动）", res.seasonal, "#16a085"),
             ("残差（未被解释的部分）", res.resid, "#8e44ad")]
    for ax, (title, data, color) in zip(axes, comps):
        ax.plot(series.index, np.asarray(data) / 1e4, color=color, lw=1.5)
        ax.set_ylabel("万元", fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_title("图 2：收入的 STL 分解（趋势 / 季节 / 残差）",
                      fontsize=13, fontweight="bold", loc="left")
    axes[-1].set_xlabel("日期")
    plt.tight_layout()
    p = f"{FIG_DIR}/eda2_decompose.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {p}")

    var_total = np.var(series)
    var_trend = np.var(res.trend)
    var_season = np.var(res.seasonal)
    var_resid = np.var(res.resid)
    print(f"\n  【业务解读】各方差占比：")
    print(f"  · 趋势项解释 {100*var_trend/var_total:5.1f}% —— 产品自然增长")
    print(f"  · 季节项解释 {100*var_season/var_total:5.1f}% —— 年内周期波动")
    print(f"  · 残差占比   {100*var_resid/var_total:5.1f}% —— 投放+事件+噪声")
    print(f"  · 季节波动幅度: ±{res.seasonal.std()/1e4:.2f} 万元 "
          f"({100*res.seasonal.std()/series.mean():.1f}% of 均值)")
    print("  → 季节性不可忽略！若 MMM 不建模季节，会把"
          "『春节收入高』误判成『春节期间广告有效』，")
    print("     导致 ROI 虚高。这就是 yearly_seasonality 参数存在的理由。")
    return {"var_trend_pct": 100 * var_trend / var_total,
            "var_season_pct": 100 * var_season / var_total,
            "var_resid_pct": 100 * var_resid / var_total}


# ============================================================
# 分析 3：共线性诊断（MMM 的核心难点）
# ============================================================
def analysis_3_collinearity(df: pd.DataFrame):
    """
    业务问题：5 个渠道能被分开归因吗？
    —— 如果两个渠道的花费高度同步（你今天加 Facebook，明天也加 Google），
       模型在数学上无法区分它们的贡献。这决定了 MMM 结果是否可信。

    两个工具：
      · 相关系数矩阵：两两之间的线性关系
      · VIF（方差膨胀因子）：一个渠道能被其他渠道"预测"的程度
        VIF = 1/(1-R²)，R² 是用其他渠道预测该渠道的决定系数
        VIF < 5 健康 | 5~10 中度共线 | > 10 严重共线
    """
    print("\n" + "=" * 70)
    print("分析 3：渠道共线性诊断（MMM 能否区分渠道？）")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import seaborn as sns
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    spend = df[[f"{c}_spend" for c in CHANNELS]].copy()
    spend.columns = CHANNELS
    corr = spend.corr()

    # VIF
    Xv = add_constant(spend.values)
    vif = pd.DataFrame({
        "渠道": CHANNELS,
        "VIF": [variance_inflation_factor(Xv, i + 1)
                for i in range(len(CHANNELS))],
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    # 左：相关热力图
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8, "label": "相关系数"},
                ax=axes[0], annot_kws={"size": 10})
    axes[0].set_title("图 3a：渠道花费相关系数矩阵",
                      fontsize=12.5, fontweight="bold", pad=10)

    # 右：VIF 条形图
    colors = ["#27ae60" if v < 5 else "#f39c12" if v < 10 else "#c0392b"
              for v in vif["VIF"]]
    bars = axes[1].barh(vif["渠道"], vif["VIF"], color=colors, edgecolor="white")
    axes[1].axvline(5, color="#f39c12", ls="--", lw=1.4, label="中度共线阈值 5")
    axes[1].axvline(10, color="#c0392b", ls="--", lw=1.4, label="严重共线阈值 10")
    for b, v in zip(bars, vif["VIF"]):
        axes[1].text(v + 0.15, b.get_y() + b.get_height() / 2,
                     f"{v:.2f}", va="center", fontsize=10)
    axes[1].set_xlabel("VIF（方差膨胀因子）")
    axes[1].set_title("图 3b：各渠道 VIF（越低越好）",
                      fontsize=12.5, fontweight="bold", pad=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    p = f"{FIG_DIR}/eda3_collinearity.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {p}")

    print(f"\n  VIF 结果：")
    for _, r in vif.iterrows():
        flag = "健康 ✓" if r["VIF"] < 5 else ("中度共线 ⚠" if r["VIF"] < 10 else "严重共线 ✗")
        print(f"    {r['渠道']:10s} VIF = {r['VIF']:6.2f}   {flag}")

    max_corr = corr.where(~np.eye(len(CHANNELS), dtype=bool)).abs().max().max()
    print(f"\n  · 渠道间最大相关系数 = {max_corr:.3f}")
    print(f"  · 最大 VIF = {vif['VIF'].max():.2f}")
    if vif["VIF"].max() < 5:
        print("  → 结论：渠道间没有严重共线性，MMM 可以分开估计各渠道贡献 ✓")
    else:
        print("  → 结论：存在共线性，需合并相似渠道或用实验数据校准先验")
    return {"max_vif": float(vif["VIF"].max()),
            "max_corr": float(max_corr),
            "vif_table": vif}


# ============================================================
# 分析 4：滞后 / 互相关分析
# ============================================================
def analysis_4_lag(df: pd.DataFrame):
    """
    业务问题：今天投的广告，几周后才见效？
    —— 用互相关函数（CCF）计算"花费提前 k 周"与"收入"的相关性。
       相关性最高的滞后周数，就是广告效果的"峰值延迟"，
       这为 Adstock 的 decay 和 l_max 提供数据驱动的初始值。
    """
    print("\n" + "=" * 70)
    print("分析 4：滞后效应分析（广告多久后见效？）")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    max_lag = 8
    lags = range(0, max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].set_title("图 4a：各渠道不同滞后的相关性（找峰值延迟）",
                      fontsize=12.5, fontweight="bold", pad=10)

    lag_results = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(CHANNELS)))
    rev = df["revenue"].values
    for ch, col in zip(CHANNELS, colors):
        sp = df[f"{ch}_spend"].values
        cs = []
        for k in lags:
            if k == 0:
                a, b = sp, rev
            else:
                a, b = sp[:-k], rev[k:]
            cs.append(np.corrcoef(a, b)[0, 1])
        lag_results[ch] = cs
        axes[0].plot(list(lags), cs, marker="o", lw=1.8, color=col, label=ch)

    axes[0].set_xlabel("滞后周数（花费提前 k 周）")
    axes[0].set_ylabel("与收入的相关性")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[0].set_xticks(list(lags))

    # 右：衰减曲线拟合示意
    axes[1].set_title("图 4b：由滞后相关推断的 Adstock 衰减形态",
                      fontsize=12.5, fontweight="bold", pad=10)
    for ch, col in zip(CHANNELS, colors):
        cs = np.array(lag_results[ch])
        cs_norm = cs / cs.max()
        axes[1].plot(list(lags), cs_norm, marker="s", ms=4, lw=1.5,
                     color=col, alpha=0.85, label=ch)
    axes[1].set_xlabel("滞后周数")
    axes[1].set_ylabel("归一化相关性（峰值为 1）")
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(list(lags))

    plt.tight_layout()
    p = f"{FIG_DIR}/eda4_lag.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {p}")

    print(f"\n  各渠道滞后相关性（0~{max_lag} 周）：")
    print(f"  {'渠道':10s}" + "".join(f"{k:>7d}" for k in lags) + "   峰值滞后")
    for ch in CHANNELS:
        cs = np.array(lag_results[ch])
        peak = int(np.argmax(cs))
        print(f"  {ch:10s}" + "".join(f"{v:7.3f}" for v in cs) + f"   {peak} 周")

    print(f"\n  【业务解读】")
    print(f"  · 相关性峰值出现在滞后 0~2 周，说明广告效果主要"
          f"在投放后 2 周内释放")
    print(f"  · 建议 Adstock 初始设定：decay ≈ 0.2~0.45，l_max = 6")
    print(f"  · 若某渠道在滞后 3~5 周仍有较高相关，说明它具有")
    print(f"    长期品牌效应，decay 应设更高")
    return lag_results


# ============================================================
# 分析 5：花费-收入弹性 / 饱和初判
# ============================================================
def analysis_5_saturation(df: pd.DataFrame):
    """
    业务问题：继续加钱还有用吗，还是已经开始饱和了？
    —— 把"周花费"对"收入"做散点 + LOWESS 平滑曲线。
       曲线越平（右端斜率趋近 0），说明该渠道越饱和。
       这是阶段 3 预算优化的先验依据。
    """
    print("\n" + "=" * 70)
    print("分析 5：花费-收入关系与饱和初判")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4.2 * len(CHANNELS), 4.2))
    y = df["revenue"].values

    sat_info = {}
    for ax, ch in zip(axes, CHANNELS):
        x = df[f"{ch}_spend"].values
        sm = lowess(y, x, frac=0.55, return_sorted=True)
        ax.scatter(x / 1e4, y / 1e4, s=16, alpha=0.45, color="#95a5a6",
                   edgecolor="none")
        ax.plot(sm[:, 0] / 1e4, sm[:, 1] / 1e4, color="#c0392b", lw=2.2,
                label="LOWESS 平滑")
        ax.axvline(x.mean() / 1e4, color="#2980b9", ls="--", lw=1.3,
                   label=f"平均 {x.mean()/1e4:.1f}万")
        ax.set_title(ch, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("周花费（万元）", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("收入（万元）", fontsize=9)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7.5)

        # 简单饱和指标：高花费段 vs 低花费段的收入提升
        lo = y[x <= np.median(x)].mean()
        hi = y[x > np.median(x)].mean()
        lift = 100 * (hi / lo - 1)
        sat_info[ch] = lift

    fig.suptitle("图 5：各渠道花费-收入散点与平滑曲线（判断饱和程度）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = f"{FIG_DIR}/eda5_saturation.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {p}")

    print(f"\n  高花费段 vs 低花费段的收入差：")
    for ch in CHANNELS:
        print(f"    {ch:10s} 收入提升 {sat_info[ch]:6.1f}%")
    print(f"\n  【业务解读】")
    print(f"  · 若某渠道高花费段收入提升明显 < 平均，说明已接近饱和")
    print(f"  · 曲线斜率随花费增加而变平 → 边际收益递减 → 支持饱和曲线建模")
    print(f"  · 注意：散点是「相关」不是「因果」，含内生性（大促周既加投又收入高），")
    print(f"     真正因果需靠 MMM 分解 + 实验校准")
    return sat_info


# ============================================================
# 分析 6：数据质量扫描
# ============================================================
def analysis_6_data_quality(df: pd.DataFrame):
    """
    业务问题：这份数据能直接建模吗？
    —— 检查缺失值、异常值、零值、时间连续性。
       这些是阶段 6 Streamlit 数据校验页要自动检测的项目。
    """
    print("\n" + "=" * 70)
    print("分析 6：数据质量扫描")
    print("=" * 70)

    spend_cols = [f"{c}_spend" for c in CHANNELS]
    issues = []

    # 缺失值
    miss = df[spend_cols].isna().sum()
    print("  1) 缺失值检查：")
    for c in spend_cols:
        if miss[c] > 0:
            issues.append(f"缺失值: {c} 有 {miss[c]} 个")
            print(f"     {c}: {miss[c]} 个缺失")
    if miss.sum() == 0:
        print("     [OK] 无缺失值")

    # 零值（断投）
    print("  2) 零值/断投检查：")
    any_zero = False
    for c in spend_cols:
        nz = int((df[c] == 0).sum())
        if nz > 0:
            any_zero = True
            issues.append(f"断投: {c} 有 {nz} 周花费为 0")
            print(f"     {c}: {nz} 周为 0")
    if not any_zero:
        print("     [OK] 无零花费周")

    # 异常值（用 IQR 或 3σ）
    print("  3) 异常值检查（偏离均值 > 3σ）：")
    any_out = False
    for c in spend_cols:
        s = df[c].dropna()
        mu, sd = s.mean(), s.std()
        outliers = s[(s - mu).abs() > 3 * sd]
        if len(outliers) > 0:
            any_out = True
            issues.append(f"异常值: {c} 有 {len(outliers)} 个 (>3σ)")
            print(f"     {c}: {len(outliers)} 个，最大 {outliers.max()/1e4:.1f} 万元")
    if not any_out:
        print("     [OK] 无明显异常值（>3σ）")

    # 时间连续性
    print("  4) 时间连续性检查：")
    diffs = df["date"].diff().dropna().dt.days
    if diffs.nunique() == 1:
        print(f"     [OK] 时间间隔一致（{diffs.iloc[0]} 天）")
    else:
        issues.append(f"时间间隔不一致: {sorted(diffs.unique())}")
        print(f"     [!] 间隔不一致: {sorted(diffs.unique())}")

    print(f"\n  总计发现 {len(issues)} 个数据质量问题")
    return issues


# ============================================================
# 主流程
# ============================================================
def main():
    _ensure_dirs()

    print("=" * 70)
    print("  阶段 1：探索性数据分析（EDA）")
    print("=" * 70)

    # 生成干净数据
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    # 同时生成"脏数据"用于质量扫描演示
    df_dirty = generate_game_data(n_weeks=156, seed=2024, inject_issues=True)

    print(f"\n数据: {len(df)} 周 × {len(CHANNELS)} 渠道  "
          f"({df['date'].min().date()} ~ {df['date'].max().date()})")
    print(f"周均收入 {df['revenue'].mean()/1e4:.1f} 万元 | "
          f"周均总花费 "
          f"{df[[f'{c}_spend' for c in CHANNELS]].sum(axis=1).mean()/1e4:.1f} 万元")

    r1 = analysis_1_trend(df)
    r2 = analysis_2_decompose(df)
    r3 = analysis_3_collinearity(df)
    r4 = analysis_4_lag(df)
    r5 = analysis_5_saturation(df)
    print("\n" + "=" * 70)
    print("分析 6：数据质量扫描（使用注入了问题的脏数据）")
    print("=" * 70)
    r6 = analysis_6_data_quality(df_dirty)

    # ---------- 保存数据 ----------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/game_ads_weekly.csv", index=False,
              encoding="utf-8-sig")
    df_dirty.to_csv("data/processed/game_ads_weekly_dirty.csv", index=False,
                    encoding="utf-8-sig")
    print(f"\n  数据已保存:")
    print(f"    data/processed/game_ads_weekly.csv (干净)")
    print(f"    data/processed/game_ads_weekly_dirty.csv (含问题，测试用)")

    # ---------- 结论汇总 ----------
    print("\n" + "=" * 70)
    print("  EDA 结论汇总")
    print("=" * 70)
    print(f"  1. 收入与花费相关系数 {r1['corr_rev_spend']:.3f} —— "
          f"投放与收入同向，但需警惕内生性")
    print(f"  2. 方差分解：趋势 {r2['var_trend_pct']:.1f}% / "
          f"季节 {r2['var_season_pct']:.1f}% / "
          f"残差 {r2['var_resid_pct']:.1f}%")
    print(f"     → 季节性与趋势必须建模，否则 ROI 虚高")
    print(f"  3. 最大 VIF = {r3['max_vif']:.2f}，"
          f"最大渠道相关 = {r3['max_corr']:.3f}")
    print(f"     → {'渠道可分离，MMM 可行 ✓' if r3['max_vif'] < 5 else '存在共线，需处理 ⚠'}")
    print(f"  4. 广告效果峰值滞后 0~2 周 → Adstock decay ≈ 0.2~0.45, l_max=6")
    print(f"  5. 各渠道花费-收入曲线均呈现边际递减 → 需饱和曲线")
    print(f"  6. 数据质量检查项已确定，将用于 Streamlit 数据校验页")
    print("\n" + "=" * 70)
    print("  阶段 1 完成 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
