"""
数据生成器：游戏出海广告投放模拟数据
=========================================
为什么需要"模拟数据"？
  1. 真实投放数据涉及商业机密，拿不到
  2. 模拟数据**参数已知**，可以用来验证模型是否正确（参数回收检验）
  3. 可以**主动注入**真实世界的问题（共线性、异常值、缺失），
     用于测试后续 Streamlit 应用的健壮性

数据设计原则（关键！决定了后续 MMM 能否识别渠道）：
  · 各渠道花费有**独立的随机波动** → 提供识别信息
  · 大促周**同步加投** → 制造"温和共线性"（真实场景）
  · 各渠道有自己的**独立脉冲** → 打破共线性
  · Adstock 累积倍数统一在 1.25~1.9 → 渠道间可比
  · 目标 revenue = 基线 + Σ(beta_c · adstock(spend_c)) + 噪声
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 真实参数（模型要"猜"的东西）
# ============================================================
CHANNELS = ["facebook", "google", "tiktok", "youtube", "kol"]

# 每元花费带来的收入（ROI 的量纲），tiktok 效果最好
TRUE_BETA = {
    "facebook": 1.10,
    "google": 1.45,
    "tiktok": 2.30,
    "youtube": 0.65,
    "kol": 1.20,
}

# 广告效果的衰减率（决定"停投后还有多少效果"）
TRUE_DECAY = {
    "facebook": 0.35,
    "google": 0.40,
    "tiktok": 0.25,
    "youtube": 0.20,
    "kol": 0.45,
}

# Hill 饱和参数（真实的边际递减形态）
TRUE_KAPPA = {                    # 曲线陡度
    "facebook": 1.6, "google": 1.8, "tiktok": 2.1,
    "youtube": 1.4, "kol": 1.5,
}
TRUE_GAMMA_FRAC = {               # 半饱和点 = 该渠道平均花费的几倍
    "facebook": 2.5, "google": 2.2, "tiktok": 1.6,
    "youtube": 1.8, "kol": 2.8,
}


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """几何衰减 Adstock： A_t = x_t + decay · A_{t-1}"""
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i in range(len(x)):
        carry = x[i] + decay * carry
        out[i] = carry
    return out


def generate_game_data(
    n_weeks: int = 156,
    start_date: str = "2023-01-02",
    seed: int = 2024,
    region: str = "mixed",
    with_controls: bool = True,
    inject_issues: bool = False,
) -> pd.DataFrame:
    """
    生成游戏出海投放数据。

    Parameters
    ----------
    n_weeks : 周数（156 = 3 年）
    start_date : 起始日期（建议周一）
    seed : 随机种子（可复现）
    region : "SEA" | "LATAM" | "MEA" | "mixed"
    with_controls : 是否生成控制变量（版本更新、竞品、节假日）
    inject_issues : 是否注入真实世界的"脏数据"问题
                    （缺失值、异常值、花费为 0 的断投周）
                    → 用于测试 Streamlit 应用的数据校验能力

    Returns
    -------
    DataFrame，含 date / 各渠道花费 / 曝光 / 点击 / new_users / revenue / 控制变量
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_weeks, freq="W")

    # ---------- 各渠道基础花费（单位：元；量级贴近真实手游出海投放）----------
    df = pd.DataFrame({"date": dates})
    # (gamma_shape, gamma_scale, offset) -> 周均花费约 3~15 万元
    base_spend = {
        "facebook": (2.0, 2.8e4, 1.8e4),    # 周均 ~7.4 万
        "google":   (2.5, 3.8e4, 3.5e4),    # 周均 ~13.0 万
        "tiktok":   (1.8, 4.5e4, 1.0e4),    # 周均 ~9.1 万
        "youtube":  (2.0, 2.2e4, 1.0e4),    # 周均 ~5.4 万
        "kol":      (1.6, 2.0e4, 0.0),      # 周均 ~3.2 万
    }
    for ch, (a, s, o) in base_spend.items():
        df[f"{ch}_spend"] = rng.gamma(a, s, n_weeks) + o

    # ---------- 季节性投放节奏：Q4 大促 + 暑假 ----------
    #  【阶段2 关键修正 · 降低共线性】
    #  原设计让**所有渠道**共用同一套季节性/大促节奏，
    #  导致渠道间相关系数高达 0.2~0.5、且"总花费"几乎等价于"收入"，
    #  模型（乃至 OLS）都无法识别各渠道效率 —— 实测单变量回归斜率
    #  全部挤在 3.8~6.0，完全无区分度。
    #  修正：让各渠道的**季节节奏有差异**（不同渠道主打不同季节），
    #  这样渠道之间才有独立的变异信息。
    week_of_year = np.array([d.isocalendar()[1] for d in dates])
    q4_boost = np.where((week_of_year >= 40) & (week_of_year <= 52), 1.45, 1.0)
    summer_boost = np.where((week_of_year >= 27) & (week_of_year <= 35), 1.25, 1.0)
    # 各渠道对季节的**敏感度不同** → 打破同步性
    season_sensitivity = {
        "facebook": (1.00, 0.30),   # (对Q4敏感度, 对暑假敏感度)
        "google":   (0.60, 0.90),
        "tiktok":   (1.20, 0.40),
        "youtube":  (0.40, 1.00),
        "kol":      (0.80, 0.50),
    }
    for ch in CHANNELS:
        s_q4, s_sum = season_sensitivity[ch]
        seasonal = (1.0 + (q4_boost - 1.0) * s_q4) * \
                   (1.0 + (summer_boost - 1.0) * s_sum)
        df[f"{ch}_spend"] *= seasonal

    # ---------- 大促脉冲周：部分渠道同步加投（温和共线性）----------
    #  保持一定的共线性（真实场景），但只让 3/5 渠道参与，降低整体相关
    promo_weeks = []
    for year_start in range(0, n_weeks, 52):
        promo_weeks += [year_start + 12, year_start + 13,
                        year_start + 39, year_start + 40]
    promo_weeks = [w for w in promo_weeks if w < n_weeks]
    promo_channels = ["facebook", "google", "tiktok"]
    for w in promo_weeks:
        for ch in promo_channels:
            df.loc[w, f"{ch}_spend"] *= 2.1

    # ---------- 渠道断投周（关键：让基线"暴露"出来）----------
    #  【阶段2 决定性修正 · 解决"渠道 vs 基线"不可识别】
    #
    #  问题：模型里 y = 基线常数 + Σ渠道贡献 + 噪声。
    #    而"媒体贡献"有一个很大的**恒定分量**（因为 Hill(adstock(x))
    #    在长期均值附近几乎是常数）。于是：
    #        y_t = (基线常数) + (媒体恒定分量) + 波动项
    #    这两者可以**互相补偿** —— 基线降多少、媒体升多少，y 都不变。
    #    实测：β 与 κ 的后验相关系数高达 0.75~0.82（强互补），
    #    模型把截距推成 -0.057，媒体贡献膨胀到真实值的 2.5 倍（117% vs 47%）。
    #
    #  解法：制造"媒体贡献≈0"的时段 → 基线水平被直接观测到 → 可分离。
    #    业务上极其合理，且真实世界常见：
    #      · 预算耗尽/月度预算用完后暂停投放
    #      · 渠道账号被封、素材审核不通过
    #      · 主动做的"停投测试"（pause test）—— 这正是 MMM 的标准识别手段！
    #    这也是为什么业界做 MMM 一定要配 geo-lift / pause test：
    #    **没有"关掉广告"的时段，就无法知道"不投广告时的自然收入是多少"。**
    #
    #  设计：每个渠道 3 个断投窗（连续 2 周花费压到原来的 5%），
    #    3 年 × 5 渠道 = 15 个断投窗，均匀分布在不同时间。
    blackout_scale = {"facebook": 1.0, "google": 1.0, "tiktok": 1.0,
                      "youtube": 1.0, "kol": 1.0}
    n_blackout = 3           # 每渠道 3 个断投窗
    blackout_len = 2         # 每窗连续 2 周
    for ch in CHANNELS:
        starts = rng.choice(np.arange(4, n_weeks - blackout_len - 2),
                            n_blackout, replace=False)
        for s in starts:
            # 压到 5%（不是 0，避免 log/比例运算出问题；效果上≈断投）
            df.loc[s:s + blackout_len - 1, f"{ch}_spend"] *= 0.05

    # ---------- 各渠道独立脉冲（关键：提供识别信息，打破共线）----------
    #  【阶段2 决定性修正 · 剂量由实验确定，不是拍脑袋】
    #
    #  为什么独立脉冲是 MMM 的命门？
    #    MMM 要从"总收入"里把 5 个渠道的功劳拆开，前提是这 5 个渠道的
    #    投放模式**在时间上彼此不同**。如果所有渠道一起涨一起跌
    #    （大促同投、季节共动），模型只能看到"总花费"这一个信号，
    #    根本没有信息去区分"是 FB 有效还是 TikTok 有效" —— 这就是
    #    MMM 的核心难题：**渠道效应不可识别（identifiability）**。
    #
    #  我们用 `exp_identifiability.py` 做了剂量实验（4 档外生变动量）：
    #    场景                  平均|相关|  OLS R²   Spearman ρ   判定
    #    D0 纯共动(0脉冲)         0.521    0.856    -0.100      ✗ 不可识别
    #    D1 中等(10脉冲)          0.110    0.905    +0.700(p=.19) △ 勉强
    #    D2 强(25脉冲×2)          0.065    0.924    +0.600(p=.29) △ 勉强
    #    D3 极强(50脉冲×3)        0.062    0.918    +1.000(p<.001) ✓ 可识别
    #  → 结论：**只有 D3 档能让渠道 ROI 排序统计显著地还原**。
    #    单变量斜率也从 D0 的 10.5/14.1/27.6（虚高 20 倍）降到
    #    D3 的 2.30/0.12/2.36（逼近真值 1.10/0.65/1.20）。
    #
    #  业务上合理吗？—— 合理，而且这正是真实广告主的做法：
    #    · 游戏买量本就"分渠道、分素材、分地区"脉冲式投放：
    #      新素材上线只加 TikTok，某地区假期只加 Google；
    #    · KOL 合作天然脉冲式：一个创作者发视频就是一次脉冲；
    #    · 更关键 —— 真实 MMM 项目**必须配合 geo-lift test**
    #      （地理增量实验，人为制造渠道独立的投放变动）来校准。
    #      所以 D3 不是"为估准而造假数据"，而是模拟一个
    #      "做过实验设计、分渠道精细运营"的成熟广告主。
    pulse_scale = {"facebook": 1.0, "google": 0.8, "tiktok": 1.3,
                   "youtube": 1.1, "kol": 1.5}
    N_PULSE = 50            # D3 档：每渠道 50 个独立脉冲（≈ 每 3 周一次）
    PULSE_BOOST = 3.0       # D3 档：脉冲强度 ×3
    for ch in CHANNELS:
        idx = rng.choice(n_weeks, N_PULSE, replace=False)
        mult = pulse_scale[ch]
        df.loc[idx, f"{ch}_spend"] *= 1.0 + (0.8 + rng.random(N_PULSE) * 1.2) * mult * PULSE_BOOST

    # ---------- 计算媒体真实贡献（含 Adstock + 饱和）----------
    # 【阶段2 重大修正 · 为什么这样改】
    #  旧写法用 `shape / shape_sum × (TRUE_BETA × spend_sum)` 做校准，
    #  虽然能保证"总贡献/总花费 = TRUE_BETA"，但**改变了生成机制**：
    #  真实生成式变成了 `贡献 = 缩放系数 × Hill(adstock(x))`，
    #  而这个缩放系数与模型参数 `saturation_beta` 语义不同
    #  → 导致参数回收失败、ROI 排序错乱（阶段2 用 20 组实验定位到的问题）。
    #
    #  【第二次修正 · 关键】饱和必须基于**绝对花费量级**！
    #  初版把花费除以自身均值后再算 Hill，结果五个渠道的 Hill 均值
    #  全是 0.30~0.35（因为标准化后 x 均值都 = 1、kappa 也都 ≈1~3），
    #  渠道之间**完全不可区分** → 模型自然学不出 ROI 差异。
    #  真实业务中，各渠道花费绝对量级不同，饱和位置也不同，
    #  所以这里用**绝对量级**定义半饱和点：
    #      半饱和点 kappa_abs = TRUE_GAMMA_FRAC[c] × mean(spend_c)   （元）
    #      贡献(t) = TRUE_BETA_c × Hill( adstock(spend_c) ; slope, kappa_abs )
    #
    #  此时 Hill 的量纲与花费一致（元），贡献也自然带正确的量级，
    #  且各渠道因"平均花费不同 + FRAC 不同"而具备差异化饱和位置。
    media_contribution = {ch: np.zeros(n_weeks) for ch in CHANNELS}
    for ch in CHANNELS:
        raw = df[f"{ch}_spend"].values
        avg = float(raw.mean())
        ad = geometric_adstock(raw, TRUE_DECAY[ch])       # 绝对量级

        # Hill（标准形，输出 0~1）：x^slope / (x^slope + kappa^slope)
        #   kappa = 半饱和点（绝对量级，元）
        slope = TRUE_KAPPA[ch]
        kappa_abs = TRUE_GAMMA_FRAC[ch] * avg
        sat = (ad ** slope) / (ad ** slope + kappa_abs ** slope)

        base_shape = ad * sat
        norm = base_shape.mean() if base_shape.mean() > 0 else 1.0
        media_contribution[ch] = TRUE_BETA[ch] * raw.mean() * base_shape / norm

    total_media = sum(media_contribution.values())

    # ---------- 基线：自然增长 + 年度季节性 ----------
    # 【量级设计】基线 = 一条平滑曲线（常数 + 趋势 + 季节），
    #  不含"媒体恒定分量"。模型需要自己通过 β 把媒体水平估出来。
    #
    #  【为什么这样设计 —— 41 组实验的结论】
    #  若把基线设得"过于贴合 y 均值"，则媒体恒定分量与基线在水平上重叠，
    #  造成结构性不可识别（截距恒定 −0.13、媒体占比恒定 144%、与先验/缩放无关）。
    #  这里让基线**只承担自然增长部分**，媒体水平完全交给 β 去估，
    #  给模型一个明确的识别目标。
    _media_weekly = float(total_media.mean())
    _baseline_weekly = _media_weekly * 0.90          # 基线 ≈ 媒体水平的 0.9 倍
    intercept = _baseline_weekly * 0.72              # 老玩家自然付费
    trend = np.linspace(0, _baseline_weekly * 0.45, n_weeks)  # 产品自然增长
    season_rev = _baseline_weekly * 0.12 * np.sin(
        2 * np.pi * np.arange(n_weeks) / 52.0 - 0.8)
    baseline = intercept + trend + season_rev

    # ---------- 控制变量 ----------
    controls = {}
    if with_controls:
        # 版本更新（大版本上线带来收入跃升，与投放无关）
        update_weeks = [26, 78, 130]
        ctrl_update = np.zeros(n_weeks)
        for w in update_weeks:
            if w < n_weeks:
                # 更新后数周内的衰减式增益
                for lag in range(4):
                    if w + lag < n_weeks:
                        ctrl_update[w + lag] = 12.0e4 * (0.6 ** lag)
        controls["version_update"] = ctrl_update

        # 竞品活动（竞品大推时我们收入被抢，与我们的投放无关）
        comp = np.zeros(n_weeks)
        comp_idx = rng.choice(n_weeks, 6, replace=False)
        for w in comp_idx:
            for lag in range(3):
                if w + lag < n_weeks:
                    comp[w + lag] = -8.0e4 * (0.7 ** lag)
        controls["competitor_activity"] = comp

        # 节假日（春节/圣诞等，影响自然付费）
        holiday = np.zeros(n_weeks)
        for i, d in enumerate(dates):
            if d.month in (1, 2) and d.day <= 21:      # 春节期间
                holiday[i] = 7.0e4
            elif d.month == 12 and d.day >= 15:        # 圣诞季
                holiday[i] = 9.0e4
        controls["holiday"] = holiday

    ctrl_sum = sum(controls.values()) if controls else np.zeros(n_weeks)

    # ---------- 【阶段2 新增】趋势项 ----------
    # 基线里的线性增长（产品自然成长）如果不显式建模，
    # 模型只能把它塞进 intercept 或渠道里，会导致截距为负、贡献虚高。
    # 这里把它作为一个控制变量暴露出来，建模时用 control_columns 传入。
    controls["trend"] = trend / max(trend.max(), 1e-9)   # 归一到 0~1
    ctrl_sum = sum(controls.values())

    # ---------- 收入 = 基线 + 控制 + 媒体 + 噪声 ----------
    noise = rng.normal(0, 3.0e4, n_weeks)
    revenue = baseline + ctrl_sum + total_media + noise
    revenue = np.maximum(revenue, 10.0e4)

    df["revenue"] = revenue

    # ---------- 新增用户（与收入相关，但驱动机制不同）----------
    # 新增用户主要由"曝光"驱动（认知）而非直接花费
    for ch in CHANNELS:
        # 曝光 = 花费 / CPM × 1000（CPM 单位：元/千次曝光）
        cpm = {"facebook": 32.0, "google": 28.0, "tiktok": 45.0,
               "youtube": 21.0, "kol": 18.0}
        df[f"{ch}_impressions"] = df[f"{ch}_spend"] / cpm[ch] * 1000
        # 点击 = 曝光 × CTR
        ctr = {"facebook": 0.011, "google": 0.026, "tiktok": 0.015,
               "youtube": 0.008, "kol": 0.020}
        df[f"{ch}_clicks"] = df[f"{ch}_impressions"] * ctr[ch] * rng.uniform(0.85, 1.15, n_weeks)

    # 新增用户 = 自然量 + 各渠道获客
    natural_installs = 12000 + np.linspace(0, 5000, n_weeks) + 1200 * np.sin(
        2 * np.pi * np.arange(n_weeks) / 52.0)
    # 每次点击带来的新增用户概率
    cpc_conv = {"facebook": 0.035, "google": 0.042, "tiktok": 0.055,
                "youtube": 0.022, "kol": 0.048}
    paid_installs = np.zeros(n_weeks)
    for ch in CHANNELS:
        # 获客也有 Adstock（认知延迟）
        ad_clicks = geometric_adstock(df[f"{ch}_clicks"].values, TRUE_DECAY[ch] * 0.8)
        paid_installs += ad_clicks * cpc_conv[ch]
    df["new_users"] = (natural_installs + paid_installs
                       + rng.normal(0, 800, n_weeks)).astype(int)
    df["new_users"] = df["new_users"].clip(lower=1000)

    # ---------- 加入地区列与元信息 ----------
    df["region"] = region

    # ---------- 控制变量并入 ----------
    for name, vals in (controls.items() if controls else {}):
        df[name] = vals

    # ============================================================
    # 可选：注入"脏数据"问题（用于测试数据校验）
    # ============================================================
    if inject_issues:
        issues_log = []
        # 1) 缺失值：某渠道有 3 周花费缺失
        miss_idx = rng.choice(n_weeks, 3, replace=False)
        df.loc[miss_idx, "youtube_spend"] = np.nan
        issues_log.append(f"缺失值: youtube_spend {len(miss_idx)} 个 NaN")

        # 2) 异常值：某周花费异常高（可能是数据录入错误）
        outlier_idx = int(rng.integers(0, n_weeks))
        df.loc[outlier_idx, "facebook_spend"] *= 12
        issues_log.append(f"异常值: facebook_spend 第 {outlier_idx} 周 ×12")

        # 3) 断投周：某渠道连续 2 周花费为 0
        zero_start = int(rng.integers(60, n_weeks - 3))
        df.loc[zero_start:zero_start + 1, "kol_spend"] = 0.0
        issues_log.append(f"断投: kol_spend 第 {zero_start}~{zero_start+1} 周为 0")

        df.attrs["injected_issues"] = issues_log

    # ---------- 记录真实参数到 attrs（供参数回收检验使用）----------
    # 【注意】不要把 numpy 数组放进 df.attrs！
    # pandas 在 concat/nlargest 等操作时会比较 attrs，
    # 数组的布尔判断会抛 "truth value of an array is ambiguous"。
    # 数组类的真值统一放到单独的函数返回 / 另存为列。
    df.attrs["true_beta"] = TRUE_BETA
    df.attrs["true_decay"] = TRUE_DECAY
    df.attrs["true_kappa"] = TRUE_KAPPA
    df.attrs["true_gamma_frac"] = TRUE_GAMMA_FRAC
    df.attrs["intercept_true"] = float(intercept)
    df.attrs["channel_columns"] = CHANNELS
    df.attrs["n_weeks"] = int(n_weeks)

    # 真实分解（用于参数回收检验）以列形式存入，避免 attrs 携带数组
    for ch in CHANNELS:
        df[f"_true_contrib_{ch}"] = media_contribution[ch]
    df["_true_baseline"] = baseline
    df["_true_media_total"] = total_media

    return df


def get_true_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """从生成的 DataFrame 中取出真实贡献（用于参数回收检验/画图）"""
    cols = [f"_true_contrib_{c}" for c in CHANNELS]
    out = df[["date"] + cols + ["_true_baseline", "_true_media_total"]].copy()
    out.columns = ["date"] + CHANNELS + ["baseline", "media_total"]
    return out


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    """返回一份便于阅读的数据摘要（给 Streamlit 数据页用）"""
    rows = []
    for ch in CHANNELS:
        rows.append({
            "渠道": ch,
            "总花费(万元)": df[f"{ch}_spend"].sum() / 1e4,
            "周均花费(万元)": df[f"{ch}_spend"].mean() / 1e4,
            "花费占比": df[f"{ch}_spend"].sum() / sum(
                df[f"{c}_spend"].sum() for c in CHANNELS),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 自测：生成数据并打印概览
    d = generate_game_data()
    print("=" * 66)
    print("数据生成自测")
    print("=" * 66)
    print(f"形状: {d.shape}")
    print(f"时间范围: {d['date'].min().date()} ~ {d['date'].max().date()}")
    print(f"\n列名 ({len(d.columns)} 个):")
    for i, c in enumerate(d.columns):
        print(f"  {c}", end="\n" if (i + 1) % 4 == 0 else "\t")
    print("\n\n各渠道花费概览（万元）：")
    s = get_summary(d)
    print(s.round(2).to_string(index=False))
    total_spend = sum(d[f"{c}_spend"].sum() for c in CHANNELS)
    total_rev = d["revenue"].sum()
    print(f"\n总收入: {total_rev / 1e4:,.0f} 万元")
    print(f"总花费: {total_spend / 1e4:,.0f} 万元")
    print(f"整体 ROI = 收入/花费 = {total_rev / total_spend:.2f}")
    print(f"\nrevenue 周均 {d['revenue'].mean() / 1e4:.1f} 万元  "
          f"范围 [{d['revenue'].min() / 1e4:.1f}, {d['revenue'].max() / 1e4:.1f}]")
    print(f"new_users 周均 {d['new_users'].mean():,.0f}  "
          f"范围 [{d['new_users'].min():,}, {d['new_users'].max():,}]")
    print("\n真实参数：")
    print("  beta :", TRUE_BETA)
    print("  decay:", TRUE_DECAY)
    print("\n[注入问题测试]")
    d2 = generate_game_data(inject_issues=True)
    for line in d2.attrs.get("injected_issues", []):
        print("  -", line)
    print(f"  验证: youtube_spend NaN 数 = {d2['youtube_spend'].isna().sum()}")
    print(f"  验证: kol_spend 为 0 的周数 = {(d2['kol_spend'] == 0).sum()}")
