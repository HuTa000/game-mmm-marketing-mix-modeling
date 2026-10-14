"""
探索性分析算法层（纯逻辑：不依赖 streamlit、不画图、不打印）。
=============================================================

为什么要有这一层，而不是直接在页面里调 statsmodels：
    1. **可离线自测**。页面的正确性一旦要"打开浏览器点一遍"才能验证，就等于没有回归测试。
       把算法抽成纯函数后，`app/selftest_core.py` 可以用人造数据直接断言
       （已知共线强度 → 已知 VIF；已知滞后期 → 已知峰值位置），一跑就知道有没有退化。
    2. **换数据源不换算法**。内置模拟数据与用户上传的 CSV 走同一套代码，
       上传数据也能得到全部图表与结论，而不是只能看内置数据的静态截图。
    3. **可复用**。阶段 8 的建模页需要"滞后峰值 → Adstock 先验"，阶段 9 的优化页
       需要"饱和形态 → Hill 先验"，两者都从这里取结论，不重复实现。

每个函数回答一个业务问题，而不是罗列统计量：

    trend()            → 收入为什么忽高忽低？投放和收入同步吗？
    decompose()        → 多少波动是规律性的（趋势+季节），多少能被投放解释？
    collinearity()     → 5 个渠道能被分开归因吗？（MMM 的核心难点）
    lag_correlation()  → 广告投下去几周见效？→ 决定 Adstock 的 l_max 与 decay 初值
    saturation()       → 加钱还有用吗？→ 毛关系 vs 控制后净关系对照，演示内生性
    identifiability()  → 渠道效应方向可信吗？（块自助法看系数是否跨 0）
    readiness()        → 综合评分卡：这份数据到底支不支持做归因？

踩过的坑（都已在实现里规避，改代码时请勿回退）：
    ① `np.corrcoef` 遇到**常数序列**（某渠道全程不投放）会返回 nan 并抛 RuntimeWarning；
       这里统一走 `_safe_corr`，先判标准差，避免污染日志、也避免 nan 传播。
    ② `variance_inflation_factor` 在**完全共线/常数列**上会返回 inf 甚至 NaN，
       且在 statsmodels 不同版本间对缺失值与常数项的处理有差异；
       这里用 OLS 的 R² 定义自己实现 `_vif`（VIF = 1/(1-R²)），行为完全可控，
       并在自测里与 statsmodels 对照验证等价。
    ③ STL 的季节周期 `period` 必须满足 `n > 2·period`，上传数据可能只有 40 周；
       这里按数据长度自适应缩短周期并回报 `period_used`，绝不静默失败。
    ④ LOWESS 需要「x 有变异且点数足够」，稀疏投放渠道会崩；这里逐渠道 try 并跳过。
    ⑤ 时间序列直接做 iid 自助法会破坏自相关、把系数区间估窄；
       这里用**移动块自助法**（moving block bootstrap），保留局部时间结构。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 判定等级（供页面配色，不用枚举类是为了让结果能直接序列化进缓存）
PASS, WARN, FAIL = "pass", "warn", "fail"

# 共线性阈值（业界通用经验值，写死在模块里避免各页各说一套）
VIF_HEALTHY = 5.0
VIF_SEVERE = 10.0


# ============================================================
# 基础工具
# ============================================================
def _num(df: pd.DataFrame, cols) -> np.ndarray:
    """取数值矩阵，非数值 → NaN → 0（与 data_io 的清洗口径一致）。"""
    cols = list(cols or [])
    if not cols:
        return np.zeros((len(df), 0), dtype=float)
    return np.column_stack([
        pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for c in cols])


def _vec(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _dates(df: pd.DataFrame, date_col: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df[date_col], errors="coerce"))


def _safe_corr(a, b) -> float:
    """相关系数；长度不足或任一序列为常数时返回 nan（不抛 warning）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cv(x: np.ndarray) -> float:
    """变异系数 = 标准差 / 均值，衡量"有没有变动"（可识别性的前提）。"""
    x = np.asarray(x, dtype=float)
    m = float(np.nanmean(x)) if x.size else 0.0
    if m <= 1e-12:
        return float("nan")
    return float(np.nanstd(x) / m)


def _pairwise_corr(X: np.ndarray) -> np.ndarray:
    """相关矩阵，常数列对应位置置 NaN（而不是 0，避免伪装成"无相关"）。"""
    k = X.shape[1]
    C = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            C[i, j] = 1.0 if i == j else _safe_corr(X[:, i], X[:, j])
    return C


def _vif(X: np.ndarray) -> list[float]:
    """
    方差膨胀因子。VIF_j = 1 / (1 - R²_j)，
    R²_j 是「用其余列回归第 j 列」的决定系数 —— 这就是 VIF 的定义式，
    与 statsmodels.variance_inflation_factor 等价（自测有对照断言）。

    返回 inf 表示该列可被其余列完全线性表示（完全共线/常数列）。
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    out: list[float] = []
    for j in range(k):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            if ss_tot <= 1e-12:            # 该列本身是常数 → 无定义
                out.append(float("nan"))
                continue
            r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
            out.append(float("inf") if r2 >= 1 - 1e-9 else 1.0 / (1.0 - r2))
        except Exception:
            out.append(float("nan"))
    return out


def _fmt_num(v: float, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _loglog_slope(x, y) -> float:
    """
    对数-对数斜率 = 弹性。"花费涨 1%，收入涨几个 %"。

    用对数斜率而非"曲线首段/末段斜率比"来度量边际递减，原因很实在：
    LOWESS 在低花费段样本稀疏、噪声大，末段斜率可能为负，比值会直接退化成
    NaN（实测 5 个渠道有 4 个算不出）。对数斜率对尺度不敏感、对噪声更稳健，
    而且天然可解释为弹性。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    if lx.std() <= 1e-9:
        return float("nan")
    return float(np.polyfit(lx, ly, 1)[0])


# ============================================================
# 分析 1：时间趋势
# ============================================================
def trend(df: pd.DataFrame, date_col: str, target_col: str,
          channel_cols) -> dict:
    """
    业务问题：收入为什么忽高忽低？投放和收入是不是同向波动？

    把「收入」与「总花费」放同一时间轴对照。同向波动说明投放确实在起作用，
    同时也暴露内生性的来源 —— 大促周既加投放又高收入，两者互相"喂"对方，
    所以这个相关系数**不能**读成"投放的因果效果"。
    """
    channel_cols = list(channel_cols)
    dates = _dates(df, date_col)
    rev = _vec(df, target_col)
    sp = _num(df, channel_cols)
    total = sp.sum(axis=1) if sp.shape[1] else np.zeros(len(df))

    n = len(rev)
    if n < 3:
        return {"ok": False, "reason": "有效观测不足 3 周，无法分析趋势"}

    # 大促周：总花费最高的 5%（至少 1 周）
    k = max(1, int(round(0.05 * n)))
    promo_idx = np.argsort(-total)[:k]
    promo_lift = (float(rev[promo_idx].mean() / rev.mean() - 1.0)
                  if rev.mean() > 0 else float("nan"))

    return {
        "ok": True,
        "dates": dates,
        "revenue": rev,
        "total_spend": total,
        "channel_spend": sp,
        "channel_cols": channel_cols,
        "corr_rev_spend": _safe_corr(rev, total),
        "promo_idx": promo_idx,
        "promo_lift": promo_lift,
        "revenue_mean": float(rev.mean()),
        "revenue_min": float(rev.min()),
        "revenue_max": float(rev.max()),
        "total_spend_mean": float(total.mean()),
        "n": n,
    }


# ============================================================
# 分析 2：趋势与季节性分解
# ============================================================
def decompose(df: pd.DataFrame, date_col: str, target_col: str,
              period: int = 52, robust: bool = True) -> dict:
    """
    业务问题：收入波动里多少是"规律性"的（趋势 + 季节），多少是投放与噪声？

    用 STL 把收入拆成 趋势 / 季节 / 残差。残差才是"投放可能解释的部分"。
    这张图存在的意义是**给 MMM 的必要性做辩护**：
    若不建模季节，模型会把「春节收入自然高」误判成「春节广告有效」→ ROI 虚高。

    period 自适应：周数据取 52；数据不足 2 年时自动缩短（STL 要求 n > 2·period），
    并通过 `period_used` / `note` 如实告知用户"季节项是按几周周期估的"。
    """
    dates = _dates(df, date_col)
    y = _vec(df, target_col)
    n = len(y)
    if n < 12 or float(np.nanstd(y)) <= 1e-12:
        return {"ok": False, "reason": "有效观测不足或收入无波动，无法做季节分解"}

    per = int(min(max(int(period), 4), max(4, (n - 1) // 2)))
    if n <= 2 * per:
        return {"ok": False, "reason": f"观测 {n} 周不足以支撑周期 {per} 的季节分解"}

    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(np.asarray(y, dtype=float), period=per, robust=robust).fit()
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"STL 分解失败：{type(e).__name__}: {e}"}

    trend = np.asarray(res.trend, dtype=float)
    seasonal = np.asarray(res.seasonal, dtype=float)
    resid = np.asarray(res.resid, dtype=float)

    # 注意：STL 三分量之和即原序列，因此用方差比做"解释占比"不会出现交叉项
    var_tot = float(np.var(y)) or 1.0
    note = ""
    if per != int(period):
        note = (f"数据仅 {n} 周，季节周期自动缩短为 {per} 周"
                f"（原设定 {period} 周）；周期越短，季节项越容易吸收真实趋势。")

    return {
        "ok": True,
        "dates": dates,
        "observed": y,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
        "period_used": per,
        "note": note,
        "var_trend_pct": 100.0 * float(np.var(trend)) / var_tot,
        "var_season_pct": 100.0 * float(np.var(seasonal)) / var_tot,
        "var_resid_pct": 100.0 * float(np.var(resid)) / var_tot,
        "seasonal_std": float(np.std(seasonal)),
        "seasonal_amp_pct": (100.0 * float(np.std(seasonal)) / float(np.mean(y))
                             if float(np.mean(y)) > 0 else float("nan")),
        "n": n,
    }


# ============================================================
# 分析 3：渠道共线性（MMM 的核心难点）
# ============================================================
def collinearity(df: pd.DataFrame, channel_cols) -> dict:
    """
    业务问题：5 个渠道能被分开归因吗？

    如果两个渠道的花费几乎同步（大促一起加投），模型在数学上**无法区分**
    谁贡献了收入 —— 这是 MMM 最根本的难题，比"拟合优度"重要得多。

    两个视角：
      · 相关系数矩阵：两两的线性同步程度；
      · VIF：一个渠道能被其余渠道"解释"掉多少（VIF = 1/(1-R²)）。
        VIF < 5 健康 | 5~10 中度 | > 10 严重。
    """
    cols = list(channel_cols)
    X = _num(df, cols)
    if len(cols) < 2:
        return {"ok": False, "reason": "至少需要 2 个渠道才能讨论共线性"}

    corr = pd.DataFrame(_pairwise_corr(X), index=cols, columns=cols)
    vif_vals = _vif(X)
    vif = pd.DataFrame({"channel": cols, "vif": vif_vals})

    off = corr.to_numpy()[~np.eye(len(cols), dtype=bool)]
    finite = off[np.isfinite(off)]
    max_corr = float(np.abs(finite).max()) if finite.size else float("nan")
    mean_abs_corr = float(np.abs(finite).mean()) if finite.size else float("nan")

    # ★ 注意：inf 与 nan 必须分开统计。
    #   inf = 该渠道能被其余渠道**完全**线性表示（最严重，例如 youtube = fb + google）；
    #   nan = 该渠道全程不投放（无变异，VIF 无定义）。
    #   早期版本用 np.isfinite 一并过滤，结果把"最严重的完全共线"排除在 max_vif 之外，
    #   判定反而变成"健康" —— 这是典型的静默错误，务必保留这段分类逻辑。
    inf_vif = [v for v in vif_vals if v == float("inf")]
    nan_vif = [v for v in vif_vals if isinstance(v, float) and np.isnan(v)]
    finite_vif = [v for v in vif_vals if np.isfinite(v)]
    n_const = len(nan_vif)

    if inf_vif:
        max_vif = float("inf")
        level, verdict = FAIL, ("存在**完全共线**的渠道（可被其余渠道精确线性表示），"
                                "其贡献与相邻渠道无法区分，必须合并或剔除后再建模")
    elif finite_vif:
        max_vif = max(finite_vif)
        if max_vif < VIF_HEALTHY:
            level, verdict = PASS, "渠道间没有严重共线性，MMM 可以分开估计各渠道贡献"
        elif max_vif < VIF_SEVERE:
            level, verdict = WARN, "存在中度共线性，渠道排序可信度下降，建议用强先验或实验数据校准"
        else:
            level, verdict = FAIL, "存在严重共线性，渠道贡献无法可靠拆分，必须先降低共动或合并渠道"
    else:
        max_vif = float("nan")
        level, verdict = WARN, "所有渠道均无变异（VIF 无定义），无法评估共线性"

    return {
        "ok": True, "corr": corr, "vif": vif,
        "max_vif": max_vif, "max_corr": max_corr, "mean_abs_corr": mean_abs_corr,
        "n_const_channel": n_const,
        "level": level, "verdict": verdict,
        "thresholds": {"healthy": VIF_HEALTHY, "severe": VIF_SEVERE},
    }


# ============================================================
# 分析 4：滞后 / 互相关
# ============================================================
def lag_correlation(df: pd.DataFrame, date_col: str, target_col: str,
                    channel_cols, max_lag: int = 8) -> dict:
    """
    业务问题：今天投的广告，几周后才见效？

    对每个渠道计算「花费提前 k 周」与「当期收入」的相关系数，
    取曲线的峰值位置作为广告效果的**峰值延迟**。这个数直接给出：
      · Adstock 的最大滞后窗 l_max ≈ 峰值延迟 + 3~4 周
      · decay 初值：衰减越慢（曲线右拖越长），decay 应越大
    并附一条近似 95% 显著性带 ±1.96/√n —— 带内的波动不可当真。

    注意：这只是**相关**，不是因果；它给的是"先验量级"，最终由模型后验决定。
    """
    rev = _vec(df, target_col)
    n = len(rev)
    if n < 10:
        return {"ok": False, "reason": "有效观测不足 10 周，滞后相关不可靠"}

    max_lag = int(max(1, min(max_lag, n // 5)))
    lags = list(range(0, max_lag + 1))
    curves: dict[str, list[float]] = {}
    peaks: dict[str, dict] = {}

    for c in list(channel_cols):
        sp = _vec(df, c)
        cs = [_safe_corr(sp[: n - k] if k else sp, rev[k:]) for k in lags]
        curves[c] = cs
        arr = np.asarray(cs, dtype=float)
        if np.isfinite(arr).any():
            absarr = np.abs(np.where(np.isfinite(arr), arr, 0.0))
            j = int(np.nanargmax(absarr))
            peak_val = float(absarr[j])
            # 半衰滞后：从峰值起，|r| 首次跌破峰值一半的滞后周数
            half_lag = None
            for k in range(j + 1, len(lags)):
                if absarr[k] <= 0.5 * peak_val:
                    half_lag = lags[k]
                    break
            peaks[c] = {"lag": lags[j], "r": float(arr[j]),
                        "pos_lag": int(np.nanargmax(arr)) if np.isfinite(arr).all() else j,
                        "half_lag": half_lag}
        else:
            peaks[c] = {"lag": None, "r": float("nan"), "pos_lag": None,
                        "half_lag": None}

    conf = float(1.96 / np.sqrt(n))
    valid = [p["pos_lag"] for p in peaks.values() if p.get("pos_lag") is not None]
    med_peak = float(np.median(valid)) if valid else float("nan")

    # 用"相关性半衰滞后"推断广告的记忆长度：
    # |r| 从峰值衰减到一半所需的周数 = 效果释放完所需时间的一阶近似。
    # 比"峰值滞后 + 常数"更有数据依据 —— 衰减慢的渠道必须给更长的 Adstock 窗口。
    halfs = [p["half_lag"] for p in peaks.values()
             if p.get("half_lag") is not None]
    med_half = float(np.median(halfs)) if halfs else float("nan")
    if np.isfinite(med_half):
        l_max_sug = int(np.clip(round(med_half) + 4, 6, 12))
    else:
        l_max_sug = 6
    # 下限取 6 是工程约定而非数据结论：Adstock 的几何衰减尾巴需要足够长的
    # 预热窗口，窗口过短会把"上期投放的残留影响"错误地算到当期。

    return {
        "ok": True, "lags": lags, "curves": curves, "peaks": peaks,
        "conf": conf, "n": n, "max_lag": max_lag,
        "median_peak_lag": med_peak, "median_half_lag": med_half,
        "suggested_l_max": l_max_sug,
    }


# ============================================================
# 分析 5：花费-收入饱和
# ============================================================
def saturation(df: pd.DataFrame, channel_cols, target_col: str,
               control_cols=None, frac: float = 0.55, period: int = 52) -> dict:
    """
    业务问题：继续加钱还有用吗，还是已经饱和了？

    ★ 这里必须做两件事，否则会得出与事实相反的结论 ★

    ① **毛关系（raw）**：直接把「渠道周花费 → 周收入」做散点 + LOWESS 平滑。
       在真实营销数据里这条曲线常常是**上凸**的（花费越高、收入越高，还涨得更快），
       看起来像"规模效应"。但这主要是**内生性**：高花费周往往同时是大促周、
       版本上新周，收入本来就高。把它当作"广告效果"，会系统性高估高投入渠道的回报。

    ② **净关系（adjusted）**：先用 OLS 把收入中「其他渠道 + 控制变量 + 趋势 + 季节」
       的部分剔除（partial residual plot / 偏回归图），再看该渠道花费与**剩余收入**
       的关系。控制之后，曲线才会显出真实的**边际递减**。

    页面把两条曲线画在同一张图上对照 —— 这是"为什么不能只看散点"最直观的演示，
    也是本页最有说服力的一段：同一个渠道，毛关系上凸、净关系下凹。

    判据（**只用净关系**判定，避免被内生性骗）：
      对净数据拟合二次式 y = a + b·x + c·x²，并算端点斜率比
      flatness = s_high / s_low（尺度无关，比二次项系数更稳健）：
        flatness < 0.75 → 递减（已进入饱和区）；
        0.75 ~ 1.33    → 近似线性；
        > 1.33         → 递增（仍在 S 形曲线的超线性段，尚未见饱和拐点）。
      低花费端斜率为负（噪声主导）时比值无定义，退回看二次项符号。

    注意：三种形状**都可能是正确答案**。本项目内置数据实测为"递增"，
    因为其半饱和点被刻意设在平均花费的 1.6~2.8 倍（见 data_generator 的
    TRUE_GAMMA_FRAC），观测区间落在 S 形的左半段 —— 这与阶段 3
    "硬边界下最优解全卡角点"的发现完全自洽，不是 bug。
    """
    rev = _vec(df, target_col)
    channel_cols = list(channel_cols)
    control_cols = list(control_cols or [])
    n = len(rev)
    out: dict[str, dict] = {}
    if n < 8:
        return {"ok": False, "reason": "有效观测不足 8 周，无法拟合花费-收入曲线"}

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"LOWESS 不可用：{type(e).__name__}: {e}"}

    # 控制矩阵：截距 | 各渠道 | 控制变量 | 季节(sin/cos) | 线性趋势
    Xch = _num(df, channel_cols)
    Xctl = _num(df, control_cols) if control_cols else np.zeros((n, 0), dtype=float)
    t = np.arange(n, dtype=float)
    w = 2.0 * np.pi * t / float(max(2, int(period)))
    seas = np.column_stack([np.sin(w), np.cos(w), t / max(1.0, n - 1.0)])
    blocks = [np.ones((n, 1), dtype=float), Xch]
    if Xctl.shape[1]:
        blocks.append(Xctl)
    blocks.append(seas)
    A = np.column_stack(blocks)          # 列 0 = 截距；列 1..k = 渠道

    for j, c in enumerate(channel_cols):
        x = Xch[:, j]
        rec: dict = {"channel": c, "ok": False, "mean_spend": float(x.mean()),
                     "x": x, "y": rev}
        if x.std() <= 1e-12:
            rec["reason"] = "该渠道全程无投放，无花费-收入关系可估"
            out[c] = rec
            continue

        # ---- ① 毛关系曲线（含内生性，只做视觉对照，不用于判定）----
        raw_curve = None
        try:
            sm = np.asarray(lowess(rev, x, frac=float(frac), return_sorted=True),
                            dtype=float)
            if sm.ndim == 2 and sm.shape[0] >= 6:
                raw_curve = sm
        except Exception:                                     # noqa: BLE001
            raw_curve = None

        # ---- ② 净关系：partial residual = y - 用"除本渠道外的一切"做的拟合 ----
        adj_xy = None
        adj_curve = None
        c2 = s_lo = s_hi = flat = float("nan")
        shape = "未知"
        diminishing = False
        try:
            keep = np.ones(A.shape[1], dtype=bool)
            keep[1 + j] = False
            Aj = A[:, keep]
            coef_j, *_ = np.linalg.lstsq(Aj, rev, rcond=None)
            part = rev - Aj @ coef_j                 # 剩下"与该渠道有关的收入 + 噪声"
            part_c = part - part.mean()
            adj_xy = np.column_stack([x, part_c])
            sm2 = np.asarray(lowess(part_c, x, frac=float(frac), return_sorted=True),
                             dtype=float)
            if sm2.ndim == 2 and sm2.shape[0] >= 6:
                adj_curve = sm2
            # 二次拟合判凹（全局拟合比分段斜率稳健得多）
            P = np.column_stack([np.ones(n), x, x * x])
            pc, *_ = np.linalg.lstsq(P, part_c, rcond=None)
            c2 = float(pc[2])
            x_lo, x_hi = (float(v) for v in np.percentile(x, [5, 95]))
            s_lo = float(pc[1] + 2.0 * pc[2] * x_lo)
            s_hi = float(pc[1] + 2.0 * pc[2] * x_hi)
            if np.isfinite(s_lo) and s_lo > 1e-12:
                flat = s_hi / s_lo
            # 形状三分类优先看"端点斜率比"（尺度无关，比二次项系数稳健）；
            # 低花费端斜率为负（噪声主导）导致比值无定义时，退回看二次项符号。
            if np.isfinite(flat):
                shape = "递减" if flat < 0.75 else ("递增" if flat > 1.33 else "近似线性")
            elif np.isfinite(c2):
                shape = "递减" if c2 < -1e-12 else ("递增" if c2 > 1e-12 else "近似线性")
            diminishing = (shape == "递减")
        except Exception:                                     # noqa: BLE001
            pass

        if raw_curve is None and adj_curve is None:
            rec["reason"] = "有效点过少，平滑曲线不稳定"
            out[c] = rec
            continue

        med = float(np.median(x))
        lo_m = float(rev[x <= med].mean()) if (x <= med).any() else float("nan")
        hi_m = float(rev[x > med].mean()) if (x > med).any() else float("nan")
        lift = (100.0 * (hi_m / lo_m - 1.0)
                if (np.isfinite(lo_m) and lo_m > 0 and np.isfinite(hi_m)) else float("nan"))

        rec.update({
            "ok": True,
            "raw_curve": raw_curve, "adj_curve": adj_curve, "adj_xy": adj_xy,
            "quad_c2": c2, "slope_low": s_lo, "slope_high": s_hi, "flatness": flat,
            "shape": shape, "diminishing": diminishing,
            "low_half_mean": lo_m, "high_half_mean": hi_m, "lift_pct": lift,
            "near_max_share_pct": (100.0 * float(np.mean(x >= x.max() * 0.85))
                                   if x.max() > 0 else float("nan")),
        })
        out[c] = rec

    ok_rows = [r for r in out.values() if r.get("ok")]
    flats = [r["flatness"] for r in ok_rows if np.isfinite(r.get("flatness", np.nan))]
    n_ok = len(ok_rows)
    n_dim = int(sum(1 for r in ok_rows if r.get("shape") == "递减"))
    n_con = int(sum(1 for r in ok_rows if r.get("shape") == "递增"))
    n_lin = int(sum(1 for r in ok_rows if r.get("shape") == "近似线性"))
    med_flat = float(np.median(flats)) if flats else float("nan")
    maj = max(1, int(np.ceil(0.6 * n_ok))) if n_ok else 1

    if n_ok == 0:
        verdict = "无可用渠道（可能全部渠道投入过于平稳）"
    elif n_dim >= maj:
        verdict = (f"控制其他因素后，{n_dim}/{n_ok} 个渠道已呈边际递减"
                   f"（中位端点斜率比 {med_flat:.2f}）→ 继续加钱效率下降，"
                   "应使用 Hill / 幂饱和曲线，否则会高估高投入渠道的回报")
    elif n_con >= maj:
        verdict = (f"控制其他因素后，{n_con}/{n_ok} 个渠道在**观测区间内为超线性**"
                   f"（中位端点斜率比 {med_flat:.2f} → 回报仍随花费上升）。"
                   "说明当前花费落在 S 形曲线的低/中段，半饱和点还在观测范围之外。"
                   "含义：① 预算优化时最优解容易顶到约束上界；"
                   "② 线性模型会低估这段回报，仍需 Hill 这类能刻画超线性段的函数；"
                   "③ **不可**把这段递增外推到大预算（区间外才出现饱和拐点）")
    else:
        verdict = (f"各渠道形状不一致（递减 {n_dim} / 近似线性 {n_lin} / 递增 {n_con}，"
                   "共 " f"{n_ok} 个）→ 逐渠道判断，不要用同一个先验一刀切")

    return {
        "ok": True, "channels": out, "frac": float(frac), "period": int(period),
        "median_flatness": med_flat,
        "n_diminishing": n_dim, "n_convex": n_con, "n_linear": n_lin,
        "n_channels_ok": n_ok, "verdict": verdict,
    }


# ============================================================
# 分析 6：渠道体检（质量 + 可识别性前提）
# ============================================================
def channel_health(df: pd.DataFrame, channel_cols, target_col: str,
                   control_cols) -> pd.DataFrame:
    """
    逐渠道体检表。回答"哪一列数据的分布异常、或几乎没有变动"。

    关键指标是 **CV（变异系数）**：一个渠道如果每周投放金额几乎一样，
    模型就没有任何"变动"可用来估计它的效果 —— 这叫缺少可识别性，
    再怎么调模型也估不准（这正是可识别性实验的结论）。
    """
    rows = []
    for c in list(channel_cols):
        x = _vec(df, c)
        mean, sd = float(x.mean()), float(x.std(ddof=0))
        rows.append({
            "列": c,
            "类型": "渠道花费",
            "周均(万元)": mean / 1e4,
            "最大单周(万元)": float(x.max()) / 1e4,
            "CV": _cv(x),
            "零值占比": float((x == 0).mean()),
            "峰值/周均": (float(x.max() / mean) if mean > 1e-12 else float("nan")),
        })
    y = _vec(df, target_col)
    rows.append({
        "列": target_col, "类型": "目标(收入)",
        "周均(万元)": float(y.mean()) / 1e4,
        "最大单周(万元)": float(y.max()) / 1e4,
        "CV": _cv(y),
        "零值占比": float((y <= 0).mean()),
        "峰值/周均": (float(y.max() / y.mean()) if y.mean() > 1e-12 else float("nan")),
    })
    for c in list(control_cols or []):
        x = _vec(df, c)
        rows.append({
            "列": c, "类型": "控制变量",
            "周均(万元)": float(x.mean()) / 1e4,
            "最大单周(万元)": float(x.max()) / 1e4,
            "CV": _cv(x),
            "零值占比": float((x == 0).mean()),
            "峰值/周均": (float(x.max() / x.mean()) if abs(x.mean()) > 1e-12 else float("nan")),
        })
    return pd.DataFrame(rows)


# ============================================================
# 分析 7：渠道效应方向的可识别性（块自助法）
# ============================================================
def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """移动块自助法的抽样下标：整块整块地抽，保留局部时间结构。"""
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def identifiability(df: pd.DataFrame, channel_cols, target_col: str,
                    control_cols=None, n_boot: int = 240, block: int = 6,
                    seed: int = 0) -> dict:
    """
    业务问题：渠道的"效果方向"可信吗？花费更多真的对应收入更高吗？

    做法：对 (收入 ~ 渠道花费 + 控制变量) 跑 OLS，然后用**移动块自助法**
    重抽 n_boot 次，看每个渠道系数的抽样分布。

    判据（比 VIF 更贴近业务）：
      · P(系数>0) ≥ 0.975 且区间不跨 0 → 方向可信；
      · 区间跨 0 → 数据分不清该渠道到底有没有正效果（常见于高度共动）；
      · 点估计为负 → 危险信号：要么共线性作祟，要么遗漏变量导致反向因果。

    这一项与 VIF 互补：VIF 是"结构性"的静态诊断，自助法区间是"实际后果"的
    动态验证 —— 面试时能同时讲出这两个，说明真的理解 MMM 的可识别性。
    """
    cols = list(channel_cols) + list(control_cols or [])
    Y = _vec(df, target_col)
    n = len(Y)
    if len(cols) < 1 or n < 20:
        return {"ok": False, "reason": "观测过少或缺少渠道列，无法做自助法诊断"}

    X = _num(df, cols)
    # 去掉常数列（它们会让设计矩阵奇异，且本身不携带信息）
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 1e-12]
    if not keep:
        return {"ok": False, "reason": "所有渠道列均为常数，无变异可供识别"}
    Xk = X[:, keep]
    cols_k = [cols[i] for i in keep]

    A = np.column_stack([np.ones(n), Xk])
    try:
        coef0, *_ = np.linalg.lstsq(A, Y, rcond=None)
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"OLS 失败：{type(e).__name__}: {e}"}
    resid = Y - A @ coef0
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-12 else float("nan")

    rng = np.random.default_rng(int(seed))
    boot = np.full((int(n_boot), Xk.shape[1]), np.nan)
    for b in range(int(n_boot)):
        idx = _block_indices(n, block, rng)
        try:
            cb, *_ = np.linalg.lstsq(A[idx], Y[idx], rcond=None)
            boot[b] = cb[1:]
        except Exception:                                     # noqa: BLE001
            continue

    rows = []
    for i, c in enumerate(cols_k):
        col = boot[:, i]
        col = col[np.isfinite(col)]
        if col.size < 20:
            rows.append({"列": c, "系数": float(coef0[i + 1]), "均值": float("nan"),
                         "CI下": float("nan"), "CI上": float("nan"),
                         "P(系数>0)": float("nan"), "跨0": True})
            continue
        lo, hi = np.percentile(col, [2.5, 97.5])
        rows.append({
            "列": c, "系数": float(coef0[i + 1]), "均值": float(col.mean()),
            "CI下": float(lo), "CI上": float(hi),
            "P(系数>0)": float((col > 0).mean()),
            "跨0": bool(lo <= 0.0 <= hi),
        })

    table = pd.DataFrame(rows)
    ch_set = set(channel_cols)
    ch_rows = [r for r in rows if r["列"] in ch_set]

    n_cross = int(sum(1 for r in ch_rows if r["跨0"]))
    n_neg = int(sum(1 for r in ch_rows if r["系数"] < 0))
    pmin = (min((r["P(系数>0)"] if np.isfinite(r["P(系数>0)"]) else 1.0)
                for r in ch_rows) if ch_rows else float("nan"))

    if not ch_rows:
        level, verdict = WARN, "未识别到渠道列，无法判断方向"
    elif n_neg > 0 or n_cross > 0:
        level = FAIL if n_neg > 0 or n_cross >= max(1, len(ch_rows) // 2) else WARN
        verdict = ("存在系数为负或区间跨 0 的渠道："
                   f"跨 0 的渠道 {n_cross}/{len(ch_rows)} 个，负系数 {n_neg} 个，"
                   "说明数据尚不足以确认其效果方向")
    else:
        level, verdict = PASS, "所有渠道系数均为正且自助区间不跨 0，方向可信"

    return {
        "ok": True, "table": table, "channel_table": table[table["列"].isin(ch_set)],
        "ols_r2": r2, "nonconst_cols": cols_k, "n_boot": int(n_boot), "block": int(block),
        "n_cross_zero": n_cross, "n_negative": n_neg, "min_p_positive": pmin,
        "level": level, "verdict": verdict,
    }


# ============================================================
# 结论卡片：这份数据支不支持做归因？
# ============================================================
def _check(key: str, label: str, status: str, value: str,
           why: str, fix: str = "") -> dict:
    return {"key": key, "label": label, "status": status,
            "value": value, "why": why, "fix": fix}


def readiness(df: pd.DataFrame, date_col: str, target_col: str,
              channel_cols, control_cols) -> dict:
    """
    把前面所有诊断收敛成一张"能不能开工"的评分卡。
    每一项都给：现值、判据、为什么重要、不合格时怎么办 —— 而不是只给一个分数。

    评分权重：可识别性相关（共线性、渠道变异、方向稳定性）权重最高，
    因为它们是"花了大力气建模也救不回来"的问题；数据规模与质量次之。
    """
    channel_cols = list(channel_cols)
    control_cols = list(control_cols or [])
    n = len(df)
    checks: list[dict] = []

    # ---- 1) 历史长度 ----
    if n >= 104:
        st_ = PASS; val = f"{n} 周"
        why = "≥2 年才能让年度季节项与趋势被分开估计"
    elif n >= 52:
        st_ = WARN; val = f"{n} 周"
        why = "满 1 年但不足 2 年，季节项与趋势易相互混淆"
    else:
        st_ = FAIL; val = f"{n} 周"
        why = "不足 1 年，无法建模年度季节性，季节性会被误当成广告效果"
    checks.append(_check("history", "历史长度", st_, val, why,
                         "尽量补齐至少 2 年的周数据" if st_ != PASS else ""))

    # ---- 2) 渠道数量 ----
    k = len(channel_cols)
    if k >= 3:
        st_ = PASS
    elif k == 2:
        st_ = WARN
    else:
        st_ = FAIL
    checks.append(_check("n_channels", "渠道数量", st_, f"{k} 个",
                         "至少 2 个渠道才有分配问题；3 个以上才体现 MMM 的价值",
                         "补充缺失渠道列" if st_ != PASS else ""))

    # ---- 3) 共线性 ----
    col = collinearity(df, channel_cols) if k >= 2 else {"ok": False, "max_vif": float("nan")}
    if col.get("ok"):
        mv = col["max_vif"]
        if mv == float("inf"):
            st_ = FAIL; val = "完全共线 (∞)"
        elif not np.isfinite(mv):
            st_ = WARN; val = "不可计算"
        elif mv < VIF_HEALTHY:
            st_ = PASS; val = f"最大 VIF {mv:.2f}"
        elif mv < VIF_SEVERE:
            st_ = WARN; val = f"最大 VIF {mv:.2f}"
        else:
            st_ = FAIL; val = f"最大 VIF {mv:.2f}"
        checks.append(_check(
            "collinearity", "渠道共线性", st_, val,
            "渠道花费高度同步时，模型无法判断收入该归给谁 —— MMM 最根本的难题",
            "把同步投放改成错峰投放、增加渠道独立脉冲，或用 geo-lift 实验校准"
            if st_ != PASS else ""))
    else:
        checks.append(_check("collinearity", "渠道共线性", FAIL, "不可计算",
                             "渠道不足 2 个，无法评估", "补充渠道列"))

    # ---- 4) 渠道变异（可识别性的前提）----
    cvs = {c: _cv(_vec(df, c)) for c in channel_cols}
    worst = min((v for v in cvs.values() if np.isfinite(v)), default=float("nan"))
    if not np.isfinite(worst):
        st_ = FAIL; val = "无有效渠道"
    elif worst >= 0.10:
        st_ = PASS; val = f"最小 CV {worst:.2f}"
    elif worst >= 0.02:
        st_ = WARN; val = f"最小 CV {worst:.2f}"
    else:
        st_ = FAIL; val = f"最小 CV {worst:.2f}"
    low = [c for c, v in cvs.items() if np.isfinite(v) and v < 0.10]
    checks.append(_check(
        "variation", "渠道投入的变动幅度", st_, val,
        "投放每周几乎不变时，数据里没有任何信息能用来估计该渠道的效果"
        "（缺少可识别性）",
        f"这些渠道投入过于平稳：{'、'.join(low)}；建议增加错峰或阶梯式投放"
        if low else ""))

    # ---- 5) 零投放 ----
    zeros = {c: float((_vec(df, c) == 0).mean()) for c in channel_cols}
    worst_z = max(zeros.values(), default=float("nan"))
    if not np.isfinite(worst_z):
        st_ = WARN; val = "—"
    elif worst_z < 0.30:
        st_ = PASS; val = f"最大零投占比 {worst_z*100:.0f}%"
    elif worst_z < 0.60:
        st_ = WARN; val = f"最大零投占比 {worst_z*100:.0f}%"
    else:
        st_ = FAIL; val = f"最大零投占比 {worst_z*100:.0f}%"
    checks.append(_check("zeros", "投放稀疏程度", st_, val,
                         "长期断投会让 Adstock 的衰减参数失真，也让该渠道的样本量偏少",
                         "稀疏渠道可考虑合并，或放宽其先验"))

    # ---- 6) 目标序列变异 ----
    ycv = _cv(_vec(df, target_col))
    if np.isfinite(ycv) and ycv >= 0.05:
        st_ = PASS
    elif np.isfinite(ycv) and ycv >= 0.02:
        st_ = WARN
    else:
        st_ = FAIL
    checks.append(_check("target_var", "收入本身的波动", st_,
                         f"CV {_fmt_num(ycv)}" if np.isfinite(ycv) else "—",
                         "收入若几乎是一条直线，就没有可供解释的变异，回归无从下手",
                         "检查是否取错了目标列或时间粒度太大"))

    # ---- 7) 控制变量 ----
    cn = len(control_cols)
    if cn >= 3:
        st_ = PASS
    elif cn >= 1:
        st_ = WARN
    else:
        st_ = FAIL
    checks.append(_check("controls", "控制变量", st_, f"{cn} 个",
                         "版本更新/节假日/竞品会同时影响收入与投放节奏；"
                         "不控制它们，模型会把「因为大促才加投」误读成「加投带来收入」"
                         "（内生性）",
                         "补充 holiday / version_update / competitor_activity / trend"
                         if st_ != PASS else ""))

    # ---- 8) 方向稳定性（块自助法）----
    ident = identifiability(df, channel_cols, target_col, control_cols)
    if ident.get("ok") and not ident["channel_table"].empty:
        st_ = ident["level"]
        val = (f"跨0渠道 {ident['n_cross_zero']}/{len(ident['channel_table'])}"
               f" · 最小 P(正) {_fmt_num(ident['min_p_positive'], 3)}")
        checks.append(_check(
            "direction", "渠道效应方向稳定性", st_, val,
            "方向都定不了，就更谈不上比较 ROI 大小了",
            "增加渠道独立外生变动，或引入历史先验 / 实验校准" if st_ != PASS else ""))
    else:
        checks.append(_check("direction", "渠道效应方向稳定性", WARN, "不可计算",
                             ident.get("reason", "数据不足"), "补充数据后重试"))
        ident = ident if ident.get("ok") else None

    # ---- 综合 ----
    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    n_warn = sum(1 for c in checks if c["status"] == WARN)
    weights = {"history": 1.0, "n_channels": 0.6, "collinearity": 1.6,
               "variation": 1.6, "zeros": 0.8, "target_var": 0.6,
               "controls": 1.0, "direction": 1.6}
    score_map = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
    wsum = sum(weights.get(c["key"], 1.0) for c in checks)
    score = 100.0 * sum(weights.get(c["key"], 1.0) * score_map[c["status"]]
                        for c in checks) / wsum if wsum else 0.0

    if n_fail == 0 and n_warn <= 2:
        level, verdict = PASS, "支持做归因"
        summary = ("数据的结构足以支撑渠道层面的归因：渠道可分离、有明显外生变动、"
                   "效应方向稳定。可以直接进入建模。")
    elif n_fail == 0:
        level, verdict = WARN, "基本支持，但有若干先验假设需要交代"
        summary = (f"没有致命问题，但有 {n_warn} 项需要留意。"
                   "建议建模时对被点名的渠道加正则/强先验，并在报告里明示这些假设。")
    elif n_fail == 1:
        level, verdict = WARN, "结论存疑，需针对性处理"
        summary = ("有 1 项硬伤。先按下方建议处理该问题再建模，"
                   "否则对应结论大概率站不住。")
    else:
        level, verdict = FAIL, "不建议直接归因"
        summary = (f"有 {n_fail} 项硬伤。在当前数据上做渠道归因，"
                   "结果主要由模型先验而非数据决定 —— 更该做的是先补齐数据，"
                   "或改用增量实验（geo-lift / holdout）验证。")

    return {
        "ok": True, "checks": checks, "score": score,
        "level": level, "verdict": verdict, "summary": summary,
        "n_fail": n_fail, "n_warn": n_warn,
        "collinearity": col,
        "identifiability": ident,
    }


# ============================================================
# 便捷入口：一次算完（供页面层整体缓存）
# ============================================================
def analyze_all(df: pd.DataFrame, date_col: str, target_col: str,
                channel_cols, control_cols, max_lag: int = 8,
                period: int = 52) -> dict:
    """一次性跑完全部诊断。页面只需缓存这一个调用。"""
    return {
        "trend": trend(df, date_col, target_col, channel_cols),
        "decompose": decompose(df, date_col, target_col, period=period),
        "collinearity": collinearity(df, channel_cols),
        "lag": lag_correlation(df, date_col, target_col, channel_cols, max_lag=max_lag),
        "saturation": saturation(df, channel_cols, target_col,
                                 control_cols, period=period),
        "health": channel_health(df, channel_cols, target_col, control_cols),
        "identifiability": identifiability(df, channel_cols, target_col, control_cols),
        "readiness": readiness(df, date_col, target_col, channel_cols, control_cols),
        "n_rows": len(df), "channel_cols": list(channel_cols),
    }


def ground_truth() -> dict | None:
    """
    内置模拟数据的真值（仅在有 data_generator 时可用）。
    上传真实数据时返回 None —— 真实世界没有"标准答案"，这也正是
    MMM 需要靠诊断而非"对答案"来判断可信度的原因。
    """
    try:
        import data_generator as dg
        return {
            "roi": {c: float(dg.TRUE_BETA[c]) for c in dg.CHANNELS},
            "decay": {c: float(dg.TRUE_DECAY[c]) for c in dg.CHANNELS},
        }
    except Exception:                                         # noqa: BLE001
        return None
