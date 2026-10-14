# -*- coding: utf-8 -*-
"""
预算优化逻辑层（纯逻辑，不依赖 streamlit，可离线自测）
=====================================================

业务问题
--------
"五个渠道的预算，如果重新分配，能多拿多少收入？我该接受多大调整幅度？"

本模块承载整个项目**最有决策价值**的产出。核心主张：

    **交付物不是"一个最优分配数字"，而是"弹性-收益曲线"。**

    因为在带硬边界的最优化问题里，最优解几乎全部卡在边界上 ——
    决定结果的是"允许每个渠道调多少"（业务方拍板的），
    不是"模型小数点后第几位"（分析师的自我感动）。

模块结构
--------
    infer_window()            优化窗口自动推断（上传数据没有预设窗口）
    ensure_channel_scales()   ★ 挂载 channel_scales —— 不挂会得到"假优化"
    build_optimizer()         构造官方优化器
    response_tables()         响应曲线 + 边际回报（复用 src/response_curve.py）
    optimize_budget()         单次约束优化
    sweep_bounds()            弹性-收益曲线（多档位各求一次）
    scenarios()               总预算 ±X% 时的最优结构
    robustness()              把后验不确定性传导到最优分配
    recommend()               生成业务建议（含理由与风险）
    analyze()                 一次跑完全部（页面调用入口）

三个必须知道的坑（阶段 3 用 11 个探针逐层验证过，这里同源修复）
------------------------------------------------------------
[坑1] ★ channel_scales 静默错配（最隐蔽，且**不报错**）
    库内 mmm.py:3090 写 `channel_scales=getattr(self, "_channel_scales", 1.0)`。
    MMM 对象默认没有这个属性（budget_optimizer.py:1096 只在内部子对象上设 1.0），
    于是回退 1.0 → 货币与模型空间量纲错配。

    实测后果（本机复现，13 期窗口 + 单渠道 ±40% 约束 —— 与页面实际跑的口径一致）：
        挂载   → nit=50，提升 +12.8301%，分配 [157.56 518.41 517.16 119.53 78.87] 万元
        不挂载 → nit= 2，提升  +0.0000%，分配 [309.53 309.53 309.53 278.91 184.02] 万元
                 其中 facebook / google / tiktok 三者**逐位相同**
    注意 **success=True**：优化器认为自己成功了，只是没找到任何增量。
    两个信号都值得警惕：① 提升恰好为 0；② 边际回报不同的渠道拿到了完全相同的预算
    （正确目标函数下这是测度为零的事件）。

    阶段 3 在无约束口径下也复现过一次（更极端）：不挂载 → nit=1、
    分配 [278.28]×5 万元、提升 -0.00%。同一个根因，只是随窗口与约束换了个面目。

    这也是本模块 `optimize_budget()` 内置「并列检测」的原因：不能只靠人眼看。

[坑2] allocate_budget / evaluate_plan 都用【货币单位】（元），不是归一化值。
      不要手工拼 _pack_decision_vector，会绕过换算。

[坑3] budget_bounds 过宽会让 SLSQP 不收敛 → 用 return_if_fail=True 兜底，
      并把"是否收敛"如实报给用户（报告层不采用未收敛结果）。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 常量（判据集中在这里，页面与自测都引用同一份，避免"两处阈值打架"）
# ------------------------------------------------------------
MIN_ROWS_FOR_OPT = 19          # 13 决策期 + 6 carry-in 的最低要求
MIN_DECISION = 6               # 决策期下限
# 【性能设计】决策期上限刻意压在 13 期（一个季度），而不是放进 26 期。
#   实测：单次 allocate_budget 在 26 期下要 12~17 秒，13 期下约 6 秒 ——
#   因为目标函数要在「后验样本 × 决策期数」上做矩阵运算，期数直接乘进成本。
#   本页要跑十几次数值优化（弹性扫描 + 情景），窗口减半 = 总耗时减半，
#   而结论几乎不变（阶段 3 用的就是 13 期，结论已与真值对齐）。
MAX_DECISION = 13
DEFAULT_DECISION_FRACTION = 0.25
DEFAULT_CARRY_IN = 6           # 与库内 carry_in_periods 默认一致

ELASTICITY_SAFE = 0.20         # "当月可执行"档
ELASTICITY_STRETCH = 0.40      # "需要 1~2 个季度"档
# 弹性扫描档位。刻意只取 5 档（而不是 7~9 档）：
# 每档都是一次完整的 SLSQP 求解（~6 秒），档位数直接乘进用户等待时间。
# 这 5 档已覆盖"几乎不动 / 温和 / 推荐 / 激进 / 接近全放开"的决策区间。
SWEEP_LEVELS = (0.05, 0.10, 0.20, 0.40, 0.60)
SCENARIO_MULTS = (0.8, 1.0, 1.2)

CORNER_TOL = 1e-6              # 角点判定相对容差

# 【性能设计·第二轮】点估计求解只用后验的一个小子集
# ------------------------------------------------------------
# 背景：13 期窗口下单次 allocate_budget 约 6~10 秒，本页要跑十几次数值优化，
#       用户要等 1~2 分钟 —— 这是阶段 9 唯一不可接受的体验问题。
#
# 本机实测（内置数据，13 期窗口，±40% 约束）：
#     full 6000 抽样   9.73s  nit=49  提升 +12.830%  分配 [157.6 518.4 517.2 119.5 78.9]
#     subset  300 抽样 0.38s  nit=52  提升 +12.865%  分配 [157.6 518.4 517.2 119.5 78.9]
#     subset  500 抽样 0.37s  nit=47  提升 +12.839%  分配 [157.6 518.4 517.2 119.5 78.9]
#     subset 1000 抽样 0.51s  nit=49  提升 +12.843%  分配 [157.6 518.4 517.2 119.5 78.9]
#     subset 2000 抽样 0.67s  nit=47  提升 +12.837%  分配 [157.6 518.4 517.2 119.5 78.9]
#     → 逐渠道分配**完全一致**（最大偏差 0 元），目标值差异 < 0.05%，快约 25 倍。
#
# 为什么安全：allocate_budget 是**点估计**优化，用的是响应曲线的后验均值；
#             几百分样本的均值早已收敛到与全量无差别。真正需要全量后验分布形状的，
#             是 evaluate_response_distribution（可信区间）与稳健性检验 ——
#             那两条路径**不**使用这个子集。
#
# 防御：若子集求解出现 nit<=1 或未收敛（假优化信号），自动退回全量后验重解一次。
FAST_DRAWS = 500               # 子集抽样数
FAST_MIN_FULL = 1200           # 全量抽样不足这么多时没必要截取（截了也省不了多少）

FREQ_LABEL = {7: "周", 1: "天", 14: "双周", 28: "月"}


# ============================================================
# 1. 优化窗口推断
# ============================================================
def infer_window(df: pd.DataFrame, date_col: str, *,
                 fraction: float = DEFAULT_DECISION_FRACTION,
                 min_decision: int = MIN_DECISION,
                 max_decision: int = MAX_DECISION,
                 carry_in: int = DEFAULT_CARRY_IN) -> dict:
    """
    自动推断"用历史数据的哪一段模拟未来规划场景"。

    为什么需要推断而不是写死：
        阶段 3 用内置数据时写死了 2025-06-08 ~ 2025-08-31。
        上传数据没有这个日期，写死必然崩。这里按数据自身的长度与频率推断。

    规则
    ----
      决策期期数 n = clip(round(总期数 × fraction), min_decision, max_decision)
      决策期 = 数据最后 n 期；start 之前至少要留 carry_in 期做 Adstock 预热。

    返回
    ----
      {ok, start, end, n_decision, freq_days, freq_label, notes, reason}
      ok=False 时 reason 说明缺什么（页面据此给出可操作提示）。
    """
    d = df.copy()
    dt = pd.to_datetime(d[date_col], errors="coerce")
    n = int(dt.notna().sum())
    if n < MIN_ROWS_FOR_OPT:
        return {"ok": False, "reason":
                f"历史仅 {n} 期，预算优化至少需要 {MIN_ROWS_FOR_OPT} 期"
                f"（{MIN_DECISION} 期决策 + {carry_in} 期预热）",
                "n_rows": n}
    dt = dt.dropna().sort_values().reset_index(drop=True)

    gaps = dt.diff().dt.days.dropna()
    freq_days = int(gaps.median()) if len(gaps) else 7
    freq_days = max(1, freq_days)

    n_dec = int(round(n * fraction))
    n_dec = int(np.clip(n_dec, min_decision, max_decision))
    # 预留预热期
    n_dec = min(n_dec, n - carry_in)
    if n_dec < min_decision:
        return {"ok": False, "reason":
                f"历史 {n} 期不足以留出 {carry_in} 期预热后再取 {min_decision} 期决策",
                "n_rows": n}

    start = dt.iloc[-n_dec]
    end = dt.iloc[-1]
    return {
        "ok": True,
        "start": str(start.date()) if hasattr(start, "date") else str(start)[:10],
        "end": str(end.date()) if hasattr(end, "date") else str(end)[:10],
        "n_decision": int(n_dec),
        "freq_days": freq_days,
        "freq_label": FREQ_LABEL.get(freq_days, f"{freq_days} 天"),
        "n_rows": n,
        "carry_in": int(carry_in),
        "notes": [
            f"数据共 {n} 期（周期约 {freq_days} 天），取最后 {n_dec} 期作为决策窗口，"
            f"其前 {carry_in} 期作为广告残留预热。"
        ],
    }


# ============================================================
# 2. ★ channel_scales 挂载（本轮最关键的修复）
# ============================================================
def ensure_channel_scales(mmm, df: pd.DataFrame, cfg) -> dict:
    """
    确保 `mmm._channel_scales` 已挂载为【各渠道训练期最大花费】。

    换算链（已在阶段 3 闭环验证）：
        X 列 = 花费 ÷ 渠道周均
        channel_scale = max(X 列)（库内 DataDerivedScaling(method="max")）
        => channel_scale × 渠道周均 = max(训练期花费)  ✅
    因此挂 `max(花费)` 与库内换算完全一致。

    返回 {attached, scales, changed, note}；changed=True 表示本次真的挂了值。
    """
    from mmm_core import get_spend_series

    scales = np.asarray([float(get_spend_series(df, c).max()) for c in cfg.channel_cols],
                        dtype=float)
    old = getattr(mmm, "_channel_scales", None)
    changed = True
    if old is not None:
        try:
            oa = np.asarray(old, dtype=float).ravel()
            if oa.shape == scales.shape and np.allclose(oa, scales, rtol=1e-9):
                changed = False
        except Exception:                                     # noqa: BLE001
            pass
    mmm._channel_scales = scales
    note = ("已挂载 channel_scales（不挂载会得到『提升恰好为 0』的假优化）"
            if changed else "channel_scales 已就位")
    return {"attached": True, "scales": scales, "changed": changed, "note": note}


def channel_scales_health(mmm, cfg) -> dict:
    """体检：判断优化器拿到的 channel_scales 是否可信（全是同一个值 = 可疑）。"""
    s = getattr(mmm, "_channel_scales", None)
    if s is None:
        return {"ok": False, "level": "fail",
                "verdict": "channel_scales 未挂载 —— 库会回退 1.0，优化结果不可信"}
    arr = np.asarray(s, dtype=float).ravel()
    if arr.size != len(cfg.channel_cols):
        return {"ok": False, "level": "fail",
                "verdict": f"channel_scales 长度 {arr.size} 与渠道数 "
                           f"{len(cfg.channel_cols)} 不一致"}
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return {"ok": False, "level": "fail",
                "verdict": "channel_scales 含非正或非有限值"}
    uniq = np.unique(np.round(arr, 6))
    if uniq.size == 1:
        return {"ok": False, "level": "fail",
                "verdict": "所有渠道的 channel_scales 完全相同 —— 极可能是回退到了 1.0"}
    return {"ok": True, "level": "pass",
            "verdict": f"channel_scales 已挂载（{uniq.size} 个不同取值）"}


# ============================================================
# 3. 构造优化器
# ============================================================
def build_optimizer(mmm, df: pd.DataFrame, cfg, window: dict) -> tuple:
    """
    构造官方优化器并顺手做 channel_scales 体检。

    返回 (opt, meta)；meta 含 num_periods / carry_in / adstock_periods /
    response_variable / scales_health，页面直接展示给用户看"这一步做对了什么"。
    """
    if not window.get("ok"):
        raise ValueError(f"优化窗口不可用：{window.get('reason')}")

    health_before = channel_scales_health(mmm, cfg)
    sc = ensure_channel_scales(mmm, df, cfg)

    opt = mmm.budget_optimizer(window["start"], window["end"])
    meta = {
        "start": window["start"], "end": window["end"],
        "num_periods": int(opt.num_periods),
        "carry_in_periods": int(getattr(opt, "carry_in_periods", DEFAULT_CARRY_IN)),
        "adstock_periods": int(getattr(opt, "adstock_periods", 0)),
        "response_variable": str(getattr(opt, "response_variable", "?")),
        "scales": np.asarray(sc["scales"], dtype=float),
        "scales_were_missing": (not health_before["ok"]),
        "scales_note": sc["note"],
        "health_before": health_before,
        "health": channel_scales_health(mmm, cfg),
    }
    return opt, meta


def make_plan(vals, channels):
    """把货币分配向量包成 evaluate_plan 需要的带标签 DataArray。"""
    import xarray as xr
    return xr.DataArray(np.asarray(vals, dtype=float),
                        coords={"channel": list(channels)},
                        dims=("channel",))


def hist_weekly_spend(df: pd.DataFrame, cfg) -> np.ndarray:
    """各渠道的周均花费（元）。这是"当前分配"的比较基准。"""
    from mmm_core import get_spend_series
    return np.asarray([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols],
                      dtype=float)


# ============================================================
# 3.5 后验子集加速（点估计专用）
# ============================================================
def fast_posterior(idata, k: int = FAST_DRAWS, min_full: int = FAST_MIN_FULL):
    """
    从完整后验里取一个小子集，**只用于点估计求解**。

    返回 (subset_idata_or_None, info)；info 说明这次到底截了多少、为什么。
    任何一步不满足条件都返回 (None, info)，调用方据此原样用全量后验 —— 宁慢不错。

    注意切片方式：必须对**整份 idata** 切片（idata.isel(chain=[0], draw=slice(0,k))），
    不能先取 .posterior 再切 —— 库内 _extract_dataset 要求容器带 "posterior" 这个 group。
    这与稳健性检验里踩过的坑是同一个。
    """
    info = {"applied": False, "k": None, "full": None, "why": ""}
    if idata is None:
        info["why"] = "没有后验样本"
        return None, info
    try:
        post = idata.posterior
    except Exception as e:                                     # noqa: BLE001
        info["why"] = f"取 posterior 失败：{type(e).__name__}"
        return None, info

    n_chain = int(post.sizes.get("chain", 1))
    n_draw = int(post.sizes.get("draw", 1))
    full = n_chain * n_draw
    info["full"] = full
    if n_draw < int(min_full):
        info["why"] = f"全量仅 {full} 抽样，无需截取"
        return None, info
    if int(k) >= n_draw:
        info["why"] = f"目标子集 {k} 不小于单链抽样数 {n_draw}"
        return None, info

    try:
        sub = idata.isel(chain=[0], draw=slice(0, int(k)))
        # 触发一次访问，确认切片可用（有些 arviz 版本对 group 切片行为不一致）
        _ = sub.posterior
    except Exception as e:                                     # noqa: BLE001
        info["why"] = f"切片失败：{type(e).__name__}: {str(e)[:100]}"
        return None, info

    info.update({"applied": True, "k": int(k),
                 "why": f"用 {k}/{full} 抽样做点估计求解"})
    return sub, info


@contextmanager
def fast_solve(opt, idata, *, k: int = FAST_DRAWS, enabled: bool = True):
    """
    临时把优化器的后验换成小子集，把单次分配求解从 ~10 秒降到 ~0.4 秒。

    用法::

        with fast_solve(opt, idata) as fs:
            res = optimize_budget(opt, cfg, hist_total, ...)   # 走子集，快
        # 出了 with 就恢复全量后验
        dist = response_distribution(opt, cfg, res["budgets"]) # 走全量，准

    为什么必须"进出都要管"：opt 是共享对象（页面里来自 cache_resource），
    只进不出会让后续所有用到后验的地方悄悄拿到子集 —— 那就是给不确定性
    分析注了水，且完全不报错。用 try/finally 兜住，异常路径也不会漏。

    产出 fs：{applied, k, full, why, restored}
    """
    info = {"applied": False, "k": None, "full": None, "why": "已禁用", "restored": False}
    if not enabled:
        yield info
        return

    sub, pinfo = fast_posterior(idata, k)
    info.update(pinfo)
    if sub is None:
        yield info
        return

    try:
        opt.set_posterior(sub)
    except Exception as e:                                     # noqa: BLE001
        info.update({"applied": False,
                     "why": f"set_posterior 失败：{type(e).__name__}: {str(e)[:100]}"})
        yield info
        return

    try:
        yield info
    finally:
        try:
            opt.set_posterior(idata)
            info["restored"] = True
        except Exception:                                      # noqa: BLE001
            # 恢复失败就如实标记；不能假装没事（否则后续区间会拿子集算）
            info["why"] += "；⚠ 后验恢复失败"


# ============================================================
# 4. 响应曲线与边际回报
# ============================================================
def response_tables(mmm, cfg, df, idata=None, *, multiple: float = 3.0,
                    n_grid: int = 60) -> dict:
    """
    复用 src/response_curve.py 算响应曲线与边际回报。

    返回 {curves, means, marginal, saturation}：
      curves     dict[channel] -> ResponseCurveResult（含货币空间网格与区间）
      marginal   DataFrame（列名与阶段 3 产物 stage3_marginal_roi.csv 一致）
      saturation DataFrame（渠道 / 当前饱和度 / 平均 ROI / 边际回报）
    """
    import response_curve as rc

    curves = rc.compute_all_curves(mmm, cfg, df, multiple=multiple,
                                   n_grid=n_grid, idata=idata)
    means = rc.compute_channel_means(df, cfg)
    marginal = rc.compute_marginal_roi(mmm, cfg, df, idata=idata)

    rows = []
    for c in cfg.channel_cols:
        cu = curves[c]
        sat = float(np.interp(cu.current_spend, cu.spend_grid, cu.saturation_pct))
        roi = (cu.current_response / cu.current_spend) if cu.current_spend else np.nan
        mr = marginal.loc[marginal["渠道"] == c, "边际回报"]
        rows.append({
            "渠道": c,
            "周均花费(元)": cu.current_spend,
            "周贡献(元)": cu.current_response,
            "饱和度": sat,
            "平均ROI": roi,
            "边际回报": float(mr.iloc[0]) if len(mr) else np.nan,
        })
    return {"curves": curves, "means": means, "marginal": marginal,
            "saturation": pd.DataFrame(rows)}


# ============================================================
# 5. 单次约束优化
# ============================================================
def optimize_budget(opt, cfg, hist_total: np.ndarray, *,
                    low: float = 0.6, high: float = 1.4,
                    total_budget: float | None = None,
                    budget_bounds: dict | None = None,
                    unconstrained: bool = False) -> dict:
    """
    在总预算不变、单渠道有上下限的约束下最大化预期媒体贡献。

    参数
    ----
      hist_total : 窗口内"当前分配"的货币金额（长度 = 渠道数）
      low / high : 单渠道允许的调整倍数（0.6 / 1.4 = 允许 -40% ~ +40%）
      budget_bounds : 直接给 dict 则忽略 low/high（用于稳健性复现同一组约束）

    返回
    ----
      {ok, budgets, total, bounds, bounds_lo, bounds_hi, corner_hit, corner_pct,
       interior_channels, residual_channel, residual_note,
       utility, hist_utility, zero_utility, lift_pct, net_utility, nit, success,
       message}

    业务读法
    --------
      · nit：优化器迭代次数。nit=1 是"没动过"的强烈信号（假优化）。
      · corner_hit：最优解卡在上下限上的渠道数。在硬边界下几乎必然全卡角点 ——
        这不是 bug，而是在告诉业务：**决定结果的是你给我的边界有多宽**。
      · residual_channel：当恰好只有 1 个渠道没卡边界时，它就是"算术残差"，
        其变化幅度**不是模型的推荐**（详见函数内注释）。
    """
    hist_total = np.asarray(hist_total, dtype=float)
    channels = list(cfg.channel_cols)
    total_budget = float(total_budget if total_budget is not None
                         else hist_total.sum())

    if unconstrained:
        # 【无约束的正确写法】必须给 allocate_budget 传 budget_bounds=None，
        # 而不是传 [(0, inf)] —— 后者会让 SLSQP 的边界矩阵出现 inf，
        # 数值条件变差、容易不收敛。阶段 3 用的是 None，这里保持一致。
        # 但对外展示的 bounds 用 (0, +inf)：语义是"只受物理约束（花费非负）"。
        bounds = {c: (0.0, float("inf")) for c in channels}
        call_bounds = None
    elif budget_bounds is not None:
        bounds = {c: (float(v[0]), float(v[1])) for c, v in budget_bounds.items()}
        call_bounds = bounds
    else:
        bounds = {c: (float(hist_total[i]) * float(low),
                      float(hist_total[i]) * float(high))
                  for i, c in enumerate(channels)}
        call_bounds = bounds

    da = lambda v: make_plan(v, channels)                      # noqa: E731

    try:
        r = opt.allocate_budget(total_budget=total_budget, budget_bounds=call_bounds)
        success = bool(r.scipy_result.success)
    except Exception as e:                                     # noqa: BLE001
        # 【坑3】边界过宽 → SLSQP 可能报错。用 return_if_fail 拿最近的可行解，
        # 并如实标注"未收敛"，绝不让页面把未收敛结果伪装成最优解。
        try:
            r = opt.allocate_budget(total_budget=total_budget,
                                    budget_bounds=call_bounds, return_if_fail=True)
            success = False
        except Exception as e2:                                # noqa: BLE001
            return {"ok": False, "low": low, "high": high, "bounds": bounds,
                    "reason": f"优化失败：{type(e2).__name__}: {str(e2)[:180]}",
                    "raw_error": f"{type(e).__name__}: {str(e)[:120]}"}

    b = np.asarray(r.budgets, dtype=float)
    ev = opt.evaluate_plan(da(b), total_budget=total_budget)
    ev_hist = opt.evaluate_plan(da(hist_total), total_budget=total_budget)
    ev_zero = opt.evaluate_plan(da(np.zeros(len(channels))), total_budget=total_budget)

    lo_v = np.asarray([bounds[c][0] for c in channels], dtype=float)
    hi_v = np.asarray([bounds[c][1] for c in channels], dtype=float)
    # 角点判定的容差必须用**有限**上界的量级来定；无约束时 hi=+inf，
    # 若直接拿 inf 算 atol，会让所有渠道都被误判成"卡在上限"。
    _fin_hi = hi_v[np.isfinite(hi_v)]
    _span = float(_fin_hi.max()) if _fin_hi.size else float(np.max(b)) if b.size else 1.0
    _atol = 1e-6 * max(1.0, _span)
    at_lo = np.isclose(b, lo_v, rtol=CORNER_TOL, atol=_atol) & np.isfinite(lo_v)
    at_hi = np.isclose(b, hi_v, rtol=CORNER_TOL, atol=_atol) & np.isfinite(hi_v)
    corner = int((at_lo | at_hi).sum())

    nit = int(getattr(r.scipy_result, "nit", -1))
    lift = (ev.utility / ev_hist.utility - 1.0) * 100.0 if ev_hist.utility else np.nan

    # ------------------------------------------------------------
    # 「残差渠道」诊断 —— 最容易误读的一处
    # ------------------------------------------------------------
    # 带等式约束（总预算不变）+ 逐渠道上下限的问题里，若 n-1 个渠道都被推到边界，
    # 第 n 个渠道的取值就**完全由算术决定**：
    #     它 = 总预算 − 其余渠道边界值之和
    # 此时它显示出的"变化幅度"与模型对它的判断无关，纯粹是守恒式配出来的。
    # 本机实测：facebook/tiktok/youtube/kol 全卡角点，google 显示 +20.88% ——
    # 而这个 20.88% 对 12 个不同后验样本完全不变（因为它不是模型算的）。
    # 业务上必须点破，否则会被当成"模型推荐把 google 加投 20.88%"。
    at_any = at_lo | at_hi
    interior = [c for i, c in enumerate(channels) if not at_any[i]]
    residual = None
    if len(interior) == 1 and len(channels) >= 3:
        residual = interior[0]
        residual_note = (
            f"只有「{residual}」没有卡在上下限上。在总预算不变的等式约束下，"
            f"它的取值 = 总预算 − 其余 {len(channels) - 1} 个渠道的边界值之和，"
            f"是**预算守恒算出来的残差**，不是模型对它的测算结论。"
            f"换句话说：它显示的变化幅度不能当作投放建议读。")
    elif not interior:
        residual_note = ("所有渠道的最优解都卡在边界上 —— 结果是「边界宽度」"
                         "决定的，模型只负责给出挤压方向。")
    else:
        residual_note = ""

    # ------------------------------------------------------------
    # 「假优化」第二道防线：并列检测
    # ------------------------------------------------------------
    # 在正确的目标函数下，两个历史花费不同、边际回报不同的渠道拿到**逐位相同**
    # 的最优预算，是测度为零的事件 —— 真实最优会被各自的边界/斜率分开。
    # 一旦出现精确并列，几乎只能是两种情况：
    #   (a) channel_scales 错配 → 模型所有渠道都用同一把尺子，无法区分；
    #   (b) 渠道列之间存在完全共线。
    # 本机实测（13 期窗口、±40%）：
    #     挂载   → nit=50，提升 +12.8301%，分配 [157.56 518.41 517.16 119.53 78.87]，无并列
    #     未挂载 → nit= 2，提升  +0.0000%，分配 [309.53 309.53 309.53 278.91 184.02]，
    #              facebook/google/tiktok 三者逐位相同、提升恰好为 0，而 success=True
    # 注意「提升恰好为 0」这一点比「分配相等」更值得警惕：优化器跑完一圈，
    # 报告说自己成功了，但一分钱增量都没找到。
    _bd = np.asarray(b, dtype=float)
    dup_pairs = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            if abs(_bd[j] - _bd[i]) <= 1e-9 * max(1.0, abs(_bd[i])):
                dup_pairs.append((channels[i], channels[j]))
    degenerate = len(dup_pairs) > 0

    msgs = []
    if degenerate:
        _names = "、".join(sorted({c for p in dup_pairs for c in p}))
        msgs.append(
            f"⚠ 检测到『并列解』：{_names} 拿到了逐位相同的最优预算。"
            f"在边际回报不同的渠道之间，这在数学上几乎不可能自然出现 —— "
            f"最可能的原因是 channel_scales 未正确挂载（模型用同一把尺子衡量所有渠道）。"
            f"该结果不应作为决策依据。")
    if nit <= 2:
        msgs.append(f"优化器只迭代了 {nit} 次 —— 高度可疑，"
                    f"请检查 channel_scales 是否正确挂载")
    if not success:
        msgs.append("数值优化未收敛（已用兜底解），该方案不应作为决策依据")
    if corner == len(channels) and not msgs:
        msgs.append("所有渠道的最优解都卡在上下限上：这是硬边界问题的正常现象，"
                    "说明真正要谈的是『预算弹性』而不是模型精度")

    return {
        "ok": True,
        "budgets": b, "total": total_budget,
        "bounds": bounds, "bounds_lo": lo_v, "bounds_hi": hi_v,
        "at_lo": at_lo, "at_hi": at_hi,
        "corner_hit": corner, "corner_pct": corner / len(channels),
        "interior_channels": interior,
        "residual_channel": residual,
        "residual_note": residual_note,
        "degenerate": degenerate,
        "dup_pairs": dup_pairs,
        "utility": float(ev.utility),
        "hist_utility": float(ev_hist.utility),
        "zero_utility": float(ev_zero.utility),
        "constant_floor_pct": (float(ev_zero.utility / ev_hist.utility * 100)
                               if ev_hist.utility else np.nan),
        "lift_pct": float(lift) if np.isfinite(lift) else np.nan,
        "net_utility": float(ev.utility - ev_zero.utility),
        "hist_net_utility": float(ev_hist.utility - ev_zero.utility),
        "nit": nit, "success": success,
        "low": float(low), "high": float(high),
        "message": "；".join(msgs),
    }


def response_distribution(opt, cfg, budgets) -> dict:
    """
    把后验不确定性传导到"预期贡献"，得到可信区间。

    这条路径与点估计用的是**同一个目标函数**（官方 evaluate_response_distribution），
    因此区间与点估计口径一致 —— 不能用简化 Hill 口径另行重算。
    """
    channels = list(cfg.channel_cols)
    zero = make_plan(np.zeros(len(channels)), channels)
    try:
        d_plan = np.asarray(opt.evaluate_response_distribution(make_plan(budgets, channels)))
        d_zero = np.asarray(opt.evaluate_response_distribution(zero))
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}

    net = d_plan - d_zero
    q = lambda a, p: float(np.quantile(a, p))                   # noqa: E731
    return {
        "ok": True,
        "raw_mean": float(np.mean(d_plan)),
        "raw_lo": q(d_plan, 0.03), "raw_hi": q(d_plan, 0.97),
        "net_mean": float(np.mean(net)),
        "net_lo": q(net, 0.03), "net_hi": q(net, 0.97),
        "samples": int(np.asarray(d_plan).size),
    }


# ============================================================
# 6. 弹性-收益曲线（核心交付物）
# ============================================================
def sweep_bounds(opt, cfg, hist_total: np.ndarray, *,
                 levels: tuple = SWEEP_LEVELS,
                 total_budget: float | None = None,
                 include_unconstrained: bool = True,
                 reuse: dict | None = None) -> dict:
    """
    对每一档"允许调整幅度"各求解一次最优分配，得到【弹性-收益曲线】。

    这是本页最重要的图。它回答的不是"最优是多少"，而是：
        **"我们愿意接受多大调整，能拿多少增量？"**

    为什么它比单个最优数字更有决策价值：
        硬边界下最优解全卡角点 → 收益几乎完全由边界宽度决定。
        业务方真正能拍板的就是"允许渠道团队调多少"。

    参数
    ----
      reuse : {弹性档位: 已有求解结果}，用于**避免重复求解**。
              主方案的弹性档往往也在扫描列表里（如 ±40%），
              复用它的结果能省下一次 ~6 秒的求解。

    返回 {ok, table, monotone, inf_point, verdict}
      table 含列：弹性(%) / 预期贡献(元) / 提升(%) / 收敛 / 角点命中 / 说明
    """
    hist_total = np.asarray(hist_total, dtype=float)
    channels = list(cfg.channel_cols)
    total_budget = float(total_budget if total_budget is not None else hist_total.sum())
    reuse = reuse or {}

    rows = []
    for e in levels:
        key = round(float(e), 6)
        if key in reuse:
            res = reuse[key]
        else:
            res = optimize_budget(opt, cfg, hist_total, low=1.0 - float(e),
                                  high=1.0 + float(e), total_budget=total_budget)
        lo, hi = 1.0 - float(e), 1.0 + float(e)
        if not res.get("ok"):
            rows.append({"弹性(%)": e * 100, "预期贡献(元)": np.nan,
                         "提升(%)": np.nan, "收敛": False, "角点命中": 0,
                         "说明": f"±{e*100:.0f}%", "reason": res.get("reason", "")})
            continue
        rows.append({
            "弹性(%)": e * 100,
            "弹性下限": lo, "弹性上限": hi,
            "预期贡献(元)": res["utility"],
            "提升(%)": res["lift_pct"],
            "收敛": res["success"],
            "角点命中": res["corner_hit"],
            "说明": f"±{e*100:.0f}%",
        })

    inf_row = None
    if include_unconstrained:
        res = optimize_budget(opt, cfg, hist_total, total_budget=total_budget,
                              unconstrained=True)
        if res.get("ok"):
            inf_row = {
                "弹性(%)": np.inf, "弹性下限": 0.0, "弹性上限": np.inf,
                "预期贡献(元)": res["utility"], "提升(%)": res["lift_pct"],
                "收敛": res["success"], "角点命中": res["corner_hit"],
                "说明": "无约束（理论上界）",
            }
            rows.append(inf_row)

    tab = pd.DataFrame(rows)

    # 单调性：放宽界限收益不应下降（数值噪声允许 1e-6 相对容差）
    finite = tab[np.isfinite(tab["提升(%)"])].sort_values("弹性(%)")
    mono = True
    vals = finite["提升(%)"].to_numpy()
    if vals.size >= 2:
        mono = bool(np.all(np.diff(vals) >= -1e-6 * max(1.0, np.abs(vals).max())))

    def at(pct):
        r = tab[np.isclose(tab["弹性(%)"].fillna(-1), pct)]
        return float(r["提升(%)"].iloc[0]) if len(r) else np.nan

    v_safe, v_stretch = at(20.0), at(40.0)
    v_inf = float(inf_row["提升(%)"]) if inf_row else np.nan
    if np.isfinite(v_safe) and np.isfinite(v_inf):
        verdict = (
            f"收益对『允许的调整幅度』高度敏感：±20% 只能拿到 {v_safe:+.2f}%，"
            f"±40% 拿到 {v_stretch:+.2f}%，无约束理论上界 {v_inf:+.2f}%。"
            f"**先和渠道团队谈弹性，比纠结模型精度更值钱。**")
    else:
        verdict = "未能算出完整的弹性-收益曲线。"

    return {"ok": bool(len(tab)), "table": tab, "monotone": mono,
            "inf_row": inf_row, "verdict": verdict,
            "lift_at_20": v_safe, "lift_at_40": v_stretch, "lift_at_inf": v_inf}


# ============================================================
# 7. 情景分析
# ============================================================
def scenarios(opt, cfg, hist_total: np.ndarray, *,
              mults: tuple = SCENARIO_MULTS,
              low: float = 0.6, high: float = 1.4) -> dict:
    """
    总预算 ×0.8 / ×1.0 / ×1.2 时，最优结构怎么变。

    业务问题："如果明年预算砍 20%，我该怎么重新排兵布阵？"

    关键设计：情景下"当前分配"也要按同一倍数缩放，否则比较的是
    "新版预算 vs 旧版预算"，而不是"新预算下的优化 vs 新预算下的维持现状"。
    """
    hist_total = np.asarray(hist_total, dtype=float)
    channels = list(cfg.channel_cols)
    base = float(hist_total.sum())

    rows, agg = [], []
    for m in mults:
        tb = base * float(m)
        hist_scaled = hist_total * float(m)
        res = optimize_budget(opt, cfg, hist_scaled, low=low, high=high,
                              total_budget=tb)
        label = f"预算 ×{m:.1f}"
        if not res.get("ok"):
            agg.append({"情景": label, "预算倍数": float(m), "总预算(元)": tb,
                        "优化贡献(元)": np.nan, "基线贡献(元)": np.nan,
                        "提升(%)": np.nan, "收敛": False, "reason": res.get("reason")})
            continue
        for i, c in enumerate(channels):
            rows.append({
                "情景": label, "预算倍数": float(m), "渠道": c,
                "优化预算(元)": res["budgets"][i],
                "当前预算(元)": hist_scaled[i],
                "变化(%)": (res["budgets"][i] / hist_scaled[i] - 1) * 100
                          if hist_scaled[i] else np.nan,
            })
        agg.append({
            "情景": label, "预算倍数": float(m), "总预算(元)": tb,
            "优化贡献(元)": res["utility"], "基线贡献(元)": res["hist_utility"],
            "净值(元)": res["net_utility"],
            "提升(%)": res["lift_pct"], "收敛": res["success"],
            "角点命中": res["corner_hit"],
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(agg)
    spread = (summary["提升(%)"].max() - summary["提升(%)"].min()
              if len(summary) and summary["提升(%)"].notna().any() else np.nan)
    if np.isfinite(spread):
        verdict = (f"三个预算情景下的提升幅度相差 {spread:.2f} 个百分点 —— "
                   f"{'结论对总预算规模不敏感，业务上比较稳' if spread < 4 else '结论对总预算规模较敏感，需谨慎外推'}。")
    else:
        verdict = "情景分析结果不完整。"
    return {"ok": bool(len(summary)), "detail": detail, "summary": summary,
            "spread_pct": float(spread) if np.isfinite(spread) else np.nan,
            "verdict": verdict}


# ============================================================
# 8. 稳健性：把后验不确定性传导到最优分配
# ============================================================
def robustness(mmm, cfg, df: pd.DataFrame, idata, window: dict,
               hist_total: np.ndarray, *, low: float = 0.6, high: float = 1.4,
               n_draws: int = 40, seed: int = 42,
               progress=None, use_existing_opt=None) -> dict:
    """
    用多个后验样本各求解一次最优分配，看分配方案的稳定性。

    【为什么必须走官方 set_posterior 路径，而不能自己用响应曲线重算】
    阶段 3 我最初用"简化 Hill 口径逐样本重算"做稳健性，结果分布系统性
    偏离点估计 —— 因为简化口径忽略了 Adstock 时序与 carry-in，
    那是在检验**另一个目标函数**，得到的"稳健性"没有意义。
    正确做法：opt.set_posterior(单样本 idata) → 用同一个 allocate_budget。

    另一个坑：必须对**整份 idata** 切片（idata.isel(chain=[ci], draw=[di])），
    不能传 .posterior —— 库内 _extract_dataset(node, "posterior") 要求容器带
    "posterior" 这个 group，切出来的 .posterior 再取 ["posterior"] 会 KeyError。

    返回 {ok, table, n_ok, n_fail, median_cv, n_extreme, verdict,
           always_corner, stable_hi, stable_lo, unstable, corner_counts,
           bounds_used, opt_reused}

    table 每行一个渠道，除金额分位数外还给出**方向**三类统计：
      · 卡下限 / 卡上限 / 未卡边界：该渠道在 n_ok 个样本里各自撞到哪一侧
      · 方向 ∈ {结构性加投, 结构性减投, 方向不稳定, 方向偏弱, 算术残差}
    为什么必须分上下限数：一个渠道可能每轮都撞边界、但有时撞上限有时撞下限，
    它的标准差会很大却"看起来在动" —— 实际上方向是乱的。
    实测（内置数据，6 个样本）：facebook 5 次下限 / 1 次上限 → 方向不稳定；
    tiktok 6 次上限、youtube 与 kol 6 次下限 → 这三条方向可以直接进决策。
    """
    cb = progress or (lambda *a, **k: None)
    hist_total = np.asarray(hist_total, dtype=float)
    channels = list(cfg.channel_cols)

    if idata is None:
        return {"ok": False, "reason": "没有后验样本，无法做稳健性检验"}
    post = idata.posterior
    n_chain = int(post.sizes.get("chain", 1))
    n_draw = int(post.sizes.get("draw", 1))
    flat = [(ci, di) for ci in range(n_chain) for di in range(n_draw)]
    k = int(min(max(1, n_draws), len(flat)))
    idx = np.linspace(0, len(flat) - 1, k).astype(int)

    # 独立实例：避免把主优化器的后验改掉，影响页面其它区块
    opt = use_existing_opt
    own = False
    if opt is None:
        opt, _ = build_optimizer(mmm, df, cfg, window)
        own = True

    bounds = {c: (float(hist_total[i]) * low, float(hist_total[i]) * high)
              for i, c in enumerate(channels)}
    total = float(hist_total.sum())

    point = optimize_budget(opt, cfg, hist_total, low=low, high=high,
                            total_budget=total)
    point_b = point["budgets"] if point.get("ok") else None

    rows, fails, first_err = [], 0, None
    # 逐渠道统计"这个渠道在多少个样本里卡在边界上"。
    # 为什么这个指标比标准差更有业务含义：标准差小可能只是"解被钉死"，
    # 而"某渠道在 N/N 个样本中都顶在上限"是一条**结构性结论** ——
    # 它说明这个渠道的加投方向不依赖后验的具体取值，可以直接进决策。
    at_lo_cnt = np.zeros(len(channels), dtype=int)
    at_hi_cnt = np.zeros(len(channels), dtype=int)
    lo_v = np.asarray([bounds[c][0] for c in channels], dtype=float)
    hi_v = np.asarray([bounds[c][1] for c in channels], dtype=float)

    for cnt, si in enumerate(idx):
        ci, di = flat[int(si)]
        try:
            one = idata.isel(chain=[ci], draw=[di])
            opt.set_posterior(one)
            r = opt.allocate_budget(total_budget=total, budget_bounds=bounds)
            bb = np.asarray(r.budgets, dtype=float)
            rows.append(bb)
            _atol = 1e-6 * max(1.0, float(np.max(bb))) if bb.size else 1.0
            at_lo_cnt += np.isclose(bb, lo_v, rtol=CORNER_TOL, atol=_atol).astype(int)
            at_hi_cnt += np.isclose(bb, hi_v, rtol=CORNER_TOL, atol=_atol).astype(int)
        except Exception as e:                                 # noqa: BLE001
            fails += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {str(e)[:140]}"
        if cnt % 5 == 0:
            cb(0.05 + 0.9 * cnt / max(1, k),
               f"稳健性检验 {cnt + 1}/{k}（每次都是一次完整求解）")

    # 恢复：把优化器的后验还原成完整样本
    try:
        opt.set_posterior(idata)
    except Exception:                                          # noqa: BLE001
        pass

    arr = np.array(rows) if rows else np.zeros((0, len(channels)))
    if not len(arr):
        return {"ok": False, "n_ok": 0, "n_fail": fails,
                "reason": f"全部样本求解失败（首次失败：{first_err}）"}

    recs = []
    n_ok = int(arr.shape[0])
    for i, c in enumerate(channels):
        col = arr[:, i]
        pct = (float((col <= point_b[i]).mean() * 100) if point_b is not None
               else np.nan)
        n_lo, n_hi = int(at_lo_cnt[i]), int(at_hi_cnt[i])
        # 方向判定：只看「有多少个样本卡在下限 / 上限」。
        # 为什么必须把上下限分开数：facebook 在 6 个样本里 6 次都卡边界，
        # 但是 3 次卡下限、3 次卡上限 —— 它的方向随样本翻来覆去，
        # 用「卡边界次数」看不出这一点（会误以为它很稳）。
        if n_ok and n_lo == n_ok:
            direct = "结构性减投"
        elif n_ok and n_hi == n_ok:
            direct = "结构性加投"
        elif n_lo + n_hi == 0:
            direct = "算术残差"
        elif n_lo + n_hi == n_ok:
            direct = "方向不稳定"
        else:
            direct = "方向偏弱"
        recs.append({
            "渠道": c,
            "点估计(元)": float(point_b[i]) if point_b is not None else np.nan,
            "均值(元)": float(col.mean()),
            "中位数(元)": float(np.median(col)),
            "p5(元)": float(np.quantile(col, 0.05)),
            "p95(元)": float(np.quantile(col, 0.95)),
            "标准差(元)": float(col.std(ddof=0)),
            "变异系数": float(col.std(ddof=0) / col.mean()) if col.mean() else np.nan,
            "点估计分位(%)": pct,
            "卡下限": n_lo,
            "卡上限": n_hi,
            "未卡边界": int(n_ok - n_lo - n_hi),
            "方向": direct,
            "样本数": n_ok,
        })
    tab = pd.DataFrame(recs)

    stable_lo = [channels[i] for i in range(len(channels))
                 if n_ok and int(at_lo_cnt[i]) == n_ok]
    stable_hi = [channels[i] for i in range(len(channels))
                 if n_ok and int(at_hi_cnt[i]) == n_ok]
    unstable = [channels[i] for i in range(len(channels))
                if n_ok and int(at_lo_cnt[i] + at_hi_cnt[i]) == n_ok
                and int(at_lo_cnt[i]) != n_ok and int(at_hi_cnt[i]) != n_ok]
    always_corner = stable_lo + stable_hi + unstable

    pcts = tab["点估计分位(%)"].to_numpy()
    cvs = tab["变异系数"].to_numpy()
    med_cv = float(np.nanmedian(cvs)) if np.isfinite(cvs).any() else np.nan
    extreme = int(((pcts < 2) | (pcts > 98)).sum())

    if med_cv < 0.02:
        verdict = ("各渠道分配的波动极小（中位变异系数 "
                   f"{med_cv:.1%}）—— 不是因为后验很确定，而是**解全被压在上下限上**。"
                   "再次印证：决定结果的是边界宽度。")
    elif extreme >= max(1, len(channels) // 2):
        verdict = (f"有 {extreme}/{len(channels)} 个渠道的点估计解落在分布的极值分位上 —— "
                   "说明这些渠道的最优解对后验样本敏感，建议结合业务刚性再定。")
    else:
        verdict = (f"分配方案的波动温和（中位变异系数 {med_cv:.1%}），"
                   f"点估计解大多落在分布中部，方案相对稳。")

    # 把「方向」这件事单独说清楚 —— 这是本检验最有决策价值的输出
    parts = []
    if stable_hi:
        parts.append(f"{'、'.join(stable_hi)} 在 {n_ok} 个样本里**次次顶在上限**"
                     f"（结构性加投）")
    if stable_lo:
        parts.append(f"{'、'.join(stable_lo)} **次次压在下限**（结构性减投）")
    if unstable:
        parts.append(f"{'、'.join(unstable)} 每次都撞边界、但**上下限来回翻**，"
                     f"方向不稳，需结合业务刚性判断")
    if parts:
        verdict += " " + "；".join(parts) + "。"

    return {"ok": True, "table": tab, "n_ok": n_ok, "n_fail": fails,
            "first_error": first_err, "median_cv": med_cv,
            "n_extreme": extreme, "verdict": verdict,
            "always_corner": always_corner,
            "stable_hi": stable_hi, "stable_lo": stable_lo, "unstable": unstable,
            "corner_counts": {channels[i]: int(at_lo_cnt[i] + at_hi_cnt[i])
                              for i in range(len(channels))},
            "bounds_used": bounds, "opt_reused": (not own)}


# ============================================================
# 9. 业务建议生成（确定性，供模板与 LLM 共用）
# ============================================================
def recommend(cfg, alloc: pd.DataFrame, marginal: pd.DataFrame,
              sweep: dict, opt_result: dict) -> dict:
    """
    生成结构化的业务建议：每个渠道 → 加/减/维持 + 理由（含数字）。
    这里**只做搬运与判定**，不做任何推断性表述。

    容错：sweep / opt_result 里的数值可能是 None（页面在缺少某一段结果时
    会传空字典或 None）。`np.isfinite(None)` 会直接抛 TypeError，
    所以这里统一用 `_num_or_nan()` 收口 —— 缺数据只是少写一段文案，不该让页面崩。
    """
    def _num_or_nan(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return f if np.isfinite(f) else float("nan")

    ch_col = "渠道"
    mr = {r[ch_col]: r for _, r in marginal.iterrows()}
    rank = sorted(mr.values(), key=lambda r: -(r.get("边际回报") or 0))
    actions = []
    for r in rank:
        c = r[ch_col]
        m = float(r.get("边际回报") or 0)
        roi = float(r.get("平均ROI") or 0)
        if m >= 1.5:
            act, why = "加大投入", f"边际回报 {m:.2f}（每多投 1 元多拿 {m:.2f} 元），明显高于盈亏线 1.00"
        elif m >= 1.0:
            act, why = "适度加投", f"边际回报 {m:.2f}，仍高于盈亏线但溢价有限"
        elif m >= 0.8:
            act, why = "维持观察", f"边际回报 {m:.2f}，已低于盈亏线，加投不再划算"
        else:
            act, why = "考虑收缩", f"边际回报 {m:.2f}，继续加投属于净亏损"
        actions.append({"渠道": c, "建议": act, "边际回报": m, "平均ROI": roi,
                        "理由": why})

    add = [a[ch_col] for a in actions if a["建议"] in ("加大投入", "适度加投")]
    cut = [a[ch_col] for a in actions if a["建议"] in ("考虑收缩", "维持观察")]

    lift_now = _num_or_nan(opt_result.get("lift_pct"))
    el_now = _num_or_nan(opt_result.get("high"))
    if not np.isfinite(el_now):
        el_now = 1.4
    headline = (
        f"在总预算不变、单渠道允许调整 ±{(el_now - 1) * 100:.0f}% 的"
        f"前提下，把预算从 {'、'.join(cut)} 向 {'、'.join(add)} 倾斜，"
        f"模型估计可把窗口内媒体贡献提升 "
        f"**{lift_now:+.2f}%**。"
        if add and cut else
        f"模型估计的预期提升为 {lift_now:+.2f}%。")

    steps = []
    _l20 = _num_or_nan(sweep.get("lift_at_20"))
    _l40 = _num_or_nan(sweep.get("lift_at_40"))
    if np.isfinite(_l20):
        steps.append(
            f"**第一步（可当月执行）**：按 ±20% 调整，预计提升 {_l20:+.2f}%。"
            f"这个幅度对渠道团队编制与平台关系冲击很小。")
    if np.isfinite(_l40):
        steps.append(
            f"**第二步（1~2 个季度逐步到位）**：若验证了模型预测方向，"
            f"可放宽到 ±40%，预计提升 {_l40:+.2f}%。")
    steps.append(
        "**验证**：调整后按周跟踪实际收入与预测的偏差；若连续 4 周偏差超过 ±10%，"
        "暂停加投并重新拟合模型。")

    # 「残差渠道」提示单独成字段：它是对上面那句 headline 的**限定**，
    # 不能混进理由列表里被淹没（否则用户会照着残差数字去调投放）。
    residual = opt_result.get("residual_channel")
    caveat = opt_result.get("residual_note", "")

    return {"headline": headline, "actions": actions,
            "add": add, "cut": cut, "steps": steps,
            "residual_channel": residual, "caveat": caveat}


# ============================================================
# 10. 一键分析（页面入口）
# ============================================================
def analyze(mmm, cfg, df: pd.DataFrame, idata, *,
            window: dict | None = None,
            elasticity: float = ELASTICITY_STRETCH,
            total_budget: float | None = None,
            sweep_levels: tuple = SWEEP_LEVELS,
            scenario_mults: tuple = SCENARIO_MULTS,
            robustness_draws: int = 0,
            seed: int = 42,
            fast: bool = True,
            fast_draws: int = FAST_DRAWS,
            progress=None) -> dict:
    """
    跑完整条优化链路，返回页面所需的全部结果。

    参数
    ----
      elasticity : 主方案允许的单渠道调整幅度（0.4 = ±40%）
      robustness_draws : 后验样本数；0 表示跳过（每次样本都是一次完整求解，耗时）
      fast : 点估计求解是否使用后验子集加速（默认开）。
             实测分配结果与全量完全一致，耗时从 ~10s/次 降到 ~0.4s/次。
             若要严格复现"全量后验点估计"，传 False。
      progress : progress(frac, msg) 回调，用于长任务反馈

    返回
    ----
      {ok, notes, window, opt_meta, fast, curves, means, marginal, saturation,
       main, dist, sweep, scenarios, robustness, recommend, elapsed}
    """
    cb = progress or (lambda *a, **k: None)
    t0 = time.time()
    notes: list[dict] = []

    window = window or infer_window(df, cfg.date_col)
    if not window.get("ok"):
        return {"ok": False, "reason": window.get("reason"), "window": window}

    cb(0.05, "校验 channel_scales（不挂载会得到假优化）")
    opt, meta = build_optimizer(mmm, df, cfg, window)
    if meta["scales_were_missing"]:
        notes.append({
            "level": "warning", "title": "已自动修复 channel_scales 缺失",
            "detail": "优化器原本会回退到 1.0，导致量纲错配、返回『五渠道完全相等』的假最优解。"
                      "已在调用前按『训练期各渠道最大花费』挂载。",
        })
    if not meta["health"]["ok"]:
        notes.append({"level": "error", "title": "channel_scales 体检不通过",
                      "detail": meta["health"]["verdict"]})

    n_periods = meta["num_periods"]
    hist_weekly = hist_weekly_spend(df, cfg)
    hist_total = hist_weekly * n_periods
    base_total = float(hist_total.sum())
    tb = float(total_budget) if total_budget is not None else base_total
    notes.append({
        "level": "info", "title": f"决策窗口 {meta['num_periods']} 期",
        "detail": f"{meta['start']} ~ {meta['end']}，前 {meta['carry_in_periods']} 期"
                  f"作广告残留预热；窗口内当前总预算 {base_total / 1e4:,.1f} 万元。",
    })

    cb(0.12, "计算响应曲线与边际回报")
    rt = response_tables(mmm, cfg, df, idata)

    # ------------------------------------------------------------
    # 点估计求解（主方案 + 弹性扫描 + 情景）统一走"后验子集"加速
    # ------------------------------------------------------------
    # 这三段都是纯点估计，用小子集后验即可 —— 实测最优分配逐渠道一致、快约 25 倍。
    # 而 response_distribution（可信区间）与稳健性检验**不**放进 with 里：
    # 它们要的是后验的分布形状，必须用完整样本。
    use_fast = bool(fast)
    fast_info: dict = {"applied": False, "why": "已禁用"}
    main: dict = {"ok": False, "reason": "未求解"}
    need_full = False

    with fast_solve(opt, idata, enabled=use_fast, k=fast_draws) as fs:
        cb(0.30, f"求解主方案（允许调整 ±{elasticity * 100:.0f}%）")
        main = optimize_budget(opt, cfg, hist_total,
                               low=1.0 - elasticity, high=1.0 + elasticity,
                               total_budget=tb)
        # 只保留引用：fast_solve 的 finally 会在退出时补上 restored 标记，
        # 若这里 dict() 快照，就拿不到真实结果（会永远显示 restored=False）。
        fast_info = fs
        # 假优化信号：nit 崩塌到 ≤2、未收敛、或出现并列解（正确目标函数下
        # 边际回报不同的渠道不该拿到逐位相同的预算）。
        need_full = (not main.get("ok")
                     or int(main.get("nit", -1)) <= 2
                     or not main.get("success", False)
                     or bool(main.get("degenerate")))
    fast_info = dict(fast_info)

    # 【假优化安全阀】子集求解若给出 nit<=1 或未收敛，退回全量后验重解一次。
    # 这是"快"与"可信"之间的取舍判据：宁可多等 10 秒，也不把可疑结果当最优解交付。
    if need_full and fast_info.get("applied"):
        notes.append({
            "level": "warning", "title": "子集加速已自动退出",
            "detail": f"子集后验下主方案出现可疑信号（nit={main.get('nit')}、"
                      f"收敛={main.get('success')}），已改用完整后验重新求解。",
        })
        use_fast = False
        main = optimize_budget(opt, cfg, hist_total,
                               low=1.0 - elasticity, high=1.0 + elasticity,
                               total_budget=tb)
        fast_info = {"applied": False, "fell_back": True,
                     "why": "主方案可疑，已退回全量后验"}

    if not main.get("ok"):
        return {"ok": False, "reason": main.get("reason"), "window": window}

    # 主方案的弹性档往往正好在扫描列表里（默认 ±40%），复用结果可省一次求解
    reuse = {round(float(e), 6): main for e in sweep_levels
             if np.isclose(float(e), float(elasticity), rtol=0.0, atol=1e-9)}

    with fast_solve(opt, idata, enabled=use_fast, k=fast_draws):
        cb(0.45, f"求解弹性-收益曲线（{len(sweep_levels)} 档 + 无约束，"
                 f"每档一次完整求解）")
        sweep = sweep_bounds(opt, cfg, hist_total, levels=sweep_levels,
                             total_budget=tb, reuse=reuse)

        cb(0.70, "情景分析：总预算 ×0.8 / ×1.0 / ×1.2")
        scen = scenarios(opt, cfg, hist_total, mults=scenario_mults,
                         low=1.0 - elasticity, high=1.0 + elasticity)

    cb(0.82, "把后验不确定性传导到预期贡献（需完整后验）")
    dist = response_distribution(opt, cfg, main["budgets"])

    if fast_info.get("applied"):
        notes.append({
            "level": "info", "title": "点估计求解走的是后验子集（加速）",
            "detail": f"{fast_info.get('why')}。本机实测子集与全量的最优分配逐渠道"
                      f"一致（最大偏差 0 元）；可信区间与稳健性检验仍用完整后验，"
                      f"不受影响。",
        })

    rob = {"ok": False, "reason": "本次未运行（默认跳过以节省时间）"}
    if robustness_draws and robustness_draws > 0:
        cb(0.85, f"后验稳健性：抽 {robustness_draws} 个样本各求解一次")
        rob = robustness(mmm, cfg, df, idata, window, hist_total,
                         low=1.0 - elasticity, high=1.0 + elasticity,
                         n_draws=robustness_draws, seed=seed,
                         progress=lambda f, m: cb(0.85 + 0.13 * f, m),
                         use_existing_opt=None)

    alloc = pd.DataFrame({
        "渠道": list(cfg.channel_cols),
        "当前周均花费(元)": hist_weekly,
        "当前窗口预算(元)": hist_total,
        "优化窗口预算(元)": main["budgets"],
        "变化(%)": (main["budgets"] / np.where(hist_total == 0, np.nan, hist_total) - 1) * 100,
    })
    mr_map = rt["marginal"].set_index("渠道")
    alloc["边际回报"] = [float(mr_map.loc[c, "边际回报"]) for c in cfg.channel_cols]
    alloc["平均ROI"] = [float(mr_map.loc[c, "平均ROI"]) for c in cfg.channel_cols]
    alloc["卡角点"] = ["上限" if h else ("下限" if l else "—")
                    for l, h in zip(main["at_lo"], main["at_hi"])]

    rec = recommend(cfg, alloc, rt["marginal"], sweep, main)

    cb(1.0, "优化完成")
    return {
        "ok": True,
        "notes": notes,
        "window": window,
        "opt_meta": meta,
        "fast": fast_info,
        "hist_weekly": hist_weekly,
        "hist_total": hist_total,
        "base_total": base_total,
        "total_budget": tb,
        "elasticity": float(elasticity),
        "curves": rt["curves"],
        "means": rt["means"],
        "marginal": rt["marginal"],
        "saturation": rt["saturation"],
        "main": main,
        "dist": dist,
        "sweep": sweep,
        "scenarios": scen,
        "robustness": rob,
        "alloc": alloc,
        "recommend": rec,
        "elapsed": time.time() - t0,
    }


def light(result: dict) -> dict:
    """剥掉大对象（响应曲线网格），只留页面表格与图所需。"""
    if not result or not result.get("ok"):
        return result or {}
    keep = {k: v for k, v in result.items() if k != "curves"}
    sat = result.get("saturation")
    keep["saturation"] = sat
    return keep


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core import data_io, model

    print("=" * 78)
    print("app/core/optimize.py 自测（复用阶段 2 后验）")
    print("=" * 78)

    ds = data_io.load_builtin()
    art = model.load_pretrained(ds)
    mmm, idata, df, cfg = art["mmm"], art["idata"], art["df"], art["cfg"]

    w = infer_window(df, cfg.date_col)
    print(f"窗口推断 : {w}")

    # ------------------------------------------------------------
    # A/B 对照：全量后验 vs 后验子集
    # 目的：证明"加速"没有把结论也一起加速掉。这是本模块唯一需要实测背书的地方。
    # ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[A/B] 点估计求解：全量后验 vs 后验子集（同一约束 ±40%）")
    print("-" * 78)
    opt, meta = build_optimizer(mmm, df, cfg, w)
    hw = hist_weekly_spend(df, cfg)
    ht = hw * meta["num_periods"]
    tb = float(ht.sum())
    sub, sinfo = fast_posterior(idata)
    print(f"  子集抽取 : applied={sinfo['applied']} — {sinfo['why']}")

    ab = {}
    ab_t = {}
    for tag, post in (("fast", sub), ("full", idata)):
        if post is None:
            continue
        opt.set_posterior(post)
        t0 = time.perf_counter()
        r = optimize_budget(opt, cfg, ht, low=0.6, high=1.4, total_budget=tb)
        ab_t[tag] = time.perf_counter() - t0
        ab[tag] = r
        print(f"  [{tag:4s}] {ab_t[tag]:6.2f}s  nit={r['nit']:3d}  success={r['success']}  "
              f"提升={r['lift_pct']:+.3f}%  分配(万元)={np.round(r['budgets'] / 1e4, 1)}")
    opt.set_posterior(idata)

    if "fast" in ab and "full" in ab:
        dev = float(np.abs(ab["fast"]["budgets"] - ab["full"]["budgets"]).max())
        scale = float(np.max(ab["full"]["budgets"])) or 1.0
        speedup = (ab_t["full"] / ab_t["fast"]) if ab_t.get("fast") else float("nan")
        print(f"  → 分配最大偏差 {dev:,.0f} 元（占最大渠道 {dev / scale:.4%}），"
              f"提升差 {ab['fast']['lift_pct'] - ab['full']['lift_pct']:+.4f} pp，"
              f"加速 {speedup:.1f}x")
        print(f"  → 判据：{'✅ 偏差 < 1%，加速可信' if dev / scale < 0.01 else '❌ 偏差偏大，应关闭 fast'}")
        assert dev / scale < 0.01, "子集加速改变了最优分配，必须关闭 fast 默认值"
    print("-" * 78)

    def cb(f, m):
        print(f"   [{f*100:5.1f}%] {m}")

    r = analyze(mmm, cfg, df, idata, elasticity=0.4, robustness_draws=12,
                progress=cb)
    print(f"\nok={r.get('ok')}  用时 {r.get('elapsed', 0):.1f}s  加速 = {r.get('fast')}")
    for n in r.get("notes", []):
        print(f"   [{n['level']}] {n['title']} — {n['detail']}")
    m = r["main"]
    print(f"\n主方案(±{r['elasticity']*100:.0f}%)：nit={m['nit']} success={m['success']} "
          f"角点 {m['corner_hit']}/{len(cfg.channel_cols)}")
    print(f"  提升 {m['lift_pct']:+.2f}%  常数底占比 {m['constant_floor_pct']:.1f}%")
    print(r["alloc"].round(3).to_string(index=False))
    print(f"\n边际回报：")
    print(r["marginal"].round(3).to_string(index=False))
    print(f"\n弹性-收益：")
    print(r["sweep"]["table"].round(2).to_string(index=False))
    print(f"  单调递增 = {r['sweep']['monotone']}")
    print(f"  {r['sweep']['verdict']}")
    print(f"\n情景：")
    print(r["scenarios"]["summary"].round(2).to_string(index=False))
    print(f"  {r['scenarios']['verdict']}")
    if r["robustness"].get("ok"):
        print(f"\n稳健性：成功 {r['robustness']['n_ok']}，失败 {r['robustness']['n_fail']}")
        print(r["robustness"]["table"].round(3).to_string(index=False))
        print(f"  {r['robustness']['verdict']}")
    print(f"\n建议：{r['recommend']['headline']}")
    for a in r["recommend"]["actions"]:
        print(f"   {a['渠道']:10s} {a['建议']:6s} — {a['理由']}")
