# -*- coding: utf-8 -*-
"""
阶段 3 · 响应曲线工具
=========================
回答的业务问题：
  "我在某个渠道多花 10 万元，大概能多拿多少收入？"
  "哪个渠道最容易饱和？我现在的花费位置在曲线的哪一段？"

核心思路
--------
MMM 的响应函数是：
    收入贡献_c(t) = β_c · Hill_kappa,slope( Adstock_α( 花费_c(t) ) )

预算优化的全部数学基础就是"边际递减"：
    花费 → 0 时，多花 1 元的边际回报最高
    花费越大，边际回报越低
    当某渠道的边际回报低于其他渠道时，把钱挪过去就赚了

本模块提供三个能力
------------------
1. `extract_channel_params`  —— 从后验里抽出每个渠道的 (β, κ, slope, α) 后验样本
2. `compute_response_curve`  —— 给定花费网格，算出"周花费 → 周收入贡献"的曲线（含量化区间）
3. `compute_marginal_roi`    —— 算边际回报曲线（当前花费点的斜率），这是优化的直接依据

注意（重要，踩过的坑）
--------------------
- 模型内部 channel_data 是【花费/渠道均值/channel_scale】三重缩放，
  本模块统一在【货币/周】空间输出，便于业务阅读。
- β 也在缩放空间，需乘 target_scale 才能回到"元"。
  模型定义：channel_contribution_orig = 缩放空间贡献 × target_scale
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================
# 1. 从后验抽取渠道参数
# ============================================================
def extract_channel_params(mmm, cfg, idata=None) -> dict:
    """
    从后验抽每个渠道的 (beta, kappa, slope, alpha) 样本矩阵。

    返回
    ----
    dict: {
        "channels": [...],
        "beta":   ndarray (n_samples, n_channels),
        "kappa":  ...,
        "slope":  ...,
        "alpha":  ...,
        "target_scale": float,
        "channel_scale": ndarray (n_channels,),   # 模型内缩放
        "sample_coords": (chain, draw) 便于追溯,
    }

    为什么要抽样本而不是用后验均值？
      因为要做"稳健性"——最优分配不是一个点，而是一个分布。
      用样本能算出"有 90% 的信心里，tiktok 该拿多少预算"。
    """
    idata = idata if idata is not None else mmm.idata
    post = idata.posterior
    channels = list(cfg.channel_cols)

    def grab(name):
        if name not in post:
            raise KeyError(f"后验里没有 {name!r}，可用: {sorted(post.data_vars)}")
        da = post[name]
        # (chain, draw, channel) -> (n_samples, channel)
        return da.stack(sample=("chain", "draw")).transpose("sample", "channel").values

    beta = grab("saturation_beta")
    kappa = grab("saturation_kappa")
    slope = grab("saturation_slope")
    alpha = grab("adstock_alpha")

    # target_scale：从模型常量数据里取（决定缩放空间的量纲）
    target_scale = 1.0
    try:
        for holder in ("constant_data", "observed_data"):
            if holder in idata and "target_scale" in idata[holder]:
                target_scale = float(np.asarray(idata[holder]["target_scale"]).ravel()[0])
                break
    except Exception:
        pass
    if target_scale == 1.0:
        try:
            target_scale = float(np.asarray(mmm.model.named_vars["target_scale"].get_value()).ravel()[0])
        except Exception:
            pass

    # channel_scale：模型内部缩放
    channel_scale = np.ones(len(channels))
    try:
        cs = mmm.model.named_vars["channel_scale"].get_value()
        channel_scale = np.asarray(cs, dtype=float).ravel()
    except Exception:
        pass

    chain_dim = post.sizes.get("chain", 1)
    draw_dim = post.sizes.get("draw", 1)

    return {
        "channels": channels,
        "beta": beta,
        "kappa": kappa,
        "slope": slope,
        "alpha": alpha,
        "target_scale": target_scale,
        "channel_scale": channel_scale,
        "n_samples": beta.shape[0],
        "sample_coords": (chain_dim, draw_dim),
    }


# ============================================================
# 2. 渠道均值（用于把货币换算进模型空间）
# ============================================================
def compute_channel_means(df, cfg) -> dict:
    """
    从原始数据算每个渠道的【周均花费（元）】。

    为什么要它：`prepare_data` 把渠道列除以了自身均值，
    所以模型内的 1.0 = 该渠道的周均花费。
    这是"货币 ↔ 模型空间"换算的第一环。
    """
    from mmm_core import get_spend_series  # 局部导入，避免循环
    out = {}
    for c in cfg.channel_cols:
        s = get_spend_series(df, c)
        out[c] = float(np.mean(s))
    return out


# ============================================================
# 3. 响应曲线（货币空间）
# ============================================================
@dataclass
class ResponseCurveResult:
    """一条渠道的响应曲线结果。"""
    channel: str
    spend_grid: np.ndarray          # (n_grid,) 周花费（元）
    response_mean: np.ndarray       # (n_grid,) 周收入贡献均值（元）
    response_lo: np.ndarray         # (n_grid,) 94% 下界
    response_hi: np.ndarray         # (n_grid,) 94% 上界
    current_spend: float            # 当前周均花费（元）
    current_response: float         # 当前花费对应的收入贡献（元）
    saturation_pct: np.ndarray      # (n_grid,) 饱和度 0~1（= 相对渐近线的比例）
    extra: dict = field(default_factory=dict)


def _hill(x, kappa, slope):
    """Hill 饱和函数，与 PyMC-Marketing 1.x 的 HillSaturation 一致。

        Hill(x) = 1 - kappa^slope / (kappa^slope + x^slope)

    参数含义：
      kappa : 半饱和点，Hill(kappa) = 0.5
      slope : 陡峭度，slope 大 → 曲线更陡，拐点更锐
    """
    kp = np.power(np.maximum(kappa, 1e-12), slope)
    xp = np.power(np.maximum(x, 0.0), slope)
    return 1.0 - kp / (kp + xp + 1e-30)


def compute_response_curve(
    mmm,
    cfg,
    df,
    channel: str,
    *,
    spend_grid: np.ndarray | None = None,
    multiple: float = 3.0,
    n_grid: int = 60,
    ci: float = 0.94,
    idata=None,
    params: dict | None = None,
    channel_means: dict | None = None,
) -> ResponseCurveResult:
    """
    计算单渠道"周花费 → 周收入贡献"响应曲线（含可信区间）。

    参数
    ----
    channel : 渠道名
    spend_grid : 自定义花费网格（元/周）。None 时自动生成 0 ~ 当前均值的 multiple 倍
    multiple : 自动网格的上限倍数
    n_grid : 网格点数
    ci : 可信区间宽度（0.94 = 94%）

    返回
    ----
    ResponseCurveResult

    数学推导（关键，面试要能讲清）
    ------------------------------
    模型内部（缩放空间）：
        z = 花费 / 渠道均值 / channel_scale      # 归一化
        contrib_scaled = β · Hill(z; κ, slope)
    回到货币空间：
        contrib_元 = contrib_scaled × target_scale
    所以：
        contrib_元(花费) = β · Hill( 花费 / (均值×channel_scale); κ, slope ) × target_scale

    这里的 (κ, slope) 是【缩放空间】的半饱和点和陡度，
    所以网格也要先换算到缩放空间再喂进 Hill。
    """
    params = params or extract_channel_params(mmm, cfg, idata)
    channel_means = channel_means or compute_channel_means(df, cfg)

    chs = params["channels"]
    if channel not in chs:
        raise ValueError(f"未知渠道 {channel!r}，可选: {chs}")
    j = chs.index(channel)

    beta = params["beta"][:, j]
    kappa = params["kappa"][:, j]
    slope = params["slope"][:, j]
    ts = params["target_scale"]
    cs = float(params["channel_scale"][j])
    mean_spend = float(channel_means[channel])

    # 货币 → 模型缩放空间的换算系数
    #   z = 花费 / (mean_spend × channel_scale)
    denom = mean_spend * cs
    if denom <= 0:
        denom = 1.0

    # 网格
    if spend_grid is None:
        # 默认在【模型空间】取 0 ~ multiple（因为 κ 是缩放空间的半饱和点，
        # 1.0 附近就是拐点，multiple 倍能覆盖到接近饱和）
        z_grid = np.linspace(0.0, multiple, n_grid)
        spend_grid = z_grid * denom
    else:
        spend_grid = np.asarray(spend_grid, dtype=float)
        z_grid = spend_grid / denom

    # 逐样本算曲线：(n_samples, n_grid)
    z2 = z_grid[None, :]
    hill = _hill(z2, kappa[:, None], slope[:, None])
    contrib_scaled = beta[:, None] * hill            # (S, G)
    contrib_yuan = contrib_scaled * ts               # (S, G)

    resp_mean = contrib_yuan.mean(axis=0)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    resp_lo = np.quantile(contrib_yuan, lo_q, axis=0)
    resp_hi = np.quantile(contrib_yuan, hi_q, axis=0)

    # 饱和度：相对"渐近线"的比例 = Hill 值本身
    sat = hill.mean(axis=0)

    # 当前花费点
    cur_resp = float(np.interp(mean_spend, spend_grid, resp_mean)) if mean_spend <= spend_grid[-1] \
        else float(resp_mean[-1])

    return ResponseCurveResult(
        channel=channel,
        spend_grid=spend_grid,
        response_mean=resp_mean,
        response_lo=resp_lo,
        response_hi=resp_hi,
        current_spend=mean_spend,
        current_response=cur_resp,
        saturation_pct=sat,
        extra={
            "beta_mean": float(beta.mean()),
            "kappa_mean": float(kappa.mean()),
            "slope_mean": float(slope.mean()),
            "alpha_mean": float(params["alpha"][:, j].mean()),
            "target_scale": ts,
            "channel_scale": cs,
            "denom": denom,
        },
    )


# ============================================================
# 4. 边际回报（优化决策的直接依据）
# ============================================================
def compute_marginal_roi(
    mmm, cfg, df, *, spend_points: dict | None = None,
    idata=None, params=None, channel_means=None, eps_frac: float = 0.02,
) -> pd.DataFrame:
    """
    算各渠道在【当前花费点】附近的边际回报 d(收入)/d(花费)。

    业务含义
    --------
    "在 facebook 再多花 1 万元，预期多拿 X 万元收入。"
    X > 1 说明还值得加投；X < 1 说明已经过度投放。

    同时也是优化算法的依据：预算应流向边际回报最高的渠道，
    直到各渠道边际回报相等（或撞到约束边界）。

    用数值微分（中心差分），因为 Hill 的解析导数在 κ 附近数值不稳。
    """
    params = params or extract_channel_params(mmm, cfg, idata)
    channel_means = channel_means or compute_channel_means(df, cfg)
    chs = params["channels"]

    if spend_points is None:
        spend_points = {c: float(channel_means[c]) for c in chs}

    rows = []
    for c in chs:
        j = chs.index(c)
        cs = float(params["channel_scale"][j])
        denom = float(channel_means[c]) * cs
        s0 = float(spend_points[c])
        h = max(s0 * eps_frac, denom * 1e-4)   # 步长=当前花费的 2%

        def resp_at(s_money):
            z = np.asarray(s_money, dtype=float) / denom
            beta = params["beta"][:, j]
            kappa = params["kappa"][:, j]
            slope = params["slope"][:, j]
            return (beta * _hill(z, kappa, slope) * params["target_scale"])

        r_plus = resp_at(s0 + h)
        r_minus = resp_at(max(s0 - h, 0.0))
        # 注意步长实际值（s0-h 可能被截断）
        h_eff = (s0 + h) - max(s0 - h, 0.0)
        slope_samples = (r_plus - r_minus) / h_eff

        rows.append({
            "渠道": c,
            "当前周花费(元)": s0,
            "当前周贡献(元)": float(resp_at(s0).mean()),
            "边际回报": float(slope_samples.mean()),
            "边际回报_p5": float(np.quantile(slope_samples, 0.03)),
            "边际回报_p95": float(np.quantile(slope_samples, 0.97)),
            "平均ROI": float(resp_at(s0).mean() / s0) if s0 > 0 else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("边际回报", ascending=False).reset_index(drop=True)
    return out


def compute_all_curves(mmm, cfg, df, *, multiple: float = 3.0, n_grid: int = 60,
                       idata=None) -> dict[str, ResponseCurveResult]:
    """一次性算所有渠道的响应曲线（共享参数抽取，省时间）。"""
    params = extract_channel_params(mmm, cfg, idata)
    means = compute_channel_means(df, cfg)
    out = {}
    for c in cfg.channel_cols:
        out[c] = compute_response_curve(
            mmm, cfg, df, c, multiple=multiple, n_grid=n_grid,
            idata=idata, params=params, channel_means=means,
        )
    return out
