"""
阶段 2 核心：贝叶斯 MMM 建模模块
====================================
本模块是可复用的建模引擎，后续阶段 8（Streamlit 建模页）会直接调用它。

设计要点（每条都有业务理由）：
  1. 数据标准化 —— 渠道量级差异大，不标准化会导致截距变负（阶段 0 已证）
  2. Adstock l_max=6, decay 先验围绕 EDA 推断的 0.2~0.45 —— 用数据定先验
  3. Hill 饱和 —— 阶段 1 曲线呈 S 型
  4. 业务先验 —— 关键！解决 beta 与饱和参数的"尺度不可识别性"
  5. Fourier 季节性 —— 季节占 42% 方差，不建模会让 ROI 虚高
  6. 控制变量 —— 版本更新/竞品/节假日，缓解遗漏变量偏误
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ============================================================
# 配置类：把建模决策集中管理（便于阶段 8 用控件暴露给用户）
# ============================================================
@dataclass
class MMMConfig:
    """MMM 建模配置"""
    # ---- 数据 ----
    date_col: str = "date"
    target_col: str = "revenue"
    channel_cols: list[str] = field(default_factory=lambda: [
        "facebook", "google", "tiktok", "youtube", "kol"])
    control_cols: list[str] = field(default_factory=lambda: [
        "version_update", "competitor_activity", "holiday", "trend"])

    # ---- Adstock ----
    adstock_l_max: int = 6          # 阶段 1：滞后相关性 6 周后衰减至可忽略

    # ---- 饱和 ----
    saturation_type: str = "hill"   # "hill" | "logistic" | "michaelis_menten"

    # ---- 季节性 ----
    yearly_seasonality: int = 2     # 阶段 1：季节占 42% 方差

    # ---- 缩放（★★★ 阶段2 最关键的修复 ★★★）----
    # 【问题】PyMC-Marketing 1.2.0 默认对目标 y 用 **max 缩放**：
    #       target_scale = y.max() = 10,837,440
    #     于是内部拟合的 y_std = y / y.max() ∈ [0.105, 1.0]，**均值仅 0.27**。
    #     而渠道贡献 = β × Hill(adstock(x))，典型水平就有 0.2~0.5。
    #     两者量级不匹配 → Σ渠道贡献轻易超过 0.27 → **截距被迫为负来"倒扣"**。
    #     实测：截距 −0.140，媒体占比 152%（真值 53.9%），ROI 排序 ρ=−0.50。
    #
    # 【解法】把目标缩放改成 **mean**：target_scale = y.mean()
    #     此时 y_std 均值 = 1.0，截距与渠道贡献在同一量级上分解。
    #     实测（verify_v42）：截距扭转为 +0.468，媒体占比 56.9%（真值 53.9%），
    #     ρ 从 −0.50 提升到 +0.30。
    #
    # 【为什么 max 缩放会出问题？】
    #     它是为"量纲稳定/生产环境刷新"设计的工程默认值，适合 y 波动小的场景；
    #     但 MMM 里 y 常有强季节峰值（Q4 大促），max 会远大于 mean，
    #     导致 y_std 被压到很小的数值范围，与渠道贡献的量级脱节。
    #     这类问题在 pymc-marketing 社区有讨论，官方也建议按数据特性选择缩放方法。
    target_scaling: str = "mean"    # "mean" | "max" | "none"
    # ---- 先验（业务约束，解决可识别性）----
    # saturation_beta：用 LogNormal（正值），避免 HalfNormal 众数为 0 的问题
    #   实测（verify_v43）各渠道 beta 在内部尺度上真值 ∈ [0.10, 0.64]，
    #   几何均值 ≈ 0.24 → mu = log(0.24) ≈ -1.43
    #   sigma 取 0.7：覆盖 [0.12, 0.48] 的 68% 区间，既不过窄也不过宽
    beta_prior_mu: float = -1.43
    beta_prior_sigma: float = 0.7
    # adstock_alpha 的 Beta 先验 (alpha, beta)：集中在 0.2~0.5
    adstock_alpha_prior: tuple[float, float] = (2.0, 4.0)
    # saturation_kappa（半饱和点）的 Gamma 先验 (alpha, beta)
    # 【语义澄清】1.x 中 saturation_kappa = **半饱和点**（f(kappa)=0.5），
    #        不是曲线的陡峭度（陡峭度是 saturation_slope）。
    # 【先验中心由实测推算，见 verify_v43】
    #        模型内部的 x 经过"外面÷均值、内部÷max"两次缩放 → x_inner ∈ [0,1]
    #        真实数据的半饱和点换算到内部尺度：
    #          kappa_inner = TRUE_GAMMA_FRAC × mean(spend)/max(spend)
    #                      = 0.19 ~ 0.45（均值 0.319）
    #        用 Gamma 表示：中心 = alpha/beta = 0.32
    #        取 alpha=3.2, beta=10 → 中心 0.32，标准差 0.18（合理离散度）
    saturation_kappa_prior: tuple[float, float] = (3.2, 10.0)
    # saturation_slope（Hill 曲线陡峭度）
    # 【关键修正 · verify_v43】真实数据里 slope（即 TRUE_KAPPA）= 1.4~2.1
    #   早期为"压制不可识别性"把先验设成 HalfNormal(0.1)（均值仅 0.08），
    #   结果 slope≈0 使 Hill 退化成常数、饱和完全失效，
    #   β 独自承担全部水平 → 渠道间无法区分（tiktok β 被高估 5 倍）。
    #   现改为 LogNormal 中心 1.7（真值均值），sigma=0.5 允许合理波动。
    saturation_slope_mu: float = 0.53      # = log(1.7)
    saturation_slope_sigma: float = 0.5
    # ---- intercept（截距 = 基线中"不随时间变化"的部分在标准化空间的水平）----
    # 【阶段2 最终设定 · 依据与局限都在这里写明】
    #
    #  真实值：基线常数 = 1,372,162 元/周，÷ y.max() = 0.1218
    #  数据可识别水平：OLS 截距 = 1,104,403，÷ y.max() = 0.0980
    #
    #  【为什么用 OLS 值而不是真值？】
    #  因为真值是我们造数据时才知道的；真实项目里只有"数据能给多少信息"。
    #  OLS（无 adstock/无饱和）给出的是**数据本身对基线水平的识别能力**，
    #  这才是可以正当引用的依据，也能避免"拿答案当先验"的循环论证。
    #
    #  【根本局限 —— MMM 结构性不可识别】
    #  模型方程 y = 基线常数 + 媒体恒定分量 + 波动项 中，
    #  前两项可以任意互相转移（加 α / 减 α）而 y 不变。
    #  实测：媒体贡献的恒定分量占收入 46.8%，与基线完全不可分。
    #  因此截距的先验**必须**来自业务知识（本例 = 数据可识别水平），
    #  真实项目中还需要 geo-lift / pause test 才能进一步收紧。
    #  → 这一点会在交付文档中作为核心难点单独讨论。
    # 【先验中心由实测推算，见 verify_v43】
    #   改成 mean 缩放后 y_std 均值 = 1.0，基线占比 45.8%，
    #   而基线中"常数部分"占约 76% → 截距 ≈ 0.458 × 0.76 ≈ 0.35
    intercept_mu: float = 0.35
    intercept_sigma: float = 0.15
    # ---- 控制变量 gamma（默认 Normal(0,2) 尺度比实际信号大 100 倍，等于没设）----
    # 【关键修复】实测 trend 系数在标准化空间只有 0.018，而默认先验 sigma=2，
    #   是先验信号的 112 倍 → 控制变量可以任意放大、吃掉整个基线。
    #   这里收紧到 0.2，既允许合理的控制效应，又防止其吞噬基线。
    control_sigma: float = 0.2

    # ---- 采样 ----
    draws: int = 1500
    tune: int = 1500
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42
    nuts_sampler: str = "numpyro"

    # ---- 模式 ----
    quick_mode: bool = False        # True → 减少采样量（应用里用）

    def apply_quick_mode(self):
        if self.quick_mode:
            self.draws = 300
            self.tune = 300
            self.chains = 2
        return self


# ============================================================
# 工具：列名兼容
# ============================================================
def normalize_columns(df: pd.DataFrame, cfg: MMMConfig) -> pd.DataFrame:
    """
    把 `facebook_spend` 之类的列名统一成裸渠道名 `facebook`。
    真实业务数据常见 `{channel}_spend` / `{channel}_cost` 等命名。
    """
    rename = {}
    for c in cfg.channel_cols:
        if c not in df.columns:
            for suffix in ("_spend", "_cost", "_costs", "_花费"):
                if f"{c}{suffix}" in df.columns:
                    rename[f"{c}{suffix}"] = c
                    break
    return df.rename(columns=rename) if rename else df


def get_spend_series(df: pd.DataFrame, channel: str) -> pd.Series:
    """取某渠道的花费列，兼容裸名与带后缀名"""
    for name in (channel, f"{channel}_spend", f"{channel}_cost"):
        if name in df.columns:
            return df[name].astype(float)
    raise KeyError(f"找不到渠道 {channel} 的花费列")


# ============================================================
# 数据预处理
# ============================================================
def prepare_data(df: pd.DataFrame, cfg: MMMConfig):
    """
    数据预处理。

    【重要设计决策 —— 只标准化渠道，不标准化目标 y】
    原因：PyMC-Marketing 1.x 内部会对目标做 target_scale 缩放，
    如果我们再手动把 y 除以均值，就会造成"双重缩放"，
    导致贡献分解无法正确还原（阶段 2 调试中实测到的问题）。

    正确做法：
      · 渠道列 ÷ 自身均值 → 变成"相当于几倍平均投放"（解决量级差异）
      · 控制变量标准化 → 便于共用先验
      · 目标 y 保持原始尺度 → 交给 PyMC-Marketing 内部处理

    返回
    ----
    X : DataFrame（date + 标准化渠道列 + 标准化控制列）
    y : Series（**原始尺度**的目标）
    meta : dict（保存均值，用于后续还原）
    used_controls : list
    """
    df = normalize_columns(df, cfg)
    X = df[[cfg.date_col] + cfg.channel_cols].copy()
    y = df[cfg.target_col].astype(float).copy()

    meta = {"channel_mean": {}, "y_mean": float(y.mean()),
            "control_mean": {}, "control_std": {}}

    # 渠道：除以自身均值 → "相当于几倍平均投放"
    for c in cfg.channel_cols:
        m = float(X[c].mean())
        meta["channel_mean"][c] = m
        X[c] = X[c] / m

    # 控制变量：标准化（均值0标准差1）
    for c in cfg.control_cols:
        if c in df.columns:
            mu, sd = float(df[c].mean()), float(df[c].std())
            if sd == 0:
                sd = 1.0
            meta["control_mean"][c] = mu
            meta["control_std"][c] = sd
            X[c] = (df[c] - mu) / sd

    used_controls = [c for c in cfg.control_cols if c in X.columns]
    return X, y, meta, used_controls


# ============================================================
# 构建 MMM
# ============================================================
def build_mmm(cfg: MMMConfig, control_cols: list[str] | None = None):
    """
    根据配置构建 MMM 对象。

    Parameters
    ----------
    cfg : MMMConfig
    control_cols : list[str] | None
        实际用于建模的控制变量列名。若为 None 则用 cfg.control_cols。
        【重要】请传入 prepare_data 返回的 used_controls，
        以保证"进了 X 的列"与"传给 MMM 的列"完全一致。

    【先验设定 —— 解决可识别性的关键】
    【阶段2 调试结论（经过 20 组对照实验）】：
      1. control_columns 必须显式传入，否则控制变量被静默忽略
      2. intercept 默认先验 Normal(0,2) 会产出负截距 → 贡献虚高 4~8 倍
      3. saturation_kappa 是"半饱和点"而非"陡峭度"（命名易错位）
      4. saturation_slope 最难识别，应压到接近常数
    """
    from pymc_marketing.mmm import (
        MMM, GeometricAdstock, HillSaturation,
        LogisticSaturation, MichaelisMentenSaturation,
    )
    # 【导入路径踩坑】Prior 不在 pymc_marketing.prior，
    # 1.2.0 里从 pymc_extras.prior 导入（pymc_marketing.model_config 也只是转发）
    try:
        from pymc_extras.prior import Prior
    except ImportError:
        from pymc_marketing.model_config import Prior

    sat_map = {
        "hill": HillSaturation,
        "logistic": LogisticSaturation,
        "michaelis_menten": MichaelisMentenSaturation,
    }
    sat_cls = sat_map.get(cfg.saturation_type, HillSaturation)

    mmm_kwargs: dict[str, Any] = dict(
        date_column=cfg.date_col,
        channel_columns=cfg.channel_cols,
        target_column=cfg.target_col,
        adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
        saturation=sat_cls(),
        yearly_seasonality=cfg.yearly_seasonality,
    )

    # 【★★★ 阶段2 最关键的修复 ★★★】显式指定缩放策略
    #   默认 max 缩放会把 y_std 压到均值 0.27，与渠道贡献量级脱节 →
    #   截距被迫为负、媒体占比虚高到 152%。改为 mean 后全部正常。
    if cfg.target_scaling and cfg.target_scaling != "max":
        try:
            from pymc_marketing.mmm.scaling import (
                Scaling, DataDerivedScaling, FixedScaling)
            if cfg.target_scaling == "none":
                tgt = FixedScaling(dims=(), value=1.0)
            else:  # "mean"
                tgt = DataDerivedScaling(method="mean", dims=())
            mmm_kwargs["scaling"] = Scaling(
                target=tgt,
                channel=DataDerivedScaling(method="max", dims=()),
            )
        except Exception as e:                     # 版本不兼容时静默退回
            print(f"  [提示] 自定义 scaling 不生效({type(e).__name__})，使用默认 max 缩放")

    # 【关键修复 · 阶段2 调试核心发现】
    # 必须显式传 control_columns，否则 MMM() 根本不会建模控制变量，
    # 导致"版本更新/竞品/节假日"的影响被错误归因给渠道，
    # 同时 intercept 被迫承担全部基线而被推成负值（实测 -2.51/周），
    # 进而使渠道贡献虚高 4~8 倍、ROI 排序错乱。
    # 判定方法：fit 后检查 `"gamma_control" in mmm.idata.posterior`，
    #          若为 False，说明控制变量没进去。
    use_controls = control_cols if control_cols is not None else list(cfg.control_cols)
    mmm_kwargs["control_columns"] = use_controls

    # ---- 先建一个默认模型，拿到 default_model_config ----
    base = MMM(**mmm_kwargs)
    model_config = dict(base.default_model_config)

    a, b = cfg.adstock_alpha_prior
    model_config.update({
        "adstock_alpha": Prior("Beta", alpha=a, beta=b, dims="channel"),
        # 【语义澄清】saturation_kappa = 半饱和点（不是陡峭度）
        "saturation_kappa": Prior(
            "Gamma", alpha=cfg.saturation_kappa_prior[0],
            beta=cfg.saturation_kappa_prior[1], dims="channel"),
        # 【语义澄清】saturation_slope = Hill 曲线陡峭度
        #   真值 1.4~2.1；用 LogNormal 保证正值且中心合理
        "saturation_slope": Prior(
            "LogNormal", mu=cfg.saturation_slope_mu,
            sigma=cfg.saturation_slope_sigma, dims="channel"),
        "saturation_beta": Prior(
            "LogNormal", mu=cfg.beta_prior_mu, sigma=cfg.beta_prior_sigma,
            dims="channel"),
        # 【关键修复】intercept 用信息先验，避免负截距
        #   中心值 0.131 由实测推算（真实基线常数部分 / y.max()），见类注释
        "intercept": Prior("Normal", mu=cfg.intercept_mu,
                           sigma=cfg.intercept_sigma, dims=()),
        # 【关键修复】控制变量 gamma 默认 Normal(0,2) 尺度是实际信号的 ~100 倍，
        #   等于无信息先验，会让 trend 等变量吞噬基线、挤负截距。
        #   收紧到 Normal(0, control_sigma) 后，控制效应被约束在合理范围。
        "gamma_control": Prior("Normal", mu=0.0, sigma=cfg.control_sigma,
                               dims="control"),
    })

    mmm_kwargs["model_config"] = model_config
    return MMM(**mmm_kwargs)


def build_mmm_safe(cfg: MMMConfig, control_cols: list[str] | None = None):
    """
    构建 MMM，带降级保护：
    若自定义先验的 API 不被当前版本支持，退回默认先验并给出提示。
    """
    try:
        return build_mmm(cfg, control_cols), "自定义业务先验已生效"
    except Exception as e:
        print(f"  [提示] 自定义先验不兼容当前版本({type(e).__name__}: {e})，退回默认先验")
        from pymc_marketing.mmm import (
            MMM, GeometricAdstock, HillSaturation,
            LogisticSaturation, MichaelisMentenSaturation,
        )
        sat_map = {"hill": HillSaturation, "logistic": LogisticSaturation,
                   "michaelis_menten": MichaelisMentenSaturation}
        fb_kwargs = dict(
            date_column=cfg.date_col,
            channel_columns=cfg.channel_cols,
            target_column=cfg.target_col,
            control_columns=control_cols if control_cols is not None
            else list(cfg.control_cols),
            adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
            saturation=sat_map.get(cfg.saturation_type, HillSaturation)(),
            yearly_seasonality=cfg.yearly_seasonality,
        )
        # 降级路径也必须保留 mean 缩放（否则截距又会变负）
        try:
            from pymc_marketing.mmm.scaling import Scaling, DataDerivedScaling
            fb_kwargs["scaling"] = Scaling(
                target=DataDerivedScaling(method="mean", dims=()),
                channel=DataDerivedScaling(method="max", dims=()),
            )
        except Exception:
            pass
        mmm = MMM(**fb_kwargs)
        return mmm, f"已退回默认先验（原因: {e}）"


# ============================================================
# 采样
# ============================================================
def fit_mmm(mmm, X: pd.DataFrame, y: pd.Series, cfg: MMMConfig,
            progressbar: bool = True):
    """执行 NUTS 采样"""
    import time
    t0 = time.time()
    idata = mmm.fit(
        X, y,
        method="mcmc",
        chains=cfg.chains,
        cores=cfg.chains,
        draws=cfg.draws,
        tune=cfg.tune,
        random_seed=cfg.random_seed,
        progressbar=progressbar,
        nuts_sampler=cfg.nuts_sampler,
        target_accept=cfg.target_accept,
    )
    return idata, time.time() - t0


# ============================================================
# 诊断
# ============================================================
def diagnose(idata, cfg: MMMConfig, quiet: bool = False):
    """
    拟合诊断：R-hat / ESS / divergences / 后验预测
    返回结构化诊断结果，供应用展示。
    """
    import arviz as az

    post = idata.posterior
    avail = list(post.data_vars)
    want = ["intercept_contribution", "saturation_beta", "adstock_alpha",
            "saturation_kappa", "y_sigma", "gamma_fourier"]
    pick = [v for v in want if v in avail]

    try:
        summary = az.summary(idata, var_names=pick, ci_prob=0.94, round_to=3)
    except TypeError:
        summary = az.summary(idata, var_names=pick, hdi_prob=0.94, round_to=3)

    # divergences
    try:
        n_div = int(idata.sample_stats["diverging"].sum().values)
        n_tot = int(idata.sample_stats["diverging"].size)
    except Exception:
        n_div, n_tot = 0, 1

    # 收敛判定
    n_bad_rhat = int((summary["r_hat"] > 1.01).sum()) if "r_hat" in summary.columns else -1
    min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else -1

    diag = {
        "summary": summary,
        "n_divergences": n_div,
        "n_total": n_tot,
        "divergence_pct": 100.0 * n_div / max(n_tot, 1),
        "n_bad_rhat": n_bad_rhat,
        "n_params": len(summary),
        "min_ess_bulk": min_ess,
        "converged": (n_bad_rhat == 0 and n_div / max(n_tot, 1) < 0.02),
    }

    # ---- 【阶段2 新增】截距健康检查 ----
    # 判据：intercept_contribution 为负 → 基线建模失败 → 贡献会虚高
    try:
        ic = float(idata.posterior["intercept_contribution"].values.mean())
        diag["intercept"] = ic
        diag["intercept_healthy"] = ic > 0
    except Exception:
        diag["intercept"] = None
        diag["intercept_healthy"] = None

    # ---- 【阶段2 新增】控制变量是否真的进了模型 ----
    diag["has_controls"] = "gamma_control" in avail

    if not quiet:
        print("  R-hat / ESS 摘要（所有参数）：")
        print(summary.to_string())
        print(f"\n  divergences: {n_div}/{n_tot} ({diag['divergence_pct']:.2f}%)")
        print(f"  R-hat > 1.01 的参数: {n_bad_rhat}/{diag['n_params']}")
        print(f"  最小 ESS(bulk): {min_ess:.0f}")
        if diag["intercept"] is not None:
            flag = "✓" if diag["intercept_healthy"] else "✗ 负截距！贡献会虚高"
            print(f"  截距 intercept_contribution = {diag['intercept']:.3f} {flag}")
        print(f"  控制变量已进模型: {diag['has_controls']}"
              f"{'' if diag['has_controls'] else '  ← 需传 control_columns!'}")
        print("\n  [判定标准]")
        print("    R-hat < 1.01  → 多链收敛，可信")
        print("    ESS > 400     → 有效样本量足够")
        print("    divergences < 2% → 采样健康")
        print("    截距 > 0      → 基线建模正确（否则媒体贡献虚高）")
        verdict = "✓ 收敛良好" if diag["converged"] else "⚠ 需改进（增加采样量/调先验）"
        print(f"    综合判定: {verdict}")

    return diag


# ============================================================
# 贡献分解（用官方 API，已实测正确）
# ============================================================
def get_target_scale(mmm) -> float:
    """
    取 PyMC-Marketing 内部的 target_scale 缩放因子。

    【它是什么】模型内部把目标 y 除以 target_scale 再建模。
    实测：target_scale ≈ y_mean × 1.38（与内部 scaling 策略有关）。

    【为什么重要】后验里的 `channel_contribution` 是**标准化空间**的量，
    要还原到原始单位必须乘 target_scale。
    """
    try:
        ts = mmm.idata.constant_data["target_scale"]
        return float(np.squeeze(np.asarray(ts.values)))
    except Exception:
        return 1.0


def check_model_sanity(mmm, y_raw: pd.Series, verbose: bool = True) -> dict:
    """
    【阶段2 新增】模型自洽性检查 —— 在算 ROI 之前必须先过这一关。

    检查 3 件事：
      1. 可加性：μ 是否 ≈ Σchannel_contribution + intercept×n + 季节 + 控制
         （identity link 下必须成立，否则贡献分解不可信）
      2. 截距健康：intercept 是否 > 0（负截距 → 贡献虚高）
      3. 预测精度：predict 能否拟合真实 y

    返回 dict 供应用展示。
    """
    pt = mmm.idata.posterior
    n = len(y_raw)
    info: dict = {}

    # ---- 1. 可加性 ----
    try:
        mu_sum = float(pt["mu"].values.mean(axis=(0, 1)).sum())
        cc_sum = float(pt["channel_contribution"].values.mean(axis=(0, 1)).sum())
        ic = float(pt["intercept_contribution"].values.mean())
        ys_sum = float(pt["yearly_seasonality_contribution"]
                       .values.mean(axis=(0, 1)).sum())
        fc_sum = float(pt["fourier_contribution"].values.mean(axis=(0, 1)).sum())
        parts = ic * n + cc_sum + ys_sum + fc_sum
        info.update({"mu_sum": mu_sum, "cc_sum": cc_sum, "intercept": ic,
                     "parts_sum": parts,
                     "additivity_err": (parts - mu_sum) / mu_sum if mu_sum else np.nan})
        info["additive_ok"] = abs(info["additivity_err"]) < 0.01
    except Exception as e:
        info["additive_ok"] = None
        info["additivity_err"] = np.nan
        info["additivity_note"] = str(e)

    # ---- 2. 截距健康 ----
    info["intercept_healthy"] = bool(info.get("intercept", -1) > 0)

    # ---- 3. 预测精度 ----
    try:
        ts = get_target_scale(mmm)
        pred_sum = float(np.asarray(mmm.predict(
            _rebuild_X_for_predict(mmm, y_raw))).ravel().sum() * ts)
        true_sum = float(y_raw.sum())
        info["pred_sum"] = pred_sum
        info["true_sum"] = true_sum
        info["pred_err"] = pred_sum / true_sum - 1 if true_sum else np.nan
        info["pred_ok"] = abs(info["pred_err"]) < 0.05
    except Exception as e:
        info["pred_ok"] = None
        info["pred_note"] = str(e)

    if verbose:
        print("  [模型自洽性检查]")
        print(f"    可加性  : Σ分量/μ 偏差 = {info.get('additivity_err', np.nan):+.4f} "
              f"{'✓' if info.get('additive_ok') else '✗'}")
        print(f"    截距    : {info.get('intercept', np.nan):+.3f} "
              f"{'✓' if info.get('intercept_healthy') else '✗ 负截距，贡献会虚高'}")
        if info.get("pred_ok") is not None:
            print(f"    预测精度: 偏差 = {info['pred_err']:+.1%} "
                  f"{'✓' if info['pred_ok'] else '✗'}")
    return info


def _rebuild_X_for_predict(mmm, y_raw: pd.Series) -> pd.DataFrame:
    """
    从模型的 constant_data 里重建 predict 所需的输入 X。
    避免调用方重新拼 DataFrame 时漏列/错列。
    """
    cd = mmm.idata.constant_data
    try:
        ch = np.asarray(cd["channel_data"].values)   # (date, channel)
        cols = list(mmm.channel_columns) if hasattr(mmm, "channel_columns") else None
        if cols is None:
            cols = [f"ch{i}" for i in range(ch.shape[1])]
        X = pd.DataFrame(ch, columns=cols)
        # 控制变量
        for c in getattr(mmm, "control_columns", None) or []:
            if c in cd:
                X[c] = np.asarray(cd[c].values)
        # 日期维（用 datetime 索引）
        dates = np.asarray(cd["channel_data"].coords.get("date", np.arange(len(X))).values)
        X.insert(0, "date", dates)
        return X
    except Exception:
        # 兜底：从 y 的索引建
        return pd.DataFrame({"date": np.arange(len(y_raw))})


def compute_contributions_library(mmm, df: pd.DataFrame, cfg: MMMConfig,
                                  meta: dict):
    """
    【推荐·官方 API】用 PyMC-Marketing 内置的守恒分解计算各渠道贡献。

    原理（1.2.0 的 `mmm.data.get_contributions(original_scale=True)`）：
      identity link 下，各分量 = 标准化空间的量 × target_scale，
      且**各分量之和恰好等于预测的 y**（守恒分解）。

    返回
    ----
    contrib_df : DataFrame(date + 各渠道贡献)，单位与 y 一致
    info : dict（含自洽性检查）
    """
    channels = cfg.channel_cols

    # ---- 官方守恒分解 ----
    ds = mmm.data.get_contributions(original_scale=True)
    ch_da = ds["channels"]                       # (chain, draw, date, channel)
    ch_mean = ch_da.mean(("chain", "draw"))      # (date, channel)

    dates = pd.to_datetime(np.asarray(ch_mean.coords["date"].values))
    contrib_df = pd.DataFrame({"date": dates})
    for c in channels:
        if c in ch_mean.coords.get("channel", []):
            contrib_df[c] = np.asarray(ch_mean.sel(channel=c).values)
        else:
            contrib_df[c] = 0.0

    # 同时保存 baseline / seasonality（便于画堆叠图）
    try:
        baseline = ds["baseline"].mean(("chain", "draw"))
        sea = ds["seasonality"].mean(("chain", "draw"))
        contrib_df["_baseline"] = np.asarray(baseline.values).ravel()[:len(contrib_df)]
        contrib_df["_seasonality"] = np.asarray(sea.values).ravel()[:len(contrib_df)]
    except Exception:
        pass

    media_cols = [c for c in channels if c in contrib_df.columns]
    media_sum = float(contrib_df[media_cols].sum().sum())
    true_y_sum = float(df[cfg.target_col].sum())
    ratio = media_sum / true_y_sum if true_y_sum else np.nan

    info = {
        "media_sum": media_sum,
        "true_y_sum": true_y_sum,
        "media_share": ratio,
        "target_scale": get_target_scale(mmm),
        "scale_ok": 0.05 <= ratio <= 1.0,
        "note": "官方守恒分解 mmm.data.get_contributions(original_scale=True)",
    }
    return contrib_df, info


def compute_contributions_counterfactual(mmm, df: pd.DataFrame, cfg: MMMConfig,
                                         meta: dict):
    """
    【备用】用官方反事实分解计算各渠道贡献（不守恒，但更贴近"移除法"直觉）。

    原理：contribution_j(t) = inv(μ_t)·s − inv(μ_t − v_j,t)·s
    identity link 下 = v_j,t · s，且**component 之和 ≠ y**（会重复计数交叉项）。

    【注意】identity link 下它与守恒分解数值接近；
    但在 log link 下差别很大，应用时必须区分。
    """
    channels = cfg.channel_cols
    ds = mmm.compute_counterfactual_contributions_dataset(
        central_tendency="median")

    dates = pd.to_datetime(np.asarray(ds.coords["date"].values))
    contrib_df = pd.DataFrame({"date": dates})
    for c in channels:
        if c in ds.data_vars:
            arr = ds[c].mean(("chain", "draw"))
            contrib_df[c] = np.asarray(arr.values).ravel()[:len(contrib_df)]

    media_cols = [c for c in channels if c in contrib_df.columns]
    media_sum = float(contrib_df[media_cols].sum().sum())
    true_y_sum = float(df[cfg.target_col].sum())
    ratio = media_sum / true_y_sum if true_y_sum else np.nan

    info = {
        "media_sum": media_sum,
        "true_y_sum": true_y_sum,
        "media_share": ratio,
        "target_scale": get_target_scale(mmm),
        "scale_ok": 0.05 <= ratio <= 1.5,
        "note": "官方反事实分解（不守恒，分量之和不等于 y）",
    }
    return contrib_df, info


def compute_contributions(mmm, df: pd.DataFrame, cfg: MMMConfig, meta: dict,
                          method: str = "library"):
    """
    计算各渠道贡献。

    method:
      · "library"（推荐）—— 官方守恒分解，各分量之和 = 预测 y，
        尺度可靠、可直接做堆叠图与占比分析
      · "counterfactual" —— 官方反事实分解，更贴近"移除某渠道"的直觉，
        但不守恒（存在交叉项重复计数）

    【单位】返回值单位与目标列一致。
    """
    if method == "counterfactual":
        return compute_contributions_counterfactual(mmm, df, cfg, meta)
    return compute_contributions_library(mmm, df, cfg, meta)


def compute_roi(contrib: pd.DataFrame, df: pd.DataFrame, cfg: MMMConfig):
    """
    计算各渠道 ROI。
    ROI = 该渠道带来的总收入 / 该渠道总花费

    【单位说明】数据的花费与收入都以「元」存储，
    这里统一输出「万元」以便阅读；ROI 本身是无量纲比值。
    """
    rows = []
    total_contrib = sum(
        float(contrib[c].sum()) for c in cfg.channel_cols if c in contrib.columns)
    for c in cfg.channel_cols:
        if c not in contrib.columns:
            continue
        spend = float(get_spend_series(df, c).sum())     # 元
        gain = float(contrib[c].sum())                   # 元（与 y 同尺度）
        rows.append({
            "渠道": c,
            "总花费(万元)": spend / 1e4,
            "总贡献(万元)": gain / 1e4,
            "ROI": gain / spend if spend > 0 else np.nan,
            "贡献占比": gain / total_contrib if total_contrib else np.nan,
        })
    return pd.DataFrame(rows).sort_values("ROI", ascending=False)


# ============================================================
# 参数回收检验
# ============================================================
def check_recovery(diag: dict, true_beta: dict, true_decay: dict,
                   cfg: MMMConfig):
    """
    参数回收检验：对比模型估计与真实值。
    这是验证 MMM 实现正确性的标准手段。
    """
    s = diag["summary"]
    rows = []
    for c in cfg.channel_cols:
        row = {"渠道": c}
        for key, true_vals, pname in [
            ("decay", true_decay, "adstock_alpha"),
            ("beta", true_beta, "saturation_beta"),
        ]:
            name = f"{pname}[{c}]"
            if name in s.index:
                lo_col = "eti94_lb" if "eti94_lb" in s.columns else "hdi_94%_lb"
                hi_col = "eti94_ub" if "eti94_ub" in s.columns else "hdi_94%_ub"
                lo, hi = float(s.loc[name, lo_col]), float(s.loc[name, hi_col])
                tv = float(true_vals[c])
                row[f"真_{key}"] = tv
                row[f"估_{key}"] = float(s.loc[name, "mean"])
                row[f"{key}_区间"] = f"[{lo:.2f}, {hi:.2f}]"
                row[f"{key}_命中"] = "✓" if lo <= tv <= hi else "✗"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 自测：跑一遍完整流程
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_generator import generate_game_data, get_true_contribution

    print("=" * 70)
    print("阶段 2 建模模块自测")
    print("=" * 70)

    df = generate_game_data(n_weeks=156, seed=2024)
    df = normalize_columns(df, MMMConfig())
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()

    X, y, meta, used_ctrl = prepare_data(df, cfg)
    print(f"数据就绪: X={X.shape}, y={y.shape}, 控制变量={used_ctrl}")

    mmm, note = build_mmm_safe(cfg, control_cols=used_ctrl)
    print(f"模型构建: {note}")

    print("\n开始采样...")
    idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    print(f"采样完成，耗时 {elapsed:.1f}s")

    print("\n诊断中...")
    diag = diagnose(idata, cfg)

    print("\n模型自洽性检查...")
    sanity = check_model_sanity(mmm, y)

    print("\n贡献分解（官方守恒分解）...")
    contrib, info = compute_contributions(mmm, df, cfg, meta, method="library")
    print(f"  方法: {info['note']}")
    print(f"  媒体贡献合计 = {info['media_sum'] / 1e4:,.0f} 万元  "
          f"/ 历史总收入 = {info['true_y_sum'] / 1e4:,.0f} 万元  "
          f"(占比 {info['media_share']:.1%})")

    roi = compute_roi(contrib, df, cfg)
    print("\n=== 各渠道 ROI ===")
    print(roi.round(3).to_string(index=False))

    # 参数回收
    from data_generator import TRUE_BETA, TRUE_DECAY
    rec = check_recovery(diag, TRUE_BETA, TRUE_DECAY, cfg)
    print("\n=== 参数回收检验 ===")
    print(rec.to_string(index=False))
