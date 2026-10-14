"""
建模与诊断逻辑层（纯逻辑，不依赖 streamlit，可离线自测）
========================================================

这一层回答的业务问题，按重要性排序：

  1. 模型可信吗？          → 收敛诊断（R-hat / ESS / divergences / 截距健康）
  2. 各渠道 ROI 多少、准吗？ → ROI 的后验分布与 94% 可信区间（不是点估计）
  3. 收入拆成了什么？       → 基线 / 媒体 / 季节 / 控制的守恒分解
  4. 参数估对了吗？         → 参数回收检验（仅内置模拟数据有真值）
  5. 模型自洽吗？           → 可加性 + 预测精度（算 ROI 之前的前置关卡）
  6. 有多少是不可识别的？   → 媒体恒定分量与基线的结构性混淆

设计原则：
  · 只做计算与结构化，不画图、不打印、不碰 st.session_state；
  · 所有"能不能算"的退化路径都要提前判断并返回可读原因，
    绝不让上层页面收到 NaN 或抛异常；
  · 两条取数路径（复用阶段 2 后验 / 应用内拟合）产出**同构**的结果字典，
    页面只认一种结构。

【为什么不用 mmm_core.compute_contributions_library】
它只返回渠道 + 基线 + 季节，缺"控制变量"分项，无法验证守恒是否闭合。
这里直接调官方 `mmm.data.get_contributions(original_scale=True)` 拿全量分项。

【上传数据的三个硬边界（实测确认，pydantic 校验会直接拒绝）】
  · `control_columns`：list[str] | None，min_length=1 → **没有控制变量时必须不传该参数**，
    传 `[]` 会 ValidationError；
  · `yearly_seasonality`：int | None，gt=0     → **要关掉季节性时必须不传**，传 0 会被拒；
  · 渠道列必须至少 1 列，且**不能有零方差列**（channel_scale=0 会让 Adstock 除零）。
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

from . import paths

paths.ensure_src_on_path()

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 常量与判据阈值
# ------------------------------------------------------------
PASS, WARN, FAIL = "pass", "warn", "fail"

MIN_ROWS = 26                 # 少于 26 周：连一年的半周期都不够，拒绝建模
SHORT_HISTORY_WEEKS = 52      # 少于一个完整年度：季节性降级
MIN_CV = 1e-9                 # 渠道花费变异系数低于此值 → 视为常量，无法识别

RHAT_MAX = 1.01               # 收敛判据（行业通用）
ESS_MIN = 400                 # 有效样本量下限
DIV_MAX_PCT = 2.0             # 发散样本占比上限（%）

MEDIA_SHARE_SANE = (0.05, 1.00)   # 媒体贡献占收入比的可接受区间


# ============================================================
# 1. 配置桥：Dataset + 界面控件 → MMMConfig
# ============================================================
DEFAULT_USER_CFG: dict = {
    "adstock_l_max": 6,
    "saturation_type": "hill",
    "yearly_seasonality": 2,
    "target_scaling": "mean",
    "draws": 1500,
    "tune": 1500,
    "chains": 4,
    "target_accept": 0.95,
    "quick_mode": False,
}


def _num(v, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:.{nd}f}"


def _note(level: str, title: str, detail: str = "", hint: str = "") -> dict:
    return {"level": level, "title": title, "detail": detail, "hint": hint}


def build_config(ds, user: dict | None = None):
    """
    把「数据集的列」与「界面上的超参」合成一份 MMMConfig。

    返回 (cfg, notes)。notes 里记录所有为了让数据可建模而做的降级动作 ——
    这一点很重要：**静默降级会让用户以为模型用了他的配置**。
    """
    from mmm_core import MMMConfig          # 延迟导入：避免页面加载时就拉起重依赖

    user = {**DEFAULT_USER_CFG, **(user or {})}
    notes: list[dict] = []
    df = ds.df

    # ---- 渠道：剔除零方差列（全零 / 常数投放都无法识别效应）----
    channels, dropped = [], []
    for c in list(ds.channel_cols):
        col = pd.to_numeric(df[c], errors="coerce").astype(float)
        mu = float(col.mean()) if len(col) else 0.0
        sd = float(col.std(ddof=0)) if len(col) else 0.0
        cv = (sd / abs(mu)) if abs(mu) > 1e-12 else (0.0 if sd <= 1e-12 else float("inf"))
        if cv <= MIN_CV:
            dropped.append((c, "全程无投放" if mu <= 1e-12 else "投放金额几乎不变"))
        else:
            channels.append(c)

    if dropped:
        detail = "；".join(f"{c}（{why}）" for c, why in dropped)
        notes.append(_note(
            WARN, f"已自动剔除 {len(dropped)} 个无法估计的渠道", detail,
            "该渠道花费没有波动，模型无法把它的贡献与基线区分开。"
            "若它确实重要，请补充有投放波动的历史数据。"))
    if not channels:
        notes.append(_note(
            FAIL, "没有任何可建模的渠道",
            "所有渠道列都是常量或全零。",
            "请回到「概览与数据」页检查数据源，或确认渠道列是否被正确识别。"))

    # ---- 控制变量：只保留数据里真实存在且有变异的列 ----
    controls = []
    for c in list(ds.control_cols):
        if c not in df.columns:
            continue
        if float(pd.to_numeric(df[c], errors="coerce").std(ddof=0) or 0.0) <= MIN_CV:
            continue
        controls.append(c)
    if not controls and ds.control_cols:
        notes.append(_note(
            WARN, "已跳过全部控制变量",
            "数据里的控制变量列都是常量或无法解析。",
            "没有控制变量的模型会把「版本更新/竞品/节假日」的影响误记到渠道头上。"))

    # ---- 历史长度 → 季节性降级 ----
    n = int(len(df))
    yearly = int(user.get("yearly_seasonality", 2) or 0)
    if n < MIN_ROWS:
        notes.append(_note(
            FAIL, f"历史长度只有 {n} 周，不足以建模",
            f"至少需要 {MIN_ROWS} 周才能让 Adstock 与季节性参数有可识别性。",
            "请补充更长的历史数据。MMM 的可靠区间通常在 1.5~3 年。"))
    if yearly > 0 and n < SHORT_HISTORY_WEEKS:
        notes.append(_note(
            WARN, "历史不足一年，已把季节性从 %d 阶降为 1 阶" % yearly,
            f"只有 {n} 周数据，{yearly} 阶 Fourier（{yearly * 2} 个参数）会过拟合噪声。",
            "补满 52 周后此降级会自动取消。"))
        yearly = 1

    cfg = MMMConfig(
        date_col=ds.date_col,
        target_col=ds.target_col,
        channel_cols=list(channels),
        control_cols=list(controls),
        adstock_l_max=int(user.get("adstock_l_max", 6)),
        saturation_type=str(user.get("saturation_type", "hill")),
        yearly_seasonality=int(yearly),
        target_scaling=str(user.get("target_scaling", "mean")),
        draws=int(user.get("draws", 1500)),
        tune=int(user.get("tune", 1500)),
        chains=int(user.get("chains", 4)),
        target_accept=float(user.get("target_accept", 0.95)),
        quick_mode=bool(user.get("quick_mode", False)),
    )
    return cfg, notes


def config_snapshot(cfg) -> dict:
    """
    把 cfg 折叠成可哈希的 dict —— 缓存键的组成部分。
    改了任何一项超参，缓存键就变，绝不会命中旧模型的结果。
    """
    return {
        "channels": list(cfg.channel_cols),
        "controls": list(cfg.control_cols),
        "l_max": int(cfg.adstock_l_max),
        "saturation": str(cfg.saturation_type),
        "yearly": int(cfg.yearly_seasonality),
        "scaling": str(cfg.target_scaling),
        "draws": int(cfg.draws),
        "tune": int(cfg.tune),
        "chains": int(cfg.chains),
        "target_accept": float(cfg.target_accept),
        "seed": int(cfg.random_seed),
    }


def sample_size_note(cfg) -> str:
    """给用户一个"这一跑要多久"的量化预期，而不是只有一个转圈图标。"""
    total = int(cfg.chains) * (int(cfg.draws) + int(cfg.tune))
    if total <= 1200:
        tier = "约 20~60 秒"
    elif total <= 4000:
        tier = "约 1~3 分钟"
    elif total <= 13000:
        tier = "约 3~10 分钟"
    else:
        tier = "10 分钟以上"
    return (f"{cfg.chains} 链 ×（{cfg.draws} 采样 + {cfg.tune} 预热）"
            f"= {total:,} 次后验抽样，典型耗时 {tier}（取决于 CPU 核数）")


# ============================================================
# 2. 构建模型（显式处理"可选参数必须省略"的边界）
# ============================================================
def build_model(cfg, ctrl: list[str]):
    return _build_mmm_direct(cfg, ctrl)


def _build_mmm_direct(cfg, ctrl: list[str], custom_prior: bool = True):
    """
    直接构造 MMM。相比 mmm_core.build_mmm 多了两件事：
      1. **按条件传参** —— 无控制变量时不传 `control_columns`，
         无季节性时不传 `yearly_seasonality`
         （两者都是 pydantic 强校验字段：空列表 / 0 会被直接拒绝）；
      2. `custom_prior=False` 可退回默认先验，作为降级兜底。

    【scaling.dims 必须传元组 ()，不能传空字符串 ""】
    库内的校验是 `set(scaling.channel.dims).difference([*self.dims, "channel", "date"])`。
    `dims=""` 会被规范化为 `("",)`，差集非空 → 抛
    `ValueError: Channel scaling dims ('',) must contain (), 'channel', and 'date'`；
    而 `dims=()` 的差集为空，通过。这个报错信息很误导（提示"必须包含空元组"），
    实际原因是把元组写成了字符串。
    """
    from pymc_marketing.mmm import (
        MMM, GeometricAdstock, HillSaturation,
        LogisticSaturation, MichaelisMentenSaturation,
    )
    from pymc_marketing.mmm.scaling import (
        Scaling, DataDerivedScaling, FixedScaling)

    sat_map = {"hill": HillSaturation, "logistic": LogisticSaturation,
               "michaelis_menten": MichaelisMentenSaturation}

    tgt = (FixedScaling(dims=(), value=1.0) if cfg.target_scaling == "none"
           else DataDerivedScaling(method="mean", dims=()))

    kwargs: dict = dict(
        date_column=cfg.date_col,
        channel_columns=list(cfg.channel_cols),
        target_column=cfg.target_col,
        adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
        saturation=sat_map.get(cfg.saturation_type, HillSaturation)(),
        scaling=Scaling(target=tgt, channel=DataDerivedScaling(method="max", dims=())),
    )
    if ctrl:                                   # ★ 有才传
        kwargs["control_columns"] = list(ctrl)
    if cfg.yearly_seasonality > 0:             # ★ 有才传
        kwargs["yearly_seasonality"] = int(cfg.yearly_seasonality)

    if not custom_prior:
        return MMM(**kwargs)

    try:
        from pymc_extras.prior import Prior
    except ImportError:
        from pymc_marketing.model_config import Prior

    base = MMM(**kwargs)
    mc = dict(base.default_model_config)
    a, b = cfg.adstock_alpha_prior
    mc.update({
        "adstock_alpha": Prior("Beta", alpha=a, beta=b, dims="channel"),
        "saturation_kappa": Prior(
            "Gamma", alpha=cfg.saturation_kappa_prior[0],
            beta=cfg.saturation_kappa_prior[1], dims="channel"),
        "saturation_slope": Prior("LogNormal", mu=cfg.saturation_slope_mu,
                                  sigma=cfg.saturation_slope_sigma, dims="channel"),
        "saturation_beta": Prior("LogNormal", mu=cfg.beta_prior_mu,
                                 sigma=cfg.beta_prior_sigma, dims="channel"),
        "intercept": Prior("Normal", mu=cfg.intercept_mu,
                           sigma=cfg.intercept_sigma, dims=()),
    })
    if ctrl:
        mc["gamma_control"] = Prior("Normal", mu=0.0, sigma=cfg.control_sigma,
                                    dims="control")
    kwargs["model_config"] = mc
    return MMM(**kwargs)


# ============================================================
# 3. 预训练后验复用（阶段 2 的产物，秒级）
# ============================================================
def pretrained_available() -> bool:
    return paths.pretrained_pkl() is not None


def pretrained_info() -> dict:
    """
    描述当前可用的后验存档（供页面如实说明「档里有什么」）。

    为什么要读 .json 旁挂文件而不是 pkl 本身：
        页面需要报「共 N 个后验样本」。若为此 pickle.load 一次 pkl，
        就等于**为了显示一行字加载 20 MB 数据**（全量版更是 109 MB）。
        所以让 `src/export_lite_posterior.py` 把链数 / 抽样数写进同名 json，
        这里只读 json（毫秒级）。json 不存在时退化为「不报具体数字」。

    `samples_note` 是给页面直接用的中文描述，避免各处再拼一遍字符串
    （之前 modeling.py 里就硬编码过「含 6,000 个后验样本」，
     换成精简版后数字对不上却仍然照显 —— 属于「不报错的错误陈述」）。
    """
    p = paths.pretrained_pkl()
    if not p:
        return {"available": False}

    info = {"available": True, "path": p, "lite": p.endswith(paths.LITE_PKL),
            "size_mb": round(os.path.getsize(p) / 1e6, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M",
                                   time.localtime(os.path.getmtime(p)))}

    meta = _read_sidecar_meta(p)
    if meta and meta.get("total_draws"):
        n_ch = int(meta.get("chains") or 0)
        n_dr = int(meta.get("draws_per_chain") or 0)
        n_tot = int(meta["total_draws"])
        info.update({"chains": n_ch, "draws_per_chain": n_dr,
                     "total_draws": n_tot})
        # 例如「4 链 × 300 抽，共 1,200 个后验样本」
        info["samples_note"] = (f"{n_ch} 链 × {n_dr} 抽，"
                                f"共 {n_tot:,} 个后验样本")
        info["samples_note_short"] = f"含 {n_tot:,} 个后验样本"
    return info


def _read_sidecar_meta(pkl_path: str) -> dict | None:
    """
    读取与 pkl 同名的 .json 元信息（不存在或损坏时返回 None，绝不抛异常）。

    刻意「静默降级」：这只是给页面文案用的锦上添花，
    不该因为它缺失就让整页崩掉。
    """
    import json
    j = os.path.splitext(pkl_path)[0] + ".json"
    if not os.path.isfile(j):
        return None
    try:
        with open(j, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                          # noqa: BLE001
        return None


def is_builtin(ds) -> bool:
    """判断当前数据集是否就是内置模拟数据（只有它才能复用阶段 2 的后验）。"""
    return getattr(ds, "source", "") == "builtin"


def load_pretrained(ds):
    """
    复用阶段 2 的后验。

    为什么只允许内置数据走这条路：
        后验是**针对某一组具体的 X/y** 采样出来的。换一份数据却注入旧后验，
        得到的贡献分解与 ROI 全是错的，而且不会报错 —— 这是最危险的一类 bug。
        因此这里用「数据源 + 行数」双重校验，宁可多花时间重拟合。

    返回结果字典；不可用时返回 {"ok": False, "reason": ...}。
    """
    if not pretrained_available():
        return {"ok": False, "reason": "缺少阶段 2 后验存档（outputs/reports/stage2_mmm_idata.pkl）"}
    if not is_builtin(ds):
        return {"ok": False, "reason": "当前使用的是上传数据，阶段 2 的后验对它无效"}
    try:
        import stage3_optimize as s3
        df, cfg, mmm, X, y, meta, used_ctrl, idata = s3.load_model_and_data()
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    if int(len(df)) != int(len(ds.df)):
        return {"ok": False,
                "reason": f"后验的样本量（{len(df)} 周）与当前数据（{len(ds.df)} 周）不一致"}
    art = assemble(mmm, idata, df, cfg, y, elapsed=0.0, source="pretrained", notes=[])
    art["ds_note"] = "复用阶段 2 已拟合好的后验（毫秒级，无需重新采样）"
    return art


# ============================================================
# 4. 应用内拟合
# ============================================================
def fit(ds, user_cfg: dict | None = None, progress=None,
        sample_step: int = 1) -> dict:
    """
    完整拟合流水线：准备数据 → 建结构 → 采样 → 诊断 → 分解 → 打包。

    progress(frac, msg) 是可选回调。注意：NUTS 采样是**阻塞**的，
    回调无法在采样过程中被触发；因此采样前后各报一次，
    并在采样前明确告知"共多少个样本、典型耗时多久"，
    这比给一个假进度条诚实。

    返回结构与 load_pretrained 完全同构。
    """
    cb = progress or (lambda *a, **k: None)
    t0 = time.time()

    cfg, notes = build_config(ds, user_cfg)
    if any(n["level"] == FAIL for n in notes):
        return {"ok": False, "reason": notes[0]["title"], "notes": notes}

    cb(0.05, f"准备数据：{len(ds.df)} 周 × {len(cfg.channel_cols)} 个渠道")
    from mmm_core import prepare_data
    df = ds.df.copy()
    X, y, meta, used_ctrl = prepare_data(df, cfg)
    if used_ctrl and len(used_ctrl) != len(cfg.control_cols):
        notes.append(_note(WARN, "部分控制变量未进入模型",
                           f"实际使用：{used_ctrl}", "检查数据列是否能被解析为数值。"))

    cb(0.12, "构建模型结构（Adstock + 饱和 + 季节性 + 先验）")
    try:
        mmm = _build_mmm_direct(cfg, used_ctrl)
        build_note = "自定义业务先验已生效"
    except Exception as e:                                    # noqa: BLE001
        # 【降级路径也必须是"按条件传参"的版本】
        # 不能退回 mmm_core.build_mmm_safe —— 它在无控制变量时会把 [] 传进去，
        # 触发同一个 pydantic 校验错误，等于降级也失败。
        notes.append(_note(WARN, "自定义先验不兼容当前版本，已退回默认先验",
                           f"{type(e).__name__}: {str(e)[:160]}"))
        mmm = _build_mmm_direct(cfg, used_ctrl, custom_prior=False)
        build_note = "已退回默认先验"
    mmm.build_model(X, y)

    cb(0.25, f"编译 PyTensor 计算图 …（{sample_size_note(cfg)}）")
    from mmm_core import fit_mmm, get_target_scale
    try:
        idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    except Exception as e:                                    # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        notes.append(_note(FAIL, "采样失败", msg,
                           "常见原因：历史过短、渠道间完全共线、或某个渠道全程零投放。"))
        return {"ok": False, "reason": msg, "notes": notes}

    cb(0.88, f"采样完成（{elapsed:.0f} 秒），计算诊断与贡献分解")
    art = assemble(mmm, idata, df, cfg, y, elapsed=elapsed,
                   source="fit", notes=notes, sample_step=sample_step)
    art["build_note"] = build_note
    art["target_scale"] = get_target_scale(mmm)
    cb(1.0, "完成")
    art["wall_seconds"] = time.time() - t0
    return art


# ============================================================
# 5. 结果组装（两条路径共用）
# ============================================================
def assemble(mmm, idata, df: pd.DataFrame, cfg, y, elapsed: float = 0.0,
             source: str = "fit", notes: list[dict] | None = None,
             sample_step: int = 1) -> dict:
    """把模型 + 后验打包成页面所需的一切（诊断、贡献、ROI、回收、自洽性）。"""
    art: dict = {
        "ok": True, "source": source, "df": df, "cfg": cfg, "mmm": mmm,
        "idata": idata, "y": y, "elapsed": float(elapsed),
        "notes": list(notes or []), "error": None,
    }
    # ★ 在建模阶段就把 channel_scales 挂好（三条路径 fit / pretrained / cache 全走这里）。
    #
    # 为什么必须在这里而不是等优化页再挂：
    #   库内 mmm.py:3090 用 `getattr(self, "_channel_scales", 1.0)`，MMM 对象默认没这个
    #   属性 → 回退 1.0 → 货币与模型空间量纲错配 → 优化器返回"五渠道完全相等"的
    #   假最优解，而且 success=True（不报错）。
    #   阶段 9 的探针实测：3 渠道上传数据 nit=1、提升 -0.00%、分配 [353.6,353.6,353.6]；
    #   挂载后同一份数据 nit=53、提升 +75.73%。
    #   注意 budget_optimizer() 在**构造时**就把 scales 读进去了，所以必须在这里
    #   （建模阶段）挂，晚了没用。
    try:
        art["channel_scales"] = _attach_channel_scales(mmm, df, cfg)
        art["scales_ok"] = True
    except Exception as e:                                    # noqa: BLE001
        art["channel_scales"] = None
        art["scales_ok"] = False
        art["scales_error"] = f"{type(e).__name__}: {e}"
    try:
        art["diag"] = diag_summary(idata, cfg)
    except Exception as e:                                    # noqa: BLE001
        art["diag"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        art["contrib"], art["cinfo"] = contributions(mmm, df, cfg)
    except Exception as e:                                    # noqa: BLE001
        art["contrib"], art["cinfo"] = None, {"ok": False,
                                              "reason": f"{type(e).__name__}: {e}"}
    try:
        art["roi"] = roi_posterior(mmm, df, cfg, sample_step=sample_step)
    except Exception as e:                                    # noqa: BLE001
        art["roi"] = None
        art["roi_error"] = f"{type(e).__name__}: {e}"
    try:
        art["sanity"] = sanity(mmm, y, idata)
    except Exception as e:                                    # noqa: BLE001
        art["sanity"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        art["mix"] = contribution_mix(art["contrib"], df, cfg)
    except Exception as e:                                    # noqa: BLE001
        art["mix"] = {"ok": False, "reason": f"{type(e).__name__}: {e}",
                      "total": None, "wave": None}
    art["recovery"] = recovery_table(idata, cfg, art.get("roi"))
    art["ident"] = identifiability(mmm, df, cfg)
    return art


# ============================================================
# 6. 收敛诊断
# ============================================================
def diag_summary(idata, cfg) -> dict:
    """
    R-hat / ESS / divergences / 截距健康 —— 决定"这份结果能不能拿去汇报"。

    判据（都是行业通用阈值，写在这里方便页面直接引用）：
        R-hat < 1.01      → 多条链采到了同一个分布（不是各自卡在局部）
        ESS_bulk > 400    → 有效样本量足够，分位数估计稳定
        divergences < 2%  → 没有系统性几何病态，参数可信
        截距 > 0          → 基线没被渠道贡献挤负（否则媒体贡献会虚高 4~8 倍）
    """
    import arviz as az

    post = idata.posterior
    avail = list(post.data_vars)
    want = ["intercept_contribution", "saturation_beta", "adstock_alpha",
            "saturation_kappa", "saturation_slope", "y_sigma", "gamma_control"]
    pick = [v for v in want if v in avail]

    try:
        summary = az.summary(idata, var_names=pick, ci_prob=0.94, round_to=4)
    except TypeError:                       # 老版 arviz 用 hdi_prob
        summary = az.summary(idata, var_names=pick, hdi_prob=0.94, round_to=4)

    n_div = n_tot = 0
    try:
        n_div = int(idata.sample_stats["diverging"].sum().values)
        n_tot = int(idata.sample_stats["diverging"].size)
    except Exception:                                         # noqa: BLE001
        pass
    div_pct = 100.0 * n_div / max(n_tot, 1)

    n_bad = int((summary["r_hat"] > RHAT_MAX).sum()) if "r_hat" in summary.columns else 0
    min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else float("nan")
    max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary.columns else float("nan")

    ic = None
    try:
        ic = float(post["intercept_contribution"].values.mean())
    except Exception:                                         # noqa: BLE001
        pass

    checks = []
    if np.isfinite(max_rhat):
        ok = n_bad == 0
        checks.append(_check("rhat", "R-hat 全部 < 1.01", PASS if ok else FAIL,
                             f"最大 {max_rhat:.4f}（{n_bad}/{len(summary)} 个参数超标）",
                             "超标说明有链没收敛到同一分布，结论不可用。"))
    if np.isfinite(min_ess):
        checks.append(_check("ess", "最小 ESS > 400", PASS if min_ess > ESS_MIN else WARN,
                             f"{min_ess:,.0f}",
                             "ESS 低说明该参数的后验分位数不稳，可信区间会偏窄。"))
    checks.append(_check("div", "发散样本 < 2%", PASS if div_pct < DIV_MAX_PCT else FAIL,
                         f"{n_div}/{n_tot}（{div_pct:.2f}%）",
                         "发散说明采样器撞上了几何病态区，通常要调先验或收紧参数空间。"))
    if ic is not None:
        checks.append(_check("intercept", "截距为正（基线健康）", PASS if ic > 0 else FAIL,
                             _num(ic, 3),
                             "负截距意味着渠道贡献被虚高，ROI 会整体偏乐观。"))

    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    n_warn = sum(1 for c in checks if c["status"] == WARN)
    return {
        "ok": True, "summary": summary, "checks": checks,
        "max_rhat": max_rhat, "min_ess": min_ess,
        "n_bad_rhat": n_bad, "n_params": int(len(summary)),
        "n_divergences": n_div, "n_total": n_tot, "divergence_pct": div_pct,
        "intercept": ic, "n_fail": n_fail, "n_warn": n_warn,
        "converged": (n_fail == 0),
        "n_samples": int(idata.posterior.sizes.get("chain", 0)
                         * idata.posterior.sizes.get("draw", 0)),
        "verdict": ("收敛良好，结果可用于决策"
                    if n_fail == 0 and n_warn == 0 else
                    ("收敛基本可用，但有一项需要注意" if n_fail == 0 else
                     "收敛未达标，结论不可直接用")),
    }


def _check(key: str, label: str, status: str, value: str, hint: str = "") -> dict:
    return {"key": key, "label": label, "status": status, "value": value, "hint": hint}


def posterior_table(idata, cfg, step: int = 1) -> pd.DataFrame:
    """所有关键参数的点估计 + 94% 可信区间（含渠道/控制变量的可读标签）。"""
    d = diag_summary(idata, cfg)
    if not d.get("ok"):
        return pd.DataFrame()
    s = d["summary"].reset_index().rename(columns={"index": "参数"})
    lo = "eti94_lb" if "eti94_lb" in s.columns else ("hdi_94%_lb" if "hdi_94%_lb" in s.columns else None)
    hi = "eti94_ub" if "eti94_ub" in s.columns else ("hdi_94%_ub" if "hdi_94%_ub" in s.columns else None)
    out = pd.DataFrame({
        "参数": s["参数"].str.replace(r"\[(.*)\]", r"（\1）", regex=True),
        "均值": s["mean"].astype(float).round(4),
        "95%区间下": s[lo].astype(float).round(4) if lo else np.nan,
        "95%区间上": s[hi].astype(float).round(4) if hi else np.nan,
        "R-hat": s["r_hat"].astype(float).round(4) if "r_hat" in s.columns else np.nan,
        "ESS": s["ess_bulk"].astype(float).round(0) if "ess_bulk" in s.columns else np.nan,
    })
    return out


# ============================================================
# 7. 贡献分解（守恒：各分量之和 = 预测 y）
# ============================================================
def contributions(mmm, df: pd.DataFrame, cfg):
    """
    官方守恒分解 `mmm.data.get_contributions(original_scale=True)`。

    为什么必须用 original_scale=True：
        后验里的 `channel_contribution` 是**标准化空间**的量，
        直接拿来当"元"会差一个 target_scale 的倍数。这个 bug 不会报错，
        只会让 ROI 系统性偏大或偏小 —— 属于最难发现的一类。

    返回 (contrib_df, info)：
        contrib_df: date + 各渠道贡献 + _baseline + _seasonality + _controls
        info: 守恒检验与占比
    """
    ds = mmm.data.get_contributions(original_scale=True)
    mean = lambda da: da.mean(("chain", "draw"))              # noqa: E731

    ch = ds["channels"]                                        # (chain,draw,date,channel)
    dates = pd.to_datetime(np.asarray(ch.coords["date"].values))
    out = pd.DataFrame({"date": dates})
    for c in cfg.channel_cols:
        out[c] = (np.asarray(mean(ch.sel(channel=c)).values)
                  if c in ch.coords.get("channel", []) else 0.0)

    out["_baseline"] = float(np.asarray(mean(ds["baseline"]).values).ravel()[0])
    try:
        out["_seasonality"] = np.asarray(mean(ds["seasonality"]).values).ravel()[:len(out)]
    except Exception:                                         # noqa: BLE001
        out["_seasonality"] = 0.0
    try:
        ct = ds["controls"]                                    # (chain,draw,date,control)
        out["_controls"] = np.asarray(mean(ct).values).sum(axis=-1)[:len(out)]
    except Exception:                                         # noqa: BLE001
        out["_controls"] = 0.0

    media_cols = [c for c in cfg.channel_cols if c in out.columns]
    y_true = pd.to_numeric(df[cfg.target_col], errors="coerce").astype(float)
    y_sum = float(y_true.sum())

    parts = out[media_cols + ["_baseline", "_seasonality", "_controls"]].sum(axis=1)
    resid = float((y_true.values[:len(out)] - parts.values).sum())
    info = {
        "ok": True,
        "media_sum": float(out[media_cols].sum().sum()),
        "baseline_sum": float(out["_baseline"].sum()),
        "season_sum": float(out["_seasonality"].sum()),
        "control_sum": float(out["_controls"].sum()),
        "true_y_sum": y_sum,
        "resid_sum": resid,
        "residual_pct": abs(resid) / y_sum if y_sum else float("nan"),
        "media_share": float(out[media_cols].sum().sum()) / y_sum if y_sum else float("nan"),
        "conservative": abs(resid) / y_sum < 0.02 if y_sum else False,
        "note": "官方守恒分解 get_contributions(original_scale=True)",
    }
    return out, info


def contribution_mix(contrib: pd.DataFrame, df: pd.DataFrame, cfg) -> dict:
    """
    收入怎么分的 —— 但必须拆成**两张口径不同的表**，混在一张里一定会讲错。

    ┌─ 表 A「总量账」─────────────────────────────────────────────┐
    │ 基线 + 各渠道。这两类有真正的"总量"含义，加总 ≈ 历史收入。      │
    └──────────────────────────────────────────────────────────┘
    ┌─ 表 B「波动账」─────────────────────────────────────────────┐
    │ 季节性 / 控制变量。它们的**年度净额恒为 0**，所以"占收入比"      │
    │ 这个指标对它们没有意义，只能看**逐周振幅**。                   │
    └──────────────────────────────────────────────────────────┘

    为什么季节项的净额必然是 0：
        Fourier 基（sin/cos）在一个完整年度上的积分等于 0，
        156 周恰好是 3 个完整年度 → 逐期季节贡献求和精确抵消。
        这不是 bug，而是"季节性只重新分配收入的时点分布、不创造总量"的数学表达。
        把它的 sum 当成"季节性贡献了 X 万元"是 MMM 汇报里最常见的错误之一。
    """
    if contrib is None or not len(contrib):
        return {"ok": False, "total": pd.DataFrame(), "wave": pd.DataFrame()}
    n = len(contrib)
    y_sum = float(pd.to_numeric(df[cfg.target_col], errors="coerce").sum())

    rows = []
    if "_baseline" in contrib.columns:
        rows.append({"分量": "基线（自然增长）", "类别": "基线",
                     "金额(万元)": float(contrib["_baseline"].iloc[0]) * n / 1e4})
    for c in cfg.channel_cols:
        if c in contrib.columns:
            rows.append({"分量": c, "类别": "媒体",
                         "金额(万元)": float(contrib[c].sum()) / 1e4})
    total = pd.DataFrame(rows)
    if len(total):
        total["占收入比"] = (total["金额(万元)"] * 1e4 / y_sum) if y_sum else np.nan
        total = total.sort_values("金额(万元)", ascending=False).reset_index(drop=True)

    wave_rows = []
    for col, lab, why in (
            ("_seasonality", "季节性（Fourier）",
             "Fourier 基在完整年度上积分为 0 → 只改变收入的时点分布，不改变年度总量"),
            ("_controls", "控制变量",
             "控制变量已标准化（均值 0），且被收缩先验收拢 → 同样只贡献波动")):
        if col in contrib.columns:
            v = pd.to_numeric(contrib[col], errors="coerce").astype(float)
            wave_rows.append({
                "分量": lab,
                "逐周振幅 ±(万元)": round(float(v.std(ddof=0)) / 1e4, 1),
                "年度净额(万元)": round(float(v.sum()) / 1e4, 1),
                "为什么净额接近 0": why,
            })
    return {"ok": True, "total": total, "wave": pd.DataFrame(wave_rows),
            "y_sum": y_sum,
            "media_sum": float(sum(contrib[c].sum() for c in cfg.channel_cols
                                   if c in contrib.columns)),
            "base_sum": (float(contrib["_baseline"].iloc[0]) * n
                         if "_baseline" in contrib.columns else float("nan"))}


# ============================================================
# 8. ROI 的后验分布（不是点估计！）
# ============================================================
def roi_posterior(mmm, df: pd.DataFrame, cfg, sample_step: int = 1) -> pd.DataFrame:
    """
    逐样本算 ROI，得到**区间**而非一个数。

    业务含义：ROI = 1.24 这句话没有信息量；"ROI 有 94% 概率落在 [1.11, 1.38]，
    且整段区间都在 1.0 以上"才能支撑"继续加预算"的决策。

    sample_step：每隔几个后验样本取一个（大模型下抽样以控内存，不影响区间位置）。
    """
    from mmm_core import get_spend_series

    ds = mmm.data.get_contributions(original_scale=True)
    ch = ds["channels"]                                        # (chain,draw,date,channel)
    if sample_step and sample_step > 1:
        ch = ch.isel(draw=slice(None, None, int(sample_step)))

    rows = []
    for c in cfg.channel_cols:
        if c not in ch.coords.get("channel", []):
            continue
        tot = np.asarray(ch.sel(channel=c).sum("date").values).ravel()   # (sample,) 元
        spend = float(get_spend_series(df, c).sum())
        if spend <= 0:
            rows.append({"渠道": c, "ROI均值": np.nan, "ROI下界": np.nan,
                         "ROI上界": np.nan, "总花费(万元)": 0.0,
                         "总贡献(万元)": float(tot.mean()) / 1e4,
                         "P(ROI>1)": np.nan, "结论": "本期无投放"})
            continue
        roi = tot / spend
        p_beat = float(np.mean(roi > 1.0))
        lo, hi = np.percentile(roi, [3, 97])
        if lo > 1.0:
            concl = "整段区间 > 1，加预算有正当理由"
        elif hi < 1.0:
            concl = "整段区间 < 1，应缩减或重做素材/定向"
        else:
            concl = "区间跨越 1，证据不足，先做小规模增量实验"
        rows.append({
            "渠道": c, "ROI均值": float(roi.mean()),
            "ROI下界": float(lo), "ROI上界": float(hi),
            "总花费(万元)": spend / 1e4, "总贡献(万元)": float(tot.mean()) / 1e4,
            "P(ROI>1)": p_beat, "结论": concl,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("ROI均值", ascending=False).reset_index(drop=True)
    out["ROI均值"] = out["ROI均值"].round(3)
    out["ROI下界"] = out["ROI下界"].round(3)
    out["ROI上界"] = out["ROI上界"].round(3)
    out["P(ROI>1)"] = out["P(ROI>1)"].round(3)
    out["总花费(万元)"] = out["总花费(万元)"].round(1)
    out["总贡献(万元)"] = out["总贡献(万元)"].round(1)
    return out


def roi_from_csv() -> pd.DataFrame | None:
    """
    读阶段 2 已落盘的 ROI 区间表（用于与页面上现算的结果做交叉核对）。
    页面会对比两者；数值不一致说明口径漂移，比直接展示一个数更安全。
    """
    p = paths.artifact_path("stage2_roi_ci")
    if not os.path.isfile(p):
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        return df.rename(columns={"channel": "渠道", "mean": "ROI均值",
                                  "lo": "ROI下界", "hi": "ROI上界",
                                  "true": "真值"})
    except Exception:                                         # noqa: BLE001
        return None


# ============================================================
# 9. 参数回收检验
# ============================================================
def recovery_table(idata, cfg, roi_df: pd.DataFrame | None = None) -> dict:
    """
    估计值 vs 真值。**只有内置模拟数据有真值**，上传数据时真值列留空。

    【口径必须对齐，否则这张表会误导】
      · ROI 要拿 `roi_posterior` 的结果（元收入/元花费）比真实 ROI，
        **不能** 拿 `saturation_beta` 去比 —— 后者是标准化空间的曲线高度参数，
        量纲与 ROI 不同（实测 beta≈4.0 而 ROI≈2.5），放在同一列会得出
        "模型全错"的错误结论。
      · 衰减 λ 直接比 `adstock_alpha` 与真值 decay，两者同尺度。

    这张表的意义不是炫技，而是**自证实现正确性**：
    连已知答案都收不回来，后面的 ROI 排序就没有可信度。
    """
    from . import eda
    truth = eda.ground_truth()
    d = diag_summary(idata, cfg)
    if not d.get("ok") or truth is None:
        return {"ok": truth is not None, "table": pd.DataFrame(),
                "reason": None if truth is None else "诊断失败", "n_hit": 0, "n_tested": 0,
                "note": "上传数据没有真值可比 —— 真实世界没有标准答案，"
                        "这正是要靠诊断而非对答案判断可信度的原因。"}
    s = d["summary"]
    lo_c = "eti94_lb" if "eti94_lb" in s.columns else "hdi_94%_lb"
    hi_c = "eti94_ub" if "eti94_ub" in s.columns else "hdi_94%_ub"

    roi_lookup = {}
    if roi_df is not None and len(roi_df):
        roi_lookup = {r["渠道"]: (float(r["ROI均值"]), float(r["ROI下界"]),
                                  float(r["ROI上界"]))
                      for _, r in roi_df.iterrows()}

    rows = []
    for c in cfg.channel_cols:
        row = {"渠道": c}
        if c in roi_lookup:
            est, lo, hi = roi_lookup[c]
            tv = float(truth.get("roi", {}).get(c, np.nan))
            row.update({
                "真_ROI": round(tv, 3) if np.isfinite(tv) else np.nan,
                "估_ROI": round(est, 3), "ROI区间": f"[{lo:.2f}, {hi:.2f}]",
                "ROI命中": ("—" if not np.isfinite(tv) else
                            ("✓" if lo <= tv <= hi else "✗"))})
        idx = f"adstock_alpha[{c}]"
        if idx in s.index:
            lo, hi = float(s.loc[idx, lo_c]), float(s.loc[idx, hi_c])
            est = float(s.loc[idx, "mean"])
            tv = float(truth.get("decay", {}).get(c, np.nan))
            row.update({
                "真_衰减λ": round(tv, 3) if np.isfinite(tv) else np.nan,
                "估_衰减λ": round(est, 3), "衰减λ区间": f"[{lo:.2f}, {hi:.2f}]",
                "衰减λ命中": ("—" if not np.isfinite(tv) else
                              ("✓" if lo <= tv <= hi else "✗"))})
        rows.append(row)
    tab = pd.DataFrame(rows)

    hits = [v for k in ("ROI命中", "衰减λ命中") if k in tab.columns
            for v in tab[k] if v in ("✓", "✗")]
    n_hit = sum(1 for v in hits if v == "✓")

    # ---- 排序正确性：这才是真正重要的（绝对值可以偏，排序不能错）----
    rank_info = {}
    if roi_lookup:
        est_order = [row["渠道"] for row in sorted(rows, key=lambda r: -r.get("估_ROI", -9))]
        true_order = [row["渠道"] for row in sorted(rows, key=lambda r: -r.get("真_ROI", -9))]
        rank_info = {"est_order": est_order, "true_order": true_order,
                     "order_match": est_order == true_order}
    return {"ok": True, "table": tab, "n_hit": n_hit, "n_tested": len(hits),
            "rank_info": rank_info,
            "note": "区间覆盖真值 = 模型没有把不确定性说得太满；"
                    "排序一致比绝对值精确更重要 —— 预算分配只依赖排序。"}


# ============================================================
# 10. 模型自洽性（算 ROI 之前的前置关卡）
# ============================================================
def sanity(mmm, y, idata=None) -> dict:
    """
    算 ROI 之前的三道前置关卡：
      1. 可加性 —— Σ分量 是否等于 μ（identity link 下必须成立）
      2. 截距   —— 是否为正（负截距 → 媒体贡献虚高 4~8 倍）
      3. 预测精度 —— 模型能否拟合历史总量（|偏差| < 5%）

    为什么必须独立做：这三条任何一条不过，后面的贡献分解与 ROI 都不可信，
    而且**不会报错**，只会静默地给出漂亮但错误的数字。

    【为什么不直接用 mmm_core.check_model_sanity】
    它内部调 `mmm.predict()`，而在"注入后验"的复用路径上，
    `constant_data` 里的数组是 MaskedArray，PyTensor 会在编译期抛
    `Unsupported array type: numpy.ma.MaskedArray` → 第 3 项检查静默消失。
    这里改为直接读后验里的 `mu`（模型对 y 的期望）并乘回 target_scale，
    数学等价、无需重新编译计算图，也就不会在复用路径上失效。
    """
    from mmm_core import check_model_sanity, get_target_scale
    info = check_model_sanity(mmm, y, verbose=False)
    checks = []

    if info.get("additive_ok") is not None:
        checks.append(_check("additive", "可加性：Σ分量 = μ",
                             PASS if info["additive_ok"] else FAIL,
                             f"偏差 {info.get('additivity_err', np.nan):+.4%}",
                             "不成立说明贡献分解口径有问题，各渠道贡献之和解释不了总预测。"))
    if info.get("intercept") is not None:
        checks.append(_check("intercept", "截距为正（基线健康）",
                             PASS if info["intercept_healthy"] else FAIL,
                             _num(info["intercept"], 3),
                             "负截距 → 媒体贡献虚高，ROI 整体偏乐观。"))

    # ---- 3. 预测精度：用 mu 直接算，绕开 predict 的 MaskedArray 问题 ----
    try:
        post = (idata if idata is not None else mmm.idata).posterior
        ts = get_target_scale(mmm)
        pred_sum = float(np.asarray(post["mu"].values).mean(axis=(0, 1)).sum()) * ts
        true_sum = float(np.asarray(pd.to_numeric(pd.Series(y), errors="coerce")).sum())
        err = pred_sum / true_sum - 1 if true_sum else float("nan")
        info.update({"pred_sum": pred_sum, "true_sum": true_sum, "pred_err": err,
                     "pred_ok": abs(err) < 0.05, "target_scale": ts})
        checks.append(_check("pred", "预测拟合历史总量（|偏差| < 5%）",
                             PASS if info["pred_ok"] else WARN, f"{err:+.2%}",
                             "偏差大说明模型没抓住主要结构，先别解读渠道。"))
    except Exception as e:                                    # noqa: BLE001
        info["pred_ok"] = None
        info["pred_note"] = f"{type(e).__name__}: {e}"

    info["checks"] = checks
    info["n_fail"] = sum(1 for c in checks if c["status"] == FAIL)
    return info


# ============================================================
# 11. 可识别性：媒体贡献里有多少是不可分的
# ============================================================
def identifiability(mmm, df: pd.DataFrame, cfg) -> dict:
    """
    MMM 最根本的结构性问题，必须主动讲出来：

        y = 基线常数 + Σ媒体贡献 + 波动项

    其中「媒体贡献的**恒定分量**」与「基线常数」在数学上可以任意互相转移
    （加 α、减 α 而 y 不变）。所以要给出这个混淆量的大小，
    让读者知道"媒体贡献占比 56.9%"这个数字的**分辨率**在哪。

    做法：把每个渠道的贡献序列对其时间均值做分解，
    恒定分量 = 各渠道时间均值的加总；波动分量 = 残下的部分。
    """
    try:
        contrib, cinfo = contributions(mmm, df, cfg)
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    media_cols = [c for c in cfg.channel_cols if c in contrib.columns]
    y_sum = cinfo["true_y_sum"]
    const_part = sum(float(contrib[c].mean()) for c in media_cols) * len(contrib)
    wave_part = cinfo["media_sum"] - const_part
    base_const = float(contrib["_baseline"].iloc[0]) * len(contrib)
    share = const_part / y_sum if y_sum else np.nan
    ratio = const_part / base_const if base_const else np.nan
    return {
        "ok": True,
        "const_part": const_part, "wave_part": wave_part,
        "const_share": share, "baseline_const": base_const,
        "confusable_ratio": ratio,
        "level": FAIL if share > 0.35 else (WARN if share > 0.20 else PASS),
        "verdict": (
            f"媒体贡献里有 {share:.1%} 是**恒定分量**，与基线不可分离（混淆量约 {ratio:.2f} 倍）。"
            "这意味着绝对水平的解释力有限；可靠的是**渠道之间的相对排名**"
            "与**预算变动带来的增量**，而不是「媒体一共赚了多少」这个绝对数。"
            + (" 要进一步收紧，需要 geo-lift 实验或暂停投放测试（pause test）做外部锚定。"
               if share > 0.35 else "")),
    }


# ============================================================
# 12. 后验抽样（供页面绘图，不依赖 matplotlib）
# ============================================================
def draws_for_plot(idata, var: str, channels=None, max_n: int = 1500) -> dict:
    """
    把后验抽成一个扁平数组，交给 Plotly 画分布。
    只取前 max_n 个样本 —— 画直方图不需要全部抽样，反而拖慢前端。
    """
    post = idata.posterior
    if var not in post:
        return {"ok": False, "reason": f"后验中没有 {var}"}
    da = post[var]
    out: dict = {"ok": True, "var": var}
    if "channel" in da.dims:
        ch_names = [str(v) for v in da.coords["channel"].values]
        sel = ch_names if channels is None else [c for c in channels if c in ch_names]
        out["channels"] = sel
        out["values"] = {c: np.asarray(da.sel(channel=c).values).ravel()[:max_n]
                         for c in sel}
    else:
        out["channels"] = None
        out["values"] = {"（整体）": np.asarray(da.values).ravel()[:max_n]}
    return out


def trace_for_plot(idata, var: str, channel: str | None = None,
                   max_points: int = 200) -> dict:
    """轨迹图数据：每条链一条折线。用于目视判断链是否平稳、是否混得好。"""
    post = idata.posterior
    if var not in post:
        return {"ok": False, "reason": f"后验中没有 {var}"}
    da = post[var]
    if "channel" in da.dims and channel:
        da = da.sel(channel=channel)
    for d in ("fourier_mode", "control"):
        if d in da.dims:
            da = da.isel({d: 0})
    arr = np.asarray(da.values)                                # (chain, draw)
    if arr.ndim == 1:
        arr = arr[None, :]
    step = max(1, arr.shape[1] // max(1, max_points))
    return {"ok": True, "var": var, "channel": channel,
            "chains": [arr[i, ::step].tolist() for i in range(arr.shape[0])]}


def contribution_series(contrib: pd.DataFrame, cfg, top_n: int | None = None) -> dict:
    """把贡献分解转成画堆叠图的宽表（date + 各分量）。"""
    if contrib is None or not len(contrib):
        return {"ok": False, "reason": "无贡献分解结果"}
    cols = [c for c in cfg.channel_cols if c in contrib.columns]
    if top_n:
        order = contrib[cols].sum().sort_values(ascending=False).index.tolist()
        cols = order[:top_n]
    out = contrib[["date"] + cols].copy()
    out["基线"] = contrib["_baseline"]
    if "_seasonality" in contrib.columns:
        out["季节性"] = contrib["_seasonality"]
    return {"ok": True, "df": out, "channels": cols}


# ============================================================
# 13. 缓存载荷（只放可序列化的部分）
# ============================================================
def light(art: dict) -> dict:
    """
    剥掉不可序列化的大对象（mmm / idata / df），只留页面展示所需的轻量结果。
    用于把结果放进 cache_data 或写进会话状态时做一次瘦身。
    """
    if not art:
        return {}
    keep = ("ok", "source", "error", "reason", "elapsed", "notes", "diag", "cinfo",
            "sanity", "mix", "recovery", "ident", "roi", "contrib", "cache_note",
            "ds_note", "build_note", "wall_seconds", "roi_error",
            "channel_scales", "scales_ok", "scales_error")
    return {k: art.get(k) for k in keep if k in art}


def to_cache_payload(art: dict) -> dict:
    """从结果字典里挑出可 pickle 的部分，用于落盘（模型对象不入盘，读时重建）。"""
    if not art or not art.get("ok"):
        return {}
    return {
        "idata": art.get("idata"),
        "cfg": art.get("cfg"),
        "elapsed": art.get("elapsed", 0.0),
        "source": art.get("source", "fit"),
        "notes": art.get("notes", []),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def from_cache_payload(payload: dict, df: pd.DataFrame):
    """
    从磁盘缓存还原：重建 MMM 结构并注入后验（不采样）。
    这是"跨重启复用"的关键 —— 否则用户重开浏览器就要再等一次 10 分钟。
    """
    if not payload or not payload.get("idata"):
        return {"ok": False, "reason": "缓存内容为空"}
    cfg = payload["cfg"]
    from mmm_core import prepare_data
    try:
        X, y, meta, used_ctrl = prepare_data(df, cfg)
        mmm = _build_mmm_direct(cfg, used_ctrl)
        mmm.build_model(X, y)
        mmm.idata = payload["idata"]
        _attach_channel_scales(mmm, df, cfg)
        art = assemble(mmm, payload["idata"], df, cfg, y,
                       elapsed=payload.get("elapsed", 0.0),
                       source="cache", notes=payload.get("notes", []))
        # 保留"最初是怎么算出来的"，但对外呈现为"磁盘缓存还原"
        art["origin_source"] = payload.get("source", "fit")
        art["cache_note"] = ("从磁盘缓存还原（重建模型结构 + 注入后验，未重新采样）。"
                             f"这份结果的原始来源：{payload.get('source', 'fit')}")
        return art
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "reason": f"缓存还原失败：{type(e).__name__}: {e}"}


def _attach_channel_scales(mmm, df: pd.DataFrame, cfg) -> list[float]:
    """
    ★ 挂载 channel_scales（阶段 3 踩过的坑，这里必须同源修复）。

    不挂的后果：库内 `getattr(self, "_channel_scales", 1.0)` 回退到 1.0，
    量纲错配 → 阶段 9 的预算优化会返回"各渠道完全相等"的假最优解。
    这里在**建模阶段**就挂上，避免优化页再踩一次。
    """
    from mmm_core import get_spend_series
    scales = [float(get_spend_series(df, c).max()) for c in cfg.channel_cols]
    mmm._channel_scales = np.asarray(scales)
    return scales


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("app/core/model.py 自测（复用阶段 2 后验，不重新采样）")
    print("=" * 72)
    from . import data_io                                     # noqa: F401

    ds = data_io.load_builtin()
    art = load_pretrained(ds)
    print(f"预训练可用 : {pretrained_info()}")
    print(f"载入结果   : ok={art.get('ok')}  原因={art.get('reason')}")
    if art.get("ok"):
        d = art["diag"]
        print(f"收敛判定   : {d['verdict']}  "
              f"maxRhat={_num(d['max_rhat'],4)}  minESS={d['min_ess']:,.0f}  "
              f"div={d['divergence_pct']:.2f}%  截距={_num(d['intercept'])}")
        print(f"样本量     : {d['n_samples']:,}")
        print()
        print("ROI（后验区间）:")
        print(art["roi"][["渠道", "ROI均值", "ROI下界", "ROI上界", "P(ROI>1)", "结论"]]
              .to_string(index=False))
        print()
        print("贡献分解守恒:", art["cinfo"]["conservative"],
              f"残差占比 {art['cinfo']['residual_pct']:.4%}",
              f"媒体占比 {art['cinfo']['media_share']:.1%}")
        print()
        print("收入构成（总量账）:")
        print(art["mix"]["total"].to_string(index=False))
        print()
        print("波动账（年度净额应接近 0）:")
        print(art["mix"]["wave"].to_string(index=False))
        print()
        r = art["recovery"]
        print(f"参数回收: {r['n_hit']}/{r['n_tested']} 命中；"
              f"ROI 排序与真值一致: {r['rank_info'].get('order_match')}")
        print(r["table"].to_string(index=False))
        print(f"   排序(估) = {r['rank_info'].get('est_order')}")
        print(f"   排序(真) = {r['rank_info'].get('true_order')}")
        print()
        print("自洽性:", [(c["label"], c["status"]) for c in art["sanity"]["checks"]])
        print()
        print("可识别性:", art["ident"]["level"])
        print(art["ident"]["verdict"])
