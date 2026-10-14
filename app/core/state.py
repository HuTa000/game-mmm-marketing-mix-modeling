"""
会话状态管理（依赖 streamlit）。

要解决的核心问题：
    Streamlit 每次交互都会**从头重跑整个脚本**。如果建模结果只放在局部变量里，
    用户一切换页面就丢了，得重新等 10~20 分钟 —— 应用直接不可用。

解法：
    1. 所有跨页共享的重结果（数据、模型、优化、报告）放进 st.session_state；
    2. 建立**依赖链** dataset → model → optimization → report，
       上游一变，自动清空下游，绝不允许"新数据配旧结论"。

第 2 点是正确性问题，不是体验问题：如果用户换了数据但优化结果没清空，
页面会展示自相矛盾的数字，比崩溃更糟。
"""
from __future__ import annotations

import streamlit as st

from . import ui

# ------------------------------------------------------------
# 会话键名（集中定义，避免字符串散落各处拼错）
# ------------------------------------------------------------
K_DATASET = "ds"                  # Dataset | None
K_DATA_SOURCE = "data_source"     # "builtin" | "upload"
K_MODEL_CFG = "model_cfg"         # dict，用户在界面上调的超参
K_ARTIFACTS = "artifacts"         # dict：mmm/idata/X/y/meta/used_ctrl/diag/...
K_ARTIFACT_KEY = "artifact_key"   # 生成 artifacts 时所用的缓存键
K_OPT_RESULT = "opt_result"       # dict：最优分配、边际回报、情景、弹性
K_OPT_KEY = "opt_key"             # str：生成 opt_result 时所用的结果键
K_OPT_CFG = "opt_cfg"             # dict：用户在优化页设的业务参数
K_REPORT = "report"               # dict：报告文本与来源
K_RUN_LOG = "run_log"             # list[str]：本次会话的关键动作留痕
K_PRETRAIN_LOADED = "pretrain_loaded"   # bool：是否已尝试加载阶段2预训练后验
K_UI_THEME = "ui_theme_cache"

DEFAULTS: dict = {
    K_DATASET: None,
    K_DATA_SOURCE: "builtin",
    K_MODEL_CFG: {},
    K_ARTIFACTS: None,
    K_ARTIFACT_KEY: None,
    K_OPT_RESULT: None,
    K_OPT_KEY: None,
    K_OPT_CFG: {},
    K_REPORT: None,
    K_RUN_LOG: [],
    K_PRETRAIN_LOADED: False,
}


def init() -> None:
    """在每个页面开头调用一次；幂等。"""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = (v.copy() if isinstance(v, (dict, list)) else v)


# ------------------------------------------------------------
# 基础读写
# ------------------------------------------------------------
def get(key: str, default=None):
    return st.session_state.get(key, default if default is not None else DEFAULTS.get(key))


def set(key: str, value) -> None:
    st.session_state[key] = value


def log(msg: str) -> None:
    """记录本次会话的关键动作（用于"运行日志"面板，方便面试演示时回溯）。"""
    st.session_state.setdefault(K_RUN_LOG, [])
    st.session_state[K_RUN_LOG].append(msg)


def clear_log() -> None:
    st.session_state[K_RUN_LOG] = []


# ------------------------------------------------------------
# 依赖链：上游变更 → 级联清空下游
# ------------------------------------------------------------
def _clear_model() -> None:
    st.session_state[K_ARTIFACTS] = None
    st.session_state[K_ARTIFACT_KEY] = None


def _clear_optimization() -> None:
    st.session_state[K_OPT_RESULT] = None
    st.session_state[K_OPT_KEY] = None


def _clear_report() -> None:
    st.session_state[K_REPORT] = None


def invalidate_from(stage: str) -> None:
    """
    stage ∈ {"dataset", "model", "optimization"}，表示该环节的数据发生了变化，
    其所有下游结论都失效。
    """
    if stage == "dataset":
        _clear_model()
        _clear_optimization()
        _clear_report()
    elif stage == "model":
        _clear_optimization()
        _clear_report()
    elif stage == "optimization":
        _clear_report()
    else:
        raise ValueError(f"未知环节: {stage}")


def set_dataset(ds, *, source: str | None = None, rebuild: bool = True) -> None:
    """写入新数据集。若指纹变化则级联清空模型/优化/报告。"""
    old = st.session_state.get(K_DATASET)
    old_fp = getattr(old, "fingerprint", None)
    new_fp = getattr(ds, "fingerprint", None)
    st.session_state[K_DATASET] = ds
    if source:
        st.session_state[K_DATA_SOURCE] = source
    if rebuild and old_fp != new_fp:
        invalidate_from("dataset")
        log(f"载入数据集：{getattr(ds, 'source_label', '未命名')}"
            f"（{getattr(ds, 'n_rows', 0)} 行，指纹 {new_fp}）→ 已清空下游结果")


def get_dataset():
    return st.session_state.get(K_DATASET)


def has_dataset() -> bool:
    ds = st.session_state.get(K_DATASET)
    return ds is not None and getattr(ds, "n_rows", 0) > 0


def has_model() -> bool:
    return st.session_state.get(K_ARTIFACTS) is not None


def has_optimization() -> bool:
    return st.session_state.get(K_OPT_RESULT) is not None


# ------------------------------------------------------------
# 依赖链的显式守卫：页面在消费上游结果前先自检
# ------------------------------------------------------------
def _guide(title: str, why: str, steps: list[str], icon: str = "📂") -> None:
    """
    统一的「缺少前置条件」引导卡。

    刻意写得比一句「请先载入数据」更具体：说明**缺什么、去哪、点哪个按钮**。
    演示与首次使用时，最让人卡住的不是报错，而是「页面空着但不知道该干嘛」。
    """
    ui.empty_state(title, why, icon)
    st.markdown("**下一步**")
    for s in steps:
        st.markdown(f"- {s}")


def require_dataset() -> bool:
    """数据未就绪时给出统一引导，返回 False 让页面提前返回。"""
    if not has_dataset():
        _guide(
            "还没有载入数据",
            "本页的所有计算都基于你载入的那份数据。请先到左侧「① 开始 · 概览与数据」。",
            ["点「🎲 内置模拟数据」标签下的**载入内置数据** —— 最快，"
             "且数据带真值标注，可用于检验模型估得准不准",
             "或切到「📤 上传我的数据」上传 CSV / Excel —— "
             "列名不必严格一致，页内会逐条给出校验结论与修正建议"],
            icon="📂")
        return False
    return True


def require_model() -> bool:
    if not require_dataset():
        return False
    if not has_model():
        _guide(
            "还没有可用的模型结果",
            "预算优化与决策报告都建立在模型估计的响应曲线上。",
            ["先去左侧「② 分析 · 建模与诊断」页",
             "在该页右侧选择一种方式：**复用阶段 2 预训练后验**（秒级）"
             "或**在本页拟合**（首次约 10~20 分钟，之后命中缓存）",
             "拟合完成后回到本页即可"],
            icon="🔬")
        return False
    return True
