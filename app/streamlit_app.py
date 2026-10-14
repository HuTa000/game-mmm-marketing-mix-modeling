"""
应用入口（Streamlit 多页路由）。

启动方式（在项目根目录执行）：
    streamlit run app/streamlit_app.py

架构说明：
    采用 `st.navigation` + `st.Page`（Streamlit 1.36+ 的现代多页方案），
    而不是传统的 `pages/` 目录自动发现。原因有三：
    ① 页面标题/图标可自由定义，不必受文件名约束；
    ② 能做分组（开始 / 分析 / 交付），导航结构即业务结构；
    ③ 页面脚本与入口在同一进程、同一 session 中运行，
       共享 st.session_state —— 这是"切换页面不重跑建模"的前提。

    侧栏的品牌区与状态灯定义在本文件，对**所有页面生效**，
    这样每个页面只需关心自己的内容，不必重复搭建框架。
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# ---- 保证 app/ 在 sys.path 上，使各页面能 `from core import ...` ----
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from core import paths, state, ui, cache          # noqa: E402

paths.ensure_src_on_path()

# ---- 页面级配置（必须是第一个 Streamlit 调用）----
st.set_page_config(
    page_title="游戏出海广告效果 · 贝叶斯 MMM",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject()
state.init()

# ============================================================
# 路由定义
# ============================================================
PAGES = {
    "① 开始": [
        st.Page("views/home.py", title="概览与数据", icon="🏠", default=True),
    ],
    "② 分析": [
        st.Page("views/data_eda.py", title="探索性分析", icon="📊"),
        st.Page("views/modeling.py", title="建模与诊断", icon="🔬"),
        st.Page("views/optimize.py", title="预算优化", icon="💰"),
    ],
    "③ 交付": [
        st.Page("views/report.py", title="决策报告", icon="📝"),
    ],
}
nav = st.navigation(PAGES)

# ============================================================
# 侧栏：品牌 + 全局状态（对所有页面生效）
# ============================================================
with st.sidebar:
    ui.sidebar_brand()

    ds = state.get_dataset()
    st.markdown("**当前状态**")
    ui.status_pills([
        ("数据已载入", state.has_dataset()),
        ("模型已拟合", state.has_model()),
        ("优化已完成", state.has_optimization()),
        ("报告已生成", state.get(state.K_REPORT) is not None),
    ])
    if ds is not None:
        st.caption(f"数据源：{getattr(ds, 'source_label', '—')}")
        st.caption(f"规模：{getattr(ds, 'n_rows', 0)} 行 · "
                   f"{len(getattr(ds, 'channel_cols', []))} 渠道")

    with st.expander("阶段产物体检", expanded=False):
        rows = paths.artifact_status()
        n_ok = sum(1 for r in rows if r["exists"])
        st.caption(f"已就绪 {n_ok}/{len(rows)} 项")
        for r in rows:
            st.markdown(f"{'🟢' if r['exists'] else '⚪'} {r['stage']} · {r['name']}"
                        + (f" `{r['size_mb']:.1f}MB`" if r["exists"] else ""))

    with st.expander("拟合缓存", expanded=False):
        cs = cache.cache_status()
        st.caption(f"磁盘拟合缓存：{cs['entries']} 份 / {cs['total_mb']} MB")
        st.caption(f"阶段 2 预训练后验：{'可用' if cs['pretrained'] else '不可用'}"
                   f"（{cs['pretrained_mb']} MB）")
        if cs["entries"] and st.button("清空拟合缓存", **ui.FULL_WIDTH):
            n = cache.clear_fit_cache()
            st.success(f"已清理 {n} 份缓存")
            st.rerun()

    with st.expander("本次会话操作记录", expanded=False):
        logs = state.get(state.K_RUN_LOG) or []
        if logs:
            for i, m in enumerate(logs[-12:], 1):
                st.caption(f"{i}. {m}")
        else:
            st.caption("暂无记录")

    st.divider()
    ui.reset_button()
    st.caption("数据说明：内置模拟数据带真值标注，可用于验证模型；"
               "上传真实数据时，绝对金额仅供同一模型内相对比较。")

# ---- 运行选中的页面 ----
nav.run()
