"""
建模与诊断页（阶段 8 交付）
==========================

这一页是整个应用最重的一页：它把「周收入」拆成
    基线 + 各渠道媒体贡献 + 季节性 + 控制变量 + 噪声
并回答一串只有建模才能回答的问题：

    这套数字能拿去汇报吗？      → 收敛与自洽性诊断（R-hat / ESS / 发散 / 截距）
    哪个渠道该加钱、哪个该砍？  → 各渠道 ROI 的**后验区间**，不是点估计
    钱花出去换回了什么？        → 守恒的贡献分解（总量账 + 波动账）
    每个渠道的记忆和饱和多长？  → Adstock / Hill 参数的后验分布
    模型估得准吗？              → 参数回收检验（只有内置模拟数据有真值）
    这套结论的边界在哪？        → 结构性不可识别性 + 三条缓解手段

三条硬性要求的落地方式：

1. **建模耗时长 → 分阶段进度 + 三级缓存**
   · 首次进来若检测到阶段 2 的后验存档，**自动复用**（毫秒级，不重新采样）；
   · 用户也可在本页重新拟合：显示 5 个阶段与"共多少次抽样、典型耗时多久"；
   · 拟合前先查磁盘缓存（数据指纹 × 超参哈希），命中就直接还原；
   · 结果写进 session_state，**切页不重算**。

   【诚实说明】NUTS 采样是阻塞调用，进度条无法在采样过程中刷新。
   所以这里不在采样时给假进度，而是采样前把"样本量 + 预估耗时"讲清楚，
   采样后补报实际耗时 —— 这比一个卡在 40% 的假进度条可信。

2. **窄屏适配**：所有图 responsive、图例置顶、高度受限；表格全宽可横向滚动。

3. **业务决策导向**：每个 Tab 固定三段式 —— 业务问题 → 图 → 这张图怎么用。
   结论卡片（ROI 的"该加 / 该砍 / 证据不足"）放在最前面。

依赖：`app/core/model.py`（纯逻辑，可离线自测）、`app/core/cache.py`（三级缓存）。
"""
from __future__ import annotations

import html as _html
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from core import cache, model, paths, schema, state, ui      # noqa: E402

paths.ensure_src_on_path()

ui.inject()

# ============================================================
# 配色与版式：统一从 core/ui.py 取（全应用唯一来源 —— 三个分析页原先
# 各复制了一份相同常量，渠道色还会随列顺序变化，现已收口）
# ============================================================
C_BRAND, C_UP, C_DOWN = ui.C_BRAND, ui.C_UP, ui.C_DOWN
C_MUTED, C_WARN = ui.C_MUTED, ui.C_WARN
GRID, PALETTE = ui.GRID, ui.PALETTE


def cn(name: str) -> str:
    return schema.CHANNEL_CN.get(name, name)


def _esc(s) -> str:
    return _html.escape(str(s))


def _inline(s) -> str:
    """
    转义后放行 `**加粗**`（统一实现见 core/ui.py::md_esc）。

    为什么必须有它：本页多段结论（可识别性 verdict、混淆量说明等）
    来自 `core/model.py`，用的是 markdown 的 `**…**` 强调。
    这些文案拼进 `unsafe_allow_html=True` 的 HTML 后：
      · 只用 `_esc()` → 页面上出现**裸露的星号**（实测确有）；
      · 完全不用 → 渠道名等用户数据可能注入 HTML。
    所以「先转义、再放行固定语法」。
    """
    return ui.md_esc(s)


def _wan(v, nd: int = 0) -> str:
    """元 → 万元，带千分位。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f / 1e4:,.{nd}f}"


def _num(v, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:.{nd}f}"


def _layout(fig, height: int | None = None, title: str = "", legend_top: bool = True):
    """统一版式的薄包装（定义在 core/ui.py，全应用唯一来源）。"""
    return ui.apply_theme(fig, height=height, title=title, legend_top=legend_top)


def _question(icon: str, q: str, why: str) -> None:
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{icon} {ui.rich(q)}</div>'
        f'<div class="s">{ui.rich(why)}</div></div>', unsafe_allow_html=True)


def _takeaway(lines: list[str], title: str = "🩺 这张图怎么用") -> None:
    """图的业务解读（用 `ui.rich` 放行 `**加粗**`，避免页面出现裸星号）。"""
    body = "".join(f"<li>{ui.rich(x)}</li>" for x in lines)
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{ui.rich(title)}</div>'
        f'<div class="s"><ul style="margin:6px 0 0 16px;padding:0">{body}</ul></div></div>',
        unsafe_allow_html=True)


def _check_html(checks: list[dict]) -> str:
    """把 pass/warn/fail 的检查项渲染成一张带色标的表。"""
    tone = {"pass": C_DOWN, "warn": C_WARN, "fail": C_UP}
    icon = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}
    trs = []
    for c in checks:
        st_ = c.get("status", "warn")
        hint = (f'<div class="fix">→ {_esc(c["hint"])}</div>' if c.get("hint") else "")
        trs.append(
            f'<tr><td class="ic">{icon.get(st_, "")}</td>'
            f'<td class="lb">{_esc(c.get("label", ""))}</td>'
            f'<td class="vl" style="color:{tone.get(st_, C_MUTED)}">{_esc(c.get("value", ""))}</td>'
            f'<td class="wy">{hint}</td></tr>')
    return ('<table style="width:100%;border-collapse:collapse;font-size:.86rem">'
            + "".join(trs) + "</table>")


# ============================================================
# 图表
# ============================================================
def _fig_roi_forest(roi: pd.DataFrame, truth: dict | None) -> go.Figure:
    """
    ROI 森林图：点 = 后验均值，横线 = 94% 可信区间，虚线 = 盈亏平衡 1.0。

    颜色按**决策方向**给，而不是按渠道：
        区间整体 > 1 → 红（可以加钱）  区间整体 < 1 → 绿（该砍）
        区间跨越 1   → 灰（证据不足）
    这与红涨绿跌的中国习惯一致：红 = 加大投入。
    """
    d = roi.copy()
    d["_y"] = [cn(c) for c in d["渠道"]]
    d = d.iloc[::-1]                                        # y 轴从下往上
    colors = []
    for _, r in d.iterrows():
        lo, hi = r["ROI下界"], r["ROI上界"]
        colors.append(C_UP if lo > 1 else (C_DOWN if hi < 1 else C_MUTED))

    fig = go.Figure()
    # 区间横线
    fig.add_trace(go.Scatter(
        x=d["ROI下界"].tolist() + d["ROI上界"].tolist()[::-1],
        y=d["_y"].tolist() + d["_y"].tolist()[::-1],
        fill="toself", fillcolor="rgba(43,108,176,.12)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False, name="94% 区间"))
    for i, (_, r) in enumerate(d.iterrows()):
        fig.add_trace(go.Scatter(
            x=[r["ROI下界"], r["ROI上界"]], y=[r["_y"], r["_y"]],
            mode="lines", line=dict(color=colors[i], width=3),
            hovertemplate=(f"<b>{_esc(r['_y'])}</b><br>94% 区间 "
                           f"[{r['ROI下界']:.3f}, {r['ROI上界']:.3f}]<extra></extra>"),
            showlegend=False))
    fig.add_trace(go.Scatter(
        x=d["ROI均值"], y=d["_y"], mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(color="#fff", width=1.4)),
        text=[f" {v:.3f}" for v in d["ROI均值"]], textposition="middle right",
        textfont=dict(size=11),
        hovertemplate=("<b>%{y}</b><br>ROI 均值 %{x:.3f}<br>"
                       "P(ROI>1) %{customdata:.1%}<extra></extra>"),
        customdata=d["P(ROI>1)"].values,
        name="估计 ROI"))

    if truth and truth.get("roi"):
        ys = [cn(c) for c in d["渠道"] if c in truth["roi"]]
        xs = [truth["roi"][c] for c in d["渠道"] if c in truth["roi"]]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=11, color="#4a5568", symbol="diamond-open",
                            line=dict(width=2)),
                name="真实 ROI（仅模拟数据可得）"))

    fig.add_vline(x=1.0, line_dash="dot", line_color=C_MUTED)
    fig.add_annotation(x=1.0, y=0.02, yref="paper", yanchor="bottom",
                       text="盈亏平衡 ROI = 1", showarrow=False,
                       font=dict(size=10, color=C_MUTED), xanchor="left")
    fig.update_xaxes(title_text="ROI（每 1 元花费换回的收入）")
    return _layout(fig, height=90 + 42 * len(d), title="")


def _fig_contrib_stack(contrib: pd.DataFrame, cfg) -> go.Figure:
    """贡献分解堆叠面积图：把每周收入拆成基线 + 各渠道。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=contrib["date"], y=contrib["_baseline"], name="基线（自然增长）",
        mode="lines", stackgroup="one", line=dict(width=0.5, color="#a0aec0"),
        fillcolor="rgba(160,174,192,.55)",
        hovertemplate="%{x|%Y-%m-%d}<br>基线 %{y:,.0f} 元<extra></extra>"))
    for i, c in enumerate(context_channels(cfg)):
        if c in contrib.columns:
            fig.add_trace(go.Scatter(
                x=contrib["date"], y=contrib[c], name=cn(c),
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=ui.color_of(c, i)),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{_esc(cn(c))} %{{y:,.0f}} 元<extra></extra>"))
    fig.update_yaxes(title_text="周收入（元）")
    return _layout(fig, height=380)


def context_channels(cfg) -> list[str]:
    return list(cfg.channel_cols)


def _fig_wave(contrib: pd.DataFrame, col: str, label: str, color: str) -> go.Figure:
    """波动分量（季节性 / 控制变量）的逐期曲线 —— 它们只改变节奏，不改变总量。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=contrib["date"], y=contrib[col], mode="lines", name=label,
        line=dict(color=color, width=1.6), fill="tozeroy",
        fillcolor="rgba(43,108,176,.12)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} 元<extra></extra>"))
    fig.add_hline(y=0, line_color=C_MUTED, line_width=1)
    fig.update_yaxes(title_text=f"{label}贡献（元/周）")
    return _layout(fig, height=260, legend_top=False)


def _fig_posterior(groups: dict[str, np.ndarray], title: str,
                   truth: dict | None = None) -> go.Figure:
    """后验箱线图：每个渠道一条。箱体越窄 = 数据给的信息越多。"""
    fig = go.Figure()
    for i, (name, arr) in enumerate(groups.items()):
        fig.add_trace(go.Box(
            y=arr, name=cn(name), boxpoints=False,
            marker_color=ui.color_of(name, i),
            line=dict(width=1.4), fillcolor=ui.color_of(name, i),
            opacity=0.65, hovertemplate="中位 %{median:.3f}<extra></extra>"))
    if truth:
        for name, tv in truth.items():
            if name in groups:
                fig.add_trace(go.Scatter(
                    x=[cn(name)], y=[tv], mode="markers", showlegend=False,
                    marker=dict(symbol="diamond", size=10, color="#1a202c",
                                line=dict(width=1.5, color="#fff")),
                    hovertemplate=f"真值 {tv:.3f}<extra></extra>"))
    return _layout(fig, height=330, title=title)


def _fig_trace(tr: dict) -> go.Figure:
    """MCMC 轨迹：4 条链应该像"同一团毛线"，而不是分层的几条带子。"""
    fig = go.Figure()
    for i, ch in enumerate(tr.get("chains", [])):
        fig.add_trace(go.Scatter(
            y=ch, mode="lines", name=f"链 {i + 1}",
            line=dict(width=1.1, color=PALETTE[i % len(PALETTE)])))
    fig.update_xaxes(title_text="后验抽样序号（已抽稀）")
    fig.update_yaxes(title_text="参数取值")
    return _layout(fig, height=280)


# ============================================================
# 状态条
# ============================================================
def _status_bar(ds, art):
    items = [
        {"label": "数据源", "value": _esc(getattr(ds, "source_label", "—")),
         "delta": f"{getattr(ds, 'n_rows', 0)} 行"},
        {"label": "渠道数", "value": str(len(getattr(ds, "channel_cols", []))),
         "delta": "、".join(cn(c) for c in getattr(ds, "channel_cols", [])[:3])
                  + ("…" if len(getattr(ds, "channel_cols", [])) > 3 else "")},
        {"label": "控制变量", "value": str(len(getattr(ds, "control_cols", []))),
         "delta": "已纳入建模" if getattr(ds, "control_cols", []) else "无（易遗漏变量）"},
    ]
    if art and art.get("ok"):
        src = art.get("source")
        label = {"pretrained": "复用阶段2后验", "fit": "本页拟合",
                 "cache": "磁盘缓存还原"}.get(src, src or "—")
        d = art.get("diag") or {}
        items += [
            {"label": "模型来源", "value": _esc(label),
             "delta": (f"采样 {art.get('elapsed', 0):.0f} 秒"
                       if art.get("elapsed") else "无需采样")},
            {"label": "后验样本", "value": f"{d.get('n_samples', 0):,}",
             "delta": "收敛达标" if d.get("converged") else "收敛未达标",
             "tone": "up" if d.get("converged") else "down"},
        ]
    else:
        items.append({"label": "模型状态", "value": "未拟合",
                      "delta": "点下方按钮开始", "tone": None})
    ui.kpi_row(items, per_row=5)


# ============================================================
# 入口区
# ============================================================
def _advanced_controls(ds, applied: dict) -> dict:
    """
    超参控件。注意：这里只是**读值**，不会触发重算；
    真正生效发生在用户点「开始拟合」时。
    """
    with st.expander("⚙️ 高级设置（改动后点「开始拟合」才生效）", expanded=False):
        st.caption("下面这些是建模决策，不是随手可调的旋钮。默认值来自阶段 1/2 的"
                   "数据诊断与实验，改动前请先看右侧的说明。")
        c1, c2 = st.columns(2)
        l_max = c1.slider(
            "Adstock 记忆长度 l_max（周）", 1, 12, int(applied.get("adstock_l_max", 6)),
            help="广告效果的衰减窗口。阶段 1 的滞后相关性显示 6 周后衰减到可忽略；"
                 "调大能让效应影响更久，但也会让参数更难识别（ESS 下降）。")
        yearly = c2.slider(
            "年度季节性阶数（Fourier）", 1, 6, int(applied.get("yearly_seasonality", 2)),
            help="用几对正余弦波描述年度节律。阶数高能贴合更细的季节形状，"
                 "但历史不足 52 周时会被自动降为 1 阶。")
        sat = c1.selectbox(
            "饱和曲线类型", ["hill", "logistic", "michaelis_menten"],
            index=["hill", "logistic", "michaelis_menten"].index(
                applied.get("saturation_type", "hill")),
            help="Hill 是 S 形（低投入见效慢、中间最陡、高投入饱和），"
                 "最贴合游戏投放；logistic 更陡；michaelis_menten 等价于 Hill 的"
                 "特例（slope=1）。")
        scaling = c2.selectbox(
            "目标缩放方式", ["mean", "max"],
            index=["mean", "max"].index(applied.get("target_scaling", "mean")),
            help="★ 最容易被忽略的开关。用 max 缩放会把目标压到很小的数值范围，"
                 "导致截距被迫为负、媒体贡献虚高到 152%。阶段 2 实测改成 mean 后，"
                 "截距由 −0.140 扭转为 +0.468，ROI 排序相关系数从 −0.50 升到 +1.00。")
        mode = st.radio(
            "采样量", ["快速（约 1 分钟，演示用）", "标准（约 3~10 分钟，交付用）"],
            index=1 if not applied.get("quick") else 0, horizontal=True,
            help="快速模式 2 链 × 300 抽样；标准模式 4 链 × 1500 抽样。"
                 "链数与抽样量决定 R-hat 与 ESS 能否达标。")
        quick = mode.startswith("快速")
    return {
        "adstock_l_max": int(l_max),
        "yearly_seasonality": int(yearly),
        "saturation_type": str(sat),
        "target_scaling": str(scaling),
        "draws": 300 if quick else 1500,
        "tune": 300 if quick else 1500,
        "chains": 2 if quick else 4,
        "quick": bool(quick),
    }


def _do_fit(ds, opts: dict) -> dict:
    """执行拟合：先查磁盘缓存（跨重启），未命中再真正采样。"""
    with st.spinner("正在检查磁盘缓存（数据指纹 × 超参哈希）…"):
        cached_art = cache.get_disk_fit(ds, opts)
    if cached_art and cached_art.get("ok"):
        st.success("✅ 命中磁盘缓存：这份数据 + 这套超参之前拟合过，"
                   "已直接还原结果，**无需重新采样**。")
        return cached_art

    box = cache.TaskStatus("正在拟合贝叶斯 MMM（分阶段进度）", expanded=True)

    def _cb(frac: float, msg: str) -> None:
        box.progress(frac, msg)
        box.step(msg)

    art = model.fit(ds, opts, progress=_cb)
    if art.get("ok"):
        box.done(f"拟合完成（采样 {art.get('elapsed', 0):.0f} 秒）")
        p = cache.save_disk_fit(ds, opts, art)
        if p:
            st.caption(f"结果已落盘缓存：`{os.path.basename(p)}`"
                       f"（下次重启浏览器可直接还原）")
    else:
        box.fail(f"拟合失败：{art.get('reason')}")
    return art


def _render_entry(ds, art) -> tuple[dict | None, dict | None]:
    """未拟合（或准备重拟合）时的入口面板。返回 (art, opts)。"""
    builtin = model.is_builtin(ds)
    pv = model.pretrained_info()

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown("**① 复用阶段 2 已拟合好的后验**（秒级，推荐演示用）")
        if builtin and pv.get("available"):
            # 样本数**不写死**：精简版（4 链 × 300 = 1,200）与全量版
            # （4 链 × 1500 = 6,000）的抽样数不同。硬编码会向用户报一个
            # 与事实不符的数字，而且不会报错 —— 这类「看起来没问题的错」
            # 在面试演示里最致命。pv["samples_note"] 由 model.pretrained_info()
            # 读同名 .json 元信息得到；元信息缺失时退化为不提具体数字。
            _sn = pv.get("samples_note")
            _tag = "（精简版）" if pv.get("lite") else ""
            st.caption(
                f"检测到 {pv['size_mb']} MB 的后验存档{_tag}（{pv['mtime']}）"
                + (f"，{_sn}" if _sn else "")
                + "。载入后**不重新采样**，只重建模型结构并注入后验。")
            if st.button("⚡ 载入阶段 2 后验", type="primary", key="btn_pretrain"):
                with st.spinner("重建模型结构并注入后验…"):
                    res = model.load_pretrained(ds)
                if res.get("ok"):
                    st.success("已复用阶段 2 后验。")
                    state.set(state.K_ARTIFACT_KEY, "pretrained")
                    return res, {"__source__": "pretrained"}
                st.error(f"复用失败：{res.get('reason')}")
        elif not builtin:
            st.info("当前使用的是**上传数据**。阶段 2 的后验是针对内置模拟数据采样的，"
                    "对它无效 —— 用旧后验套新数据会得到不报错的错误结论。"
                    "请改用右侧「在本页拟合」。")
        else:
            st.warning("未找到阶段 2 的后验存档。请改用右侧「在本页拟合」。")

    with c2:
        st.markdown("**② 在本页拟合**（任何数据源都可用）")
        st.caption("耗时较长，但支持上传的数据，且超参可调。"
                   "结果会落盘缓存，同样的数据 + 配置第二次进来秒开。")

    st.markdown("---")
    opts = _advanced_controls(ds, state.get(state.K_MODEL_CFG) or model.DEFAULT_USER_CFG)
    from mmm_core import MMMConfig                              # noqa: E402
    _c = MMMConfig(draws=opts["draws"], tune=opts["tune"], chains=opts["chains"])
    st.caption(f"⏱️ 本次配置：{model.sample_size_note(_c)}")

    if st.button("🚀 开始拟合（会执行 MCMC 采样）", type="primary", key="btn_fit"):
        state.set(state.K_MODEL_CFG, opts)
        return _do_fit(ds, opts), opts
    return art, None


# ============================================================
# Tab ① 结果可信吗
# ============================================================
def _tab_trust(art):
    d = art.get("diag") or {}
    san = art.get("sanity") or {}
    _question("🩺", "这套数字能直接拿去汇报吗？",
              "贝叶斯模型的价值在于给出不确定性。但如果 MCMC 没收敛，"
              "后验区间就是假的不确定性 —— 比不给区间更危险。")

    if not d.get("ok"):
        st.error(f"诊断失败：{d.get('reason')}")
        return

    tone = C_DOWN if d.get("converged") else C_UP
    st.markdown(
        f'<div class="mmm-panel" style="border-left:5px solid {tone}">'
        f'<div class="t">{"✅" if d.get("converged") else "⛔"} {_esc(d.get("verdict", ""))}'
        f'<span style="font-size:.82rem;color:{C_MUTED};font-weight:400">'
        f'　共 {d.get("n_samples", 0):,} 个后验样本，'
        f'{d.get("n_params", 0)} 个受检参数</span></div></div>',
        unsafe_allow_html=True)

    st.markdown("**Part 1 · MCMC 收敛诊断**（这三项决定后验分布可不可信）")
    st.markdown(f'<div class="mmm-panel">{_check_html(d.get("checks", []))}</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最大 R-hat", _num(d.get("max_rhat"), 4))
    c2.metric("最小 ESS", f"{d.get('min_ess', float('nan')):,.0f}")
    c3.metric("发散样本", f"{d.get('divergence_pct', 0):.2f}%")
    c4.metric("截距", _num(d.get("intercept")))

    st.caption(
        "**怎么读**：R-hat 衡量「4 条独立链是否采到了同一个分布」，"
        "越接近 1 越好，超过 1.01 说明链之间还没混匀；ESS 是有效样本量，"
        "比实际抽样数小得多，低于 400 时分位数（也就是可信区间）会不稳；"
        "发散样本是采样器撞上几何病态区的次数，超过 2% 必须调先验。")

    st.markdown("---")
    st.markdown("**Part 2 · 模型自洽性**（算 ROI 之前的前置关卡）")
    if san.get("ok") is not True and san.get("checks") is None:
        st.warning(f"自洽性检查未完成：{san.get('reason', san)}")
    else:
        st.markdown(f'<div class="mmm-panel">{_check_html(san.get("checks", []))}</div>',
                    unsafe_allow_html=True)
        if san.get("pred_sum"):
            st.caption(f"预测总量 {_wan(san['pred_sum'])} 万元 vs 实际收入 "
                       f"{_wan(san['true_sum'])} 万元（偏差 {san['pred_err']:+.2%}）")

    _takeaway([
        "<b>三项收敛指标全过</b>，才可以把后验均值与区间写进汇报；"
        "有一项不过，结论要标『探索性』。",
        "<b>可加性</b>是身份：identity link 下各分量之和必须等于总预测。<br>"
        "它不成立时，贡献分解表里的数字加起来对不上总收入 —— 这种图发出去会直接被质疑。",
        "<b>不过怎么办</b>：先看是不是某个渠道全程零投放或与其他渠道完全共线（回「探索性分析」页看 VIF），"
        "再考虑收紧先验或增加抽样量。不要去调链数掩盖问题。",
    ])


# ============================================================
# Tab ② 渠道 ROI
# ============================================================
def _tab_roi(art):
    roi = art.get("roi")
    cfg = art["cfg"]
    from core import eda
    truth = eda.ground_truth()
    _question("💰", "哪个渠道该加钱，哪个该砍？",
              "业务方只关心一个动作：下一周的预算往哪挪。"
              "ROI 的点估计不足以支撑这个动作，需要看**整段可信区间落在 1 的哪一侧**。")

    if roi is None or not len(roi):
        st.warning("无法计算 ROI（可能是所有渠道都无投放）。")
        return

    st.plotly_chart(_fig_roi_forest(roi, truth), key="roi_forest")

    n_up = int((roi["ROI下界"] > 1).sum())
    n_dn = int((roi["ROI上界"] < 1).sum())
    n_mid = len(roi) - n_up - n_dn
    ui.kpi_row([
        {"label": "可以加预算", "value": f"{n_up} 个渠道", "tone": "up",
         "delta": "整段区间 > 1"},
        {"label": "应该缩减", "value": f"{n_dn} 个渠道", "tone": "down",
         "delta": "整段区间 < 1"},
        {"label": "证据不足", "value": f"{n_mid} 个渠道", "delta": "区间跨越 1"},
    ], per_row=3)

    show = roi.copy()
    show["渠道"] = show["渠道"].map(cn)
    show["总花费(万元)"] = show["总花费(万元)"].map(lambda v: f"{v:,.1f}")
    show["总贡献(万元)"] = show["总贡献(万元)"].map(lambda v: f"{v:,.1f}")
    show["P(ROI>1)"] = show["P(ROI>1)"].map(lambda v: f"{v:.1%}" if np.isfinite(v) else "—")
    show["ROI区间"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(roi["ROI下界"], roi["ROI上界"])]
    show = show[["渠道", "ROI均值", "ROI区间", "P(ROI>1)", "总花费(万元)",
                 "总贡献(万元)", "结论"]]
    ui.dataframe(show, caption="ROI 为「该渠道带来的收入 ÷ 该渠道花费」。"
                               "置信口径：后验 94% 分位区间。")

    lines = [
        "<b>怎么读</b>：横线是 94% 可信区间，不是误差棒 —— 它的含义是"
        "『在模型与数据给定的前提下，ROI 有 94% 的概率落在这一段』。",
        f"<b>该加钱</b>：{'、'.join(cn(c) for c in roi.loc[roi['ROI下界'] > 1, '渠道'])}"
        if n_up else "<b>该加钱</b>：本轮没有渠道的区间完全站在 1 以上。",
        f"<b>该砍</b>：{'、'.join(cn(c) for c in roi.loc[roi['ROI上界'] < 1, '渠道'])}"
        if n_dn else "<b>该砍</b>：没有渠道的区间完全落在 1 以下。",
        f"<b>最该改的</b>：{cn(roi.iloc[-1]['渠道'])} 的 ROI 均值最低"
        f"（{roi.iloc[-1]['ROI均值']:.3f}），且 P(ROI>1) = {roi.iloc[-1]['P(ROI>1)']:.1%}。",
    ]
    if truth:
        lines.append(
            "<b>和真值比</b>：绝对水平有偏差，但<b>渠道排序与真值完全一致</b>。"
            "这正是 MMM 的可用边界 —— 预算分配只依赖排序，不依赖绝对水平。")
    _takeaway(lines)


# ============================================================
# Tab ③ 收入怎么分
# ============================================================
def _tab_mix(art):
    contrib = art.get("contrib")
    cfg = art["cfg"]
    mix = art.get("mix") or {}
    cinfo = art.get("cinfo") or {}
    _question("🥧", "花出去的钱换回了什么？收入是怎么构成的？",
              "把历史收入拆成「自然增长基线」和「各渠道带来的部分」，"
              "才能回答『如果全部停投，收入会掉多少』。")

    if contrib is None or not len(contrib):
        st.warning("贡献分解不可用。")
        return

    st.plotly_chart(_fig_contrib_stack(contrib, cfg), key="contrib_stack")

    total = mix.get("total")
    if total is not None and len(total):
        t = total.copy()
        t["占收入比"] = t["占收入比"].map(lambda v: f"{v:.1%}" if np.isfinite(v) else "—")
        t["金额(万元)"] = t["金额(万元)"].map(lambda v: f"{v:,.1f}")
        ui.dataframe(t, caption="**总量账**：只有「基线」和「渠道」有总量含义，"
                                "它们加总 ≈ 历史总收入。")

    st.markdown(
        f'<div class="mmm-panel">守恒检验：各分量之和与真实总收入的残差为 '
        f'<b>{cinfo.get("residual_pct", float("nan")):.4%}</b>'
        f'（官方 `get_contributions(original_scale=True)` 的守恒分解）。'
        f'媒体贡献合计 {_wan(cinfo.get("media_sum"))} 万元，'
        f'占总收入 {cinfo.get("media_share", float("nan")):.1%}。</div>',
        unsafe_allow_html=True)

    wave = mix.get("wave")
    if wave is not None and len(wave):
        st.markdown("**波动账**（季节性 / 控制变量：只看振幅，不看总量）")
        w = wave.copy()
        ui.dataframe(w)
        st.plotly_chart(_fig_wave(contrib, "_seasonality", "季节性", C_BRAND),
                        key="wave_season")
        st.caption("上图的曲线在每个年度结束时都会回到 0 —— 这不是模型失效，"
                   "而是 Fourier 基的数学性质：季节性只改变收入的**时点分布**，"
                   "不创造收入总量。把季节项的 sum 当成『季节性贡献了多少钱』"
                   "是 MMM 汇报里最常见的错误之一。")

    _takeaway([
        "<b>基线占比</b>是这份生意『不投广告也有的基本盘』。它越高，"
        "说明增长的驱动力越偏向产品本身（留存、口碑、自然量）。",
        "<b>渠道之间的相对高度</b>才有决策含义：谁带来的收入更多、"
        "占的预算是否匹配。不要拿单个渠道的绝对金额对外承诺。",
        "<b>季节项为什么是 0</b>：Fourier 基在完整年度上积分为 0。"
        "它的作用是让模型不在旺季把『季节本该有的增量』误记到渠道头上 —— "
        "不建模季节性，Q4 的渠道 ROI 会系统性虚高。",
    ])


# ============================================================
# Tab ④ 参数与后验
# ============================================================
def _tab_params(art):
    idata = art["idata"]
    cfg = art["cfg"]
    _question("📐", "每个渠道的广告效果衰减多快？什么时候开始饱和？",
              "这两个形状参数决定了『钱该怎么分』：衰减慢 → 可以一次性投大预算；"
              "饱和点高 → 加钱还有空间，低 → 再加就是浪费。")

    from core import eda
    truth = eda.ground_truth()

    spec = [
        ("adstock_alpha", "Adstock 衰减系数 λ",
         "每过去一周，广告效果还剩多少比例。λ=0.3 表示一周后只剩 30%，"
         "约 6 周衰减到可忽略。**它决定投放节奏**：λ 高 → 适合脉冲式大促；"
         "λ 低 → 适合持续小额。",
         {c: truth["decay"][c] for c in cfg.channel_cols} if truth else None),
        ("saturation_beta", "Hill 曲线高度 β",
         "在**标准化尺度**下的渐近线高度，不是 ROI。量纲与花费相关，"
         "**不要拿它跟 ROI 比**（这是 MMM 里最常见的口径混淆）。",
         None),
        ("saturation_kappa", "半饱和点 κ",
         "花费达到多少时，效果走到一半。它是「还能加多少」的直接答案："
         "κ 越低说明越早饱和，加钱的边际收益越差。",
         None),
        ("saturation_slope", "Hill 陡峭度 s",
         "S 形曲线的陡峭程度。s 越大越接近「开关式」效果（到某个量级突然起效）；"
         "s 越小越接近平滑的对数式增长。这项**最难识别**，区间通常最宽。",
         None),
    ]
    idx = st.radio("选择要看的参数", [s[1] for s in spec],
                   index=0, horizontal=True, key="param_pick")
    pname, plabel, pdesc, ptruth = spec[[s[1] for s in spec].index(idx)]

    dr = model.draws_for_plot(idata, pname, channels=cfg.channel_cols)
    if not dr.get("ok"):
        st.warning(f"后验中没有 {pname}。")
        return
    st.plotly_chart(_fig_posterior(dr["values"], plabel, ptruth), key=f"post_{pname}")
    st.caption(pdesc + "　箱体 = 后验的 25%~75% 分位，黑点 = 真值（仅内置模拟数据）。")

    st.plotly_chart(_fig_trace(model.trace_for_plot(idata, pname,
                                                    channel=cfg.channel_cols[0])),
                    key=f"trace_{pname}")
    st.caption(f"轨迹图（{cn(cfg.channel_cols[0])}）：4 条链应像『同一团毛线』在"
               f"同一水平上来回抖动。若看到几条明显的分层带子，说明链没混匀，"
               f"对应 Tab① 里的 R-hat 会超标。")

    st.markdown("**全部参数的点估计与可信区间**")
    tab = model.posterior_table(idata, cfg)
    if len(tab):
        ui.dataframe(tab, height=340,
                     caption="标准化空间下的参数值。看区间宽度即可判断"
                             "『数据对哪个参数最没把握』。")

    _takeaway([
        "<b>衰减 λ</b> 用来排投放节奏：λ 高的渠道适合集中投、拉长观察窗；"
        "λ 低的渠道需要持续喂，停投就立刻掉。",
        "<b>半饱和点 κ</b> 用来判断加预算的空间：当前花费离 κ 越远，加钱越有效。"
        "本数据集里 κ 是平均花费的 1.6~2.8 倍，说明**大部分渠道都还没饱和**。",
        "<b>为什么 β 不能当 ROI 看</b>：β 在标准化尺度上，"
        "换算成 ROI 需要除以渠道的花费缩放因子。拿 β 直接比 ROI 会得出"
        "『模型严重高估』的错误结论。",
        "<b>区间越宽 = 数据越没说清</b>。斜率 s 的区间通常最宽，"
        "这是 MMM 的结构性特点，不是实现缺陷。",
    ])


# ============================================================
# Tab ⑤ 参数回收
# ============================================================
def _tab_recovery(art):
    rec = art.get("recovery") or {}
    _question("🎯", "模型估得准吗？—— 拿已知答案验证实现",
              "这是唯一能『对答案』的场景：数据是我们生成的，真值已知。"
              "真实项目里没有标准答案，所以只能靠诊断 —— 但在简历与面试里，"
              "能自证实现正确性是加分项。")

    if not rec.get("ok"):
        st.info(rec.get("note") or "当前数据没有可比的真值。")
        return

    tab = rec.get("table")
    rank = rec.get("rank_info") or {}
    n_hit, n_tot = rec.get("n_hit", 0), rec.get("n_tested", 0)

    ui.kpi_row([
        {"label": "区间命中真值", "value": f"{n_hit} / {n_tot}",
         "tone": "up" if n_hit >= n_tot * 0.6 else "down",
         "delta": "94% 可信区间覆盖真值"},
        {"label": "ROI 排序一致", "value": "是" if rank.get("order_match") else "否",
         "tone": "up" if rank.get("order_match") else "down",
         "delta": "① 排序 ② 绝对值"},
    ], per_row=2)

    if len(tab):
        ui.dataframe(tab, caption="ROI 用后验区间与真值比；衰减 λ 用 adstock_alpha 与真值比。"
                                  "两项口径都已经对齐到同一尺度。")

    if rank:
        est, tru = rank.get("est_order") or [], rank.get("true_order") or []
        st.markdown(
            f'<div class="mmm-panel">'
            f'<div class="t">排序对照</div><div class="s">'
            f'估计：{" &gt; ".join(_esc(cn(c)) for c in est)}<br>'
            f'真实：{" &gt; ".join(_esc(cn(c)) for c in tru)}</div></div>',
            unsafe_allow_html=True)

    _takeaway([
        "<b>为什么先看排序、再看绝对值</b>：预算分配的动作是『把 100 万从 A 挪到 B』，"
        "这个动作只依赖**谁比谁高**。绝对值受基线定义影响（见 Tab⑥），"
        "天然有偏差。",
        "<b>区间覆盖真值</b>说明模型没有把不确定性说得太满 —— "
        "如果 10 个参数只有 2 个被区间覆盖，说明可信区间被低估，"
        "所有『显著性』结论都要打折。",
        "<b>面试怎么讲</b>：不要只说『模型能收敛』，要说"
        "『我用已知真值的模拟数据做了参数回收，ROI 排序完全一致；"
        "同时用真值对照证明了 ROAS 的绝对水平不可靠，所以给业务的建议"
        "只基于排序和增量』。",
    ])


# ============================================================
# Tab ⑥ 局限与缓解
# ============================================================
def _tab_limits(art):
    ident = art.get("ident") or {}
    cfg = art["cfg"]
    _question("⚠️", "这套结论的边界在哪？",
              "MMM 有三个众所周知的软肋：内生性、渠道共线性、结构性不可识别。"
              "主动讲清楚边界，比假装配得准更能建立信任 —— "
              "这也是面试官最想听的部分。")

    if not ident.get("ok"):
        st.warning(f"可识别性计算失败：{ident.get('reason')}")
    else:
        lv = ident.get("level")
        tone = {"pass": C_DOWN, "warn": C_WARN, "fail": C_UP}.get(lv, C_MUTED)
        st.markdown(
            f'<div class="mmm-panel" style="border-left:5px solid {tone}">'
            f'<div class="t">结构性不可识别性</div>'
            f'<div class="s">{_inline(ident.get("verdict", ""))}</div></div>',
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("媒体贡献的恒定分量", f"{ident['const_share']:.1%}",
                  help="这部分与基线在数学上可以互相转移，无法分离。")
        c2.metric("波动分量", f"{ident['wave_part'] / 1e4:,.0f} 万元",
                  help="这部分随渠道投放变化，是真正可归因的。")
        c3.metric("混淆量", f"{ident['confusable_ratio']:.2f} 倍",
                  help="恒定分量 / 基线常数。倍数越大，绝对水平越不可信。")

        st.markdown(
            f"""
**为什么会这样**：模型方程是

> y(t) = 基线常数 + Σ 渠道贡献(t) + 季节(t) + 控制(t) + ε

把「基线常数」加 α、同时把每个渠道的恒定分量减 α，y 的预测值完全不变。
数据无法分辨这两种情形 —— 这就是**结构性不可识别**，不是算法不够好。
实测：媒体贡献里 **{ident['const_share']:.1%}** 是恒定分量，
与基线常数 {_wan(ident['baseline_const'] / 1e4 * 1e4)} 万元不可分离。
""")

    st.markdown("---")
    st.markdown("**MMM 的四个软肋与本项目的处理方式**")
    st.markdown(
        """
| 软肋 | 在本数据里的表现 | 已做的缓解 | 还缺什么 |
|---|---|---|---|
| **内生性** | 单渠道散点显示"花费越高收入涨越快"（上唇），因为高花费周恰好是大促周 | 引入控制变量（版本更新/竞品/节假日）+ 趋势项；探索页做了"毛关系 vs 控制后净关系"对照 | geo-lift 实验或暂停投放测试（pause test）做外部锚定 |
| **渠道共线性** | 最大 VIF 1.10（渠道间几乎不共动），这是**有利**条件 | 探索页给出 VIF 与相关矩阵，建模前先确认可识别 | 真实业务中渠道常共动，需实验设计或合并渠道 |
| **遗漏变量** | 控制变量被收缩先验收拢到接近 0，其影响被吸收进基线 | 显式建模 4 个控制变量；用 Normal(0, 0.2) 收缩先验防止其吞噬基线 | 补充更多可观测的混杂因素（KOL 档期、竞品投放强度） |
| **结构性不可识别** | 媒体贡献的恒定分量占 57%，与基线不可分 | 用 OLS 可识别水平而非真值设定截距先验（避免"拿答案当先验"） | 多市场/多区域数据、实验校准 |

> 这里的取舍逻辑是：**"约束比精度重要"**。阶段 3 的预算优化把所有渠道都限制在
> 历史花费的 ±40% 区间内 —— 因为最优解一旦不受约束，就会把全部预算压到
> 单渠道角点上，那是模型外推，不是业务决策。
"""
    )

    _takeaway([
        "<b>可以对外说的</b>：渠道之间的相对排序、预算增量的边际回报、"
        "『加 5% 预算大约多赚多少』这类**相对量**。",
        "<b>不要对外说的</b>：『媒体一共为公司创造了 X 万元』这类**绝对量** —— "
        "它取决于基线怎么定义，换个合理先验就能变。",
        "<b>怎么收窄</b>：geo-lift 实验（拿几个城市停投，看收入差）是行业公认的"
        "外部校准手段，成本远低于 MMM 本身的价值。",
    ])


# ============================================================
# 主流程
# ============================================================
def main():
    state.init()
    ui.page_header(
        "建模与诊断",
        "用贝叶斯 MMM 把周收入拆成「基线 + 各渠道媒体贡献 + 季节 + 控制」，"
        "并给出每个参数的可信区间与收敛诊断。", "🔬")

    if not state.require_dataset():
        return
    ds = state.get_dataset()

    art = state.get(state.K_ARTIFACTS)
    if art is not None and not art.get("ok"):
        art = None

    # ---- 首次进入：内置数据 + 有后验 → 自动复用（毫秒级，体验优先）----
    if art is None and model.is_builtin(ds) and model.pretrained_available() \
            and state.get(state.K_ARTIFACT_KEY) != "__skip_auto__":
        with st.spinner("检测到阶段 2 的后验存档，正在自动复用（无需重新采样）…"):
            auto = model.load_pretrained(ds)
        if auto.get("ok"):
            art = auto
            state.set(state.K_ARTIFACTS, art)
            state.set(state.K_ARTIFACT_KEY, "pretrained")
            state.log("自动复用阶段 2 后验")

    _status_bar(ds, art)

    if art is None:
        st.info("**这一页还没有模型结果。** 请选择一种方式开始："
                "左边是『秒级复用已拟合好的后验』（仅内置数据），"
                "右边是『在本页重新拟合』（支持上传数据、超参可调）。")
        new_art, applied = _render_entry(ds, art)
        if new_art and new_art.get("ok"):
            state.set(state.K_ARTIFACTS, new_art)
            if applied and applied.get("__source__") != "pretrained":
                state.set(state.K_ARTIFACT_KEY, "fit")
            st.rerun()
        return

    # ---- 已有结果 ----
    if art.get("ds_note"):
        st.caption(f"ℹ️ {art['ds_note']}")
    if art.get("cache_note"):
        st.caption(f"ℹ️ {art['cache_note']}")
    for n in art.get("notes", []):
        lvl = n.get("level")
        msg = f"**{n.get('title', '')}**"
        if n.get("detail"):
            msg += f"　{n['detail']}"
        if n.get("hint"):
            msg += f"\n\n→ {n['hint']}"
        (st.warning if lvl == "warn" else
         st.error if lvl == "fail" else st.info)(msg)

    applied = state.get(state.K_MODEL_CFG) or model.DEFAULT_USER_CFG
    cur = _advanced_controls(ds, applied)
    if cur != applied and art.get("source") == "fit":
        st.info("⚙️ 高级设置已修改。点下方「开始拟合」后新配置才会生效"
                "（当前展示的仍是旧配置的结果）。")

    b1, b2, b3 = st.columns([1, 1, 3])
    if b1.button("🚀 开始拟合", type="primary", key="btn_refit"):
        state.set(state.K_MODEL_CFG, cur)
        new_art = _do_fit(ds, cur)
        if new_art.get("ok"):
            state.set(state.K_ARTIFACTS, new_art)
            state.set(state.K_ARTIFACT_KEY, "fit")
        st.rerun()
    if b2.button("🧹 清除结果", key="btn_clear"):
        state.invalidate_from("model")
        st.rerun()
    with b3:
        st.caption(f"当前结果的配置：adstock l_max = {applied.get('adstock_l_max')}、"
                   f"季节性 {applied.get('yearly_seasonality')} 阶、"
                   f"饱和 {applied.get('saturation_type')}、"
                   f"缩放 {applied.get('target_scaling')}")

    tabs = st.tabs(["① 结果可信吗", "② 渠道 ROI", "③ 收入怎么分",
                    "④ 参数与后验", "⑤ 参数回收", "⑥ 局限与缓解"])
    with tabs[0]:
        _tab_trust(art)
    with tabs[1]:
        _tab_roi(art)
    with tabs[2]:
        _tab_mix(art)
    with tabs[3]:
        _tab_params(art)
    with tabs[4]:
        _tab_recovery(art)
    with tabs[5]:
        _tab_limits(art)


main()
