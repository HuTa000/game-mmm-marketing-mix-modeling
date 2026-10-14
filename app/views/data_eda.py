"""
探索性分析页（阶段 7 交付）
===========================

这一页要回答一个问题链：

    收入为什么忽高忽低？ → 有多少波动是规律性的？
    → 5 个渠道能被分开归因吗？ → 广告几周后见效？
    → 加钱还有用吗？ → 综合起来：**这份数据到底支不支持做归因？**

设计上的四个刻意的选择：

1. **全部图表现算现画（Plotly 交互式）**，而不是内嵌阶段 1 的静态 PNG。
   原因很实在：静态图只对应内置数据，用户上传自己的 CSV 后完全失效。
   现算现画则两个数据源都覆盖，还能缩放/悬浮看数值。

2. **结论卡片放在最上面**。业务方关心"能不能开工"，而不是先看八张图。
   论证过程放在下面，想看细节的人往下翻。

3. **每个区块固定三段式**：业务问题 → 图 → 这张图告诉我们什么（含该怎么用）。
   每张图回答一个业务问题，不罗列统计量。

4. **饱和那一段做"毛关系 vs 控制后净关系"的双视角对照**。
   单渠道散点在这份数据上呈"上凸"（看起来是规模效应），控制其他因素后才
   露出真实形态 —— 这是"内生性"最直观的演示，也是本页最有说服力的内容。

依赖：`app/core/eda.py`（纯算法，可离线自测）、`app/core/cache.py`（结果缓存）。
"""
from __future__ import annotations

import html as _html
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from core import cache, paths, schema, state, ui      # noqa: E402

paths.ensure_src_on_path()

ui.inject()

# ============================================================
# 配色与版式：统一从 core/ui.py 取（全应用唯一来源）
# ============================================================
C_REV, C_SPEND = ui.C_REV, ui.C_SPEND
C_NET, C_RAW = ui.C_NET, ui.C_RAW
C_WARN = ui.C_WARN
C_OK, C_BAD = ui.C_DOWN, ui.C_UP
GRID = ui.GRID


def cn(name: str) -> str:
    return schema.CHANNEL_CN.get(name, name)


def _esc(s) -> str:
    return _html.escape(str(s))


def _inline(s) -> str:
    """
    转义后放行 `**加粗**`（统一实现见 core/ui.py::md_esc）。

    为什么必须有它：`core/eda.py` 生成的 verdict / summary 全部用
    markdown 的 `**…**` 做强调（该文件里 0 个 `<b>`、29 处 `**`）。
    这些文案被拼进 `unsafe_allow_html=True` 的 HTML 里，
    若只做 `_esc()`，用户看到的就是**裸露的星号**（实测确有此问题）；
    若完全不转义，上传数据里的渠道名又可能注入 HTML。
    所以「先转义、再放行固定语法」。
    """
    return ui.md_esc(s)


def _vif_txt(v) -> str:
    """VIF 显示：∞ 表示完全共线（比任何数字都严重），NaN 表示该渠道无变异。"""
    try:
        v = float(v)
    except Exception:                                          # noqa: BLE001
        return "—"
    if v == float("inf"):
        return "∞"
    return "—" if not np.isfinite(v) else f"{v:.2f}"


def _infer_period(ds) -> tuple[int, str]:
    """按数据粒度推断季节周期（周数据 = 52）。"""
    try:
        gran = float(getattr(ds, "granularity_days", 7.0) or 7.0)
    except Exception:                                          # noqa: BLE001
        gran = 7.0
    if gran <= 10:
        return 52, "周"
    if gran <= 20:
        return 26, "半月"
    if gran <= 45:
        return 12, "月"
    return 4, "季"


def _layout(fig, height: int | None = None, legend_top: bool = True, title: str = ""):
    """统一版式的薄包装（定义在 core/ui.py，全应用唯一来源）。"""
    return ui.apply_theme(fig, height=height, title=title, legend_top=legend_top)


def _question(icon: str, q: str, why: str) -> None:
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{icon} {ui.rich(q)}</div>'
        f'<div class="s">{ui.rich(why)}</div></div>', unsafe_allow_html=True)


def _takeaway(lines: list[str], title: str = "🩺 这张图告诉我们什么") -> None:
    """图的业务解读（用 `ui.rich` 放行 `**加粗**`，避免页面出现裸星号）。"""
    body = "".join(f"<li>{ui.rich(x)}</li>" for x in lines)
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{ui.rich(title)}</div>'
        f'<div class="s"><ul style="margin:6px 0 0 16px;padding:0">{body}</ul></div></div>',
        unsafe_allow_html=True)


# ============================================================
# 结论卡片
# ============================================================
def _readiness_html(rr: dict) -> str:
    tone = {"pass": C_OK, "warn": C_WARN, "fail": C_BAD}
    icon = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}
    trs = []
    for c in rr["checks"]:
        fix = (f'<div class="fix">→ {_inline(c["fix"])}</div>' if c.get("fix") else "")
        trs.append(
            f'<tr><td class="ic">{icon.get(c["status"], "")}</td>'
            f'<td class="lb">{_inline(c["label"])}</td>'
            f'<td class="vl" style="color:{tone.get(c["status"], "#718096")}">{_inline(c["value"])}</td>'
            f'<td class="wy">{_inline(c["why"])}{fix}</td></tr>')
    return (
        f'<div class="mmm-panel" style="border-left:5px solid {tone.get(rr["level"], "#718096")}">'
        f'<div class="t">{icon.get(rr["level"], "")} 结论：{_inline(rr["verdict"])}'
        f'<span style="font-size:.82rem;color:#718096;font-weight:400">'
        f'　支持度评分 {rr["score"]:.0f} / 100</span></div>'
        f'<div class="s">{_inline(rr["summary"])}</div></div>'
        f'<div class="mmm-checkscroll"><table class="mmm-check"><thead><tr>'
        f'<th></th><th>检查项</th><th>现值</th><th>为什么重要 / 不合格怎么办</th>'
        f'</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')


# ============================================================
# 图 1：收入与花费的趋势
# ============================================================
def fig_trend(t: dict):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=t["dates"], y=t["revenue"] / 1e4, name="周收入",
        line=dict(color=C_REV, width=2.2), mode="lines",
        hovertemplate="%{x|%Y-%m-%d}<br>收入 %{y:,.1f} 万元<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t["dates"], y=t["total_spend"] / 1e4, name="总投放花费",
        line=dict(color=C_SPEND, width=1.5), mode="lines", opacity=0.9,
        hovertemplate="%{x|%Y-%m-%d}<br>花费 %{y:,.1f} 万元<extra></extra>"),
        secondary_y=True)
    pi = np.asarray(t["promo_idx"])
    fig.add_trace(go.Scatter(
        x=np.asarray(t["dates"])[pi], y=t["revenue"][pi] / 1e4,
        mode="markers", name="投放高峰周",
        marker=dict(color="#dd6b20", size=9, symbol="triangle-up",
                    line=dict(width=0)),
        hovertemplate="投放高峰周<br>收入 %{y:,.1f} 万元<extra></extra>"),
        secondary_y=False)
    fig.update_yaxes(title_text="收入（万元）", secondary_y=False,
                     title_font=dict(color=C_REV), tickfont=dict(color=C_REV))
    fig.update_yaxes(title_text="总花费（万元）", secondary_y=True, showgrid=False,
                     title_font=dict(color=C_SPEND), tickfont=dict(color=C_SPEND))
    fig.update_xaxes(title_text="")
    return _layout(fig, height=380)


# ============================================================
# 图 2：STL 分解
# ============================================================
def fig_decompose(d: dict):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    titles = ("原始收入", "趋势项（长期增长）", "季节项（年内周期）", "残差（未被解释的部分）")
    colors = ("#4a5568", C_REV, C_OK, C_NET)
    series = (d["observed"], d["trend"], d["seasonal"], d["resid"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=[f"　{t}" for t in titles])
    for i, (s, c) in enumerate(zip(series, colors), start=1):
        fig.add_trace(go.Scatter(
            x=d["dates"], y=np.asarray(s) / 1e4, mode="lines", showlegend=False,
            line=dict(color=c, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f} 万元<extra></extra>"),
            row=i, col=1)
    fig.update_annotations(font_size=11.5)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, ticksuffix="")
    return _layout(fig, height=620, legend_top=False)


# ============================================================
# 图 3：渠道相关热力图 + VIF
# ============================================================
def fig_corr(c: dict):
    import plotly.graph_objects as go

    corr = c["corr"]
    labels = [cn(x) for x in corr.columns]
    z = corr.to_numpy(dtype=float)
    text = np.where(np.isfinite(z), np.round(np.nan_to_num(z), 2).astype(str), "—")
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=text, texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="%{y} × %{x}<br>相关系数 %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="相关系数", side="right"), thickness=12,
                      len=0.9, tickfont=dict(size=10))))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=350, legend_top=False)


def fig_vif(c: dict):
    import plotly.graph_objects as go

    v = c["vif"]
    labels = [cn(x) for x in v["channel"]]
    raw = [float(x) for x in v["vif"]]
    vals = [(x if np.isfinite(x) else None) for x in raw]
    bar_colors = [C_OK if (np.isfinite(x) and x < 5) else
                  (C_WARN if (np.isfinite(x) and x < 10) else C_BAD) for x in raw]
    shown = ["∞" if not np.isfinite(x) else f"{x:.2f}" for x in raw]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=bar_colors,
        text=shown, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}：VIF %{text}<extra></extra>"))
    fig.add_vline(x=5, line_dash="dash", line_color=C_WARN, line_width=1.4,
                  annotation_text="中度共线 5", annotation_position="top")
    fig.add_vline(x=10, line_dash="dash", line_color=C_BAD, line_width=1.4,
                  annotation_text="严重共线 10", annotation_position="top")
    fig.update_xaxes(title_text="VIF（方差膨胀因子，越低越好）", range=[0, 12])
    return _layout(fig, height=350, legend_top=False)


# ============================================================
# 图 4：滞后互相关
# ============================================================
def fig_lag(l: dict):
    import plotly.graph_objects as go

    fig = go.Figure()
    conf = float(l["conf"])
    fig.add_hrect(y0=-conf, y1=conf, fillcolor="rgba(128,140,160,.16)",
                  line_width=0, layer="below",
                  annotation_text=f"95% 显著性带 ±{conf:.2f}",
                  annotation_position="top left",
                  annotation_font_size=11)
    for i, (ch, cs) in enumerate(l["curves"].items()):
        col = ui.color_of(ch, i)
        fig.add_trace(go.Scatter(
            x=l["lags"], y=cs, mode="lines+markers", name=cn(ch),
            line=dict(color=col, width=2), marker=dict(size=6),
            hovertemplate=f"{cn(ch)}<br>滞后 %{{x}} 周 · r=%{{y:.3f}}<extra></extra>"))
    for ch, p in l["peaks"].items():
        if p.get("lag") is not None and np.isfinite(p.get("r", np.nan)):
            fig.add_trace(go.Scatter(
                x=[p["lag"]], y=[p["r"]], mode="markers", showlegend=False,
                marker=dict(size=14, symbol="star", color="rgba(0,0,0,0)",
                            line=dict(color="#dd6b20", width=2)),
                hovertemplate=f"{cn(ch)} 峰值：滞后 {p['lag']} 周<extra></extra>"))
    fig.add_hline(y=0, line_color="#718096", line_width=1)
    fig.update_xaxes(title_text="滞后周数（花费提前 k 周）", dtick=1)
    fig.update_yaxes(title_text="与当期收入的相关系数")
    return _layout(fig, height=380)


# ============================================================
# 图 5：花费-收入（毛关系 vs 控制后净关系）
# ============================================================
def fig_saturation(ch: str, rec: dict, title: str):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = np.asarray(rec["x"]) / 1e4
    y = np.asarray(rec["y"]) / 1e4

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="毛关系（未控制）",
        marker=dict(color=C_RAW, size=5, opacity=0.45, line=dict(width=0)),
        hovertemplate="花费 %{x:,.1f} 万<br>收入 %{y:,.1f} 万<extra></extra>"),
        secondary_y=False)
    adj = rec.get("adj_xy")
    if adj is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(adj)[:, 0] / 1e4, y=np.asarray(adj)[:, 1] / 1e4,
            mode="markers", name="净关系（控制后）",
            marker=dict(color=C_NET, size=5, opacity=0.40, line=dict(width=0)),
            hovertemplate="花费 %{x:,.1f} 万<br>净收入 %{y:,.1f} 万<extra></extra>"),
            secondary_y=True)
    rc = rec.get("raw_curve")
    if rc is not None:
        rc = np.asarray(rc)
        fig.add_trace(go.Scatter(
            x=rc[:, 0] / 1e4, y=rc[:, 1] / 1e4, mode="lines", name="毛 LOWESS",
            line=dict(color=C_RAW, width=3), hoverinfo="skip"), secondary_y=False)
    ac = rec.get("adj_curve")
    if ac is not None:
        ac = np.asarray(ac)
        fig.add_trace(go.Scatter(
            x=ac[:, 0] / 1e4, y=ac[:, 1] / 1e4, mode="lines", name="净 LOWESS",
            line=dict(color=C_NET, width=3), hoverinfo="skip"), secondary_y=True)

    fig.update_yaxes(title_text="收入（万元）", secondary_y=False,
                     title_font=dict(color="#718096"), tickfont=dict(color="#718096"))
    fig.update_yaxes(title_text="净收入（万元）", secondary_y=True, showgrid=False,
                     title_font=dict(color=C_NET), tickfont=dict(color=C_NET))
    fig.update_xaxes(title_text="周花费（万元）")
    return _layout(fig, height=360, title=title)


# ============================================================
# 图 6：渠道系数的自助法区间
# ============================================================
def fig_bootstrap(ident: dict):
    import plotly.graph_objects as go

    t = ident["channel_table"].copy()
    if t.empty:
        return None
    t = t.sort_values("系数", ascending=True)
    labels = [cn(r["列"]) for _, r in t.iterrows()]
    coef = [float(r["系数"]) for _, r in t.iterrows()]
    lo = [float(r["CI下"]) for _, r in t.iterrows()]
    hi = [float(r["CI上"]) for _, r in t.iterrows()]
    cross = [bool(r["跨0"]) for _, r in t.iterrows()]
    colors = [C_WARN if c else C_OK for c in cross]
    fig = go.Figure(go.Scatter(
        x=coef, y=labels, mode="markers",
        error_x=dict(type="data", symmetric=False,
                     array=[h - c for h, c in zip(hi, coef)],
                     arrayminus=[c - l for c, l in zip(coef, lo)],
                     color="#a0aec0", thickness=2.2, width=0),
        marker=dict(size=12, color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>点估计 %{x:.3f}<extra></extra>",
        showlegend=False))
    fig.add_vline(x=0, line_color="#718096", line_dash="dash", line_width=1.4,
                  annotation_text="0（无效果）", annotation_position="top")
    fig.update_xaxes(title_text="花费对收入的净效应（元收入 / 元花费）")
    return _layout(fig, height=140 + 46 * max(3, len(labels)), legend_top=False)


# ============================================================
# 页面
# ============================================================
ui.page_header(
    "探索性分析：这份数据支不支持做归因？",
    "在建模之前，先用数据回答五个问题：波动里多少是规律性的、渠道能否被分开、"
    "广告几周见效、加钱还有没有用、哪一列数据缺少可识别性。"
    "本页所有图表都基于你当前载入的数据实时计算 —— 换成你自己的 CSV 同样有效。",
    "📊")

ui.steps(["载入数据", "探索性分析", "建模与诊断", "预算优化", "决策报告"],
         current=1, done_upto=0)

if not state.require_dataset():
    st.stop()

ds = state.get_dataset()
period_default, gran_label = _infer_period(ds)

with st.sidebar.expander("探索性分析参数", expanded=False):
    max_lag = int(st.slider("滞后分析窗口", 2, 16, 8, 1,
                            help="看广告效果在投放后多少周内释放。窗口太短会截断长尾，"
                                 "太长则尾部噪声大。"))
    period_use = int(st.number_input(
        f"季节周期（按{ds.granularity_days:.0f}天粒度推断为 {period_default}）",
        min_value=2, max_value=104, value=int(period_default), step=1,
        help="周数据用 52；月数据用 12。数据不足 2 个周期时算法会自动缩短并提示。"))
    st.caption(f"数据粒度：约 {ds.granularity_days:.0f} 天 / {gran_label}")

with st.spinner("正在计算诊断（趋势分解 / 共线性 / 滞后 / 饱和 / 自助法）…"):
    res = cache.cached_eda(ds.fingerprint, ds.df, ds.date_col, ds.target_col,
                           tuple(ds.channel_cols), tuple(ds.control_cols),
                           max_lag, period_use)

rr = res["readiness"]
lz = res["lag"]
cl = res["collinearity"]
st_ = res["saturation"]

# ---- 顶部 KPI ----
ui.kpi_row([
    {"label": "支持归因评分", "value": f"{rr['score']:.0f} / 100",
     "delta": rr["verdict"], "tone": {"pass": "down", "warn": None, "fail": "up"}.get(rr["level"])},
    {"label": "观测周期", "value": f"{ds.n_rows} 行",
     "delta": f"{ds.date_range()[0]} ~ {ds.date_range()[1]}", "tone": "brand"},
    {"label": "渠道最大 VIF",
     "value": (_vif_txt(cl.get("max_vif")) if cl.get("ok") else "—"),
     "delta": (f"渠道间最大 |相关| {cl['max_corr']:.2f}"
               if cl.get("ok") and np.isfinite(cl.get("max_corr", np.nan)) else "不可计算")},
    {"label": "广告效果峰值滞后", "value": (f"{lz['median_peak_lag']:.0f} 周"
                                      if lz.get("ok") and np.isfinite(lz.get("median_peak_lag", np.nan))
                                      else "—"),
     "delta": f"建议 Adstock 窗口 l_max = {lz.get('suggested_l_max', '—')}" if lz.get("ok") else ""},
    {"label": "边际递减渠道", "value": (f"{st_['n_diminishing']} / {st_['n_channels_ok']}"
                                   if st_.get("ok") else "—"),
     "delta": (f"递增 {st_['n_convex']} · 线性 {st_['n_linear']}" if st_.get("ok") else "")},
], per_row=5)

# ---- 结论卡片（放在最上面）----
st.subheader("一、结论：这份数据支不支持做归因")
st.caption("八项检查逐条给出「现值 + 为什么重要 + 不合格怎么办」。"
           "评分权重偏向**可识别性**类问题 —— 它们属于「再复杂的模型也救不回来」的那一类。")
st.markdown(_readiness_html(rr), unsafe_allow_html=True)

if rr["level"] == "fail":
    st.error("⛔ 当前数据的硬伤较多。若继续建模，报告里必须显式说明这些限制，"
             "否则结论会被误读。")
elif rr["level"] == "warn":
    st.warning("⚠️ 数据基本可用，但被点名的项目需要在建模时用先验约束，"
               "并在报告里交代清楚。")

st.divider()

# ============================================================
# 二、五个诊断
# ============================================================
st.subheader("二、诊断过程（每个区块回答一个业务问题）")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 趋势与节律", "🧩 渠道可分离性", "⏳ 广告滞后效应",
    "📉 花费-收入与饱和", "🧪 可识别性与数据质量"])

# ------------------------------------------------------------
# Tab 1 趋势与节律
# ------------------------------------------------------------
with tab1:
    _question("📈", "收入为什么忽高忽低？投放真的在驱动收入吗？",
              "把「周收入」和「总投放花费」放在同一时间轴对照。两者同向波动说明投放确实在起作用；"
              "但也要看到——大促周两者同时升高，这正是**内生性**的来源，"
              "所以这个相关系数不能读成因果效果。")

    t = res["trend"]
    if t.get("ok"):
        ui.plotly_show(fig_trend(t), key="eda_trend")
        _takeaway([
            f"收入与总花费的相关系数 = <b>{t['corr_rev_spend']:.3f}</b>；"
            f"周均收入 {t['revenue_mean']/1e4:,.1f} 万元，"
            f"周均总花费 {t['total_spend_mean']/1e4:,.1f} 万元。",
            f"投放高峰周的收入比平均高 <b>{100*t['promo_lift']:.0f}%</b> —— "
            f"看起来投放很有效，但这部分提升里有大促自然需求的成分。",
            "结论：收入确实随投放波动，但**不能**由这张图给投放记功。"
            "拆开「多少是投放、多少是大促和季节」正是下一步 MMM 要做的事。",
        ])
    else:
        st.warning(f"趋势分析不可用：{t.get('reason', '数据不足')}")

    st.markdown("#### 波动里有多少是「规律性」的？")
    d = res["decompose"]
    if d.get("ok"):
        ui.plotly_show(fig_decompose(d), key="eda_decomp")
        if d.get("note"):
            st.info(d["note"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="mmm-kpi"><div class="l">趋势项（产品自然增长）</div>'
                        f'<div class="v">{d["var_trend_pct"]:.1f}%</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mmm-kpi"><div class="l">季节项（年内周期）</div>'
                        f'<div class="v">{d["var_season_pct"]:.1f}%</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="mmm-kpi"><div class="l">残差（投放+事件+噪声）</div>'
                        f'<div class="v">{d["var_resid_pct"]:.1f}%</div></div>',
                        unsafe_allow_html=True)
        _takeaway([
            f"季节项解释了收入方差的 <b>{d['var_season_pct']:.1f}%</b>"
            f"（季节波动幅度 ±{d['seasonal_std']/1e4:.1f} 万元，"
            f"约为均值的 {d['seasonal_amp_pct']:.1f}%）。",
            "这就是必须显式建模季节性的理由：否则模型会把「春节收入自然高」"
            "误判成「春节广告特别有效」，导致 ROI 虚高。",
            f"残差占 {d['var_resid_pct']:.1f}% —— 这部分才是投放与突发事件能解释的空间。"
            "残差占比越高，说明越要用「控制了趋势与季节之后」的模型来归因。",
        ])
    else:
        st.warning(f"季节分解不可用：{d.get('reason', '数据不足')}")

# ------------------------------------------------------------
# Tab 2 渠道可分离性
# ------------------------------------------------------------
with tab2:
    _question("🧩", "5 个渠道能被分开归因吗？",
              "如果两个渠道的花费几乎同步（大促一起加投），模型在数学上就无法判断"
              "收入该算给谁。这不是估计方法的问题，而是数据本身的信息量问题 —— "
              "MMM 最根本的难点。")

    if cl.get("ok"):
        col_l, col_r = st.columns(2)
        with col_l:
            ui.plotly_show(fig_corr(cl), key="eda_corr")
            st.caption("两两之间的线性同步程度。越接近 ±1 越危险。")
        with col_r:
            ui.plotly_show(fig_vif(cl), key="eda_vif")
            st.caption("VIF = 1/(1−R²)：一个渠道能被其余渠道『解释』掉多少。"
                       "VIF < 5 健康，5~10 中度，> 10 严重。")

        vif_show = cl["vif"].copy()
        vif_show["渠道"] = [cn(x) for x in vif_show["channel"]]
        vif_show["VIF"] = [("∞（可被完全解释）" if not np.isfinite(v) else f"{v:.2f}")
                         for v in vif_show["vif"]]
        vif_show["判定"] = ["健康 ✓" if (np.isfinite(v) and v < 5)
                          else ("中度共线 ⚠" if (np.isfinite(v) and v < 10) else "严重 ✗")
                          for v in vif_show["vif"]]
        ui.dataframe(vif_show[["渠道", "VIF", "判定"]], caption=None)

        _takeaway([
            f"渠道间最大 |相关系数| = <b>{cl['max_corr']:.3f}</b>，"
            f"平均 |相关系数| = {cl['mean_abs_corr']:.3f}；最大 VIF = "
            f"<b>{_vif_txt(cl['max_vif'])}</b>。",
            f"判定：<b>{_esc(cl['verdict'])}</b>",
            "如果这里显示严重共线，可做的事（按性价比排序）："
            "① 把同步投放改成错峰投放；② 合并高度相似的渠道；"
            "③ 用历史先验/实验数据（geo-lift）约束渠道贡献；"
            "④ 明确告诉业务方：此时只能比较渠道『组合』的效果，不能比较单渠道 ROI。",
        ])
    else:
        st.warning(f"共线性诊断不可用：{cl.get('reason', '')}")

# ------------------------------------------------------------
# Tab 3 广告滞后效应
# ------------------------------------------------------------
with tab3:
    _question("⏳", "今天投的广告，几周后才见效？",
              "对每个渠道计算「花费提前 k 周」与「当期收入」的相关性。"
              "相关性最高的滞后位置就是广告效果的峰值延迟 —— 它直接决定 Adstock 的窗口长度，"
              "也决定预算调整后多久才能看到效果。")

    if lz.get("ok"):
        ui.plotly_show(fig_lag(lz), key="eda_lag")
        rows = []
        for ch, p in lz["peaks"].items():
            # 统一成字符串：两列都可能缺值（某渠道无有效峰值），
            # 混着 int / str 会让 Arrow 序列化失败并触发一次无谓的类型回退。
            lag_v = p.get("lag")
            half_v = p.get("half_lag")
            rows.append({
                "渠道": cn(ch),
                "峰值滞后(周)": str(lag_v) if lag_v is not None else "—",
                "峰值相关系数": (f"{p['r']:.3f}" if np.isfinite(p.get("r", np.nan)) else "—"),
                "半衰滞后(周)": (str(half_v) if half_v is not None
                            else f">{lz['max_lag']}"),
            })
        ui.dataframe(pd.DataFrame(rows),
                     caption=f"半衰滞后 = 相关性从峰值衰减到一半所需周数，用来判断记忆长度。"
                             f"显著性带约 ±{lz['conf']:.3f}（n={lz['n']}），带内波动不可当真。")
        _takeaway([
            f"多数渠道的峰值滞后在 <b>{lz['median_peak_lag']:.0f} 周</b>；"
            f"相关性半衰滞后中位数 {lz['median_half_lag']:.0f} 周。",
            f"→ 建议 Adstock 最大滞后窗 <b>l_max = {lz['suggested_l_max']}</b> 周"
            f"（下限 6 周是工程约定：几何衰减的尾巴需要足够预热，窗口太短会把上期残留算到当期）。",
            "业务含义：调整预算后，效果大致需要 2~3 周才充分显现。"
            "所以不要用「上周加了预算、这周收入没变」来否定投放效果。",
            "注意这仍是**相关**：广告可能只是提前释放了本就会发生的需求。"
            "真正的因果形状由模型后验给出。",
        ])
    else:
        st.warning(f"滞后分析不可用：{lz.get('reason', '数据不足')}")

# ------------------------------------------------------------
# Tab 4 花费-收入与饱和
# ------------------------------------------------------------
with tab4:
    _question("📉", "继续加钱还有用吗，还是已经开始饱和？",
              "先看「毛关系」（灰）：直接画渠道花费对收入。再叠加「净关系」（紫）："
              "用回归把其他渠道、控制变量、趋势和季节都剔除后，剩下的收入里与该渠道有关的部分。"
              "两条曲线形状往往不同 —— 差别就是内生性造成的假象。")

    if st_.get("ok"):
        st.markdown(
            '<div class="mmm-panel" style="border-left:4px solid #a0aec0">'
            '<div class="t">怎么看这张图</div><div class="s">'
            '<b>灰色</b>是原始散点，<b>紫色</b>是控制其他因素后的净关系。'
            '两条 LOWESS 平滑线分别对应各自的散点（左右两个纵轴刻度不同，'
            '左轴=收入，右轴=净收入）。<b>形状</b>才是重点：变平=饱和，变陡=还没到拐点。</div></div>',
            unsafe_allow_html=True)

        sat_chs = [c for c, r in st_["channels"].items() if r.get("ok")]
        skipped = [c for c, r in st_["channels"].items() if not r.get("ok")]
        cols_per_row = 3 if len(sat_chs) >= 3 else max(1, len(sat_chs))
        for i in range(0, len(sat_chs), cols_per_row):
            row = sat_chs[i:i + cols_per_row]
            cols = st.columns(len(row))
            for col, ch in zip(cols, row):
                rec = st_["channels"][ch]
                shape_icon = {"递减": "🔻", "递增": "🔺", "近似线性": "➖"}.get(rec["shape"], "•")
                with col:
                    ui.plotly_show(
                        fig_saturation(ch, rec, f"{shape_icon} {cn(ch)} · {rec['shape']}"),
                        key=f"eda_sat_{ch}")
        for ch in skipped:
            st.caption(f"⚠️ {cn(ch)}：{st_['channels'][ch].get('reason', '无法拟合')}")

        rows = []
        for ch, rec in st_["channels"].items():
            if not rec.get("ok"):
                continue
            rows.append({
                "渠道": cn(ch),
                "形态": rec["shape"],
                "净边际效应(元/元)": (f"{rec['slope_high']:.2f}"
                                 if np.isfinite(rec.get("slope_high", np.nan)) else "—"),
                "端点斜率比": (f"{rec['flatness']:.2f}"
                           if np.isfinite(rec.get("flatness", np.nan)) else "—"),
                "高低花费段收入差": (f"{rec['lift_pct']:.1f}%"
                              if np.isfinite(rec.get("lift_pct", np.nan)) else "—"),
            })
        ui.dataframe(pd.DataFrame(rows),
                     caption="净边际效应 = 控制其他因素后，该渠道每多花 1 元带来的收入增量（初判，"
                             "正式数值见「预算优化」页的边际回报表）。端点斜率比 <0.75 变平（饱和）、"
                             ">1.33 变陡（未到拐点）。")

        st.markdown(f'<div class="mmm-panel"><div class="t">📌 结论</div>'
                    f'<div class="s">{_inline(st_["verdict"])}</div></div>',
                    unsafe_allow_html=True)

        _takeaway([
            "对比灰紫两条曲线：如果毛关系明显比净关系更陡，说明该渠道的高花费周"
            "恰好也是大促/上新周 —— 毛关系把大促的自然增长算进了广告头上。",
            "判断要不要用饱和曲线，看的是**净关系**的端点斜率比，而不是原始散点的样子。",
            "阶段 3 的预算优化会用到这个形状信息：若多数渠道仍递增，"
            "最优解容易顶到预算上界（这在本项目的优化结果里确实观察到了）。",
        ])

        # ---- 与阶段 3 的交叉验证（产物存在时才显示）----
        mp = paths.artifact_path("stage3_marginal")
        if os.path.isfile(mp):
            try:
                mg = pd.read_csv(mp, encoding="utf-8-sig")
                key_c = next((c for c in mg.columns if "渠道" in c or c == "channel"), None)
                key_v = next((c for c in mg.columns if "边际回报" in c), None)
                eda_slope = {c: r["slope_high"] for c, r in st_["channels"].items()
                             if r.get("ok") and np.isfinite(r.get("slope_high", np.nan))}
                if key_c and key_v and eda_slope:
                    mg = mg[[key_c, key_v]].dropna()
                    mg = mg[mg[key_c].isin(eda_slope)]
                    if len(mg) >= 3:
                        a = [eda_slope[r[key_c]] for _, r in mg.iterrows()]
                        b = [float(r[key_v]) for _, r in mg.iterrows()]
                        cmp = pd.DataFrame({
                            "渠道": [cn(r[key_c]) for _, r in mg.iterrows()],
                            "EDA 净末段斜率": [round(v, 2) for v in a],
                            "阶段3 模型边际回报": [round(v, 2) for v in b],
                        })
                        try:
                            from scipy.stats import spearmanr
                            rho, _p = spearmanr(a, b)
                        except Exception:                          # noqa: BLE001
                            rho = float("nan")
                        st.markdown("#### 交叉验证：EDA 初判 vs 贝叶斯模型的边际回报")
                        ui.dataframe(cmp, caption=(
                            "同一套数据、两条完全独立的路径：左边是『不建模，只用回归控制的净斜率』，"
                            "右边是『完整贝叶斯 MMM 的边际回报』。"
                            + (f"两者排序的 Spearman ρ = <b>{rho:+.3f}</b>"
                               if np.isfinite(rho) else "")
                            + "。排序一致说明结论不是某个模型捏出来的，而是数据里本就有的信号。"))
            except Exception as _e:                                # noqa: BLE001
                st.caption(f"（交叉验证暂不可用：{type(_e).__name__}）")
    else:
        st.warning(f"饱和分析不可用：{st_.get('reason', '数据不足')}")

# ------------------------------------------------------------
# Tab 5 可识别性与数据质量
# ------------------------------------------------------------
with tab5:
    _question("🧪", "哪一列数据缺少信息量？渠道的效果方向可信吗？",
              "两个互补的检查：① 逐列体检 —— 有没有渠道投入过于平稳、"
              "断投太频繁、或几乎没有波动；② 方向稳定性 —— 用移动块自助法重抽 240 次，"
              "看渠道系数的抽样区间是否跨过 0。")

    st.markdown("#### 1）逐列体检")
    health = res["health"].copy()
    health["列"] = [cn(x) if x in schema.CHANNELS else
                    schema.CONTROL_CN.get(x, x) for x in health["列"]]
    health_out = pd.DataFrame({
        "列": health["列"], "类型": health["类型"],
        "周均(万元)": health["周均(万元)"].round(2),
        "CV(变异系数)": health["CV"].round(3),
        "零值占比": (health["零值占比"] * 100).round(1).astype(str) + "%",
        "峰值/周均": health["峰值/周均"].round(2),
    })
    ui.dataframe(health_out,
                 caption="CV 是「有没有变动」的度量，也是可识别性的前提："
                         "一个渠道若每周投放金额几乎一样，数据里就没有任何信息"
                         "能用来估计它的效果。经验阈值：CV < 0.10 需警惕。")

    st.markdown("#### 2）渠道效应方向稳定性（移动块自助法）")
    ident = res["identifiability"]
    if ident.get("ok"):
        fb = fig_bootstrap(ident)
        if fb is not None:
            ui.plotly_show(fb, key="eda_boot")
        st.caption(f"点估计来自 OLS（收入 ~ 渠道 + 控制变量，R² = {ident['ols_r2']:.3f}）；"
                   f"误差棒是块自助法（{ident['n_boot']} 次，块长 {ident['block']} 周，"
                   "保留局部时间结构）得到的 95% 区间。区间跨过 0 → 该渠道的效果方向"
                   "无法从数据中确认。")
        cnt = ident["channel_table"]
        n_cross = ident["n_cross_zero"]
        _takeaway([
            f"跨 0 的渠道：<b>{n_cross} / {len(cnt)}</b>；"
            f"最小 P(系数>0) = {ident['min_p_positive']:.3f}。",
            f"判定：{_esc(ident['verdict'])}",
            "为什么用块自助法而不是普通自助法：广告数据有自相关，"
            "独立重抽会把区间估得过窄（假显著）。整块重抽保留了周与周之间的依赖结构。",
            "如果某个渠道区间很宽，建模时应给它更强的先验（例如参考行业基准 ROI），"
            "否则它的 ROI 报告出来也没人敢用。",
        ])
        with st.expander("完整系数表（含控制变量）", expanded=False):
            tb = ident["table"].copy()
            tb["列"] = [cn(x) if x in schema.CHANNELS else
                       schema.CONTROL_CN.get(x, x) for x in tb["列"]]
            tb = tb.rename(columns={"系数": "点估计", "均值": "自助均值",
                                    "CI下": "95%CI 下", "CI上": "95%CI 上",
                                    "P(系数>0)": "P(系数>0)", "跨0": "区间跨0"})
            ui.dataframe(tb.round(3))
    else:
        st.warning(f"方向稳定性诊断不可用：{ident.get('reason', '数据不足')}")

    st.markdown("#### 3）下一步")
    st.markdown(
        "如果上面的结论是「支持做归因」，可以直接进入 **🔬 建模与诊断** 页；"
        "如果被点名了共线性或方向不稳，建议先去侧栏把渠道投放改成错峰，"
        "或在建模时对被点名的渠道加更强先验。")
    if st.button("刷新诊断结果", **ui.FULL_WIDTH):
        st.cache_data.clear()
        st.rerun()
