"""
预算优化页（阶段 9 交付）
========================

这一页承载整个项目**最有决策价值的产出**。

核心主张（也是面试最该讲的一句话）：

    交付物不是「一个最优分配数字」，而是「弹性-收益曲线」。

    因为在带硬边界的最优化问题里，最优解几乎全部卡在边界上 ——
    决定结果的是「允许每个渠道调多少」（业务方拍板的），
    不是「模型小数点后第几位」（分析师的自我感动）。

页面回答五个业务问题（每个 Tab 一个）：

    ① 钱该往哪挪？          → 边际回报 + 响应曲线（当前花费点的斜率）
    ② 具体怎么分？          → 优化前后分配对比 + 卡角点标注
    ③ 该接受多大调整？      → **弹性-收益曲线**（本页核心）
    ④ 预算整体变了怎么办？  → 情景分析（×0.8 / ×1.0 / ×1.2）
    ⑤ 结论有多稳？          → 后验不确定性传导 + channel_scales 对照实验

三条硬性要求的落地：
    长任务 → 求解前明确档位数与预计耗时；结果写进 session_state，切页不重算；
    窄屏   → 所有图 responsive、图例置顶、高度按内容算、表格横向滚动；
    决策导向 → 结论卡片置顶；颜色按**决策方向**给（红=加投、绿=减投）。
"""
from __future__ import annotations

import html as _html
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from core import optimize as O, paths, schema, state, ui     # noqa: E402

paths.ensure_src_on_path()

ui.inject()

# ============================================================
# 配色与版式：统一从 core/ui.py 取（全应用唯一来源 —— 三个分析页原先
# 各复制了一份相同常量，渠道色还会随列顺序变化，现已收口）
# ============================================================
C_BRAND, C_UP, C_DOWN = ui.C_BRAND, ui.C_UP, ui.C_DOWN
C_MUTED, C_WARN, C_GOLD = ui.C_MUTED, ui.C_WARN, ui.C_GOLD
GRID, PALETTE = ui.GRID, ui.PALETTE


def cn(name: str) -> str:
    return schema.CHANNEL_CN.get(name, name)


def _esc(s) -> str:
    return _html.escape(str(s))


def _inline(s) -> str:
    """
    转义后放行 **加粗**（统一实现见 core/ui.py::md_esc）。

    为什么需要：结论文本由 `optimize.recommend()` 生成，里面带 `**…**` 强调。
    如果整段 escape，用户会看到裸露的星号；如果整段不 escape，
    上传数据里的渠道名可能注入 HTML。所以「先转义、后放行固定语法」。
    """
    return ui.md_esc(s)


def _wan(v, nd: int = 1) -> str:
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


def _pct(v, nd: int = 2, sign: bool = True) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{'+' if (sign and f > 0) else ''}{f:.{nd}f}%"


def _layout(fig, height=None, title="", legend_top=True):
    """统一版式的薄包装（定义在 core/ui.py，全应用唯一来源）。"""
    return ui.apply_theme(fig, height=height, title=title, legend_top=legend_top)


def _question(icon: str, q: str, why: str) -> None:
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{icon} {ui.rich(q)}</div>'
        f'<div class="s">{ui.rich(why)}</div></div>', unsafe_allow_html=True)


def _takeaway(lines, title="🩺 这张图怎么用"):
    """
    图的业务解读。

    用 `ui.rich()` 放行 `**加粗**`：这些文案是拼进 HTML 的，
    若不处理，页面上的 `**…**` 会原样显示成裸星号。
    调用方若在 lines 里嵌入**含用户数据**的片段，需自行先用 `_esc()` 转义。
    """
    body = "".join(f"<li>{ui.rich(x)}</li>" for x in lines)
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{ui.rich(title)}</div>'
        f'<div class="s"><ul style="margin:6px 0 0 16px;padding:0">{body}</ul></div></div>',
        unsafe_allow_html=True)


# ============================================================
# 图表
# ============================================================
def _fig_marginal(marg: pd.DataFrame) -> go.Figure:
    """
    边际回报条形图：当前多投 1 元能拿回几元。

    颜色按**决策方向**给（不是按渠道）：>1 红=值得加投、<1 绿=该收缩。
    误差棒是 94% 后验区间 —— 区间跨 1 的渠道要标注「证据不足」。
    """
    d = marg.sort_values("边际回报", ascending=True).reset_index(drop=True)
    labels = [cn(c) for c in d["渠道"]]
    val = d["边际回报"].to_numpy(dtype=float)
    lo = d["边际回报_p5"].to_numpy(dtype=float)
    hi = d["边际回报_p95"].to_numpy(dtype=float)
    colors = [C_UP if v >= 1 else C_DOWN for v in val]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=val, y=labels, orientation="h", marker_color=colors, opacity=0.85,
        error_x=dict(type="data", symmetric=False,
                     array=np.maximum(hi - val, 0), arrayminus=np.maximum(val - lo, 0),
                     color=C_MUTED, thickness=1.3, width=5),
        text=[f"{v:.2f}" for v in val], textposition="outside",
        textfont=dict(size=11.5),
        hovertemplate="%{y}<br>边际回报 %{x:.3f}<extra></extra>",
        name="边际回报",
    ))
    fig.add_vline(x=1.0, line_dash="dot", line_color=C_MUTED, line_width=1.4)
    fig.add_annotation(x=1.0, y=1.02, yref="paper", yanchor="bottom",
                       text="盈亏平衡 ROI = 1", showarrow=False,
                       font=dict(size=10.5, color=C_MUTED), xanchor="left")
    xmax = float(np.nanmax(hi)) if np.isfinite(hi).any() else 1.0
    fig.update_xaxes(title_text="边际回报（每多投 1 元带来的收入，元）",
                     range=[0, max(1.25, xmax * 1.18)])
    fig.update_layout(showlegend=False)
    return _layout(fig, height=90 + 42 * len(d), legend_top=False)


def _fig_response_grid(curves: dict, ncol: int = 3) -> go.Figure:
    """
    响应曲线网格：每格一个渠道，「周花费 → 周收入贡献」，含 94% 区间。
    竖直虚线 = 当前花费点。曲线右端越平 → 越接近饱和。
    """
    chs = list(curves.keys())
    n = len(chs)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig = make_subplots(rows=nrow, cols=ncol,
                        subplot_titles=[cn(c) for c in chs],
                        horizontal_spacing=0.09, vertical_spacing=0.16)
    for i, c in enumerate(chs):
        cu = curves[c]
        r, col = divmod(i, ncol)
        x = cu.spend_grid / 1e4
        fig.add_trace(go.Scatter(x=x, y=cu.response_hi / 1e4, mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), row=r + 1, col=col + 1)
        fig.add_trace(go.Scatter(x=x, y=cu.response_lo / 1e4, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(43,108,176,.16)",
                                 name="94% 区间" if i == 0 else None,
                                 showlegend=(i == 0), hoverinfo="skip"),
                      row=r + 1, col=col + 1)
        fig.add_trace(go.Scatter(x=x, y=cu.response_mean / 1e4, mode="lines",
                                 line=dict(color=C_BRAND, width=2.4),
                                 name="期望响应" if i == 0 else None,
                                 showlegend=(i == 0),
                                 hovertemplate="花费 %{x:.1f} 万 → 贡献 %{y:.1f} 万<extra></extra>"),
                      row=r + 1, col=col + 1)
        fig.add_trace(go.Scatter(x=[cu.current_spend / 1e4],
                                 y=[cu.current_response / 1e4], mode="markers",
                                 marker=dict(color=C_UP, size=9, symbol="diamond"),
                                 name="当前花费" if i == 0 else None,
                                 showlegend=(i == 0),
                                 hovertemplate="当前：花费 %{x:.1f} 万<br>贡献 %{y:.1f} 万<extra></extra>"),
                      row=r + 1, col=col + 1)
        fig.add_vline(x=cu.current_spend / 1e4, line_dash="dash",
                      line_color=C_UP, line_width=1.1,
                      row=r + 1, col=col + 1)
        fig.update_xaxes(title_text="周花费（万元）", title_font=dict(size=10),
                         row=r + 1, col=col + 1)
        fig.update_yaxes(title_text="周贡献（万元）", title_font=dict(size=10),
                         row=r + 1, col=col + 1)
    fig.update_layout(showlegend=True)
    return _layout(fig, height=250 * nrow, legend_top=True)


def _fig_alloc(main, channels, hist_total) -> go.Figure:
    """
    优化前后分配对比。**这里刻意画了"允许区间"**：
    浅色带 = [下限, 上限]，实心点 = 优化解。
    一眼就能看出"最优解全部贴在允许区间的边界上"—— 这是本页最重要的洞察。
    """
    hist = np.asarray(hist_total, dtype=float) / 1e4
    optb = np.asarray(main["budgets"], dtype=float) / 1e4
    lo = np.asarray(main["bounds_lo"], dtype=float) / 1e4
    hi = np.asarray(main["bounds_hi"], dtype=float) / 1e4
    labels = [cn(c) for c in channels]

    fig = go.Figure()
    # 允许区间：用误差棒表示 [lo, hi]
    mid = (lo + hi) / 2.0
    fig.add_trace(go.Scatter(
        x=labels, y=mid, mode="markers",
        marker=dict(size=13, color="rgba(43,108,176,.22)",
                    symbol="line-ns-open", line=dict(width=11, color="rgba(43,108,176,.22)")),
        error_y=dict(type="data", symmetric=False,
                     array=hi - mid, arrayminus=mid - lo,
                     color=C_BRAND, thickness=2.2, width=8),
        name="允许区间（模型给的弹性边界）",
        hovertemplate="%{x}<br>允许区间中点 %{y:.1f} 万<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=labels, y=hist, mode="markers+lines",
        marker=dict(size=11, color=C_MUTED, symbol="circle"),
        line=dict(color=C_MUTED, width=1.4, dash="dot"),
        name="当前分配（维持现状）",
        hovertemplate="%{x}<br>当前 %{y:.1f} 万<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=labels, y=optb, mode="markers+lines",
        marker=dict(size=13, color=C_UP, symbol="diamond"),
        line=dict(color=C_UP, width=2.2),
        name="优化分配（模型建议）",
        hovertemplate="%{x}<br>建议 %{y:.1f} 万<extra></extra>"))
    fig.update_yaxes(title_text="窗口内预算（万元）")
    return _layout(fig, height=440)


def _fig_elasticity(sweep: dict, current_e: float) -> go.Figure:
    """
    ★ 弹性-收益曲线（本页核心交付物）。

    x 轴 = 允许单渠道调整的幅度（±%），y 轴 = 模型估计的收益提升。
    右端加一个「无约束理论上界」作为参照点。

    这张图的价值：让业务方**自己选**能承受多大调整、对应能多拿多少，
    而不是被塞一个孤零零的最优数字。
    """
    tab = sweep["table"].copy()
    fin = tab[np.isfinite(tab["提升(%)"])].sort_values("弹性(%)")
    xs = fin["弹性(%)"].to_numpy(dtype=float)
    ys = fin["提升(%)"].to_numpy(dtype=float)

    fig = go.Figure()
    # 无约束单独画在右端（用 log 尺度不合适，就用一个"虚拟"位置）
    inf_row = sweep.get("inf_row")
    inf_lift = float(inf_row["提升(%)"]) if inf_row else None
    x_inf = (xs.max() * 1.32) if len(xs) else 50.0

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=C_BRAND, width=3),
        marker=dict(size=8, color=C_BRAND),
        name="约束下的最优（每档一次完整求解）",
        hovertemplate="允许 ±%{x:.0f}%<br>预计提升 %{y:+.2f}%<extra></extra>"))
    if inf_lift is not None:
        fig.add_trace(go.Scatter(
            x=[x_inf], y=[inf_lift], mode="markers",
            marker=dict(size=15, color=C_GOLD, symbol="star"),
            name="无约束理论上界（业务不可行）",
            hovertemplate="无约束（全部预算押单一渠道）<br>理论提升 %{y:+.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[xs.max(), x_inf] if len(xs) else [x_inf],
            y=[ys.max(), inf_lift] if len(ys) else [inf_lift],
            mode="lines", line=dict(color=C_GOLD, width=1.4, dash="dot"),
            showlegend=False, hoverinfo="skip"))

    # 标记当前选择的档位
    if np.isfinite(current_e):
        ce = current_e * 100
        fig.add_vline(x=ce, line_dash="dash", line_color=C_UP, line_width=1.6)
        sel = fin[np.isclose(fin["弹性(%)"], ce)]
        if len(sel):
            fig.add_annotation(
                x=ce, y=float(sel["提升(%)"].iloc[0]),
                text=f"当前选择 ±{ce:.0f}%<br>{float(sel['提升(%)'].iloc[0]):+.2f}%",
                showarrow=True, arrowhead=2, ax=44, ay=-38,
                font=dict(size=11, color=C_UP), bgcolor="rgba(255,255,255,.75)")

    for pct, lab, col in [(20, "可当月执行", C_DOWN), (40, "需 1~2 个季度", C_WARN)]:
        fig.add_vline(x=pct, line_dash="dot", line_color=col, line_width=1.0)
        fig.add_annotation(x=pct, y=1.0, yref="paper", yanchor="bottom",
                           text=lab, showarrow=False,
                           font=dict(size=10, color=col))
    fig.update_xaxes(title_text="允许的单渠道调整幅度（±%）")
    fig.update_yaxes(title_text="模型估计的媒体贡献提升（%）")
    return _layout(fig, height=460)


def _fig_scenarios(summary: pd.DataFrame, detail: pd.DataFrame,
                   channels) -> go.Figure:
    """情景分析：左=各情景下的最优结构（堆叠），右=预期贡献与提升。"""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=["不同预算下的最优分配结构", "预算变化下的预期收益"],
                        horizontal_spacing=0.12)
    labels = [str(s) for s in summary["情景"]]
    bottom = np.zeros(len(summary))
    for i, c in enumerate(channels):
        vals = []
        for s in summary["情景"]:
            sub = detail[(detail["情景"] == s) & (detail["渠道"] == c)]
            vals.append(float(sub["优化预算(元)"].iloc[0]) / 1e4 if len(sub) else 0.0)
        vals = np.asarray(vals, dtype=float)
        fig.add_trace(go.Bar(x=labels, y=vals, name=cn(c),
                             marker_color=ui.color_of(c, i),
                             hovertemplate=cn(c) + "：%{y:.1f} 万<extra></extra>"),
                      row=1, col=1)
        bottom += vals
    fig.update_yaxes(title_text="最优预算（万元）", row=1, col=1)

    lift = summary["提升(%)"].to_numpy(dtype=float)
    base = summary["基线贡献(元)"].to_numpy(dtype=float) / 1e4
    optv = summary["优化贡献(元)"].to_numpy(dtype=float) / 1e4
    fig.add_trace(go.Scatter(x=labels, y=base, mode="markers+lines",
                             line=dict(color=C_MUTED, dash="dot", width=1.6),
                             marker=dict(size=9), name="维持现状",
                             hovertemplate="%{x}<br>维持现状 %{y:.0f} 万<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=labels, y=optv, mode="markers+lines+text",
                             line=dict(color=C_UP, width=2.6),
                             marker=dict(size=11, color=C_UP),
                             text=[f"{v:+.1f}%" for v in lift],
                             textposition="top center",
                             textfont=dict(size=11, color=C_UP),
                             name="优化后", 
                             hovertemplate="%{x}<br>优化后 %{y:.0f} 万<extra></extra>"),
                  row=1, col=2)
    fig.update_yaxes(title_text="预期媒体贡献（万元）", row=1, col=2)
    fig.update_layout(showlegend=True)
    return _layout(fig, height=470, legend_top=True)


def _fig_robustness(rob: dict, channels) -> go.Figure:
    """
    稳健性：把后验不确定性传导到最优分配。
    误差棒 = [p5, p95]，方点 = 点估计解。
    若箱体被压成一条线 → 说明解全被边界卡住，**不是后验很确定**。
    """
    tab = rob["table"].set_index("渠道")
    labels = [cn(c) for c in channels]
    p5 = np.asarray([tab.loc[c, "p5(元)"] for c in channels], dtype=float) / 1e4
    p95 = np.asarray([tab.loc[c, "p95(元)"] for c in channels], dtype=float) / 1e4
    med = np.asarray([tab.loc[c, "中位数(元)"] for c in channels], dtype=float) / 1e4
    pt = np.asarray([tab.loc[c, "点估计(元)"] for c in channels], dtype=float) / 1e4
    mid = (p5 + p95) / 2.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=mid, mode="markers",
        marker=dict(size=13, color="rgba(43,108,176,.55)",
                    symbol="line-ns-open", line=dict(width=11, color="rgba(43,108,176,.25)")),
        error_y=dict(type="data", symmetric=False,
                     array=p95 - mid, arrayminus=mid - p5,
                     color=C_BRAND, thickness=2, width=8),
        name="p5 ~ p95 区间",
        hovertemplate="%{x}<br>区间中点 %{y:.0f} 万<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=med, mode="markers",
                             marker=dict(size=10, color=C_BRAND, symbol="circle"),
                             name="中位数",
                             hovertemplate="%{x}<br>中位数 %{y:.0f} 万<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=pt, mode="markers",
                             marker=dict(size=14, color=C_UP, symbol="diamond"),
                             name="点估计解",
                             hovertemplate="%{x}<br>点估计 %{y:.0f} 万<extra></extra>"))
    fig.update_yaxes(title_text="最优预算（万元）")
    return _layout(fig, height=430)


# ============================================================
# 结论卡片（置顶）
# ============================================================
def _conclusion_card(res: dict):
    main = res["main"]
    sweep = res["sweep"]
    dist = res.get("dist") or {}
    rec = res["recommend"]

    lift = main.get("lift_pct", np.nan)
    tone = C_UP if (np.isfinite(lift) and lift > 0.5) else C_MUTED

    st.markdown(
        f'<div class="mmm-panel" style="border-left:5px solid {tone}">'
        f'<div class="t">💰 结论</div>'
        f'<div class="s" style="font-size:.92rem">'
        f'{_inline(rec["headline"])}'
        f'<br><span class="mmm-muted">可信区间：增量 '
        f'{_wan(dist.get("net_lo"))} ~ {_wan(dist.get("net_hi"))} 万元'
        f'（94%，来自后验不确定性传导）</span>'
        f'</div></div>', unsafe_allow_html=True)

    # 「残差渠道」提示必须紧跟结论 —— 不能沉到页面底部，
    # 否则用户会照着那个纯算术数字去调投放。
    rc = main.get("residual_channel")
    if rc and main.get("residual_note"):
        st.markdown(
            f'<div class="mmm-issue warn"><div class="ti">'
            f'🧮 「{_esc(cn(rc))}」的幅度是算术残差，不是模型建议</div>'
            f'<div class="de">{_inline(main["residual_note"])}</div></div>',
            unsafe_allow_html=True)

    # 分渠道动作
    rows = []
    for a in rec["actions"]:
        m = a["边际回报"]
        col = C_UP if m >= 1 else C_DOWN
        icon = "🔴" if a["建议"] in ("加大投入",) else \
               ("🟠" if a["建议"] == "适度加投" else
                ("🟢" if a["建议"] == "考虑收缩" else "⚪"))
        rows.append(
            f'<tr><td class="lb">{_inline(cn(a["渠道"]))}</td>'
            f'<td class="vl" style="color:{col}">{_num(m, 2)}</td>'
            f'<td class="vl">{_num(a["平均ROI"], 2)}</td>'
            f'<td>{icon} {_inline(a["建议"])}</td>'
            f'<td class="wy">{_inline(a["理由"])}</td></tr>')
    st.markdown(
        '<div class="mmm-checkscroll"><table class="mmm-check">'
        '<thead><tr><th>渠道</th><th>边际回报</th><th>平均ROI</th>'
        '<th>建议</th><th>为什么</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)

    # 关键提醒：约束比精度重要
    if sweep.get("verdict"):
        st.markdown(
            f'<div class="mmm-issue info"><div class="ti">⚡ 最该记住的一句话</div>'
            f'<div class="de">{_inline(sweep["verdict"])}</div></div>',
            unsafe_allow_html=True)

    if main.get("message"):
        st.markdown(
            f'<div class="mmm-issue warn"><div class="ti">⚠️ 求解提示</div>'
            f'<div class="de">{_inline(main["message"])}</div></div>',
            unsafe_allow_html=True)


def _status_bar(ds, art, res):
    items = [
        {"label": "数据源", "value": _esc(getattr(ds, "source_label", "—")),
         "delta": f"{getattr(ds, 'n_rows', 0)} 期"},
        {"label": "渠道数", "value": str(len(getattr(ds, "channel_cols", []))),
         "delta": "、".join(cn(c) for c in getattr(ds, "channel_cols", [])[:3])},
    ]
    if res and res.get("ok"):
        meta = res["opt_meta"]
        main = res["main"]
        items += [
            {"label": "决策窗口", "value": f"{meta['num_periods']} 期",
             "delta": f"{meta['start']} ~ {meta['end']}"},
            {"label": "窗口内总预算", "value": f"{_wan(res['base_total'])} 万",
             "delta": f"约束 ±{res['elasticity'] * 100:.0f}%"},
            {"label": "预期提升", "value": _pct(main.get("lift_pct")),
             "tone": "up" if (main.get("lift_pct") or 0) > 0 else None,
             "delta": f"角点 {main['corner_hit']}/{len(ds.channel_cols)}"},
        ]
    else:
        items.append({"label": "优化状态", "value": "未求解",
                      "delta": "点下方按钮开始"})
    ui.kpi_row(items, per_row=5)


# ============================================================
# 求解入口
# ============================================================
def _opt_key(ds, art, cfg: dict) -> str:
    """结果键 = 数据指纹 × 模型来源 × 参数哈希。任一变化都要重算。"""
    from core import cache
    fp = getattr(ds, "fingerprint", "nofp")
    art_tag = f"{art.get('source')}-{int(art.get('elapsed', 0))}"
    return f"{fp}|{art_tag}|{cache.config_hash(cfg)}"


def _controls(default_e: float, default_mult: float, default_rob: int) -> dict:
    """优化参数控件（读取值，不触发重算）。"""
    with st.expander("⚙️ 优化设置（改完点「重新求解」生效）", expanded=False):
        st.caption("这三个都是**业务决定**，不是技术参数 —— "
                   "它们直接决定能拿到多少收益，比模型精度重要得多。")
        c1, c2, c3 = st.columns(3)
        e = c1.slider("允许的单渠道调整幅度（±%）", 5, 80,
                      int(round(default_e * 100)), step=5,
                      help="业务方真正能拍板的东西。±20% 基本不影响团队编制与平台关系，"
                           "可当月执行；±40% 需要 1~2 个季度逐步到位。"
                           "**这个滑块往右拉，收益会明显放大** —— 这正是本页想传达的。")
        mult = c2.slider("总预算相对当前的比例", 0.5, 1.5, float(default_mult), step=0.05,
                         help="用来回答「明年预算砍 20% 该怎么排兵布阵」。"
                              "总预算不变时保持 1.0。")
        rob = c3.selectbox(
            "稳健性抽样数（每个样本一次完整求解）", [0, 20, 40, 60],
            index=[0, 20, 40, 60].index(default_rob),
            help="把后验不确定性传导到最终分配，看方案会不会翻盘。"
                 "0 = 跳过（最快）；40 个样本约需 10~20 秒。")
        st.caption("💡 弹性档位扫描（±5%~±60% 各求一次 + 无约束）是必跑的，"
                   "因为它是本页的核心交付物；它占掉大部分求解时间。")
    return {"elasticity": e / 100.0, "budget_mult": float(mult),
            "robustness_draws": int(rob)}


def _solve_with_budget(ds, art, cfg: dict) -> dict:
    """
    真正的求解入口：总预算 = 当前窗口预算 × 用户倍数。

    注意顺序：先用默认参数推断窗口与基准预算，再按倍数传 total_budget 求解。
    这样"总预算比例"这个业务旋钮才有意义（而不是让模型自己算一个基准）。
    """
    # 用 ui.StatusBox 而不是直接用 st.status(...)：后者的容器**没有**
    # .step() / .done() / .fail()（streamlit 1.65 实测），直接调用会让
    # 「点按钮 → 页面抛异常」。收口到 ui 层做能力探测，见 core/ui.py。
    box = ui.StatusBox("正在求解预算最优分配（分阶段进度）", expanded=True)

    def _cb(frac, msg):
        box.progress(frac, msg)
        box.step(msg)

    t0 = time.time()
    try:
        window = O.infer_window(art["df"], art["cfg"].date_col)
        if not window.get("ok"):
            box.fail(f"无法推断优化窗口：{window.get('reason')}")
            return {"ok": False, "reason": window.get("reason"), "window": window}
        # 先算基准预算（不求解，只做乘数）
        hist_w = O.hist_weekly_spend(art["df"], art["cfg"])
        meta_opt, meta = O.build_optimizer(art["mmm"], art["df"], art["cfg"], window)
        base_total = float((hist_w * meta["num_periods"]).sum())
        total_budget = base_total * float(cfg["budget_mult"])

        res = O.analyze(
            art["mmm"], art["cfg"], art["df"], art.get("idata"),
            window=window, elasticity=cfg["elasticity"],
            total_budget=total_budget,
            robustness_draws=cfg.get("robustness_draws", 0),
            progress=_cb,
        )
        res["base_total_of_mult"] = base_total
    except Exception as e:                                  # noqa: BLE001
        box.fail(f"求解失败：{type(e).__name__}: {e}")
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    if not res.get("ok"):
        box.fail(f"求解失败：{res.get('reason')}")
        return res
    res["wall_seconds"] = time.time() - t0
    box.done(f"求解完成（{res['wall_seconds']:.0f} 秒）")
    return res


# ============================================================
# Tab ① 钱该往哪挪
# ============================================================
def _tab_where(res):
    _question("🎯", "现在每多投 1 元，哪个渠道回报最高？",
              "这是决定「钱该往哪挪」的直接依据。边际回报 > 1 说明还值得加投，"
              "< 1 说明当前水平上加投已经不划算。")
    marg = res["marginal"]
    ui.plotly_show(_fig_marginal(marg), key="opt_marginal")

    top = marg.iloc[0]
    bot = marg.iloc[-1]
    _takeaway([
        f"最高的 <b>{_esc(cn(top['渠道']))}</b> 边际回报 <b>{_num(top['边际回报'], 3)}</b>："
        f"再多投 1 元能多拿回 {_num(top['边际回报'], 2)} 元。",
        f"最低的 <b>{_esc(cn(bot['渠道']))}</b> 只有 <b>{_num(bot['边际回报'], 3)}</b>"
        + ("，已低于盈亏线 1.00 —— 属于该往外挪的对象。"
           if bot["边际回报"] < 1 else "。"),
        "误差棒是该渠道边际回报的 94% 后验区间。**区间跨越 1.00 的渠道，"
        "说明「该加还是该砍」的证据不足**，此时优先看业务判断而不是数字。",
        "边际回报是**局部斜率**（当前花费点附近），不是平均 ROI。"
        "所以可能出现「平均 ROI 很高但边际回报已很低」的渠道 —— 那意味着它已经投放充分了。",
    ])

    st.markdown("---")
    _question("📈", "每个渠道「花多少钱 → 拿回多少」的整条曲线长什么样？",
              "边际回报只告诉你当前点的情况；响应曲线让你看到**未来**："
              "再加 50% 会不会饱和？现在处在曲线的哪一段？")
    ui.plotly_show(_fig_response_grid(res["curves"]), key="opt_curves")

    sat = res["saturation"].sort_values("饱和度", ascending=False)
    s0 = sat.iloc[0]
    _takeaway([
        f"当前饱和度最高的渠道是 <b>{_esc(cn(s0['渠道']))}</b>（{s0['饱和度']:.1%}）。"
        "饱和度接近 100% 表示继续加钱几乎没有回报。",
        "曲线右端越平 = 越接近饱和。图中各渠道的曲线都还在上升段 —— "
        "**这意味着「加钱有回报」，但也意味着最优解会被边界卡住**"
        "（因为模型会说「全都加上去」，而业务不允许）。",
        "竖直虚线是当前花费点。它离曲线拐点越远，说明还有越多空间可调。",
    ])
    with st.expander("📋 响应曲线与饱和度的底层数字"):
        show = res["saturation"].copy()
        show["渠道"] = [_esc(cn(c)) for c in show["渠道"]]
        for c in ("周均花费(元)", "周贡献(元)"):
            show[c] = (show[c] / 1e4).round(1)
        show = show.rename(columns={"周均花费(元)": "周均花费(万元)",
                                    "周贡献(元)": "周贡献(万元)"})
        show["饱和度"] = (show["饱和度"] * 100).round(1).astype(str) + "%"
        show["平均ROI"] = show["平均ROI"].round(3)
        show["边际回报"] = show["边际回报"].round(3)
        ui.dataframe(show)
        st.caption("饱和度 = 当前响应 ÷ 曲线渐近线高度。这是模型内部口径的相对位置，"
                   "不是『还能提升多少』的绝对承诺。")


# ============================================================
# Tab ② 优化方案
# ============================================================
def _tab_plan(res):
    main = res["main"]
    _question("🧮", "总预算不变的前提下，钱具体该怎么分？",
              f"在「单渠道允许调整 ±{res['elasticity'] * 100:.0f}%」的约束下求最优解。"
              f"浅色带是允许区间，菱形点是模型建议 —— 注意看它们的位置。")
    ui.plotly_show(_fig_alloc(main, list(res["alloc"]["渠道"]), res["hist_total"]),
                   key="opt_alloc")

    n_ch = len(res["alloc"])
    _takeaway([
        f"最优解有 <b>{main['corner_hit']}/{n_ch}</b> 个渠道贴在允许区间的边界上。"
        "这不是 bug —— 在『还要加钱才有回报』的数据上，"
        "数学最优一定是『把预算全给最高效的渠道』，"
        "**是业务约束把它拉住了**。",
        f"优化后预期媒体贡献 {_num(main['utility'] / 1e4, 1)} 万元，"
        f"当前分配 {_num(main['hist_utility'] / 1e4, 1)} 万元，"
        f"提升 <b>{_pct(main['lift_pct'])}</b>。",
        f"窗口内还有一个不可控的『常数底』（历史投放的广告残留余温），"
        f"约占当前贡献的 {main['constant_floor_pct']:.1f}%。"
        "所以请关注**优化与维持现状的差值**，而不是绝对总量。",
    ])

    with st.expander("📋 分配明细表"):
        d = res["alloc"].copy()
        d["渠道"] = [_esc(cn(c)) for c in d["渠道"]]
        for c in ("当前周均花费(元)", "当前窗口预算(元)", "优化窗口预算(元)"):
            d[c] = (d[c] / 1e4).round(1)
        d = d.rename(columns={"当前周均花费(元)": "当前周均花费(万元)",
                              "当前窗口预算(元)": "当前窗口预算(万元)",
                              "优化窗口预算(元)": "优化窗口预算(万元)"})
        d["变化(%)"] = d["变化(%)"].round(1)
        d["边际回报"] = d["边际回报"].round(3)
        d["平均ROI"] = d["平均ROI"].round(3)
        ui.dataframe(d)

    st.markdown("---")
    _question("📐", "这一步的约束条件是什么？会不会把最优解卡死？",
              "算法的收敛状态与约束设置必须透明 —— "
              "如果优化器没收敛，那个『最优解』就不该被当成决策依据。")
    meta = res["opt_meta"]
    checks = [
        {"label": "优化窗口", "status": "pass",
         "value": f"{meta['start']} ~ {meta['end']}（{meta['num_periods']} 期）",
         "hint": f"前 {meta['carry_in_periods']} 期作广告残留预热；"
                 f"adstock 溢出 {meta['adstock_periods']} 期。"},
        {"label": "channel_scales 已挂载", "status": "pass" if meta["health"]["ok"] else "fail",
         "value": "是" if meta["health"]["ok"] else "否",
         "hint": meta["health"]["verdict"] +
                 "　★ 不挂载会静默回退 1.0，返回『五渠道完全相等』的假最优解。"},
        {"label": "数值优化收敛", "status": "pass" if main["success"] else "fail",
         "value": f"nit={main['nit']}",
         "hint": ("迭代次数 > 1 说明优化器真的在搜索。"
                  "nit=1 是『没动过』的强烈信号。" if main["nit"] > 1 else
                  "只迭代 1 次 —— 高度可疑，请检查 channel_scales。")},
        {"label": "解的角点占比", "status": "warn" if main["corner_hit"] else "pass",
         "value": f"{main['corner_hit']}/{n_ch}",
         "hint": "角点越多，说明结果越由『边界宽度』决定，而不是由模型精度决定。"},
        {"label": "残差渠道", "status": "warn" if main.get("residual_channel") else "pass",
         "value": cn(main["residual_channel"]) if main.get("residual_channel") else "无",
         "hint": main.get("residual_note") or
                 "没有出现『唯一不卡边界』的渠道，每个渠道的取值都来自模型测算。"},
    ]
    tone = {"pass": C_DOWN, "warn": C_WARN, "fail": C_UP}
    icon = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}
    trs = []
    for c in checks:
        trs.append(
            f'<tr><td class="ic">{icon[c["status"]]}</td>'
            f'<td class="lb">{_inline(c["label"])}</td>'
            f'<td class="vl" style="color:{tone[c["status"]]}">{_inline(c["value"])}</td>'
            f'<td class="wy">{_inline(c["hint"])}</td></tr>')
    st.markdown('<div class="mmm-checkscroll"><table class="mmm-check"><tbody>'
                + "".join(trs) + "</tbody></table></div>", unsafe_allow_html=True)

    if res.get("dist", {}).get("ok"):
        d = res["dist"]
        st.caption(f"后验不确定性传导：预期贡献的 94% 区间 "
                   f"{_wan(d['raw_lo'])} ~ {_wan(d['raw_hi'])} 万元（均值 "
                   f"{_wan(d['raw_mean'])} 万元，来自 {d['samples']:,} 个后验样本）。"
                   f"净媒体增量区间 {_wan(d['net_lo'])} ~ {_wan(d['net_hi'])} 万元。")


# ============================================================
# Tab ③ 弹性-收益曲线（核心）
# ============================================================
def _tab_elasticity(res):
    sweep = res["sweep"]
    _question("🎚️", "我们愿意接受多大调整，能换多少增量？",
              "这是本页最重要的图。它把『约束』从一句技术设定变成一条可读的曲线 —— "
              "让业务方自己选档位，而不是被塞一个孤零零的最优数字。")
    ui.plotly_show(_fig_elasticity(sweep, res["elasticity"]), key="opt_elast")

    tab = sweep["table"]
    fin = tab[np.isfinite(tab["提升(%)"])].sort_values("弹性(%)")
    first_lift = float(fin["提升(%)"].iloc[0]) if len(fin) else np.nan
    _takeaway([
        f"±5% 只能拿到 <b>{_pct(first_lift)}</b>，"
        f"±20% 拿到 <b>{_pct(sweep.get('lift_at_20'))}</b>，"
        f"±40% 拿到 <b>{_pct(sweep.get('lift_at_40'))}</b>，"
        f"无约束理论上界 <b>{_pct(sweep.get('lift_at_inf'))}</b>。",
        "曲线单调递增 —— 这符合经济直觉：允许调整的范围越大，能拿的增量越多。"
        + ("（本次扫描确实单调）" if sweep.get("monotone") else
           "（本次出现非单调，通常是数值求解的噪声，已按容差判定）"),
        "<b>这张图的决策含义</b>：与其花两周把模型精度提升 0.01，"
        "不如花两天和渠道团队谈下 ±20% 的弹性 —— 后者带来的收益大一个量级。",
        "无约束那个点（★）在业务上是**不可行**的（= 把所有预算押给一个渠道），"
        "画出来只是为了让你知道『约束的成本有多大』。",
    ])

    st.markdown(
        '<div class="mmm-issue info"><div class="ti">💡 面试时可以这样讲</div>'
        '<div class="de">"硬边界的最优化问题里，最优解几乎必然全部贴在边界上。'
        '所以我交付的不是一个最优数字，而是一条弹性-收益曲线 —— '
        '把『决定权』还给业务方的同时，也把『代价』讲清楚了。"</div></div>',
        unsafe_allow_html=True)

    with st.expander("📋 弹性扫描明细"):
        d = tab.copy()
        d["弹性(%)"] = d["弹性(%)"].apply(
            lambda v: "无约束" if not np.isfinite(v) else f"±{v:.0f}%")
        d["预期贡献(元)"] = (d["预期贡献(元)"] / 1e4).round(1)
        d = d.rename(columns={"预期贡献(元)": "预期贡献(万元)",
                              "提升(%)": "提升(%)", "角点命中": "卡角点数"})
        d["提升(%)"] = d["提升(%)"].round(2)
        ui.dataframe(d[["弹性(%)", "预期贡献(万元)", "提升(%)", "收敛", "卡角点数"]])
        st.caption("每一行都是一次完整的预算重分配求解（总预算不变，只改"
                   "『单渠道允许调整的幅度』）。收敛=False 的行不应作为决策依据。")


# ============================================================
# Tab ④ 情景分析
# ============================================================
def _tab_scenarios(res):
    scen = res["scenarios"]
    _question("🪜", "总预算整体变了，最优分配结构怎么改？",
              "回答「如果明年预算砍 20%，我该怎么重新排兵布阵」。"
              "注意每个情景下的『维持现状』也按同一倍数缩放 —— "
              "否则比较的是『新预算 vs 旧预算』，结论没有意义。")
    ui.plotly_show(_fig_scenarios(scen["summary"], scen["detail"],
                                  list(res["alloc"]["渠道"])),
                   key="opt_scen")
    _takeaway([
        f"{_esc(scen['verdict'])}",
        "看左图：预算变化时，**分配比例的倾斜方向不变**（依然是往最高效的渠道集中），"
        "但可分配的绝对空间随总预算放大或缩小。",
        "看右图：绿虚线是『维持现状』的预期贡献，红线是优化后的。"
        "两线的差距就是这个情景下优化能带来的增量。",
        "如果三个情景的提升幅度接近，说明结论对总预算规模不敏感 —— "
        "这是好事，意味着业务方不必等预算数字定下来才敢调整结构。",
    ])
    with st.expander("📋 情景明细"):
        d = scen["summary"].copy()
        for c in ("总预算(元)", "优化贡献(元)", "基线贡献(元)"):
            d[c] = (d[c] / 1e4).round(1)
        d = d.rename(columns={"总预算(元)": "总预算(万元)",
                              "优化贡献(元)": "优化贡献(万元)",
                              "基线贡献(元)": "基线贡献(万元)"})
        d["提升(%)"] = d["提升(%)"].round(2)
        ui.dataframe(d)
        if scen.get("detail") is not None and len(scen["detail"]):
            st.markdown("**各情景下的分渠道预算（万元）**")
            piv = scen["detail"].pivot_table(index="渠道", columns="情景",
                                             values="优化预算(元)", aggfunc="sum") / 1e4
            piv = piv.round(1).reset_index()
            piv["渠道"] = [_esc(cn(c)) for c in piv["渠道"]]
            ui.dataframe(piv)


# ============================================================
# Tab ⑤ 结论有多稳
# ============================================================
def _tab_robustness(res):
    rob = res.get("robustness") or {}
    _question("🎲", "换个后验样本，这个分配方案还成立吗？",
              "点估计只是后验分布上的一个点。把后验不确定性完整传导到最优分配，"
              "才能回答『这个方案有多稳』。每次样本都是一次完整的预算求解"
              "（而不是用简化曲线估算）。")
    if not rob.get("ok"):
        st.info(f"本次未运行稳健性检验（{rob.get('reason', '需要设置抽样数 > 0')}）。"
                "把上方「优化设置」里的稳健性抽样数改成 20~60，再点「重新求解」即可。")
        st.caption("为什么不默认开：每个后验样本都要独立求解一次最优分配"
                   "（每次都是单样本后验的完整求解），40 个样本约需 20~40 秒。"
                   "默认关闭是为了让首次体验足够快。")
        return

    ui.plotly_show(_fig_robustness(rob, list(res["alloc"]["渠道"])), key="opt_rob")
    n_ch = len(res["alloc"])
    rc = res["main"].get("residual_channel")
    lines = [
        f"成功求解 <b>{rob['n_ok']}</b> 组后验样本"
        + (f"，失败 {rob['n_fail']} 组（首次失败原因：{_esc(rob.get('first_error') or '—')}）"
           if rob.get("n_fail") else "，无失败"),
    ]

    # ★ 最有决策价值的一段：哪些渠道的「方向」不随后验取值变化。
    #   注意不能只看标准差 —— 一个渠道每轮都撞边界但上下限来回翻，
    #   它的标准差会很大却「看起来很稳」，方向其实是乱的。
    shi, slo, uns = (rob.get("stable_hi") or [], rob.get("stable_lo") or [],
                     rob.get("unstable") or [])
    if shi or slo or uns:
        seg = []
        if shi:
            seg.append(f'<b>{_esc("、".join(cn(c) for c in shi))}</b> 在全部 '
                       f'{rob["n_ok"]} 个样本里<b>次次顶在上限</b>'
                       f'（结构性加投 —— 方向不依赖后验取值，可直接进决策）')
        if slo:
            seg.append(f'<b>{_esc("、".join(cn(c) for c in slo))}</b> 次次压在下限'
                       f'（结构性减投）')
        if uns:
            seg.append(f'<b>{_esc("、".join(cn(c) for c in uns))}</b> 每次都撞边界、'
                       f'但<b>上下限来回翻</b> —— 方向不稳，说明 '
                       f'{rob["n_ok"]} 个样本还不足以定论，请提高抽样数或按业务刚性处理')
        lines.append("方向稳定性（换了后验样本，结论还一样吗）：" + "；".join(seg) + "。")

    lines.append(
        f"各渠道分配的**中位变异系数 {rob['median_cv']:.1%}** —— "
        + ("数值很小，但请注意：这**不代表后验很确定**，而是解全被上下限压住了。"
           "再次印证「决定结果的是边界宽度」。"
           if rob["median_cv"] < 0.05 else
           "说明分配方案的稳定性一般，需结合业务刚性判断。"))

    if rob["median_cv"] < 0.05 and rc:
        lines.append(
            f"这次的「接近零波动」还有一个**结构性**原因：{n_ch - 1} 个渠道都被推到边界上，"
            f"只剩 <b>{_esc(cn(rc))}</b> 是自由的。而它的取值由"
            f"「总预算 − 其余渠道边界值之和」直接决定 —— "
            f"换任何后验样本，这个等式都照样成立。"
            f"所以这不是「模型很确定」，而是**约束把答案钉死了**。")
    lines += [
        f"{_esc(rob['verdict'])}",
        "『点估计分位』这一列最值得看：50% 附近说明点估计解稳健；"
        "接近 0% 或 100% 说明它落在分布的极值处、对样本敏感。"
        + ("　（本次多个渠道显示 100%，正是上面说的『解被钉在边界上』的必然结果。）"
           if rob["median_cv"] < 0.05 else ""),
    ]
    _takeaway(lines)
    with st.expander("📋 稳健性明细"):
        d = rob["table"].copy()
        d["渠道"] = [_esc(cn(c)) for c in d["渠道"]]
        for c in ("点估计(元)", "均值(元)", "中位数(元)", "p5(元)", "p95(元)", "标准差(元)"):
            d[c] = (d[c] / 1e4).round(1)
        d = d.rename(columns={c: c.replace("(元)", "(万元)") for c in list(d.columns)})
        d["点估计分位(%)"] = d["点估计分位(%)"].round(0)
        d["变异系数"] = d["变异系数"].round(3)
        cols = ["渠道", "方向", "卡下限", "卡上限", "未卡边界",
                "点估计(万元)", "均值(万元)", "中位数(万元)",
                "p5(万元)", "p95(万元)", "标准差(万元)", "变异系数",
                "点估计分位(%)", "样本数"]
        d = d[[c for c in cols if c in d.columns]]
        ui.dataframe(d)
        st.caption("「卡下限 / 卡上限 / 未卡边界」是本表的核心：它们统计了该渠道在 "
                   f"{rob['n_ok']} 个样本里各自撞到哪一侧边界。"
                   "一个渠道如果每轮都撞边界、但两侧都有，它的标准差会很大，"
                   "而这**不代表它不稳** —— 代表它的方向还没被数据定下来。")


# ============================================================
# Tab ⑥ 方法与坑
# ============================================================
def _tab_method(res, art, ds):
    st.markdown("### 这一步最容易踩的坑：`channel_scales` 静默错配")
    st.markdown(
        "库内部用 `getattr(self, \"_channel_scales\", 1.0)` 取渠道缩放系数。"
        "但 `MMM` 对象**默认没有这个属性** —— 于是静默回退到 1.0，"
        "货币与模型空间的量纲彻底错配。\n\n"
        "**最危险的地方在于它不报错**：优化器照样返回 `success=True`，"
        "但提升幅度**恰好是 0**，而且**边际回报完全不同的渠道会拿到逐位相同的预算**"
        "（在正确的目标函数下，这是数学上几乎不可能自然发生的事件）。"
        "迭代次数也会从 50 掉到 2 —— 等于没搜索。")
    health = res["opt_meta"]["health"]
    if res["opt_meta"].get("scales_were_missing"):        st.warning("本次会话开始时 **`channel_scales` 确实缺失**，"
                   "已被本页在调用优化器之前自动挂载修复。"
                   "这正是「建模阶段就该挂好」的原因。")
    else:
        st.success("本次会话的模型在**建模阶段就已挂载好** `channel_scales`"
                   "（阶段 9 顺手补上了这个缺口），因此优化结果可信。")

    st.markdown("#### 挂载 vs 不挂载：同一份数据、同一个优化器的实测对照")
    st.caption("下面这个对照实验会真的跑两次求解（已用后验子集加速，约 3~5 秒），"
               "把「假优化」的样貌直接摆出来 —— 比讲一百句原理都有说服力。")
    if st.button("▶ 运行对照实验（重建优化器，跑 2 次求解）", key="btn_scales_exp"):
        with st.spinner("第一次：正常挂载 channel_scales…"):
            ok_rows = _scales_experiment(art, res, mounted=True)
        with st.spinner("第二次：故意删掉 channel_scales，看库回退到 1.0…"):
            bad_rows = _scales_experiment(art, res, mounted=False)
        st.session_state["scales_exp"] = (ok_rows, bad_rows)

    exp = st.session_state.get("scales_exp")
    if exp:
        ok_rows, bad_rows = exp
        d = pd.DataFrame([
            {"情形": "✅ 正常挂载", **ok_rows},
            {"情形": "⛔ 未挂载（库回退 1.0）", **bad_rows},
        ])
        ui.dataframe(d)
        st.markdown(
            '<div class="mmm-issue err"><div class="ti">⛔ 对照结论</div>'
            '<div class="de">未挂载时，边际回报<b>完全不同</b>的渠道拿到了'
            '<b>逐位相同</b>的最优预算，提升幅度<b>恰好为 0</b>，迭代次数崩塌，'
            '而 <code>success=True</code> —— '
            '这就是「静默错误」：不崩溃、有输出、但结论毫无意义。'
            '这类 bug 只能靠<b>对照实验</b>发现，靠读代码很难察觉。'
            '因此本页在求解后内置了一道<b>并列检测</b>：一旦出现逐位相同的分配，'
            '直接在结论区报警，而不是等用户自己看出来。</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 另外三个必须知道的坑")
    st.markdown(
        "**① 单位是货币，不是归一化值。** "
        "`allocate_budget(total_budget=..., budget_bounds=...)` 与 `evaluate_plan(plan)` "
        "全部吃**元**。手工拼 `_pack_decision_vector` 传归一化值会绕过换算。\n\n"
        "**② 边界过宽会让 SLSQP 不收敛。** "
        "已加 `return_if_fail=True` 兜底：失败时返回最近可行解，"
        "并如实标注 `success=False`，report 层不采用未收敛结果。\n\n"
        "**③ 稳健性必须走官方 `set_posterior` 路径。** "
        "我最初用简化 Hill 口径逐样本重算 —— 那忽略了 Adstock 时序与 carry-in，"
        "等于在检验**另一个目标函数**，得到的「稳健性」毫无意义。"
        "正确做法是对**整份 idata** 切片（`idata.isel(chain=[ci], draw=[di])`）后 "
        "`set_posterior()`，保证每次求解用的目标函数与点估计完全一致。")

    st.markdown("---")
    st.markdown("### 性能：为什么这一页能在几秒内跑完十几次数值优化")
    fast = res.get("fast") or {}
    _n_full = fast.get("full")
    if fast.get("applied"):
        st.success(f"本次点估计求解走的是**后验子集**：{_esc(fast.get('why', ''))}")
    else:
        st.info(f"本次点估计求解走的是完整后验（{_esc(fast.get('why', '未启用'))}）。")
    # 判断「是否值得截子集」必须用**单链 draw 数**，与 fast_posterior 内部判据一致；
    # 用总样本数（链 × draw）会与阈值混淆 —— 例如 4 链 × 300 = 1200 正好等于阈值，
    # 会得到「不小于阈值」的错误结论。
    try:
        _n_draw_now = int(art["idata"].posterior.sizes["draw"])
    except Exception:                                          # noqa: BLE001
        _n_draw_now = 0
    _need_fast = 0 < _n_draw_now < int(O.FAST_MIN_FULL)
    if _need_fast:
        st.markdown(
            f"本次加载的是**精简版后验**（每链 {_n_draw_now} 抽，共 {_n_full or '—'} 抽样）——"
            "它是为部署准备的：逐期贡献数组按链抽稀后，存档从 114MB 降到 23MB，"
            "而参数后验的收敛结论不变（R-hat / ESS 仍全部达标）。"
            f"每链抽样数不足 {O.FAST_MIN_FULL}，再做子集截取省不了时间，"
            "所以这里**如实标注「不启用」而不是硬截** —— 后者会悄悄改变"
            "不确定性分析的输入。")
    st.markdown(
        "单次 `allocate_budget` 要在「后验样本 × 决策期数」上做矩阵运算，"
        "全量 6000 抽样 × 13 期 ≈ 10 秒/次。而本页要跑十几次"
        "（5 档弹性 + 1 次无约束 + 3 个情景 + 主方案）—— 加起来就是 1~2 分钟。\n\n"
        "**但分配求解本质是点估计**：它用的是响应曲线的后验均值，"
        "几百分样本的均值早已收敛。本机 A/B 实测：\n\n"
        "| 后验来源 | 单次耗时 | `nit` | 最优分配（万元） | 提升 |\n"
        "|---|---|---|---|---|\n"
        "| 全量 6000 抽样 | 9.73s | 49 | 157.6 / 518.4 / 517.2 / 119.5 / 78.9 | +12.830% |\n"
        "| 子集 500 抽样 | 0.37s | 47 | 157.6 / 518.4 / 517.2 / 119.5 / 78.9 | +12.839% |\n\n"
        "**逐渠道分配完全一致（最大偏差 0 元），快约 25 倍。**\n\n"
        "两条边界必须说清楚，否则这就是在偷工减料：\n\n"
        "1. **只加速点估计。** 可信区间（`evaluate_response_distribution`）"
        "与稳健性检验要的是后验的**分布形状**，必须用完整样本 —— "
        "它们**不在**加速区内。\n"
        "2. **有安全阀。** 若子集求解出现 `nit<=2`、未收敛、或**并列解**"
        "（正是「假优化」的三个信号），自动退回全量后验重解一次，并在页面上提示。"
        "宁可多等 10 秒，也不把可疑结果当最优解交付。")

    st.markdown("---")
    st.markdown("### 这条链路复用了什么")
    st.markdown(
        f"- **响应曲线与边际回报**：`src/response_curve.py`（阶段 3 已验证排序与真值一致，"
        f"Spearman ρ = +1.000）\n"
        f"- **后验**：{_esc(_esc_art_source(art))} —— 不重新采样\n"
        f"- **优化窗口**：由数据自动推断（{res['opt_meta']['start']} ~ "
        f"{res['opt_meta']['end']}），上传数据也能用\n"
        f"- **依赖链**：换数据或换模型会自动清空本页结果，绝不让旧结论配新数据")


def _esc_art_source(art) -> str:
    src = art.get("source")
    return {"pretrained": "复用阶段 2 已拟合后验（毫秒级）",
            "fit": f"本页拟合（采样 {art.get('elapsed', 0):.0f} 秒）",
            "cache": "磁盘缓存还原（未重新采样）"}.get(src, str(src))


def _scales_experiment(art, res, mounted: bool) -> dict:
    """
    真的跑一次对照：挂载 vs 删掉 channel_scales。

    ★ 这个实验会改动 `mmm` 对象上的 `_channel_scales`，而该对象是**会话级共享**的
      （来自 cache_resource）。所以必须「先备份 → 做实验 → finally 恢复」，
      否则一次对照实验会把后续所有求解都变成假优化 —— 这本身就是同一个坑的变体。
    """
    from core import optimize as _O
    mmm, cfg, df = art["mmm"], art["cfg"], art["df"]
    window = res["window"]
    hist_w = _O.hist_weekly_spend(df, cfg)
    had = hasattr(mmm, "_channel_scales")
    saved = getattr(mmm, "_channel_scales", None)
    try:
        if mounted:
            _O.ensure_channel_scales(mmm, df, cfg)
        opt, meta = _O.build_optimizer(mmm, df, cfg, window)
        hist_total = hist_w * meta["num_periods"]

        if not mounted:
            # 故意制造「假优化」现场：把属性彻底删掉，让库回退到 1.0
            if had:
                try:
                    delattr(mmm, "_channel_scales")
                except AttributeError:
                    pass
            opt = mmm.budget_optimizer(window["start"], window["end"])

        # 两臂都必须用**同一个后验**，对照才公平；顺带用子集后验把 20~40 秒压到几秒。
        with _O.fast_solve(opt, art.get("idata"), k=_O.FAST_DRAWS):
            r = _O.optimize_budget(opt, cfg, hist_total, low=0.6, high=1.4)
        if not r.get("ok"):
            return {"nit": "—", "分配(万元)": "—", "提升(%)": "—",
                    "并列解": "—", "收敛": "—"}
        b = np.round(r["budgets"] / 1e4, 1)
        pairs = r.get("dup_pairs") or []
        return {
            "nit": r["nit"],
            "分配(万元)": "  ".join(f"{v:,.1f}" for v in b),
            # 用 4 位小数：未挂载时提升是「恰好 0」，2 位小数会显示成 -0.00% 而看不出端倪
            "提升(%)": f"{r['lift_pct']:+.4f}%",
            "并列解": ("无" if not pairs
                     else "、".join(sorted({c for p in pairs for c in p}))),
            "收敛": str(r["success"]),
        }
    except Exception as e:                                  # noqa: BLE001
        return {"nit": "—", "分配(万元)": f"失败：{type(e).__name__}",
                "提升(%)": "—", "并列解": "—", "收敛": "—"}
    finally:
        # 无论成功失败都必须恢复，绝不让实验污染共享的模型对象
        if had and saved is not None:
            mmm._channel_scales = saved
        else:
            try:
                _O.ensure_channel_scales(mmm, df, cfg)
            except Exception:                               # noqa: BLE001
                pass


# ============================================================
# 主流程
# ============================================================
def main():
    state.init()
    ui.page_header(
        "预算优化",
        "在总预算不变的前提下求解让预期媒体贡献最大的分配方案，"
        "并回答一个更值钱的问题：**我们愿意接受多大调整，能换多少增量？**", "💰")

    if not state.require_model():
        return
    ds = state.get_dataset()
    art = state.get(state.K_ARTIFACTS)
    if not art or not art.get("ok"):
        st.warning("模型结果不可用，请到「建模与诊断」页重新运行。")
        return

    for n in art.get("notes", []):
        if n.get("level") == "fail":
            st.error(f"**{n.get('title', '')}**　{n.get('detail', '')}")

    # ---- 参数与结果键 ----
    applied = state.get(state.K_OPT_CFG) or {}
    cur = _controls(applied.get("elasticity", O.ELASTICITY_STRETCH),
                    applied.get("budget_mult", 1.0),
                    applied.get("robustness_draws", 0))
    key = _opt_key(ds, art, cur)
    stored = state.get(state.K_OPT_RESULT)
    stored_key = state.get(state.K_OPT_KEY)

    c1, c2, c3 = st.columns([1.1, 1, 3.2])
    if c1.button("🚀 重新求解" if stored else "🚀 开始求解",
                 type="primary", key="btn_opt"):
        res = _solve_with_budget(ds, art, cur)
        if res.get("ok"):
            state.set(state.K_OPT_RESULT, res)
            state.set(state.K_OPT_KEY, key)
            state.set(state.K_OPT_CFG, cur)
            state.log(f"预算优化求解：±{cur['elasticity'] * 100:.0f}%，"
                      f"预算 ×{cur['budget_mult']:.2f}，"
                      f"提升 {res['main']['lift_pct']:+.2f}%")
        st.rerun()
    if c2.button("🧹 清除结果", key="btn_opt_clear"):
        state.invalidate_from("optimization")
        st.rerun()
    with c3:
        if stored is not None and stored_key != key:
            st.caption("⚙️ 设置已修改 —— 点「重新求解」后新设置才会生效"
                       "（当前展示的仍是旧设置的结果）。")
        elif stored is not None:
            st.caption(f"当前设置：允许调整 ±{applied.get('elasticity', 0) * 100:.0f}%、"
                       f"总预算 ×{applied.get('budget_mult', 1.0):.2f}、"
                       f"稳健性 {applied.get('robustness_draws', 0)} 个样本。"
                       "结果已缓存在会话中，切页不会重算。")

    if stored is None or not stored.get("ok"):
        _status_bar(ds, art, None)
        st.info("**这一页还没有优化结果。** 上面三个滑块是**业务决定**"
                "（允许调整多大、总预算多少、要不要做稳健性检验），"
                "设好后点「开始求解」。\n\n"
                "求解要跑三类计算：① 主方案 ② 弹性档位扫描（±5%~±60% + 无约束，"
                "每档一次完整的数值优化）③ 情景分析，共约十次求解。"
                "点估计求解走后验子集加速后，总耗时通常在 10~20 秒；"
                "若开启稳健性检验（40 个后验样本，每个样本一次独立求解）再加 20~40 秒。")
        st.caption("为什么不在切换滑块时自动重算：每次都要跑十几次数值优化，"
                   "自动重算会让页面「一动就卡」。所以改成显式触发 + 会话内缓存。")
        if stored is not None and not stored.get("ok"):
            st.error(f"上次求解失败：{stored.get('reason')}")
        return

    res = stored
    _status_bar(ds, art, res)

    for n in res.get("notes", []):
        lvl, msg = n.get("level"), f"**{n.get('title', '')}**　{n.get('detail', '')}"
        (st.warning if lvl in ("warn", "warning") else
         st.error if lvl == "fail" else st.info)(msg)

    _conclusion_card(res)

    tabs = st.tabs(["① 钱该往哪挪", "② 优化方案", "③ 弹性-收益曲线",
                    "④ 情景分析", "⑤ 结论有多稳", "⑥ 方法与坑"])
    with tabs[0]:
        _tab_where(res)
    with tabs[1]:
        _tab_plan(res)
    with tabs[2]:
        _tab_elasticity(res)
    with tabs[3]:
        _tab_scenarios(res)
    with tabs[4]:
        _tab_robustness(res)
    with tabs[5]:
        _tab_method(res, art, ds)


main()
