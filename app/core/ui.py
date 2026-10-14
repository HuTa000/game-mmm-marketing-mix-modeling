"""
共享 UI 组件（依赖 streamlit）。

统一收纳：主题样式、页头、KPI 卡片、面板、校验问题清单、图片内嵌、空状态。

设计原则：
    - 只用自己的类名（mmm-*）做样式，不依赖 Streamlit 内部 DOM 结构 ——
      内部类名随版本变化，依赖它等于给自己埋雷。
    - 配色遵循中国财经习惯：**红 = 增加，绿 = 减少**（与欧美相反）。
    - 窄屏适配：所有图表走 FULL_WIDTH（自动适配新旧 Streamlit 的全宽参数），
      表格与图片外层给横向滚动容器。
"""
from __future__ import annotations

import base64
import html
import os
import re
import time

import streamlit as st

# ============================================================
# 配色与图表主题（全应用唯一来源）
# ------------------------------------------------------------
# 为什么集中在这里：`data_eda` / `modeling` / `optimize` 三个页面原先各自
# 复制了一份**完全相同**的配色常量与 `_layout()` 函数（连行号都几乎一致）。
# 后果有两个，都很难在界面上立刻看出来：
#   ① 任何一次版式调整都要改三处，漏一处就出现「同一张图两种风格」；
#   ② 渠道配色按「渠道在列表里的出现顺序」取 `PALETTE[i]`，于是**同一个渠道
#      在不同页面可能是不同颜色**（上传数据改变列顺序时更明显），
#      读者得反复回头对照图例才能确认哪条线是哪个渠道。
# 这里一次收口：版式统一、渠道色按**渠道名**固定。
# ============================================================

# ---- 语义色：承载「好坏 / 方向」含义，遵循中国财经习惯（红=增加，绿=减少）----
C_BRAND = "#2b6cb0"   # 品牌蓝：中性强调
C_UP = "#c53030"      # 红：增加 / 收入 / 正向
C_DOWN = "#2f855a"    # 绿：减少 / 成本 / 负向
C_MUTED = "#718096"   # 次要文本
C_WARN = "#b7791f"    # 警示黄
C_GOLD = "#6b46c1"    # 紫：理论上界 / 参照线
C_REV = "#c53030"     # 收入
C_SPEND = "#2b6cb0"   # 花费
C_NET = "#6b46c1"     # 控制其他因素后的「净关系」
C_RAW = "#a0aec0"     # 未控制的「毛关系」

GRID = "rgba(128,140,160,.22)"

# 中文优先字体栈：画布内文本与外层 HTML 用同一套，
# 避免中英文混排时字重 / 字宽跳变（图表里出现中文标签时尤其明显）。
FONT_STACK = ("PingFang SC, Microsoft YaHei, Noto Sans SC, Source Han Sans SC, "
              "Hiragino Sans GB, Heiti SC, sans-serif")

# ---- 分类色板：只用于区分渠道 / 类别，不承载好坏语义 ----
PALETTE = ["#2b6cb0", "#c53030", "#b7791f", "#2f855a", "#6b46c1",
           "#dd6b20", "#319795", "#d53f8c", "#4a5568", "#38a169"]

# ---- 渠道固定色：同一渠道在全局所有图里颜色一致 ----
CH_COLOR: dict[str, str] = {
    "facebook": PALETTE[0],
    "google":   PALETTE[1],
    "tiktok":   PALETTE[2],
    "youtube":  PALETTE[3],
    "kol":      PALETTE[4],
}

# 别名 → 标准渠道名（上传数据的列名被标准化前，画图时也要能对上色）
_CH_ALIAS: dict[str, str] = {
    "fb": "facebook", "meta": "facebook", "脸书": "facebook",
    "googleads": "google", "adwords": "google", "谷歌": "google",
    "tt": "tiktok", "douyin": "tiktok", "抖音": "tiktok",
    "yt": "youtube", "油管": "youtube",
    "influencer": "kol", "网红": "kol", "达人": "kol",
}


def color_of(channel: str, idx: int = 0) -> str:
    """
    取渠道的固定色；认不出的渠道回退到色板（保证任意上传渠道也能被区分开）。
    """
    key = re.sub(r"[^a-z0-9]", "", str(channel).strip().lower())
    if key in CH_COLOR:
        return CH_COLOR[key]
    mapped = _CH_ALIAS.get(key)
    if mapped:
        return CH_COLOR[mapped]
    for name in CH_COLOR:                        # 例：facebook_spend → facebook
        if key.startswith(name):
            return CH_COLOR[name]
    return PALETTE[int(idx) % len(PALETTE)]


def colors_of(channels) -> list[str]:
    """按渠道名批量取色（跨页面一致）。"""
    return [color_of(c, i) for i, c in enumerate(channels)]


def apply_theme(fig, height: int | None = None, title: str = "",
                legend_top: bool = True, hovermode: str | None = None,
                barmode: str | None = None, showlegend: bool | None = None):
    """
    统一的 Plotly 版式（三个分析页共用）。

    参数
    ----
      height     : 固定像素高度；不传则随容器自适应（配合 responsive 更利于窄屏）
      title      : 左上角小标题。刻意做得比 Plotly 默认标题轻 —— 图的重点是数据本身
      legend_top : 图例横排到图上方。窄屏时图例留在右侧会挤压绘图区，置顶可避免
      hovermode  : 悬浮模式，时间序列建议 "x unified"（一条竖线上显示全部系列）
    """
    fig.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        margin=dict(l=8, r=8, t=34 if title else 16, b=8),
        font=dict(size=12, family=FONT_STACK),
        hoverlabel=dict(font_size=12, font_family=FONT_STACK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13.5),
                                     x=0, xanchor="left"))
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      x=0, font=dict(size=11)))
    if showlegend is not None:
        fig.update_layout(showlegend=bool(showlegend))
    if hovermode:
        fig.update_layout(hovermode=hovermode)
    if barmode:
        fig.update_layout(barmode=barmode)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if height:
        fig.update_layout(height=int(height))
    return fig

_CSS = """
<style>
:root{
  --mmm-brand:#2b6cb0; --mmm-up:#c53030; --mmm-down:#2f855a;
  --mmm-border:rgba(128,140,160,.28); --mmm-surface:rgba(128,140,160,.07);
  --mmm-fg:var(--text-color,#1a202c); --mmm-muted:#718096;
  --mmm-font:__FONT__; --mmm-radius:12px;
  --mmm-shadow:0 1px 2px rgba(26,32,44,.05), 0 6px 18px rgba(26,32,44,.05);
}
/* 全局字体：与图表画布共用同一套，避免中英文混排时字重/字宽跳变 */
html, body, .stApp, [data-testid="stAppViewContainer"]{font-family:var(--mmm-font)}
/* 页面留白：窄屏不贴边 */
.block-container{padding-top:2.0rem;padding-bottom:3rem;max-width:1500px}
@media (max-width:900px){.block-container{padding-left:.8rem;padding-right:.8rem}}
h2{font-size:1.22rem;margin:.55rem 0 .45rem}
h3{font-size:1.04rem}
/* 侧栏视觉稍作收敛：与主区区分，但不抢注意力 */
[data-testid="stSidebar"]{border-right:1px solid var(--mmm-border)}
[data-testid="stSidebar"] .block-container{padding-top:1.1rem}

/* 控件统一（只碰外观，不碰布局：即使选择器失效也只是回落到原生样式，不会坏页） */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button{
  border-radius:9px; font-weight:600; border:1px solid var(--mmm-border);
  transition:transform .06s ease, box-shadow .16s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  box-shadow:0 3px 12px rgba(43,108,176,.20); transform:translateY(-1px);
}
.stTabs [data-baseweb="tab-list"]{gap:4px; border-bottom:1px solid var(--mmm-border)}
.stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0; padding:7px 13px; font-size:.87rem}
.stTabs [aria-selected="true"]{font-weight:650}
[data-testid="stExpander"] summary{font-weight:600; font-size:.88rem}
::-webkit-scrollbar{height:9px; width:9px}
::-webkit-scrollbar-thumb{background:rgba(128,140,160,.42); border-radius:6px}
::-webkit-scrollbar-track{background:transparent}

.mmm-hero{
  border:1px solid var(--mmm-border); border-radius:14px; padding:18px 22px;
  background:linear-gradient(135deg, rgba(43,108,176,.18), rgba(43,108,176,.03));
  margin-bottom:16px;
}
.mmm-hero h1{margin:0;font-size:1.42rem;line-height:1.35;color:var(--mmm-fg)}
.mmm-hero p{margin:6px 0 0;font-size:.90rem;color:var(--mmm-muted);line-height:1.6}

.mmm-kpi{
  border:1px solid var(--mmm-border); border-radius:12px; padding:13px 15px;
  background:var(--mmm-surface); height:100%;
}
.mmm-kpi .l{font-size:.75rem;color:var(--mmm-muted);margin-bottom:5px;letter-spacing:.02em}
.mmm-kpi .v{font-size:1.28rem;font-weight:700;color:var(--mmm-fg);line-height:1.25;
            font-variant-numeric:tabular-nums;word-break:break-all}
.mmm-kpi .d{font-size:.74rem;color:var(--mmm-muted);margin-top:4px}

.mmm-up{color:var(--mmm-up) !important}       /* 红：增加 */
.mmm-down{color:var(--mmm-down) !important}   /* 绿：减少 */
.mmm-brand{color:var(--mmm-brand) !important}
.mmm-muted{color:var(--mmm-muted) !important}

.mmm-panel{
  border:1px solid var(--mmm-border); border-radius:12px; padding:16px 18px;
  background:var(--mmm-surface); margin:10px 0 14px;
}
.mmm-panel .t{font-weight:700;font-size:1.0rem;color:var(--mmm-fg);margin-bottom:2px}
.mmm-panel .s{font-size:.80rem;color:var(--mmm-muted);line-height:1.55}

.mmm-issue{border-left:4px solid var(--mmm-muted);border-radius:8px;padding:9px 13px;
           margin:7px 0;background:var(--mmm-surface);font-size:.84rem;line-height:1.6}
.mmm-issue .ti{font-weight:600;color:var(--mmm-fg)}
.mmm-issue .de{color:var(--mmm-muted);margin-top:3px}
.mmm-issue .hi{color:var(--mmm-brand);margin-top:3px;font-size:.80rem}
.mmm-issue.err{border-left-color:#c53030}
.mmm-issue.warn{border-left-color:#d69e2e}
.mmm-issue.info{border-left-color:#2b6cb0}

.mmm-empty{border:1px dashed var(--mmm-border);border-radius:12px;padding:26px 22px;
           text-align:center;color:var(--mmm-muted)}
.mmm-empty .e-t{font-size:1.0rem;font-weight:600;color:var(--mmm-fg);margin:6px 0 4px}
.mmm-empty .e-h{font-size:.83rem;line-height:1.6}

.mmm-fig{border:1px solid var(--mmm-border);border-radius:12px;padding:10px;
         background:var(--mmm-surface);overflow-x:auto}
.mmm-fig img{width:100%;height:auto;display:block;border-radius:6px}
.mmm-fig .cap{font-size:.78rem;color:var(--mmm-muted);margin-top:7px;line-height:1.55}

/* 表格横向滚动，窄屏不撑破布局 */
[data-testid="stDataFrame"]{overflow-x:auto}

/* 侧栏品牌区 */
.mmm-brandbox{border:1px solid var(--mmm-border);border-radius:10px;padding:10px 12px;
              background:var(--mmm-surface);margin-bottom:10px}
.mmm-brandbox .n{font-weight:700;font-size:.92rem;color:var(--mmm-fg)}
.mmm-brandbox .d{font-size:.74rem;color:var(--mmm-muted);margin-top:3px;line-height:1.5}

/* 结论卡片的检查表（阶段 7 起使用） */
.mmm-checkscroll{overflow-x:auto;margin:2px 0 6px}
.mmm-check{width:100%;border-collapse:collapse;font-size:.83rem;min-width:540px}
.mmm-check th{text-align:left;font-weight:600;color:var(--mmm-muted);font-size:.75rem;
  padding:6px 9px;border-bottom:1px solid var(--mmm-border);white-space:nowrap}
.mmm-check td{padding:7px 9px;border-bottom:1px solid var(--mmm-border);
  vertical-align:top;color:var(--mmm-fg);line-height:1.55}
.mmm-check tr:last-child td{border-bottom:none}
.mmm-check td.ic{width:26px;text-align:center}
.mmm-check td.lb{white-space:nowrap;font-weight:600}
.mmm-check td.vl{white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:600}
.mmm-check td.wy{color:var(--mmm-muted);font-size:.80rem}
.mmm-check .fix{color:var(--mmm-brand);margin-top:3px}

/* 步骤条 */
.mmm-steps{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 14px}
.mmm-step{font-size:.76rem;padding:4px 11px;border-radius:20px;
          border:1px solid var(--mmm-border);color:var(--mmm-muted)}
.mmm-step.on{background:var(--mmm-brand);border-color:var(--mmm-brand);color:#fff;font-weight:600}
.mmm-step.done{border-color:var(--mmm-down);color:var(--mmm-down)}

/* KPI 卡片：悬停轻微抬升，暗示"可解读"；数值用等宽数字避免跳动 */
.mmm-kpi{transition:box-shadow .16s ease, transform .16s ease}
.mmm-kpi:hover{box-shadow:var(--mmm-shadow);transform:translateY(-1px)}

/* 结论 / 警示区块：左侧粗色条 + 标题，比 st.info 更能承载"业务结论" */
.mmm-callout{border:1px solid var(--mmm-border);border-left:5px solid var(--mmm-brand);
  border-radius:10px;padding:12px 15px;margin:10px 0;background:var(--mmm-surface)}
.mmm-callout .ct{font-weight:700;font-size:.95rem;color:var(--mmm-fg);margin-bottom:4px}
.mmm-callout .cb{font-size:.86rem;color:var(--mmm-fg);line-height:1.65}
.mmm-callout .cb ul{margin:6px 0 0 18px;padding:0}
.mmm-callout .cb li{margin:2px 0}
.mmm-callout.up{border-left-color:var(--mmm-up)}
.mmm-callout.down{border-left-color:var(--mmm-down)}
.mmm-callout.warn{border-left-color:#d69e2e}

/* 小标签：用于渠道名 / 状态 / 口径提示 */
.mmm-badge{display:inline-block;font-size:.74rem;padding:1px 8px;border-radius:20px;
  border:1px solid var(--mmm-border);color:var(--mmm-muted);margin-right:5px;white-space:nowrap}
.mmm-badge.brand{border-color:rgba(43,108,176,.45);color:var(--mmm-brand)}
.mmm-badge.up{border-color:rgba(197,48,48,.45);color:var(--mmm-up)}
.mmm-badge.down{border-color:rgba(47,133,90,.45);color:var(--mmm-down)}

/* 键值对列表：比两列表格更省空间，窄屏不挤 */
.mmm-kv{margin:6px 0;font-size:.86rem;line-height:1.75}
.mmm-kv .k{color:var(--mmm-muted);display:inline-block;min-width:7.5em}
.mmm-kv .v{color:var(--mmm-fg);font-weight:600;font-variant-numeric:tabular-nums}

/* 打印：面试/汇报时把页面存成 PDF，去掉交互外壳 */
@media print{
  [data-testid="stSidebar"], [data-testid="stToolbar"], header{display:none !important}
  .block-container{max-width:100% !important;padding:0 !important}
  .mmm-fig, .mmm-kpi, .mmm-callout{break-inside:avoid}
}
</style>
"""


def inject() -> None:
    """注入样式（每次页面重跑都会注入，Streamlit 会去重）。"""
    st.markdown(_CSS.replace("__FONT__", FONT_STACK), unsafe_allow_html=True)


def _detect_width_api() -> dict:
    """
    全宽参数的兼容层。

    Streamlit 1.65 起用 `width="stretch"` 取代 `use_container_width=True`
    （旧参数会在 2025-12-31 后移除）。这里**探测运行时签名**而不是猜版本号：
    签名里有 width 就用新写法，否则回退旧写法 —— 两边都不会报错、也不会有弃用警告。
    """
    try:
        import inspect
        import streamlit as _st
        if "width" in inspect.signature(_st.dataframe).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


FULL_WIDTH: dict = _detect_width_api()


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """
    页面主标题区。

    title / subtitle 支持两种写法，都会生效：内联 HTML（如 `<b>…</b>`）与
    `**加粗**`。因此传入的应当是**开发者可控的静态文案**；若内容来自用户上传
    的数据，请先用 `md_esc()` 处理。
    """
    st.markdown(
        f'<div class="mmm-hero"><h1>{icon} {rich(title)}</h1>'
        f'{f"<p>{rich(subtitle)}</p>" if subtitle else ""}</div>',
        unsafe_allow_html=True)


def kpi_row(items: list[dict], per_row: int = 5) -> None:
    """
    KPI 卡片行。
    items 每项：{"label": str, "value": str, "delta": str|None,
                "tone": "up"|"down"|"brand"|None, "help": str|None}
    tone 遵循中国习惯：up=红（增加）、down=绿（减少）。
    """
    if not items:
        return
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        cols = st.columns(len(chunk), gap="small")
        for col, it in zip(cols, chunk):
            tone = it.get("tone")
            cls = {"up": "mmm-up", "down": "mmm-down", "brand": "mmm-brand"}.get(tone, "")
            delta = it.get("delta")
            with col:
                st.markdown(
                    f'<div class="mmm-kpi"><div class="l">{it.get("label","")}</div>'
                    f'<div class="v {cls}">{it.get("value","—")}</div>'
                    f'{f"<div class=\"d\">{delta}</div>" if delta else ""}'
                    f'</div>',
                    unsafe_allow_html=True)
                if it.get("help"):
                    st.caption(it["help"])


def panel(title: str, subtitle: str = "") -> None:
    """区块标题（支持 `<b>` 与 `**加粗**`）。"""
    st.markdown(
        f'<div class="mmm-panel"><div class="t">{rich(title)}</div>'
        f'{f"<div class=\'s\'>{rich(subtitle)}</div>" if subtitle else ""}</div>',
        unsafe_allow_html=True)


class StatusBox:
    """
    长任务进度条（`st.status` 的兼容包装）。

    为什么需要包装，而不是直接用 `st.status(...)`
    --------------------------------------------
    `st.status()` 返回的 `StatusContainer` **只有** `update(label=, expanded=, state=)`
    与普通的 `write / progress / markdown / error` 等方法；
    它**没有** `.step()` / `.done()` / `.fail()`。

    我一开始按直觉写了 `box.done("完成")`，结果「点按钮 → 页面直接抛异常」，
    而且**建模页 / 优化页 / 报告页三处全中招** —— 这类"方法名想当然"的错误
    在流式组件上特别容易犯，而它只有在真正点按钮时才会暴露
    （页面首次渲染完全不报错）。所以统一收口到这里，并做能力探测：
    若某个 streamlit 版本真的提供了 `.step()/.done()/.fail()`，就优先用它；
    否则回退到受支持的写法。这样升级/降级 streamlit 都不会崩。

    用法::

        box = ui.StatusBox("正在求解…")
        box.progress(0.3, "第一阶段")
        box.step("已完成 channel_scales 校验")
        box.done("求解完成（18 秒）")     # 或 box.fail("求解失败：…")
    """

    def __init__(self, label: str, expanded: bool = True):
        self._t0 = time.time()
        self._box = st.status(label, expanded=expanded)
        self.last_msg = ""

    # ---- 进度 ----
    def progress(self, frac: float, msg: str | None = None) -> None:
        if msg:
            self.last_msg = str(msg)
        try:
            self._box.progress(max(0.0, min(1.0, float(frac))), text=msg or None)
        except Exception:                                      # noqa: BLE001
            pass

    def step(self, msg: str) -> None:
        """追加一行进度旁白。"""
        self.last_msg = str(msg)
        if hasattr(self._box, "step"):                         # 未来版本若真有
            try:
                self._box.step(msg)
                return
            except Exception:                                  # noqa: BLE001
                pass
        try:
            self._box.write(f"▸ {msg}")
        except Exception:                                      # noqa: BLE001
            pass

    def write(self, msg: str) -> None:
        try:
            self._box.write(msg)
        except Exception:                                      # noqa: BLE001
            pass

    # ---- 收尾 ----
    def done(self, label: str) -> None:
        self.close(label, "complete")

    def fail(self, label: str) -> None:
        self.close(label, "error")

    def close(self, label: str, state: str = "complete") -> None:
        """state ∈ {"running", "complete", "error"}。"""
        if hasattr(self._box, "update"):
            try:
                self._box.update(label=label, state=state,
                                 expanded=(state == "error"))
                return
            except Exception:                                  # noqa: BLE001
                pass
        try:
            (self._box.error if state == "error" else self._box.success)(label)
        except Exception:                                      # noqa: BLE001
            pass

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0


def steps(labels: list[str], current: int, done_upto: int = -1) -> None:
    html = ['<div class="mmm-steps">']
    for i, lb in enumerate(labels):
        cls = "on" if i == current else ("done" if i <= done_upto else "")
        html.append(f'<span class="mmm-step {cls}">{i+1}. {lb}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def issues_panel(report) -> None:
    """
    渲染数据校验的问题清单。**分级配色**：
        error 红（阻断） / warning 黄（已自动修复） / info 蓝（提示）
    这是"上传兜底"面向用户的出口 —— 有问题要说清楚"是什么、为什么、怎么办"。
    """
    if report is None:
        return
    c = report.counts()
    if c["error"] == 0 and c["warning"] == 0 and c["info"] == 0:
        st.success("数据校验通过，未发现问题。")
        return

    order = {"error": 0, "warning": 1, "info": 2}
    icon = {"error": "⛔", "warning": "⚠️", "info": "ℹ️"}
    for it in sorted(report.issues, key=lambda x: order.get(x.level, 3)):
        detail = f'<div class="de">{it.detail}</div>' if it.detail else ""
        hint = f'<div class="hi">→ {it.hint}</div>' if it.hint else ""
        st.markdown(
            f'<div class="mmm-issue {"err" if it.level=="error" else ("warn" if it.level=="warning" else "info")}">'
            f'<div class="ti">{icon.get(it.level,"")} {it.title}</div>{detail}{hint}</div>',
            unsafe_allow_html=True)


def empty_state(title: str, hint: str = "", icon: str = "🗂️") -> None:
    st.markdown(
        f'<div class="mmm-empty"><div style="font-size:2rem">{icon}</div>'
        f'<div class="e-t">{title}</div>'
        f'{f"<div class=\'e-h\'>{hint}</div>" if hint else ""}</div>',
        unsafe_allow_html=True)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def rich(s) -> str:
    """
    把 `**加粗**` 转成 `<b>…</b>`，但**不做 HTML 转义**。

    为什么需要它：页面标题、图的解读、结论卡这类文案大量通过
    `st.markdown(..., unsafe_allow_html=True)` 拼进 HTML。HTML 里 `**` 不会被解析，
    结果是页面上直接出现**裸星号**（实测：优化页副标题、
    报告页副标题、以及各页「这张图怎么用」的列表都有这个问题）。

    与 `md_esc()` 的分工
    -------------------
      · `rich()`   —— 不转义，允许内联 `<b>` 等标签。
        用于**开发者可控的静态文案**（如首页标题里的 `<b>过去的钱花得值不值</b>`）。
      · `md_esc()` —— 先转义再放行 `**`。
        用于**含用户数据**的文本（如上传数据里的渠道名），防止 HTML 注入。
    """
    return _BOLD_RE.sub(r"<b>\1</b>", str(s))


def md_esc(s) -> str:
    """
    转义 HTML 后**仅**放行 `**加粗**`。

    用于会混入用户数据的文案：直接 escape 会让用户看到裸露的星号；
    而完全不 escape 时，**上传数据里的渠道名**可能被当成 HTML 解析（注入）。
    所以「先整体转义、再放行固定语法」。
    """
    return _BOLD_RE.sub(r"<b>\1</b>", html.escape(str(s)))


def callout(title: str, body: str = "", tone: str = "") -> None:
    """
    结论 / 警示区块 —— 比 `st.info` 更适合承载「业务结论」。

    tone ∈ {"", "up", "down", "warn"}：分别对应品牌蓝 / 红（增加）/ 绿（减少）/ 黄（注意）。
    """
    tone = tone if tone in ("up", "down", "warn") else ""
    b = f'<div class="cb">{md_esc(body)}</div>' if body else ""
    st.markdown(f'<div class="mmm-callout {tone}"><div class="ct">{md_esc(title)}</div>{b}</div>',
                unsafe_allow_html=True)


def badge_html(text: str, tone: str = "") -> str:
    """行内小标签的 HTML 片段（供嵌进其他 markdown 一起渲染）。"""
    tone = tone if tone in ("brand", "up", "down") else ""
    return f'<span class="mmm-badge {tone}">{md_esc(text)}</span>'


def kv_list(items: list[tuple[str, str]]) -> None:
    """紧凑的键值对清单（比两列表格更省横向空间，窄屏友好）。"""
    if not items:
        return
    rows = "".join(
        f'<div><span class="k">{md_esc(k)}</span><span class="v">{md_esc(v)}</span></div>'
        for k, v in items)
    st.markdown(f'<div class="mmm-kv">{rows}</div>', unsafe_allow_html=True)


def reset_button() -> None:
    """
    侧栏「重置会话」按钮。

    为什么单独抽出来：白板演示时前一位演示者留下的状态（已载入数据 / 已拟合 /
    已优化）会成为下一位的干扰；而 MMM 的拟合结果又相当贵，不能每次刷新就清。
    这里给一个显式的、可解释的出口。
    """
    if st.button("♻️ 重置本次会话", help="清空已载入的数据、模型、优化与报告结果"
                                        "（不会删除磁盘上的阶段产物与缓存）",
                 **FULL_WIDTH):
        st.session_state.clear()
        st.rerun()


def _b64(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return None


def figure(path: str, caption: str = "", missing_hint: str = "") -> bool:
    """
    内嵌展示图片（base64，避免路径失效导致图裂）。
    返回是否成功展示 —— 便于调用方在缺图时给出降级提示。
    """
    if not os.path.isfile(path):
        empty_state("图表尚未生成", missing_hint or f"缺少文件：{os.path.basename(path)}", "📉")
        return False
    b64 = _b64(path)
    if b64 is None:
        empty_state("图表读取失败", f"无法读取：{os.path.basename(path)}", "📉")
        return False
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    st.markdown(f'<div class="mmm-fig"><img src="data:image/png;base64,{b64}"/>{cap}</div>',
                unsafe_allow_html=True)
    return True


def plotly_config(height: int | None = None) -> dict:
    """
    Plotly 统一配置。
    - responsive=True：容器变化时重绘（窄屏适配的关键）
    - displayModeBar=False：去掉悬浮工具条，避免遮挡窄屏内容
    - 导出按钮保留：面试演示时方便把图存下来
    """
    return {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
        "toImageButtonOptions": {"format": "png", "scale": 2,
                                 "filename": "mmm_chart"},
    }


def plotly_show(fig, height: int | None = None, key: str | None = None,
                *, theme: bool = True) -> None:
    """
    统一的 Plotly 渲染入口（保证全宽与响应式配置一致）。

    默认套用全应用统一版式（`theme=True`）—— 这样即使某张图忘了调
    `apply_theme()`，也不会出现「这一张的字体/网格跟别人不一样」。
    调用方若已自行设置过版式，重复套用等价设置是幂等的。
    """
    if theme:
        try:
            apply_theme(fig, height=height)
        except Exception:                                      # noqa: BLE001
            pass
    elif height:
        fig.update_layout(height=height)
    for kw in ({**FULL_WIDTH, "config": plotly_config(), "key": key},
               {**FULL_WIDTH, "config": plotly_config()},
               {"config": plotly_config()}):
        try:
            st.plotly_chart(fig, **kw)
            return
        except TypeError:
            continue
        except Exception:
            continue


def dataframe(df, *, height: int | None = None, caption: str = "", hide_index: bool = True):
    kw = dict(FULL_WIDTH)
    if height:
        kw["height"] = height
    st.dataframe(df, hide_index=hide_index, **kw)
    if caption:
        st.caption(caption)


def placeholder_page(stage_no: int, title: str, subtitle: str, bullets: list[str],
                     ready: list[str], icon: str = "🚧") -> None:
    """
    未完成页面的占位。
    明确告诉用户"这一页做什么、依赖哪些已就绪能力"，
    而不是一句"敬请期待" —— 演示时这里也是说明架构的窗口。
    """
    page_header(f"{title}（阶段 {stage_no}）", subtitle, icon)
    st.info(f"本页将在**阶段 {stage_no}** 实现。当前先说明它的职责与依赖，便于自检架构。")
    st.markdown("**本页要回答的业务问题**")
    for b in bullets:
        st.markdown(f"- {b}")
    st.markdown("**已就绪、可直接复用的能力**")
    for r in ready:
        st.markdown(f"- ✅ {r}")
    st.markdown("**与其它页面的数据契约**")
    st.markdown("→ 读取 `st.session_state` 中的共享结果；写入前会做依赖链检查，"
                "上游变更会自动清空下游，避免出现自相矛盾的结论。")


def sidebar_brand() -> None:
    st.markdown(
        '<div class="mmm-brandbox"><div class="n">🎯 游戏出海 · 广告效果 MMM</div>'
        '<div class="d">贝叶斯营销组合建模<br>Adstock + Hill 饱和 · NUTS 采样</div></div>',
        unsafe_allow_html=True)


def status_pills(items: list[tuple[str, bool]]) -> None:
    """侧栏状态灯：一眼看出哪些阶段产物已就绪。"""
    for name, ok in items:
        st.markdown(f"{'🟢' if ok else '⚪'} {name}")
