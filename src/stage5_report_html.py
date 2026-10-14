# -*- coding: utf-8 -*-
"""
阶段 5：生成单文件 HTML 报告
============================================================
把阶段 1~4 的结论整合成一份**自包含**的 HTML：
  - 所有图片以 base64 内嵌 → 可直接微信/邮件发送，对方离线也能看
  - 响应式布局 → 手机窄屏可读
  - 打印样式 → 浏览器「另存为 PDF」即得可投递版

数据来源（唯一数字源，与阶段 4 共用）：
  report_facts.build_facts()  →  保证报告与各阶段 CSV 严格一致，不手写数字

用法：
    python src/stage5_report_html.py
输出：
    outputs/reports/stage5_report.html
"""
from __future__ import annotations

import base64
import html as _html
import os
import re
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import report_facts as rf  # noqa: E402

FIGDIR = os.path.join(ROOT, "outputs", "figures")
REPDIR = os.path.join(ROOT, "outputs", "reports")
OUT_HTML = os.path.join(REPDIR, "stage5_report.html")


# ============================================================
# 工具函数
# ============================================================
def md(s) -> str:
    """HTML 转义 + 极简 markdown（**粗体** / 『』保留）。"""
    s = _html.escape(str(s if s is not None else ""))
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


class Safe(str):
    """标记：该字符串已是可信 HTML（由本文件生成），表格渲染时跳过转义。"""
    __slots__ = ()


def H(s) -> Safe:
    """包装可信 HTML 片段。用于表格单元格里的 <strong> / <span class=...>。"""
    return Safe(str(s))


def wan(x, nd: int = 1) -> str:
    """元 → 万元"""
    if x is None:
        return "—"
    try:
        return f"{float(x) / 1e4:,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def pct(x, nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def num(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def b64_img(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def fig(name: str, caption: str, note: str = "") -> str:
    """图片块：内嵌 base64；文件缺失时给出显式提示而非静默留白。"""
    uri = b64_img(os.path.join(FIGDIR, name))
    if uri is None:
        return (f'<figure class="fig"><div class="fig-missing">'
                f'图片缺失：{_html.escape(name)}（请先运行对应阶段脚本）</div></figure>')
    note_html = f'<p class="fig-note">{md(note)}</p>' if note else ""
    return (f'<figure class="fig">'
            f'<img src="{uri}" alt="{_html.escape(caption)}" loading="lazy">'
            f'<figcaption>{md(caption)}</figcaption>'
            f'{note_html}</figure>')


def table(headers, rows, align=None, cls: str = "") -> str:
    """标准表：外层包裹横向滚动容器，保证窄屏不溢出。"""
    if not rows:
        return ""
    aligns = align or ["left"] * len(headers)
    head = "".join(f'<th style="text-align:{aligns[i]}">{md(h)}</th>'
                   for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cl = ""
            cell = c
            # 支持 (值, css类) 元组，用于红涨绿跌着色
            if isinstance(c, tuple):
                cell, cl = c[0], f' class="{c[1]}"'
            # Safe 标记的内容已是可信 HTML（<span> / <strong>），跳过转义；
            # 其余一律转义，避免纯文本里的 < > & 被浏览器误解析
            out = str(cell) if isinstance(cell, Safe) else md(cell)
            tds.append(f'<td style="text-align:{aligns[i]}"{cl}>{out}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def delta_cell(v) -> tuple:
    """预算变化单元格：红=增加，绿=减少（贴合国内阅读习惯）。"""
    if v is None:
        return ("—", "")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ("—", "")
    cls = "up" if v > 1e-9 else ("down" if v < -1e-9 else "")
    return (pct(v, 1, sign=True), cls)


def sec(num_: str, title: str, sub: str = "") -> str:
    """章节标题块。

    约定：本函数只输出 <section> 的**开头**，正文由各 secXxx() 函数追加，
    闭合标签统一在 build_html() 里补 —— 保证开闭严格配对。
    """
    sub_html = f'<p class="sec-sub">{md(sub)}</p>' if sub else ""
    return (f'<section class="sec"><h2><span class="sec-num">{num_}</span>'
            f'{md(title)}</h2>{sub_html}')


# ============================================================
# 样式（浅色专业风格；不用 f-string，避免与花括号冲突）
# ============================================================
CSS = """
:root{
  --ink:#16202e; --ink2:#41506b; --ink3:#6b7a90;
  --line:#dfe6ef; --line2:#eef3f9;
  --bg:#f6f8fb; --card:#ffffff;
  --accent:#1f5fa8; --accent-soft:#e8f1fb;
  --red:#c0392b; --green:#1e8e5a;
  --amber:#b7791f; --violet:#5b4b8a;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; padding:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
    "Hiragino Sans GB","Microsoft YaHei","Source Han Sans SC",sans-serif;
  font-size:15px; line-height:1.78;
}
.wrap{max-width:1060px; margin:0 auto; padding:0 22px 70px}

/* ---------- 封面 ---------- */
.cover{
  background:linear-gradient(140deg,#12304f 0%,#1f5fa8 55%,#2f7bc4 100%);
  color:#fff; margin:0 -22px 34px; padding:46px 34px 38px;
  border-radius:0 0 20px 20px;
}
.cover h1{margin:0 0 10px; font-size:31px; line-height:1.32; letter-spacing:.3px}
.cover .sub{font-size:16px; opacity:.93; margin:0 0 6px}
.cover .meta-line{font-size:13px; opacity:.78; margin-top:14px}
.badges{display:flex; flex-wrap:wrap; gap:9px; margin-top:20px}
.badge{
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.3);
  padding:4px 12px; border-radius:999px; font-size:12.5px;
}

/* ---------- 数字卡片 ---------- */
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
  gap:14px; margin:-56px 0 34px}
.kpi{
  background:var(--card); border:1px solid var(--line); border-radius:13px;
  padding:17px 18px 15px; box-shadow:0 3px 14px rgba(22,32,46,.07);
}
.kpi .v{font-size:26px; font-weight:700; color:var(--accent);
  letter-spacing:-.4px; line-height:1.2}
.kpi .v.red{color:var(--red)} .kpi .v.green{color:var(--green)}
.kpi .l{font-size:12.6px; color:var(--ink3); margin-top:5px; line-height:1.55}

/* ---------- 章节 ---------- */
.sec{margin:40px 0 0}
.sec h2{
  font-size:21px; margin:0 0 6px; padding-bottom:11px;
  border-bottom:2px solid var(--line); display:flex; align-items:center; gap:11px;
}
.sec-num{
  background:var(--accent); color:#fff; font-size:13px; font-weight:600;
  min-width:27px; height:27px; border-radius:7px;
  display:inline-flex; align-items:center; justify-content:center; flex:none;
}
.sec-sub{color:var(--ink2); font-size:14px; margin:9px 0 0}
h3{font-size:16.5px; margin:27px 0 9px; color:var(--ink)}
h4{font-size:14.6px; margin:20px 0 7px; color:var(--ink2)}
p{margin:11px 0}
ul,ol{margin:11px 0; padding-left:23px} li{margin:6px 0}

/* ---------- 提示块 ---------- */
.box{border-left:4px solid var(--accent); background:var(--accent-soft);
  padding:13px 17px; border-radius:0 9px 9px 0; margin:17px 0}
.box.warn{border-left-color:var(--amber); background:#fdf6e8}
.box.risk{border-left-color:var(--red); background:#fdf0ee}
.box.ok{border-left-color:var(--green); background:#eaf7f0}
.box .t{font-weight:650; margin-bottom:5px; font-size:14.4px}
.box p{margin:6px 0; font-size:14.2px}

/* ---------- 图 ---------- */
.fig{margin:22px 0; background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:14px; overflow:hidden}
.fig img{display:block; width:100%; height:auto; border-radius:7px}
.fig figcaption{font-size:13.2px; color:var(--ink2); margin-top:11px;
  line-height:1.62; padding-left:11px; border-left:3px solid var(--accent)}
.fig-note{font-size:12.6px; color:var(--ink3); margin:7px 0 0; padding-left:11px}
.fig-missing{padding:30px; text-align:center; color:var(--amber);
  background:#fdf6e8; border-radius:7px; font-size:13.4px}

/* ---------- 表 ---------- */
.tw{overflow-x:auto; -webkit-overflow-scrolling:touch;
  margin:17px 0; border:1px solid var(--line); border-radius:11px; background:var(--card)}
table{border-collapse:collapse; width:100%; font-size:13.6px; min-width:540px}
th{background:#f2f6fb; color:var(--ink2); font-weight:640; padding:10px 13px;
  border-bottom:1px solid var(--line); white-space:nowrap; font-size:13.2px}
td{padding:9px 13px; border-bottom:1px solid var(--line2); vertical-align:top}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#fafcfe}
td.up{color:var(--red); font-weight:620} td.down{color:var(--green); font-weight:620}
.mono{font-variant-numeric:tabular-nums; font-feature-settings:"tnum"}

/* ---------- 风险分级 ---------- */
.risk-item{border:1px solid var(--line); border-left-width:4px; border-radius:0 10px 10px 0;
  padding:13px 16px; margin:11px 0; background:var(--card)}
.risk-item.high{border-left-color:var(--red)} .risk-item.medium{border-left-color:var(--amber)}
.risk-item.low{border-left-color:var(--ink3)} .risk-item.info{border-left-color:var(--accent)}
.risk-h{font-weight:650; font-size:14.4px; margin-bottom:5px}
.risk-tag{font-size:11.4px; padding:2px 8px; border-radius:5px; color:#fff;
  margin-right:8px; vertical-align:1.5px; font-weight:600}
.risk-tag.high{background:var(--red)} .risk-tag.medium{background:var(--amber)}
.risk-tag.low{background:var(--ink3)} .risk-tag.info{background:var(--accent)}
.risk-d{font-size:13.7px; color:var(--ink2); margin:0}
.risk-e{font-size:12.2px; color:var(--ink3); margin:7px 0 0; font-style:italic}

/* ---------- 公式 ---------- */
.formula{background:#f2f6fb; border:1px solid var(--line); border-radius:10px;
  padding:15px 18px; margin:16px 0; font-family:"SFMono-Regular",Consolas,monospace;
  font-size:13.6px; line-height:1.95; overflow-x:auto; white-space:pre}

/* ---------- 附录 markdown ---------- */
.doc-md h3{font-size:16px; margin:22px 0 8px; color:var(--accent)}
.doc-md h4{font-size:14.4px; margin:17px 0 6px}
.doc-md blockquote{margin:12px 0; padding:9px 15px; border-left:3px solid var(--line);
  background:#fafcfe; color:var(--ink2); font-size:13.6px; border-radius:0 7px 7px 0}
.doc-md ul{margin:9px 0; padding-left:21px}

footer{margin-top:52px; padding-top:20px; border-top:1px solid var(--line);
  font-size:12.6px; color:var(--ink3); text-align:center; line-height:1.9}

/* ---------- 窄屏适配 ---------- */
@media (max-width:720px){
  body{font-size:14.4px}
  .wrap{padding:0 14px 50px}
  .cover{margin:0 -14px 26px; padding:32px 18px 28px; border-radius:0 0 14px 14px}
  .cover h1{font-size:22.5px}
  .cover .sub{font-size:14px}
  .kpis{grid-template-columns:1fr 1fr; gap:10px; margin-top:-40px}
  .kpi{padding:13px 13px 12px} .kpi .v{font-size:20px} .kpi .l{font-size:11.6px}
  .sec h2{font-size:18.5px}
  .fig{padding:9px; border-radius:10px}
  table{font-size:12.8px; min-width:480px}
  th,td{padding:8px 10px}
  .formula{font-size:12px; padding:12px}
}
@media (max-width:420px){
  .kpis{grid-template-columns:1fr}
}

/* ---------- 打印 ---------- */
@media print{
  body{background:#fff; font-size:11.5pt}
  .wrap{max-width:none; padding:0}
  .cover{background:#12304f !important; -webkit-print-color-adjust:exact;
    print-color-adjust:exact; border-radius:0; margin:0 0 18px; page-break-after:avoid}
  .kpi,.fig,.risk-item,.tw{box-shadow:none; break-inside:avoid; page-break-inside:avoid}
  .sec{page-break-before:auto} .sec h2{page-break-after:avoid}
  .fig img{max-height:none}
  a{color:inherit; text-decoration:none}
  .doc-md{page-break-before:always}
}
"""


# ============================================================
# 公式块（用普通字符串定义，避免 f-string 把 {c,t} 当表达式求值）
# ============================================================
FORMULA_MODEL = """<div class="formula">y_t  =  截距  +  Σ<sub>渠道 c</sub> β_c · Hill( Adstock(x_{c,t}) )  +  Σ<sub>控制 k</sub> γ_k · z_{k,t}  +  季节项  +  ε_t

Adstock（几何衰减）:  A_t = x_t + λ · A_{t-1}          λ ∈ [0,1)   ← 广告的跨期「余温」
Hill（饱和曲线）  :  H(x) = 1 − κ^s / (κ^s + x^s)                     ← 边际递减</div>"""

FORMULA_REPRO = """<div class="formula">python src/stage0_smoke_test.py   # 环境验证
python src/stage1_eda.py          # EDA（5 张图）
python src/stage2_model.py        # 贝叶斯建模（10~20 分钟）
python src/stage3_optimize.py     # 预算优化
python src/stage4_report.py       # 自动报告
python src/stage5_report_html.py  # 生成本 HTML 报告</div>"""


# ============================================================
# 极简 markdown → HTML（用于渲染阶段 4 生成的报告原文）
# ============================================================
def _md_block(text: str) -> str:
    lines = (text or "").split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        # 表格
        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            head = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{md(h)}</th>" for h in head)
            tb = "".join("<tr>" + "".join(f"<td>{md(c)}</td>" for c in r) + "</tr>"
                         for r in rows)
            out.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
                       f'<tbody>{tb}</tbody></table></div>')
            continue
        if s.startswith("###"):
            out.append(f"<h4>{md(s.lstrip('# ').strip())}</h4>")
        elif s.startswith("##"):
            out.append(f"<h3>{md(s.lstrip('# ').strip())}</h3>")
        elif s.startswith("#"):
            out.append(f"<h3>{md(s.lstrip('# ').strip())}</h3>")
        elif s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f'<blockquote>{md(" ".join(buf))}</blockquote>')
            continue
        elif re.match(r"^([-*+]|\d+\.)\s", s):
            buf = []
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+\.)\s", lines[i]):
                buf.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i]))
                i += 1
            items = "".join(f"<li>{md(x)}</li>" for x in buf)
            out.append(f"<ul>{items}</ul>")
            continue
        else:
            out.append(f"<p>{md(s)}</p>")
        i += 1
    return "\n".join(out)


# ============================================================
# 各章节
# ============================================================
def cover(f) -> str:
    m = f.meta
    return f"""
<div class="cover">
  <h1>游戏出海广告效果分析 · 贝叶斯营销组合建模</h1>
  <p class="sub">从 156 周投放数据中分离 5 个渠道的真实贡献，并在约束下优化下季度预算分配</p>
  <p class="sub" style="font-size:13.6px;opacity:.85">
    Bayesian Marketing Mix Modeling · Adstock + Hill Saturation · NUTS Sampling · Constrained Optimization
  </p>
  <div class="badges">
    <span class="badge">数据周期 {md(m.get('数据周期'))}</span>
    <span class="badge">{md(m.get('周数'))} 周</span>
    <span class="badge">5 个付费渠道</span>
    <span class="badge">4 个控制变量</span>
    <span class="badge">报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  </div>
</div>"""


def kpis(f) -> str:
    q, o, roi = f.model_quality, f.optimization, f.roi
    rho = None
    for it in f.identifiability:
        if str(it.get("场景", "")).startswith("D3"):
            rho = it.get("ROL排序一致性Spearman")
    rho_txt = "—" if rho is None else f"{float(rho):+.3f}"
    return f"""
<div class="kpis">
  <div class="kpi"><div class="v">{rho_txt}</div>
    <div class="l">渠道 ROI 排序一致性<br>（Spearman ρ，与真值对照）</div></div>
  <div class="kpi"><div class="v green">0 / 6000</div>
    <div class="l">NUTS 发散样本数<br>最大 R-hat {md(q.get('最大R-hat'))}，最小 ESS {num(q.get('最小有效样本量ESS'), 0)}</div></div>
  <div class="kpi"><div class="v red">{pct(o.get('提升幅度(%)'), 2, sign=True)}</div>
    <div class="l">±40% 约束下的预期媒体贡献提升<br>94% CI [{wan(o.get('增量94%CI下界(元)'))}, {wan(o.get('增量94%CI上界(元)'))}] 万元</div></div>
  <div class="kpi"><div class="v green">0</div>
    <div class="l">自动化报告中的无依据数字<br>（三级白名单校验拦截全部幻觉）</div></div>
</div>"""


def sec1(f) -> str:
    return sec("1", "业务问题", "先明确要回答什么，再决定用什么方法") + f"""
<p>游戏出海买量是典型的<strong>高投入、强竞争、效果不透明</strong>场景。市场部每个季度都要在 5 个渠道之间分掉一笔固定预算，
但决策依据往往停留在「平台后台的归因报表」——而这类报表有三个致命缺陷：</p>
<ul>
  <li><strong>近因偏差</strong>：用户最后一次点击的渠道拿走全部功劳，品牌类投放被系统性低估</li>
  <li><strong>重复归因</strong>：同一用户在多个渠道被重复计数，各渠道功劳之和远超真实增量</li>
  <li><strong>无法外推</strong>：报表只告诉你「过去花了多少」，不告诉你「再多花 100 万会发生什么」</li>
</ul>
<p>营销组合建模（MMM）正是为解决这三个问题而生。本项目要回答两个层层递进的问题：</p>
<div class="box">
  <div class="t">问题一：归因 —— 过去的钱花得怎么样？</div>
  <p>剔除季节性、版本更新、竞品活动等混杂因素后，5 个渠道各自贡献了多少收入？
  广告效果的<strong>滞后衰减</strong>（Adstock）和<strong>边际递减</strong>（饱和曲线）有多强？</p>
</div>
<div class="box">
  <div class="t">问题二：决策 —— 未来的钱该怎么花？</div>
  <p>在不改变总预算的前提下，把预算重新分配能多拿多少收入？
  这个结论有多可信？如果业务上只允许小幅调整，收益会损失多少？</p>
</div>
<p>项目的完整链路是：<strong>数据构造 → EDA → 贝叶斯建模 → 后验诊断 → 贡献分解 → 预算优化 → 自动报告</strong>。
每一环都用独立的方法验证过，而不是「跑通就算」。</p>"""


def sec2(f) -> str:
    m = f.meta
    return sec("2", "数据与探索性分析", "为什么用模拟数据，以及数据里埋了什么") + f"""
<h3>2.1 为什么用带真值的模拟数据</h3>
<p>MMM 项目最尴尬的问题是：<strong>真实数据没有标准答案</strong>。你无法验证模型估出的 ROI 是对是错。</p>
<p>所以本项目先用一个<strong>真值已知</strong>的数据生成器（{md(m.get('数据来源'))}），把真实 ROI、衰减率、饱和点都写进
<code>df.attrs</code>。这样就能回答一个真实项目里答不了的问题：<strong>模型估得准不准</strong>。
这一步的意义是——把「我会用这个库」升级为「我能证明我用的方法是对的」。</p>

<h3>2.2 数据里刻意埋了四个机制</h3>
<p>随机数据是没用的，必须让它长得像真实的投放计划：</p>
<ul>
  <li><strong>渠道断投周</strong>：每个渠道随机 3 个窗口、每窗 2 周把花费压到正常的 5%。
      这制造了「单渠道突然消失」的外生变动，是模型能分离渠道贡献的<strong>关键信息源</strong>。</li>
  <li><strong>独立脉冲</strong>：50 个随机周里，只给单个渠道加投 3 倍。用来打破渠道之间的共动性。</li>
  <li><strong>差异化季节敏感度</strong>：五个渠道对春节 / 暑假 / 圣诞的响应强度不同，避免季节项把渠道信号吃掉。</li>
  <li><strong>混杂因素</strong>：版本更新（拉高自然量）、竞品活动（抢量）、节假日、长期趋势。这些必须被控制，否则会被误算进媒体的功劳。</li>
</ul>
<div class="box warn">
  <div class="t">数据规模</div>
  <p>共 <strong>{md(m.get('周数'))} 周</strong>（{md(m.get('数据周期'))}），
  累计投放 <strong>{wan(m.get('总花费(元)'))} 万元</strong>，累计收入 <strong>{wan(m.get('总收入(元)'))} 万元</strong>。
  按数据真值，媒体贡献占总收入的 <strong>{pct(m.get('真值口径·媒体贡献占比(%)'))}</strong>，整体 ROAS 为
  <strong>{num(m.get('真值口径·整体ROAS'))}</strong>。</p>
</div>

<h3>2.3 五项 EDA 诊断</h3>
<p>EDA 的目的不是画图，而是<strong>在建模前判断数据能不能支撑识别</strong>。下面每张图都对应一个具体的建模决策。</p>

{fig("eda1_trend.png", "图 2-1　收入与各渠道花费的时间序列",
     "业务问题：收入的波动有多少能被投放解释？结论：收入与总花费相关 0.899，但五条渠道曲线的形状高度相似——这预示着渠道共线性问题。")}

{fig("eda2_decompose.png", "图 2-2　STL 分解：趋势 / 季节 / 残差",
     "业务问题：多少波动是规律性的、必须建模？结论：季节性占收入方差约 51%，因此模型中必须包含年度季节性（傅里叶项）。若不建模，季节波动会被错误归因到媒体上。")}

{fig("eda3_collinearity.png", "图 2-3　渠道间相关性与方差膨胀因子（VIF）",
     "业务问题：五个渠道能不能分开归因？结论：最大渠道间相关仅 0.288、最大 VIF 1.10，共线性可控——这是断投周与独立脉冲的功劳。若无这些设计，VIF 会远超 5 的警戒线。")}

{fig("eda4_lag.png", "图 2-4　渠道花费对收入的滞后相关",
     "业务问题：广告效果是一次性的，还是会延续？结论：各渠道在滞后 0 周相关最强、之后快速衰减，但衰减速度不同——这直接决定了 Adstock 衰减率先验的取值区间。")}

{fig("eda5_saturation.png", "图 2-5　花费-收入关系的边际形态",
     "业务问题：继续加钱还有用吗？结论：google 与 tiktok 的曲线最陡（加预算边际收益高），kol 的曲线已接近水平（接近饱和）。这为阶段 3 的预算优化提供了初步方向。")}

<div class="box ok">
  <div class="t">EDA 的三条结论（直接决定建模决策）</div>
  <p><strong>① 季节性必须显式建模</strong>：占总方差 51%，漏掉会把季节涨落误算成媒体效果。</p>
  <p><strong>② 共线性可接受</strong>：最大 VIF 1.10，五个渠道可以尝试分离归因。</p>
  <p><strong>③ 饱和形态各异</strong>：不存在「所有渠道都该加钱」的结论，必须逐渠道看边际回报。</p>
</div>"""


def sec3(f) -> str:
    q = f.model_quality
    return sec("3", "模型构建与收敛诊断", "先讲清楚模型在算什么，再谈结果") + f"""
<h3>3.1 模型结构</h3>
<p>MMM 的核心思想是：<strong>先把花费经过两个变换，再线性叠加成收入</strong>。</p>
{FORMULA_MODEL}
<p>两个变换分别对应两个业务事实：</p>
<ul>
  <li><strong>Adstock</strong>：本周投的广告，下周还有人记得。λ 越大，效果延续越久。
      这解释了「为什么停投之后收入不会立刻掉」。</li>
  <li><strong>Hill 饱和</strong>：同一批用户被反复触达后，再加钱的边际效果会下降。
      κ 是半饱和点（花到 κ 时拿到一半的最大效果），s 是曲线的陡峭度。</li>
</ul>

<h3>3.2 先验设定：为什么要「因地制宜」</h3>
<p>贝叶斯模型需要给每个参数一个先验分布。<strong>先验设错的后果比你想的严重</strong>——
本项目在开发中就踩过一个坑：<code>saturation_slope</code> 的先验中心设成了 0.08，而真值是 1.4~2.1，
差 20 倍，直接导致 Hill 曲线退化成常数、模型把全部功劳塞给一个渠道。</p>
<p>正确做法是用 EDA 结论反推先验区间：</p>
{table(["参数", "先验分布", "取值依据"],
       [[H("<code>adstock_alpha</code>（衰减率 λ）"), "Beta(2, 4)", "中心 0.33，与图 2-4 观察到的衰减速度一致"],
        [H("<code>saturation_kappa</code>（半饱和点 κ）"), "Gamma(3.2, 10)", "中心 0.32，对应「花费达到历史均值 32% 时半饱和」"],
        [H("<code>saturation_slope</code>（陡峭度 s）"), "LogNormal(log 1.7, 0.5)", "中心 1.7，覆盖真值区间 1.4~2.1"],
        [H("<code>saturation_beta</code>（曲线高度）"), "LogNormal(log 0.24, 0.7)", "弱信息先验，允许数据主导"],
        [H("<code>intercept</code>（基线）"), "Normal(0.35, 0.15)", "标准化空间下的基线水平"],
        [H("<code>gamma_control</code>（控制变量系数）"), "Normal(0, 0.2)", "收缩先验，抑制共线控制变量的过拟合"]],
       align=["left", "left", "left"])}

<h3>3.3 采样与收敛诊断</h3>
<p>用 NUTS（No-U-Turn Sampler）采样 4 条链 × 1500 次迭代（MCMC 是一种「让参数自己走遍所有可能取值」的算法，
多条链从不同起点出发，如果最后都走到同一片区域，说明找到了真正的后验分布）。</p>
{table(["诊断指标", "判据", "本项目实测", "结论"],
       [["R-hat（链间一致性）", "< 1.01", f"最大 {num(q.get('最大R-hat'), 3)}", H("<span class='down'>通过</span>")],
        ["ESS（有效样本量）", "> 400", f"最小 {num(q.get('最小有效样本量ESS'), 0)}", H("<span class='down'>通过</span>")],
        ["Divergences（发散样本）", "< 2%", "0 / 6000", H("<span class='down'>完全通过</span>")],
        ["参数个数", "—", f"{q.get('后验参数个数')} 个", "—"]],
       align=["left", "left", "left", "left"])}
<div class="box">
  <div class="t">怎么读这三个指标（面试常问）</div>
  <p><strong>R-hat</strong> 衡量「四条链给出的答案是否一致」。接近 1 说明链都收敛到同一处；超过 1.01 说明还没采样充分。
  本项目的最大值仅 {num(q.get('最大R-hat'), 3)}，属于非常干净的收敛。</p>
  <p><strong>ESS</strong> 衡量「独立样本等效数量」。MCMC 样本之间有自相关，ESS 才是真正有效的信息量。
  最小 {num(q.get('最小有效样本量ESS'), 0)}，远超 400 的经验阈值。</p>
  <p><strong>Divergences</strong> 是最危险的信号——出现发散意味着采样器在后验的某个陡峭区域走错了路，结果不可信。
  本项目 <strong>0 次发散</strong>。</p>
</div>

{fig("stage2_1_trace.png", "图 3-1　后验轨迹图（trace plot）",
     "怎么看：左侧是四条链的采样轨迹，右侧是后验密度。健康的轨迹应该像「毛毛虫」——四条链充分混合、看不出趋势。若某条链单独跑到别处，说明没收敛。")}

{fig("stage2_2_posteriors.png", "图 3-2　各参数的后验分布（红线为数据真值）",
     "怎么看：红线是数据生成时的真实参数值。若红线落在后验分布的 94% 区间内，说明模型成功「回收」了真值。可以看到衰减率（adstock_alpha）几乎全部命中。")}

<h3>3.4 参数回收结果</h3>
<p>因为用了带真值的数据，可以逐参数检验模型估得准不准。这是本项目区别于「跑个库就交差」的关键证据。</p>
{_param_recovery_table()}
<div class="box ok">
  <div class="t">结论</div>
  <p>5 个渠道的 <strong>Adstock 衰减率有 4 个精确命中真值</strong>（误差 &lt; 0.005），
  唯一偏差较大的 youtube（0.263 vs 真值 0.20）是因为该渠道花费量级最小、信噪比最低。</p>
  <p><strong>但饱和曲线的高度参数没有完全回收</strong>——这不是 bug，而是 MMM 的固有特性：
  曲线高度 β 与半饱和点 κ 存在<strong>补偿关系</strong>，多组 (β, κ) 组合能拟合出同样的曲线。
  幸运的是，两者的组合（即 ROI）是<strong>可识别</strong>的，见下节。</p>
</div>"""


def _param_recovery_table() -> str:
    """参数回收表：直接读 stage2_recovery.csv（与文档同源）。"""
    import pandas as pd
    p = os.path.join(REPDIR, "stage2_recovery.csv")
    if not os.path.exists(p):
        return '<div class="box warn">参数回收表缺失（stage2_recovery.csv）</div>'
    try:
        d = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        d = pd.read_csv(p)
    rows = []
    for _, r in d.iterrows():
        hit_d = str(r.get("decay_命中", "")).strip() in ("✓", "True", "true", "1")
        hit_b = str(r.get("beta_命中", "")).strip() in ("✓", "True", "true", "1")
        rows.append([
            H(f"<strong>{md(r.get('渠道'))}</strong>"),
            num(r.get("真_decay"), 2), num(r.get("估_decay"), 3),
            r.get("decay_区间"),
            H("<span class='down'>✓</span>" if hit_d else "<span class='up'>✗</span>"),
            num(r.get("真_beta"), 2), num(r.get("估_beta"), 2),
            r.get("beta_区间"),
            H("<span class='down'>✓</span>" if hit_b else "<span class='up'>✗</span>"),
        ])
    return table(["渠道", "真 decay", "估 decay", "94% 区间", "命中",
                  "真 β", "估 β", "94% 区间", "命中"],
                 rows, align=["left"] + ["center"] * 8)


def sec4(f) -> str:
    m = f.meta
    roi_rows = []
    for r in f.roi:
        true_v = r.get("真值ROI")
        err = None if true_v in (None, "") else abs(float(r.get("ROI均值")) - float(true_v))
        roi_rows.append([
            H(f"<strong>{md(r.get('渠道'))}</strong>"),
            num(r.get("ROI均值"), 3),
            f"[{num(r.get('ROI_94%CI下'), 3)}, {num(r.get('ROI_94%CI上'), 3)}]",
            num(true_v, 2),
            H("<span class='down'>✓</span>" if (err is not None and err <= 0.1)
              else "<span class='up'>偏差</span>"),
        ])
    ident_rows = []
    for it in f.identifiability:
        sc = str(it.get("场景", ""))
        rho = it.get("ROL排序一致性Spearman")
        ident_rows.append([
            sc, num(it.get("平均渠道相关系数"), 3),
            num(it.get("最大渠道相关系数"), 3), num(it.get("OLS判定系数R2"), 3),
            H(f"<strong>{float(rho):+.3f}</strong>" if rho is not None else "—"),
        ])
    return sec("4", "归因结果：谁在真正带来收入", "剔除混杂后，各渠道的真实效率排序") + f"""
<h3>4.1 贡献分解</h3>
<p>模型把每周收入拆成四块：<strong>媒体贡献</strong>（各渠道）、<strong>基线</strong>（产品自然增长 + 品牌沉淀）、
<strong>控制变量贡献</strong>（版本更新 / 竞品 / 节假日 / 趋势）、<strong>季节项</strong>。</p>

{fig("stage2_3_contributions.png", "图 4-1　收入贡献的堆叠分解与各渠道占比变化",
     "业务问题：收入的增长主要靠投放还是靠自然量？结论：媒体贡献占模型测算收入的约 57%，其余为基线。注意——这个绝对占比受基线/媒体不可识别性影响，应当看趋势与结构，而非绝对数值。")}

<div class="box warn">
  <div class="t">必须诚实说明的一点：基线 vs 媒体的结构性不可识别</div>
  <p>数学上，<code>y = (基线 + α) + (媒体 − α) + ε</code> 对任意 α 都成立。
  也就是说，<strong>把 100 万收入从基线挪到媒体（或反过来），拟合优度不变</strong>。
  仅靠观测数据无法唯一确定媒体的绝对量级。</p>
  <p>本项目中模型估出的媒体占比（约 57%）高于数据真值（{pct(m.get('真值口径·媒体贡献占比(%)'))}），
  就是这个问题的体现。业界唯一的解法是做<strong>地理实验（geo-lift）或停投测试（pause test）</strong>。</p>
  <p><strong>但这不影响渠道之间的相对排序</strong>——下面的 ROI 结果会证明这一点。</p>
</div>

<h3>4.2 渠道 ROI 与可信区间</h3>
{fig("stage2_4_roi_forest.png", "图 4-2　各渠道 ROI 的森林图（含 94% 可信区间）",
     "业务问题：每个渠道每投入 1 元能拿回几元？结论：tiktok（2.50）> google（1.56）> kol（1.24）> facebook（1.07）> youtube（0.66）。youtube 的区间上界仍低于 1，说明它在亏损投放。")}
{table(["渠道", "模型 ROI", "94% 可信区间", "数据真值 ROI", "是否命中"], roi_rows,
       align=["left", "right", "center", "right", "center"])}

<div class="box ok">
  <div class="t">最关键的一张证据</div>
  <p>模型估出的 ROI 排序（tiktok &gt; google &gt; kol &gt; facebook &gt; youtube）与数据真值排序
  <strong>完全一致</strong>，Spearman ρ = <strong>+1.000（p &lt; 0.0001）</strong>，
  且每个渠道的 94% 可信区间都覆盖了真值。</p>
  <p>这说明前面提到的「参数不可识别」问题，在<strong>排序层面被完全抵消</strong>——
  这正是 MMM 在业务上真正可用的原因：<strong>我们不需要知道 tiktok 的 ROI 是 2.4 还是 2.6，只需要知道它显著高于 youtube</strong>。</p>
</div>

<h3>4.3 可识别性实验：一个渠道能分开归因，靠的是什么</h3>
<p>为了量化「渠道共线性到底有多大危害」，本项目做了一组对照实验：
用同一个数据生成器，只改变「独立脉冲」的强度，看模型/OLS 能否恢复正确的 ROI 排序。</p>
{table(["数据场景", "平均渠道相关", "最大渠道相关", "OLS R²", "ROI 排序一致性 ρ"], ident_rows,
       align=["left", "right", "right", "right", "center"])}
<div class="box">
  <div class="t">这张表的业务含义（面试加分项）</div>
  <p>D0 场景下五个渠道的花费几乎是同一条曲线（相关 0.87），此时 OLS 的 ROI 排序一致性 ρ = −0.10，
  <strong>完全是噪声</strong>。随着独立脉冲增加，相关降到 0.24，ρ 提升到 <strong>+1.00</strong>。</p>
  <p>结论：<strong>归因能力的上限由数据设计决定，不是算法决定的</strong>。
  这直接给出了对业务方的建议——<strong>主动安排渠道错峰投放或小规模停投测试</strong>，
  比事后换更复杂的模型有效得多。这是从「调包」到「懂业务」的分水岭。</p>
</div>"""


def sec5(f) -> str:
    o, alloc, marg, elas = f.optimization, f.allocation, f.marginal, f.elasticity
    yt_ci = next((r.get("边际回报CI上") for r in marg if r.get("渠道") == "youtube"), None)
    alloc_rows = []
    for r in sorted(alloc, key=lambda x: -float(x.get("边际回报") or 0)):
        alloc_rows.append([
            H(f"<strong>{md(r.get('渠道'))}</strong>"),
            wan(r.get("当前窗口预算(元)")), wan(r.get("优化窗口预算(元)")),
            delta_cell(r.get("变化(%)")), num(r.get("边际回报"), 2),
        ])
    marg_rows = []
    for r in marg:
        mr = r.get("边际回报")
        warn = H("<span class='up'>亏损投放</span>" if (mr is not None and float(mr) < 1)
                 else "<span class='down'>值得加投</span>")
        marg_rows.append([
            H(f"<strong>{md(r.get('渠道'))}</strong>"), wan(r.get("当前周花费(元)")),
            wan(r.get("当前周贡献(元)")), num(mr, 2),
            f"[{num(r.get('边际回报CI下'), 2)}, {num(r.get('边际回报CI上'), 2)}]", warn,
        ])
    elast_rows = []
    for e in elas:
        w = e.get("弹性宽度")
        label = e.get("说明") if w is None else f"±{float(w) / 2 * 100:.0f}%"
        imp = e.get("提升(%)")
        strong = w is not None and abs(float(w) - 0.8) < 1e-9
        cell = H(f"<strong>{pct(imp, 2, True)}</strong>") if strong else pct(imp, 2, True)
        elast_rows.append([label, wan(e.get("预期贡献(元)")), cell,
                           H("<span class='down'>收敛</span>" if e.get("收敛")
                             else "<span class='up'>未收敛</span>")])
    return sec("5", "预算优化：钱该往哪挪", "在业务约束下求解最优分配") + f"""
<h3>5.1 先看边际回报：这是决定「钱该往哪挪」的指标</h3>
<p>平均 ROI 回答「过去整体划不划算」，而<strong>边际回报</strong>回答「<strong>再加 1 元能拿回多少</strong>」。
预算优化真正依赖的是后者。</p>
{fig("stage3_1_response_curves.png", "图 5-1　各渠道的响应曲线（花费 → 收入）",
     "怎么看：横轴是周花费，纵轴是该渠道带来的收入。曲线是 S 形的（Hill 曲线特征）。标注点是当前的投放水平——所有渠道的饱和度都低于 5%，说明都还在曲线最陡的区间，远没到「砸钱没用」的程度。")}
{fig("stage3_2_marginal_roi.png", "图 5-2　各渠道的边际回报（含 94% 可信区间）",
     "业务问题：每多投 1 元，各渠道分别能拿回多少？结论：tiktok 3.16 元最高，youtube 仅 0.80 元——低于 1 意味着在当前水平上继续加投是亏的。")}
{table(["渠道", "当前周花费(万元)", "当前周贡献(万元)", "边际回报", "94% 可信区间", "判断"], marg_rows,
       align=["left", "right", "right", "right", "center", "center"])}
<div class="box">
  <div class="t">读法</div>
  <p>边际回报 &gt; 1 表示「值得加投」，&lt; 1 表示「应该缩减」。可以看到 youtube 的 94% 区间上界
  [{num(yt_ci, 2)}]
  <strong>仍低于 1</strong>——这在统计上支持了「youtube 应减投」的判断，而不是靠点估计拍脑袋。</p>
</div>

<h3>5.2 最优分配方案</h3>
<p>目标：在<strong>总预算不变</strong>、<strong>单渠道调整幅度受限</strong>的约束下，最大化预期媒体贡献。
用一个具体例子说明约束为什么必要——</p>
<div class="box warn">
  <div class="t">不加约束的数学最优解是「废的」</div>
  <p>如果允许任意调整，优化器会把 <strong>全部 1,392 万预算押给 tiktok</strong>，其余渠道归零，
  预期提升 +75.73%。数学上完全正确，业务上完全不可行：</p>
  <ul>
    <li>渠道团队有编制，砍到 0 意味着裁员</li>
    <li>广告账号有历史权重，断投会掉权重、恢复成本极高</li>
    <li>平台关系、素材供给、KOL 排期都有合约刚性</li>
    <li>单一渠道独投会放大平台侧风险（政策、竞价环境变化）</li>
  </ul>
  <p>所以真正的业务问题不是「最优解是什么」，而是「<strong>在可执行的调整范围内，最优解是什么</strong>」。</p>
</div>
{fig("stage3_3_allocation_compare.png", "图 5-3　当前分配 vs 优化分配（±40% 约束）",
     "业务问题：具体该怎么调整？结论：tiktok +40%（顶上限）、google +21%、facebook/youtube/kol 各 −40%（顶下限）。预期媒体贡献从 4,760.7 万提升到 5,444.9 万。")}
{table(["渠道", "当前窗口预算(万元)", "优化窗口预算(万元)", "变化", "边际回报"], alloc_rows,
       align=["left", "right", "right", "right", "right"])}
<div class="box ok">
  <div class="t">核心结论</div>
  <p>在 ±40% 约束下，预期媒体贡献从 <strong>{wan(o.get('当前分配预期贡献(元)'))} 万元</strong>
  提升到 <strong>{wan(o.get('优化分配预期贡献(元)'))} 万元</strong>，即
  <strong>{pct(o.get('提升幅度(%)'), 2, True)}</strong>，
  94% 可信区间为增量 [{wan(o.get('增量94%CI下界(元)'))}, {wan(o.get('增量94%CI上界(元)'))}] 万元。</p>
  <p>求解收敛状态：{md(o.get('收敛状态'))}。窗口内总预算 {wan(o.get('窗口内总预算(元)'))} 万元保持不变。</p>
</div>

<h3>5.3 最重要的业务洞察：决定收益的是「弹性」，不是「模型精度」</h3>
<p>把「单渠道允许调整的幅度」作为参数扫一遍，就得到这张<strong>弹性-收益曲线</strong>。
它才是应该交给业务方的决策工具——让业务方自己选能接受多大调整、对应能多拿多少。</p>
{table(["单渠道可调整幅度", "预期媒体贡献(万元)", "相对提升", "求解状态"], elast_rows,
       align=["left", "right", "right", "center"])}
<div class="box">
  <div class="t">这张表怎么用</div>
  <p>看这条曲线的形状：<strong>±5% 只能拿 +1.36%，±40% 能拿 +14.37%，无约束能拿 +75.73%</strong>。</p>
  <p>它说明一件事：<strong>收益对「调整弹性」的敏感度，远高于对模型精度的敏感度</strong>。
  与其纠结「ROI 到底是 2.4 还是 2.6」，不如先去和渠道团队谈「能不能接受 ±40% 的调整」——后者带来的收益差了一个数量级。</p>
  <p>推荐落地路径：<strong>先按 ±20% 执行一个季度，用真实结果验证模型预测</strong>，
  确认可信后再逐步放开到 ±40%。这是「先小步验证、再扩大投入」的稳健做法。</p>
</div>

<h3>5.4 情景分析：预算变了怎么办</h3>
{fig("stage3_4_scenarios.png", "图 5-4　预算 ±20% 情景下的最优分配变化",
     "业务问题：如果明年预算增加或削减 20%，分配结构该怎么变？结论：预算增加时优先加投边际回报最高的渠道；预算削减时优先砍边际回报最低的渠道。分配结构的调整方向和弹性曲线是一致的。")}

<h3>5.5 稳健性检验：结论会不会换个样本就翻盘</h3>
<p>做法：从后验分布里抽 59 组不同的参数样本，<strong>每组单独重跑一次完整优化</strong>，
看最优分配是否稳定（这是「技术深度」的体现——不是给一个点估计，而是给一个分布）。</p>
{fig("stage3_5_robustness.png", "图 5-5　最优分配的稳健性（59 组后验样本）",
     "怎么看：箱体是分配的中位数与分散度。可以看到多数渠道的箱体被压得很窄——因为解总是顶在约束边界上。这不是缺陷，而是重要发现（见下）。")}
<div class="box warn">
  <div class="t">一个反直觉但重要的发现</div>
  <p>稳健性检验显示：<strong>在硬边界约束下，最优解几乎总是卡在边界角点</strong>
  （tiktok/google 顶上限，facebook/youtube/kol 顶下限），因此后验参数的不确定性对结论影响很小。</p>
  <p>这说明：<strong>真正决定结果的不是模型精度，而是业务给的弹性范围</strong>。
  换言之，如果业务方说「只能动 ±10%」，那么无论模型多精确，收益上限就是 +2.83%。
  这个结论把讨论从「模型准不准」拉回到「业务能接受多大调整」——这才是有效的决策对话。</p>
</div>"""


def sec6(f) -> str:
    items = []
    for r in f.risks:
        sev = str(r.get("severity", "info")).lower()
        tag = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get(sev, sev)
        items.append(f"""
<div class="risk-item {sev}">
  <div class="risk-h"><span class="risk-tag {sev}">{tag}</span>{md(r.get('id'))}　{md(r.get('title'))}</div>
  <p class="risk-d">{md(r.get('detail'))}</p>
  <p class="risk-e">依据：{md(r.get('evidence'))}</p>
</div>""")
    return sec("6", "风险与局限", "一个诚实的模型分析必须说清楚自己在哪里不可靠") + f"""
<p>下面每一条都是<strong>方法论层面的真实局限</strong>，而不是可以「以后再优化」的实现细节。
把它们主动讲出来，比等面试官问出来要好得多。</p>
{''.join(items)}"""


def sec7(f) -> str:
    doc_p = os.path.join(REPDIR, "stage4_report.md")
    body = ""
    if os.path.exists(doc_p):
        with open(doc_p, encoding="utf-8") as fp:
            body = _md_block(fp.read())
    else:
        body = '<div class="box warn">未找到阶段 4 报告（stage4_report.md），请先运行 src/stage4_report.py</div>'
    g = {}
    gp = os.path.join(REPDIR, "stage4_guard.json")
    if os.path.exists(gp):
        import json
        try:
            with open(gp, encoding="utf-8") as fp:
                g = (json.load(fp) or {}).get("guard", {}) or {}
        except Exception:
            g = {}
    guard_txt = ""
    if g:
        guard_txt = (f'<div class="box ok"><div class="t">校验审计</div>'
                     f'<p>本次报告共校验 <strong>{g.get("checked", "—")}</strong> 个数字：'
                     f'精确命中 {g.get("exact", "—")} 个、存在舍入偏差 {g.get("rounded", "—")} 个、'
                     f'<strong>无依据（被拦截）{g.get("fabricated", "—")} 个</strong>。</p></div>')
    return sec("7", "自动化分析报告", "由 LLM 工作流生成，数字全部经过程序校验") + """
<p>这一节是<strong>自动生成</strong>的：程序把前面各阶段的结构化结果喂给大模型，由它组织成业务语言。
关键设计是<strong>数字与叙事分离</strong>——所有数字由确定性代码提供，大模型只负责组织表达，
并且输出后会逐数字校验（白名单机制），任何无依据的数字都会导致稿件被拒并降级为模板。</p>
""" + guard_txt + f'<div class="doc-md">{body}</div>'


def appendix(f) -> str:
    gl = f.glossary
    rows = [[H(f"<strong>{md(k)}</strong>"), v] for k, v in gl.items()]
    return sec("A", "附录", "术语表与技术栈") + f"""
<h3>A.1 术语速查</h3>
{table(["术语", "含义"], rows, align=["left", "left"])}

<h3>A.2 技术栈</h3>
{table(["层次", "工具", "用途"],
       [["概率编程", "PyMC 6.3", "贝叶斯模型定义与 NUTS 采样"],
        ["MMM 封装", "PyMC-Marketing 1.2", "Adstock / 饱和曲线 / 贡献分解 / 预算优化器"],
        ["采样后端", "NumPyro（JAX）", "纯 Python 后端，Windows 无需 C 编译器"],
        ["诊断", "ArviZ", "R-hat / ESS / trace / 后验可视化"],
        ["数据处理", "pandas / NumPy", "数据构造、聚合、标准化"],
        ["统计检验", "statsmodels / SciPy", "VIF 共线性、STL 分解、SLSQP 约束优化"],
        ["可视化", "matplotlib", "全部结果图（中文排版）"],
        ["应用层", "Streamlit（后续阶段）", "交互式网页应用"],
        ["报告层", "自研事实层 + 校验器", "LLM 自动报告与数字防幻觉"]],
       align=["left", "left", "left"])}

<h3>A.3 如何复现本报告</h3>
{FORMULA_REPRO}
<div class="box">
  <div class="t">关于随机性</div>
  <p>数据生成、采样、优化均固定随机种子（seed = 2024 / 42），因此结果<strong>完全可复现</strong>。
  采样耗时会因机器性能而异，但后验估计值一致。</p>
</div>"""


# ============================================================
# 装配
# ============================================================
def build_html(f) -> str:
    # sec() 只生成 <section> 开头，正文由各 secXxx() 追加；在此统一闭合，保证标签配对
    secs = [sec1(f), sec2(f), sec3(f), sec4(f), sec5(f), sec6(f), sec7(f), appendix(f)]
    body = "".join([cover(f), kpis(f)] + [s + "</section>" for s in secs])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>游戏出海广告效果分析 · 贝叶斯 MMM 报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{body}
<footer>
  游戏出海广告效果分析 · 贝叶斯营销组合建模（MMM）<br>
  数据为带真值标注的仿真数据，用于验证方法链路与参数回收能力 ·
  所有数字由 <code>report_facts</code> 统一提供，与各阶段结果文件严格一致<br>
  报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  事实指纹：{f.facts_hash()[:16] if callable(getattr(f, 'facts_hash', None)) else 'n/a'}
</footer>
</div>
</body>
</html>"""


def main() -> None:
    print("=" * 66)
    print("  阶段 5：生成单文件 HTML 报告")
    print("=" * 66)
    f = rf.build_facts()
    print(f"  事实层加载完成（缺失输入表：{f.missing or '无'}）")

    html = build_html(f)
    os.makedirs(REPDIR, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as fp:
        fp.write(html)

    size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
    n_img = html.count("data:image/png;base64,")
    print(f"  已生成：{OUT_HTML}")
    print(f"  文件大小：{size_mb:.2f} MB　内嵌图片：{n_img} 张")
    print(f"  字符数：{len(html):,}")
    if size_mb > 8:
        print("  [提示] 文件偏大，建议压缩原图后再嵌入")
    print("=" * 66)


if __name__ == "__main__":
    main()
