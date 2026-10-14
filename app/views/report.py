"""
决策报告页（阶段 9 交付）
=========================

把前面所有结论合成一份「业务方读得懂、且数字可信」的报告。

核心矛盾
--------
    LLM 表达能力强，但**会编数字**；模板不会编，但读起来像机器。

解法（本页展示的就是这个流程）：

    确定性代码算数  →  大模型只组织语言  →  白名单逐个数校验
        →  违规就带清单重试  →  仍违规就降级为模板，并把被拒稿件留档

页面提供三种生成方式，**并把校验过程完整暴露出来**：

    ① 模板渲染（秒出）        —— 零依赖、零幻觉风险，可直接发出去
    ② 大模型润色（需配密钥）  —— 表达能力更强，但每个数字都要过白名单
    ③ 离线模拟演示            —— 不调 API，用注入的幻觉验证"校验器真的能拦住"

三种方式共用同一个**事实层**：所有数字都来自模型与优化结果，
不存在"这一版报告的数字和上一版对不上"的可能。

双数据源
--------
    内置模拟数据 → 事实可从磁盘阶段产物构建（也优先用会话现算，保证与页面一致）
    上传数据     → 事实只能从当前会话的模型/优化结果**现算**
                   （磁盘上没有它的阶段产物 CSV）
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from core import facts as F, optimize as O, paths, report as R, state, ui   # noqa: E402
from core import cache                                                    # noqa: E402

paths.ensure_src_on_path()

ui.inject()

# 配色统一从 core/ui.py 取（全应用唯一来源）
C_BRAND, C_UP, C_DOWN = ui.C_BRAND, ui.C_UP, ui.C_DOWN
C_MUTED, C_WARN = ui.C_MUTED, ui.C_WARN


def _wan(v, nd: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f / 1e4:,.{nd}f}"


def _num(v, nd: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:,.{nd}f}"


def _pct(v, nd: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{'+' if f > 0 else ''}{f:.{nd}f}%"


# ============================================================
# 事实层来源与体检
# ============================================================
def _resolve_facts():
    """返回 (facts, source_tag, note)。结果按（数据指纹 × 优化结果键）缓存。"""
    art = state.get(state.K_ARTIFACTS)
    opt = state.get(state.K_OPT_RESULT)
    ds = state.get_dataset()
    if art and art.get("ok") and opt and opt.get("ok") and ds is not None:
        f = cache.cached_session_facts(
            getattr(ds, "fingerprint", "nofp"),
            str(state.get(state.K_OPT_KEY) or "nokey"), art, opt, ds)
        f.meta["事实来源"] = "当前会话（模型与优化结果现算）"
        return f, "session", ("事实层由**当前会话**的模型后验与优化结果现算 —— "
                              "页面上看到什么，报告里就说什么。")
    f = cache.get_facts()          # 磁盘产物，带版本化缓存
    return f, "disk", ("当前没有会话内的优化结果，事实层回退到**磁盘阶段产物**"
                       "（对应内置模拟数据的阶段 2/3 结果）。"
                       "在「预算优化」页跑一次求解，报告就会切到会话现算口径。")


def _source_badge(source: str) -> str:
    return ("🟢 会话现算" if source == "session" else "🟡 磁盘产物")


# ============================================================
# 事实层浏览（证明"每个数字都有出处"）
# ============================================================
def _facts_panel(f: "F.ReportFacts"):
    with st.expander("🔍 数字底稿：这份报告用到的全部事实（可逐项展开）", expanded=False):
        st.caption("报告里的每个数字都必须在这份底稿里找得到。"
                   "单击展开即可人工核对 —— 这是「报告可信」的最终依据。")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**① 数据与口径**")
            st.json({k: v for k, v in (f.meta or {}).items()}, expanded=False)
            st.markdown("**② 模型质量**")
            st.json(f.model_quality or {}, expanded=False)
        with c2:
            st.markdown("**③ 优化结论**")
            st.json(f.optimization or {}, expanded=False)

        tabs = st.tabs(["渠道分配", "边际回报", "渠道 ROI", "弹性-收益",
                        "情景分析", "稳健性", "风险声明"])
        # 注意：这里不用 `A if cond else st.info("无")` 这种跨行三元表达式。
        # streamlit 的 st.info / st.write 会去 ast.parse **调用方的源码行**来
        # 猜变量名；表达式一旦跨行，那一行本身就是不完整的 Python，
        # 于是抛 `SyntaxError: '(' was never closed` —— 整个页面挂掉。
        # 这类错误只在真正渲染到该分支时才出现，静态检查看不出来。
        with tabs[0]:
            if f.allocation:
                ui.dataframe(pd.DataFrame(f.allocation))
            else:
                st.info("无渠道分配数据")
        with tabs[1]:
            if f.marginal:
                ui.dataframe(pd.DataFrame(f.marginal))
            else:
                st.info("无边际回报数据")
        with tabs[2]:
            if f.roi:
                ui.dataframe(pd.DataFrame(f.roi))
            else:
                st.info("无渠道 ROI 数据")
        with tabs[3]:
            if f.elasticity:
                ui.dataframe(pd.DataFrame(f.elasticity))
            else:
                st.info("无弹性-收益数据")
        with tabs[4]:
            st.json(f.scenarios or [], expanded=False)
        with tabs[5]:
            if f.robustness:
                ui.dataframe(pd.DataFrame(f.robustness))
            else:
                st.info("本次未运行稳健性检验（在优化页把抽样数设为 20~60 即可）")
        with tabs[6]:
            for r in f.risks or []:
                st.markdown(f"- **[{ {'high':'高','medium':'中','low':'低','info':'说明'}.get(r.get('severity'),'?') }]"
                            f" {r.get('title')}**：{r.get('detail')}")

        st.markdown("---")
        st.markdown(f"**数字白名单规模**：{len(f.allowed_numbers()):,} 个可引用值"
                    "（含元/万元/亿元与小数/百分数的自动换算形式）。"
                    "大模型输出里出现这之外的数字，就会被判为「无依据」并拦截。")


# ============================================================
# 校验审计
# ============================================================
def _audit_panel(rep: dict):
    guard = rep.get("guard")
    if guard is None:
        st.markdown(
            '<div class="mmm-panel"><div class="t">🔒 数字校验</div>'
            '<div class="s">本次为<b>模板渲染</b> —— 所有数字由确定性代码直接写入，'
            '不经过大模型，因此不存在幻觉风险，无需白名单校验。</div></div>',
            unsafe_allow_html=True)
        return

    ok = guard.ok
    tone = C_DOWN if ok else C_UP
    icon = "✅" if ok else "⛔"
    st.markdown(
        f'<div class="mmm-panel" style="border-left:5px solid {tone}">'
        f'<div class="t">{icon} 数字校验审计</div>'
        f'<div class="s">检查了 <b>{guard.n_checked}</b> 个数字：'
        f'精确命中白名单 <b>{guard.n_exact}</b>，'
        f'合理舍入 <b>{guard.n_rounded}</b>，'
        f'<span style="color:{tone}"><b>无依据 {guard.n_fabricated}</b></span>。'
        f'{"大模型输出全部可追溯。" if ok else "已拦截并降级为模板报告。"}'
        f'</div></div>', unsafe_allow_html=True)

    rows = rep.get("audit_rows") or []
    if rows:
        df = pd.DataFrame(rows)
        df["上下文"] = df["上下文"].astype(str)
        ui.dataframe(df)
        st.caption("「无依据」= 该数字与事实层中任何值（含单位换算形式）的相对误差都 > 1%。"
                   "「舍入」= 相对误差在 0.3%~1% 之间，属于正常四舍五入，予以放行。"
                   "**这条分界线是实测校准出来的**：正常舍入的误差约 0.0005%，"
                   "而编造数字动辄 200%+，中间留了三个数量级的安全余量。")

    st.markdown("**三级判据（阈值不是拍的，是实测校准的）**")
    st.markdown(
        "| 级别 | 判据 | 处理 |\n|---|---|---|\n"
        "| ✅ 精确 | 相对误差 < 0.3% | 直接采用 |\n"
        "| ⚠️ 舍入 | 0.3% ~ 1.0% | 采用（正常四舍五入）|\n"
        "| ⛔ 无依据 | > 1.0% | **拦截** |")


def _rejected_panel(rep: dict):
    doc = rep.get("rejected_doc")
    if not doc:
        return
    st.markdown("---")
    st.error(f"**大模型原稿已被拒绝**（{rep.get('audit', {}).get('fabricated', 0)} 处无依据数字）。"
             "页面展示的是**降级后的模板报告** —— 宁可用正确但朴素的稿子，"
             "也不用漂亮但数字错的稿子。")
    with st.expander("📄 查看被拒的大模型原稿（留档以便审计）", expanded=False):
        st.markdown(doc)


# ============================================================
# 生成入口
# ============================================================
def _generate(f, source_tag: str, mode: str, extra: str) -> dict:
    """按模式生成报告，把过程日志实时显示出来。"""
    # 用 ui.StatusBox：st.status 的容器没有 .step()/.done()/.fail()，
    # 直接用会在点按钮时抛 StreamlitAPIException（见 core/ui.py 的说明）。
    box = ui.StatusBox("正在生成决策报告", expanded=True)
    logs: list[str] = []

    def _cb(frac, msg):
        if frac is not None:
            box.progress(frac, msg)
        box.write(f"▸ {msg}")

    kw = {"extra_requirement": extra, "progress": _cb}
    if mode == "template":
        kw["use_llm"] = False
    elif mode == "mock_ok":
        kw.update(use_llm=True, force_mock=True, mock_fabricate=False)
    elif mode == "mock_bad":
        kw.update(use_llm=True, force_mock=True, mock_fabricate=True)
    else:                                   # llm
        kw.update(use_llm=True, force_mock=False)

    t0 = time.time()
    try:
        rep = R.generate(f, **kw)
    except Exception as e:                                  # noqa: BLE001
        box.fail(f"生成失败：{type(e).__name__}: {e}")
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    rep["facts_source"] = source_tag
    rep["elapsed"] = time.time() - t0
    for m in rep.get("log", []):
        box.write(f"　{m}")
    if rep.get("source") == "llm":
        box.done(f"完成（采用大模型稿，{rep['elapsed']:.1f} 秒）")
    elif rep.get("source") == "llm_rejected":
        box.close(f"完成（大模型稿被拦截，已降级为模板，{rep['elapsed']:.1f} 秒）",
                  state="error")
    else:
        box.done(f"完成（模板渲染，{rep['elapsed']:.1f} 秒）")
    return rep


# ============================================================
# 主流程
# ============================================================
def main():
    state.init()
    ui.page_header(
        "决策报告",
        "把模型结论翻译成一页纸的决策建议：结论、依据、风险、行动。"
        "可选择用大模型润色叙事，但**所有数字都必须先通过白名单校验**。", "📝")

    if not state.require_model():
        return
    ds = state.get_dataset()

    f, source_tag, note = _resolve_facts()

    # ---- 顶部状态 ----
    opt_ok = bool(state.get(state.K_OPT_RESULT) and state.get(state.K_OPT_RESULT, {}).get("ok"))
    ui.kpi_row([
        {"label": "事实层来源", "value": _source_badge(source_tag),
         "delta": f"指纹 {f.facts_hash()}"},
        {"label": "数字白名单", "value": f"{len(f.allowed_numbers()):,}",
         "delta": "可引用的数字集合"},
        {"label": "优化结论", "value": _pct((f.optimization or {}).get("提升幅度(%)")),
         "tone": "up" if ((f.optimization or {}).get("提升幅度(%)") or 0) > 0 else None,
         "delta": "窗口内媒体贡献提升"},
        {"label": "风险条目", "value": str(len(f.risks or [])),
         "delta": f"其中高危 {sum(1 for r in (f.risks or []) if r.get('severity') == 'high')} 条"},
    ], per_row=4)

    st.caption(f"ℹ️ {note}")
    if not opt_ok:
        st.warning("**还没有会话内的优化结果。** 报告会基于磁盘阶段产物生成"
                   "（内置模拟数据的阶段 2/3 结果）。"
                   "如果你用的是上传数据，请先到「预算优化」页跑一次求解 —— "
                   "否则报告里的数字与你的数据无关。")

    # ---- 生成方式 ----
    st.markdown("### ① 选择生成方式")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        mode = st.radio(
            "生成方式",
            ["模板渲染（秒出，零幻觉风险）",
             "大模型润色（数字需过白名单）",
             "离线模拟：正常稿（验证链路）",
             "离线模拟：注入幻觉（验证拦截）"],
            index=0, key="rep_mode",
            help="前两种是真实可用的方式；后两种是**离线演示**，"
                 "不调用任何 API，用来证明「校验器真的能拦住编造的数字」。")
        mode_key = {"模板渲染（秒出，零幻觉风险）": "template",
                    "大模型润色（数字需过白名单）": "llm",
                    "离线模拟：正常稿（验证链路）": "mock_ok",
                    "离线模拟：注入幻觉（验证拦截）": "mock_bad"}[mode]
        extra = st.text_input(
            "额外要求（可选，会附加给大模型）", value="",
            placeholder="例：把结论压缩到三句话，面向 CEO 汇报。",
            help="模板渲染时此项无效（模板不做推断性表述）。")
    with c2:
        import llm_client as _lc
        lcfg = _lc.from_env()
        ok, why = _lc.is_configured(lcfg)
        st.markdown("**大模型配置状态**")
        st.markdown(f"- 状态：{'🟢 可用' if ok else '⚪ 未配置'}\n"
                    f"- 说明：{why}\n"
                    f"- 端点：`{lcfg.base_url or '—'}`\n"
                    f"- 模型：`{lcfg.model or '—'}`")
        st.caption("配置方式（环境变量，不写死在代码里）："
                   "`LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL`。"
                   "任何 OpenAI 兼容端点都可以（DeepSeek / 通义 / 智谱 / 本地 vLLM）。"
                   "未配置时选「大模型润色」会自动降级为模板，页面不会白屏。")

    b1, b2, b3 = st.columns([1, 1, 3])
    stored = state.get(state.K_REPORT)
    if b1.button("🚀 生成报告", type="primary", key="btn_report"):
        rep = _generate(f, source_tag, mode_key, extra)
        if rep.get("ok"):
            state.set(state.K_REPORT, rep)
            state.log(f"生成决策报告：{rep['source']}，{rep['chars']} 字符")
        st.rerun()
    if b2.button("🧹 清除报告", key="btn_rep_clear"):
        state.set(state.K_REPORT, None)
        st.rerun()
    with b3:
        if stored:
            src = {"llm": "大模型生成（数字已校验）",
                   "llm_rejected": "大模型稿被拒 → 模板兜底",
                   "template": "模板渲染"}.get(stored.get("source"), stored.get("source"))
            st.caption(f"当前展示：{src}　|　{stored.get('chars', 0):,} 字符　|　"
                       f"事实指纹 `{stored.get('facts_hash')}`")

    _facts_panel(f)

    if not stored or not stored.get("ok"):
        st.markdown("---")
        ui.empty_state("还没有生成的报告",
                       "选好生成方式后点「生成报告」。模板渲染约 1 秒，"
                       "大模型润色通常 10~40 秒（取决于模型与网络）。", "📄")
        return

    # ---- 报告正文 ----
    st.markdown("---")
    st.markdown(f"### ② 报告正文（{stored.get('chars', 0):,} 字符）")
    _audit_panel(stored)
    st.markdown(stored["text"])

    _rejected_panel(stored)

    # ---- 下载 ----
    st.markdown("---")
    st.markdown("### ③ 导出")
    name = f"MMM决策报告_{stored.get('facts_hash', 'na')}.md"
    d1, d2 = st.columns([1, 3])
    with d1:
        st.download_button("⬇️ 下载 Markdown", data=stored["text"].encode("utf-8"),
                           file_name=name, mime="text/markdown",
                           key="dl_report")
    with d2:
        st.caption(f"文件名带事实指纹（`{stored.get('facts_hash')}`）—— "
                   "数据或模型一变，指纹就变，不会出现「报告与数据对不上却看不出来」的情况。"
                   "报告里也写明了生成方式与校验结论，可直接转发给业务方。")

    with st.expander("🧾 本次生成的完整过程日志", expanded=False):
        for m in stored.get("log", []):
            st.markdown(f"- {m}")


main()
