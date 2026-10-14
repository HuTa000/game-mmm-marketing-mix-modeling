# -*- coding: utf-8 -*-
"""
决策报告生成层（纯逻辑，不依赖 streamlit，可离线自测）
=====================================================

核心矛盾
--------
LLM 表达能力强但**会编数字**；模板不会编但读起来像机器。
业务方要的是「一页纸能读懂的决策报告」，而编造的数字会让预算决策直接出错。

解法：**把「算数」与「叙事」彻底分离**

    确定性代码算数（facts） → 交给 LLM 组织语言 → 白名单逐个数校验
        → 违规就带清单重试一次 → 仍违规就降级为模板，并把被拒稿件留档

真实失效模式（两种，都在阶段 4 实测拦截过）
------------------------------------------
1. **记错**：把事实里的 `14.37%` 写成 `43.7%`
2. **编造**：凭空写出「行业平均 ROI 约为 88.88」「竞品投放约 5,000 万元」

三级判据（阈值实测校准，见 src/report_guard.py）
    exact      相对误差 < 0.3%      —— 正常引用
    rounded    相对误差 < 1.0%      —— 正常四舍五入（如 4760.6769 → 4760.7）
    fabricated 相对误差 > 1.0%      —— 拦截

与 src/stage4_report.py 的分工
-----------------------------
    src/stage4_report.py  离线批处理：读盘事实 → 写死到 outputs/reports/stage4_report.md
    app/core/report.py    运行时编排：接受任意 facts（含上传数据现算的），
                          返回结构化结果，由页面决定怎么展示与下载

本模块复用 src 下的 report_prompts / report_guard / report_templates / llm_client，
不重复实现任何一块。
"""
from __future__ import annotations

import os
import re

from . import paths

paths.ensure_src_on_path()

import llm_client as lc                                        # noqa: E402
import report_guard as rg                                      # noqa: E402
import report_prompts as rp                                    # noqa: E402
import report_templates as rt                                  # noqa: E402


# ============================================================
# 配置桥接：把 Streamlit secrets 补成环境变量
# ============================================================
_SECRET_KEYS = (
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT", "LLM_MOCK",
    "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY",
    "LLM_PROMPT_PRICE", "LLM_COMPLETION_PRICE",
)


def bridge_secrets() -> list[str]:
    """
    把 `st.secrets` 里存在、而环境变量里没有的 LLM 配置补进 `os.environ`。

    为什么需要这一步
    ----------------
    `src/llm_client.py` **只读 `os.environ`** —— 这是刻意的：它让主链路
    `src/` 完全脱离 Streamlit，可以离线自测（见 selftest_core.py 的 T41）。
    但部署到托管环境（如 Streamlit Community Cloud）时，密钥只能写在
    `st.secrets` 里，不会自动出现在环境变量中 —— 结果是「本地能生成报告、
    部署后静默降级成模板」，且没有任何报错。

    因此在这里做一次**单向桥接**：环境变量优先，secrets 只做兜底补齐。
    这样本地用 `.env`、云端用 `secrets.toml` 两条路都能工作。

    返回被补齐的键名（供日志/自检查看）。
    """
    filled: list[str] = []
    try:
        import streamlit as st
        for k in _SECRET_KEYS:
            if os.environ.get(k):
                continue
            try:
                v = st.secrets.get(k)                          # type: ignore[attr-defined]
            except Exception:                                  # noqa: BLE001
                v = None          # 没有 secrets.toml 时 st.secrets 会抛异常
            if v not in (None, ""):
                os.environ[k] = str(v)
                filled.append(k)
    except Exception:                                          # noqa: BLE001
        pass
    return filled


# ============================================================
# 离线模拟：注入"真实世界最常见的两类幻觉"
# ============================================================
def mock_text(facts, fabricate: bool = False) -> str:
    """
    用模板文本冒充 LLM 回复。

    fabricate=True 时注入两类真实幻觉，用于**演示校验器真的能拦住**：
      A. 把事实里某个真实百分比改错（×3 量级，模拟"记错"）
      B. 编造事实中不存在的外部基准（行业 ROI、竞品投放额，模拟"编造"）

    设计要点：不改写阶段 4 那种写死的 "14.37 → 43.7"，
    因为上传数据的 facts 里根本没有 14.37，替换会静默失效 ——
    演示就变成了"看起来跑了但没注入"，比不做还误导。
    这里改成**从当前 facts 里动态挑一个数字**来改错。
    """
    base = rt.render_fallback(facts)
    if not fabricate:
        return base

    t = base
    # A. 记错：挑一个事实里的百分数（≥5% 的，避免改出荒谬值）放大 3 倍
    m = re.search(r"(\d{1,2}\.\d{1,2})%", t)
    if m:
        real = float(m.group(1))
        fake = round(real * 3 + 0.37, 2)
        t = t.replace(m.group(0), f"{fake}%", 1)

    # B. 编造：插入没有事实依据的外部基准
    badge = ("\n\n据行业基准，同类出海游戏的整体 ROI 约为 88.88，"
             "同期竞品投放约 5,000 万元（数据来源：行业报告）。\n\n")
    idx = t.find("\n", t.find("## 一、"))
    t = (t[:idx] + badge + t[idx:]) if idx > 0 else (badge + t)
    return t


# ============================================================
# 审计块渲染
# ============================================================
def audit_rows(guard) -> list[dict]:
    """把 GuardResult 转成可放进表格的行。"""
    if guard is None:
        return []
    rows = []
    for h in guard.fabricated:
        rows.append({"级别": "⛔ 无依据", "原文": h.raw,
                     "最接近的事实值": (f"{h.closest:,.4f}"
                                    if h.closest is not None else "—"),
                     "相对误差": (f"{h.rel_err * 100:.2f}%" if h.rel_err is not None else "—"),
                     "上下文": (h.context or "")[:70]})
    for h in guard.rounded:
        rows.append({"级别": "⚠️ 舍入", "原文": h.raw,
                     "最接近的事实值": (f"{h.closest:,.4f}"
                                    if h.closest is not None else "—"),
                     "相对误差": (f"{h.rel_err * 100:.2f}%" if h.rel_err is not None else "—"),
                     "上下文": (h.context or "")[:70]})
    return rows


def render_header_md(source: str, facts_hash: str, llm=None, guard=None) -> str:
    """报告头部：生成方式 / 事实指纹 / 数字校验结论。"""
    tag = {"llm": "大模型生成（数字已校验）", "template": "模板渲染",
           "cache": "缓存命中", "llm_rejected": "大模型稿被拒 → 模板兜底"}.get(source, source)
    lines = [f"> **生成方式**：{tag}　|　**事实指纹**：`{facts_hash}`",
             "> **数字来源**：本次分析的事实层（所有数字可追溯、可复算）"]
    if llm is not None and getattr(llm, "ok", False):
        lines.append(f"> **模型信息**：{llm.usage_line()}")
    if guard is not None:
        if guard.ok:
            lines.append(f"> **数字校验**：✅ 通过（检查 {guard.n_checked} 个数字，"
                         f"精确命中 {guard.n_exact}，舍入偏差 {guard.n_rounded}，无依据 0）")
        else:
            lines.append(f"> **数字校验**：⛔ 发现 {guard.n_fabricated} 处无依据数字，"
                         f"已拦截并降级为模板（原稿另存，见页面下方）")
    return "\n".join(lines) + "\n\n---\n"


def render_audit_md(source: str, guard, llm=None) -> str:
    """数字校验审计块（报告正文的附录）。"""
    lines = ["", "---", "", "### 附：数字校验审计", ""]
    if guard is None:
        lines.append(f"本次为**模板渲染**（生成方式：{source}），"
                     "所有数字直接取自事实层，不经过大模型，无需白名单校验。")
    else:
        lines.append(f"- 检查数字总数：**{guard.n_checked}**")
        lines.append(f"- 精确命中事实白名单：**{guard.n_exact}**")
        lines.append(f"- 存在舍入偏差（<1%，可接受）：**{guard.n_rounded}**")
        lines.append(f"- 无事实依据（已拦截）：**{guard.n_fabricated}**")
        if guard.rounded:
            lines.append("")
            lines.append("舍入偏差明细（证明校验器并非「一刀切拒绝」）：")
            for h in guard.rounded[:10]:
                lines.append(f"- `{h.raw}` ↔ 事实值 `{h.closest:,.4f}`"
                             f"（相对偏差 {h.rel_err * 100:.3f}%）")
    if llm is not None and getattr(llm, "ok", False):
        lines.append(f"- 模型：`{llm.model}`　耗时 {llm.latency:.1f}s　"
                     f"tokens {llm.prompt_tokens}+{llm.completion_tokens}")
    elif llm is not None and getattr(llm, "error", ""):
        lines.append(f"- 模型调用失败：`{llm.error}`")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 主编排
# ============================================================
def generate(facts, *, use_llm: bool = True, force_mock: bool = False,
             mock_fabricate: bool = False, extra_requirement: str = "",
             verify_retries: int = 1, progress=None) -> dict:
    """
    生成决策报告。

    参数
    ----
      facts             : ReportFacts（来自 core/facts.py 现算，或 src/report_facts 读盘）
      use_llm           : 是否尝试调用大模型（False 直接模板渲染）
      force_mock        : 强制离线模拟（不真调 API），用于演示与验证
      mock_fabricate    : 在模拟回复里注入幻觉，用于演示**校验器拦截**
      verify_retries    : 数字校验失败后的重试次数
      progress(frac,msg): 进度回调

    返回
    ----
      {ok, source, text, guard, llm, rejected, facts_hash, audit, log, chars}
      source ∈ {"template", "llm"}
      text 是**完整的 Markdown 文档**（含头部与审计块），可直接下载。
    """
    cb = progress or (lambda *a, **k: None)
    log: list[str] = []

    def _log(m: str) -> None:
        # 只记录，不推进进度条 —— 进度由显式的 cb(frac, msg) 控制，
        # 混在一起会让进度条在"日志消息"上乱跳。
        log.append(m)

    allowed = facts.allowed_numbers()
    payload = facts.to_prompt_payload()
    facts_hash = facts.facts_hash()
    _log(f"事实指纹 {facts_hash}；数字白名单 {len(allowed)} 个值")
    if getattr(facts, "missing", None):
        _log(f"⚠️ 事实层缺失输入：{facts.missing}")

    tpl = rt.render_fallback(facts)

    def _pack(source, body, guard=None, llm=None, rejected=None, note=""):
        doc = render_header_md(source, facts_hash, llm, guard) + "\n" + \
              (body if body.lstrip().startswith("#") else
               f"# 游戏出海广告效果分析报告\n\n{body}") + "\n" + \
              render_audit_md(source, guard, llm)
        return {"ok": True, "source": source, "text": doc, "guard": guard,
                "llm": llm, "rejected": rejected, "facts_hash": facts_hash,
                "audit": (guard.to_dict() if guard is not None else None),
                "audit_rows": audit_rows(guard),
                "log": list(log), "chars": len(doc), "note": note}

    if not use_llm:
        _log("未启用大模型 → 直接模板渲染（所有数字取自事实层）")
        return _pack("template", tpl, note="未启用大模型")

    # 部署到托管环境时密钥可能只写在 st.secrets 里，先桥接成环境变量，
    # 否则会出现「本地能生成、部署后静默降级为模板」且不报错的情况。
    _bridged = bridge_secrets()
    if _bridged:
        _log(f"已从 secrets 补入配置：{', '.join(_bridged)}")

    lcfg = lc.from_env()
    if force_mock:
        lcfg.mock = True
    ok, why = lc.is_configured(lcfg)
    _log(f"大模型状态：{ok}　{why}")

    if not ok:
        _log(f"大模型不可用 → 模板降级（原因：{why}）")
        return _pack("template", tpl, note=f"大模型不可用：{why}")

    user_prompt = rp.build_user_prompt(payload, extra_requirement)
    cb(0.2, "调用大模型生成叙事（数字由事实层提供，模型只负责组织语言）")
    resp = lc.chat(lcfg, rp.SYSTEM_PROMPT, user_prompt,
                   mock_response=(mock_text(facts, mock_fabricate) if lcfg.mock else None))

    if not resp.ok:
        _log(f"调用失败 → 模板降级：{resp.error}")
        return _pack("template", tpl, llm=resp, note=f"调用失败：{resp.error}")

    _log(f"第 1 轮生成完成：{resp.usage_line()}")
    cb(0.6, "逐个数校验大模型输出（三级白名单）")
    guard = rg.check(resp.text, allowed)
    _log(f"第 1 轮校验：检查 {guard.n_checked}，精确 {guard.n_exact}，"
         f"舍入 {guard.n_rounded}，无依据 {guard.n_fabricated}")

    for attempt in range(max(0, verify_retries)):
        if guard.ok:
            break
        hint = rg.build_retry_hint(guard)
        _log(f"第 {attempt + 1} 次重试：把 {len(guard.fabricated)} 处违规数字清单回灌给模型")
        cb(0.75, "把违规数字清单回灌给模型，要求改正")
        hist = rp.build_retry_messages(user_prompt, resp.text, hint)
        retry = lc.chat(lcfg, rp.RETRY_SYSTEM, "",
                        mock_response=(mock_text(facts, mock_fabricate) if lcfg.mock else None),
                        history=hist)
        if retry.ok:
            g2 = rg.check(retry.text, allowed)
            _log(f"第 {attempt + 2} 轮校验：无依据 {g2.n_fabricated}")
            resp, guard = retry, g2

    if guard.ok:
        _log("✅ 采用大模型报告（数字校验通过）")
        cb(1.0, "完成")
        return _pack("llm", resp.text, guard=guard, llm=resp)

    # 宁可用正确但朴素的模板，也不用漂亮但数字错的稿子
    _log(f"⛔ 大模型稿仍有 {guard.n_fabricated} 处无依据数字 → 降级为模板，原稿留档")
    rejected = resp.text
    cb(1.0, "完成（已拦截并降级）")
    out = _pack("llm_rejected", tpl, guard=guard, llm=resp, rejected=rejected,
                note=f"拦截 {guard.n_fabricated} 处无依据数字，已降级")
    # 把被拒稿件也渲染成完整文档，供页面展示对比
    out["rejected_doc"] = (
        f"# ⛔ 被拒的大模型原稿（数字校验未通过）\n\n"
        f"> 事实指纹：`{facts_hash}`　|　无依据数字：**{guard.n_fabricated}** 处\n\n"
        f"> 这份稿件**没有**被采用。保留它是为了让「为什么会拦截」可审计。\n\n---\n\n"
        + resp.text)
    return out


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import report_facts as rf

    print("=" * 78)
    print("app/core/report.py 自测（磁盘事实层 + 离线模拟）")
    print("=" * 78)
    facts = rf.build_facts()

    for tag, kw in [
        ("模板（不用 LLM）", dict(use_llm=False)),
        ("模拟 LLM 正常", dict(use_llm=True, force_mock=True)),
        ("模拟 LLM 注入幻觉", dict(use_llm=True, force_mock=True, mock_fabricate=True)),
    ]:
        r = generate(facts, **kw)
        print(f"\n[{tag}] source={r['source']} 长度={r['chars']}")
        for m in r["log"]:
            print(f"    {m}")
        if r["guard"] is not None:
            print(f"    校验：检查 {r['guard'].n_checked} / 精确 {r['guard'].n_exact} / "
                  f"舍入 {r['guard'].n_rounded} / 无依据 {r['guard'].n_fabricated}")
        if r.get("rejected_doc"):
            print(f"    被拒稿件长度 {len(r['rejected_doc'])} 字符")
