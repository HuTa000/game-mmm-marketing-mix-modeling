# -*- coding: utf-8 -*-
"""
阶段 4 · LLM 自动报告工作流（主流程）
======================================

业务问题
--------
模型跑完了，但业务方（市场负责人、CEO）不会去看后验分布表和森林图。
分析师要交的是**一页纸能读懂的决策报告**。手工写要 1~2 小时，且每次数据更新都要重写。

本流程把「结构化结果 → 业务叙事」自动化，同时保证三件事：
  1. **不编数字**：所有数字来自 report_facts，LLM 输出经 report_guard 白名单校验；
  2. **不白屏**：LLM 不可用或不通过校验时，自动降级为模板报告；
  3. **不重复烧钱**：按事实指纹缓存，数据没变就不重新调用。

流程图
------
    各阶段 CSV
        │
        ▼
    report_facts.build_facts()  ──►  facts.json（唯一数字源）
        │                                │
        │                                └─► allowed_numbers() ──┐
        ▼                                                          │
    事实指纹 hash ──► 缓存命中？ ── 是 ──► 直接返回                │
        │ 否                                                       │
        ▼                                                          │
    report_prompts 组装 prompt                                     │
        │                                                          │
        ▼                                                          │
    llm_client.chat()  ── 失败 ──┐                                 │
        │ 成功                    │                                 │
        ▼                        │                                 │
    report_guard.check() ────────┼──► 违规 ──► 重试 1 次           │
        │ 通过                    │               │ 仍违规          │
        ▼                        ▼               ▼                 │
    采用 LLM 文本            模板降级 ◄──────────┘                 │
        │                        │                                 │
        └────────┬───────────────┘                                 │
                 ▼                                                  │
        写入 stage4_report.md  ◄────────────────────────────────────┘
        （含数字校验审计块 + 被拒稿件另存）

运行
----
    .venv/Scripts/python.exe src/stage4_report.py            # 正常
    .venv/Scripts/python.exe src/stage4_report.py --force    # 忽略缓存
    LLM_MOCK=1 ... stage4_report.py                          # 离线验证链路
    LLM_MOCK=1 LLM_MOCK_FABRICATE=1 ...                      # 验证幻觉拦截
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import report_facts as rf
import report_guard as rg
import report_prompts as rp
import report_templates as rt
from llm_client import from_env, is_configured, chat

REPDIR = os.path.join(ROOT, "outputs", "reports")
CACHEDIR = os.path.join(REPDIR, ".cache")
OUT_MD = os.path.join(REPDIR, "stage4_report.md")
OUT_FACTS = os.path.join(REPDIR, "stage4_facts.json")
OUT_GUARD = os.path.join(REPDIR, "stage4_guard.json")
OUT_REJECTED = os.path.join(REPDIR, "stage4_report.llm_rejected.md")

SEP = "=" * 78


def sec(t):
    print("\n" + SEP + "\n" + t + "\n" + SEP)


@dataclass
class ReportConfig:
    use_llm: bool = True
    verify_retries: int = 1        # 数字校验失败后的重试次数
    force: bool = False            # 忽略缓存
    cache: bool = True
    extra_requirement: str = ""


# ================================================================ 头部/尾部

def _header(source: str, facts_hash: str, llm_line: str, guard: rg.GuardResult | None) -> str:
    tag = {"llm": "大模型生成", "template": "模板降级", "cache": "缓存命中"}.get(source, source)
    lines = [
        f"> **生成方式**：{tag}　|　**事实指纹**：`{facts_hash}`",
        f"> **数字来源**：`outputs/reports/stage4_facts.json`（全部数字可追溯）",
    ]
    if llm_line:
        lines.append(f"> **模型信息**：{llm_line}")
    if guard is not None:
        if guard.ok:
            lines.append(f"> **数字校验**：通过（检查 {guard.n_checked} 个数字，"
                         f"精确命中 {guard.n_exact}，舍入偏差 {guard.n_rounded}，无依据 0）")
        else:
            lines.append(f"> ⚠️ **数字校验**：发现 {guard.n_fabricated} 处无依据数字，"
                         f"已自动降级为模板报告（大模型原稿见 `stage4_report.llm_rejected.md`）")
    return "\n".join(lines) + "\n\n---\n"


def _audit_block(guard: rg.GuardResult | None, llm) -> str:
    lines = ["", "---", "", "### 附：数字校验审计", ""]
    if guard is None:
        lines.append("本次为模板渲染，所有数字直接取自模型输出文件，无需白名单校验。")
    else:
        lines.append(f"- 检查数字总数：**{guard.n_checked}**")
        lines.append(f"- 精确命中事实白名单：**{guard.n_exact}**")
        lines.append(f"- 存在舍入偏差（<5%，可接受）：**{guard.n_rounded}**")
        lines.append(f"- 无事实依据（已拦截）：**{guard.n_fabricated}**")
        if guard.rounded:
            lines.append("")
            lines.append("舍入偏差明细：")
            for h in guard.rounded[:10]:
                lines.append(f"- `{h.raw}` ↔ 事实值 `{h.closest:.4f}`"
                             f"（相对偏差 {h.rel_err * 100:.2f}%）")
    if llm is not None and getattr(llm, "ok", False):
        lines.append(f"- 模型：`{llm.model}`　耗时 {llm.latency:.1f}s　"
                     f"tokens {llm.prompt_tokens}+{llm.completion_tokens}")
    lines.append("")
    return "\n".join(lines)


# ================================================================ 生成

def _mock_text(facts_obj, fabricate: bool) -> str:
    """
    离线模拟：用模板文本冒充 LLM 回复。
    fabricate=True 时注入两类**真实世界最常见的幻觉**：
      A. 把事实里的数字记错（14.37% -> 43.7%）
      B. 编造事实数据中不存在的外部基准（行业 ROI、竞品投放额）
    """
    base = rt.render_fallback(facts_obj)
    if not fabricate:
        return base
    t = base.replace("14.37", "43.7", 1)                     # 幻觉 A：数字记错
    t = t.replace("## 一、一句话结论\n",
                  "## 一、一句话结论\n\n据行业基准，同类出海游戏的整体 ROI 约为 88.88，"
                  "同期竞品投放约 5,000 万元。\n", 1)          # 幻觉 B：编造外部基准
    return t


def generate_report(cfg: ReportConfig | None = None, facts_obj=None,
                    log=print) -> dict:
    """生成报告。返回 dict（含文本、来源、校验结果、路径）。"""
    cfg = cfg or ReportConfig()
    os.makedirs(REPDIR, exist_ok=True)
    os.makedirs(CACHEDIR, exist_ok=True)

    # ---------- 1. 事实 ----------
    facts_obj = facts_obj or rf.build_facts()
    facts_hash = facts_obj.facts_hash()
    payload = facts_obj.to_prompt_payload()
    allowed = facts_obj.allowed_numbers()
    with open(OUT_FACTS, "w", encoding="utf-8") as f:
        f.write(facts_obj.to_json())
    log(f"  事实指纹 = {facts_hash}  白名单大小 = {len(allowed)}")
    if facts_obj.missing:
        log(f"  [警告] 缺失输入表: {facts_obj.missing}")

    # ---------- 2. 缓存 ----------
    cache_llm = os.path.join(CACHEDIR, f"report_{facts_hash}_llm.md")
    cache_tpl = os.path.join(CACHEDIR, f"report_{facts_hash}_template.md")
    if cfg.cache and not cfg.force:
        for src, path in (("llm", cache_llm), ("template", cache_tpl)):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                log(f"  缓存命中（{src}）: {os.path.relpath(path, ROOT)}")
                g = None
                if os.path.exists(OUT_GUARD):
                    try:
                        with open(OUT_GUARD, encoding="utf-8") as f:
                            g = json.load(f).get("guard")
                    except Exception:
                        g = None
                return _finalize(text, facts_obj, "cache", facts_hash, g, None, cfg)

    # ---------- 3. LLM 生成 ----------
    lcfg = from_env()
    llm_ok, llm_why = is_configured(lcfg)
    log(f"  LLM 状态: {llm_ok} | {llm_why}")

    result = {"source": "template", "text": "", "guard": None, "llm": None}

    if cfg.use_llm and llm_ok:
        user_prompt = rp.build_user_prompt(payload, cfg.extra_requirement)
        fabricate = str(os.environ.get("LLM_MOCK_FABRICATE", "")).strip() in ("1", "true", "yes")
        mock_txt = _mock_text(facts_obj, fabricate) if lcfg.mock else None

        resp = chat(lcfg, rp.SYSTEM_PROMPT, user_prompt, mock_response=mock_txt)
        log(f"  LLM 调用: {resp.usage_line()}")
        if resp.ok:
            guard = rg.check(resp.text, allowed)
            log(f"  数字校验(第1轮): 检查 {guard.n_checked} / 精确 {guard.n_exact} / "
                f"舍入 {guard.n_rounded} / 无依据 {guard.n_fabricated}")

            # 违规 -> 带纠正指令重试
            for attempt in range(cfg.verify_retries):
                if guard.ok:
                    break
                hint = rg.build_retry_hint(guard)
                log(f"  [重试 {attempt + 1}] 注入 {len(guard.fabricated)} 处违规数字的纠正指令")
                hist = rp.build_retry_messages(user_prompt, resp.text, hint)
                # history 末尾已含纠正指令，user 传空避免重复追加
                retry = chat(lcfg, rp.RETRY_SYSTEM, "", mock_response=mock_txt, history=hist)
                log(f"  LLM 重试: {retry.usage_line()}")
                if retry.ok:
                    g2 = rg.check(retry.text, allowed)
                    log(f"  数字校验(第{attempt + 2}轮): 无依据 {g2.n_fabricated}")
                    resp, guard = retry, g2

            if guard.ok:
                result = {"source": "llm", "text": resp.text, "guard": guard, "llm": resp}
                log("  ✅ 采用大模型报告（数字校验通过）")
            else:
                # 宁可用正确但朴素的模板，也不用漂亮但数字错的稿子
                with open(OUT_REJECTED, "w", encoding="utf-8") as f:
                    f.write(f"# 被拒稿件（数字校验未通过）\n\n"
                            f"事实指纹: {facts_hash}\n\n---\n\n{resp.text}")
                log(f"  ⚠️ 大模型稿仍有 {guard.n_fabricated} 处无依据数字 -> 降级为模板")
                log(f"     被拒稿件已另存: {os.path.relpath(OUT_REJECTED, ROOT)}")
                result = {"source": "template", "text": rt.render_fallback(facts_obj),
                          "guard": guard, "llm": resp}
        else:
            log(f"  LLM 调用失败 -> 降级为模板（原因：{resp.error}）")
            result = {"source": "template", "text": rt.render_fallback(facts_obj),
                      "guard": None, "llm": resp}
    else:
        reason = "未启用 LLM" if not cfg.use_llm else llm_why
        log(f"  跳过 LLM -> 直接模板渲染（{reason}）")
        result = {"source": "template", "text": rt.render_fallback(facts_obj),
                  "guard": None, "llm": None}

    # ---------- 4. 缓存与落盘 ----------
    cache_path = cache_llm if result["source"] == "llm" else cache_tpl
    if cfg.cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
    return _finalize(result["text"], facts_obj, result["source"], facts_hash,
                     result["guard"], result["llm"], cfg)


def _finalize(text, facts_obj, source, facts_hash, guard, llm, cfg) -> dict:
    """拼装最终文档并写盘。"""
    hdr = _header(source, facts_hash, (llm.usage_line() if llm is not None else ""), guard)
    body = text if text.lstrip().startswith("#") else f"# 游戏出海广告效果分析报告\n\n{text}"
    doc = hdr + "\n" + body + "\n" + _audit_block(guard, llm)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(doc)
    if guard is not None:
        with open(OUT_GUARD, "w", encoding="utf-8") as f:
            json.dump({"facts_hash": facts_hash, "source": source,
                       "guard": guard.to_dict()}, f, ensure_ascii=False, indent=2)
    return {"text": doc, "source": source, "facts_hash": facts_hash,
            "guard": guard, "llm": llm, "path": OUT_MD,
            "chars": len(doc)}


# ================================================================ 入口

def main():
    force = "--force" in sys.argv
    print(SEP)
    print("阶段 4 · LLM 自动报告工作流")
    print(SEP)

    sec("STEP 1. 构建事实层")
    facts_obj = rf.build_facts()
    print(f"  元信息: {facts_obj.meta.get('数据周期')} / {facts_obj.meta.get('周数')} 周")
    print(f"  优化结论: 提升 {facts_obj.optimization.get('提升幅度(%)')}%  "
          f"增量 {facts_obj.optimization.get('预期增量(元)')} 元")
    print(f"  风险条目: {len(facts_obj.risks)}")

    sec("STEP 2. 生成报告")
    r = generate_report(ReportConfig(force=force))
    print(f"\n  来源 = {r['source']}  长度 = {r['chars']} 字符")
    print(f"  输出 = {os.path.relpath(r['path'], ROOT)}")

    sec("STEP 3. 报告预览（前 1200 字）")
    print(r["text"][:1200])

    sec("产出")
    for p in [OUT_MD, OUT_FACTS, OUT_GUARD]:
        print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {os.path.relpath(p, ROOT)}")
    if os.path.exists(OUT_REJECTED):
        print(f"  OK  {os.path.relpath(OUT_REJECTED, ROOT)}（有大模型稿被拒）")


if __name__ == "__main__":
    main()
