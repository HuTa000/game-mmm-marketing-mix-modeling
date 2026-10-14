# -*- coding: utf-8 -*-
"""
阶段 4 · 数字校验器（Report Guard）
====================================

解决的问题
----------
LLM 生成报告时最危险的失效模式是 **幻觉数字**：
把 14.37% 写成 43.7%、把 5445 万写成 6455 万、或凭空编一个"行业平均 ROI=4.2"。

本模块做一件事：**把 LLM 输出里的每个数字，逐个对照事实白名单**。

分级判据
--------
  exact       误差 < 0.5%   -> 通过
  rounded     误差 < 5%     -> 通过，但标注"存在舍入偏差"
  fabricated  无法对应任何白名单值 -> **违规**

设计要点
--------
1. 白名单来自 report_facts.allowed_numbers()，已覆盖万元/亿元/百分数等换算形式；
2. 匹配用相对误差 + 绝对容差双判据，避免"13 周"这类小整数误报；
3. 输出结构化结果，供报告标注与自动重试使用。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# 匹配数字：支持千分位、小数、百分号、负号
_NUM_TOKEN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?")

# 不参与数字校验的上下文（代码块、URL、路径、行内代码、hex 指纹）
_SKIP_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),
    re.compile(r"https?://\S+"),
    re.compile(r"[A-Za-z]:\\\S+"),
    re.compile(r"[\w./]+\.(?:py|csv|png|md|json|nc|pkl)\b"),
    re.compile(r"`[^`]*`"),                       # 行内代码（含事实指纹）
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),          # hex 指纹串，会被误切成数字
]


@dataclass
class NumberHit:
    raw: str          # 原文 token
    value: float      # 归一化数值
    percent: bool     # 是否带 %
    closest: float | None = None   # 最接近的白名单值
    rel_err: float | None = None   # 相对误差
    level: str = "fabricated"      # exact / rounded / fabricated
    context: str = ""              # 上下文片段（便于人工核对）


@dataclass
class GuardResult:
    ok: bool = True
    hits: list = field(default_factory=list)
    n_checked: int = 0
    n_exact: int = 0
    n_rounded: int = 0
    n_fabricated: int = 0

    @property
    def fabricated(self) -> list:
        return [h for h in self.hits if h.level == "fabricated"]

    @property
    def rounded(self) -> list:
        return [h for h in self.hits if h.level == "rounded"]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checked": self.n_checked,
            "exact": self.n_exact,
            "rounded": self.n_rounded,
            "fabricated": self.n_fabricated,
            "fabricated_detail": [
                {"原文": h.raw, "最接近白名单值": h.closest,
                 "相对误差": None if h.rel_err is None else round(h.rel_err, 4),
                 "上下文": h.context}
                for h in self.fabricated
            ],
            "rounded_detail": [
                {"原文": h.raw, "最接近白名单值": h.closest,
                 "相对误差": round(h.rel_err, 4) if h.rel_err is not None else None}
                for h in self.rounded
            ],
        }


def _strip_skippable(text: str) -> str:
    for p in _SKIP_PATTERNS:
        text = p.sub(" ", text)
    return text


def _to_float(tok: str) -> tuple[float, bool]:
    percent = tok.endswith("%")
    s = tok.rstrip("%").replace(",", "")
    return float(s), percent


def expand_units(values, extra=()):
    """
    **条件化**单位展开（不能无条件展开！）：
      - 金额类（|v| >= 1e3）：同时接受 元 / 万元 / 百万元 / 亿元
      - 比例类（0 < |v| <= 1）：同时接受 小数 / 百分数
      - 中间量（比值、分位、1~1e3）：只保留原值

    踩过的坑：早期对**所有**数值都做 ×100 / ÷100，
    相当于把白名单铺满整条实数轴，于是编造的「88.88」能"擦边"命中
    真实的「89.83」（相对差 1.06%）、「43.7%」命中「42.9」——
    校验器形同虚设。条件化后假阴性消失。
    """
    out = set()
    for v in values:
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        out.add(v)
        a = abs(v)
        if a >= 1e3:
            out.add(v / 1e4)
            out.add(v / 1e6)
            out.add(v / 1e8)
        elif 0 < a <= 1.0:
            out.add(v * 100.0)
    out.update(float(x) for x in extra)
    return sorted(out)


def classify(value: float, allowed: list, allowed_set: set | None = None,
             rel_tol_exact: float = 0.003, rel_tol_round: float = 0.01):
    """
    分级判据：
      1) 小整数（序号 / 周数 / 渠道数）：精确匹配，容忍 ±0；
      2) 其余（金额、百分比）：相对误差 < 0.3% 视为精确命中，
         < 1.0% 视为存在舍入偏差（可接受），否则判为**无依据**。

    容差为何这么紧：实测真实舍入（4760.6769 -> "4760.7"）误差仅 0.0005%，
    而编造数字（14.37 -> "43.7"）误差达 204%。0.3%/1.0% 的分界线
    既能容纳正常四舍五入，又能拦住"看起来像但不是"的数字。
    早期用 ±0.51 绝对容差，会让「7.777」被「7.5」放行。
    """
    aset = allowed_set if allowed_set is not None else set(allowed)

    # 1) 小整数通道：覆盖"5 个渠道""13 周""第 3 大"这类表述
    rv = int(round(value))
    if abs(value - rv) < 1e-9 and -60 <= rv <= 200 and float(rv) in aset:
        return float(rv), 0.0, "exact"

    # 2) 相对容差通道
    best, best_err, best_rel = None, None, None
    for a in allowed:
        if a == 0:
            err, denom = abs(value), 1.0
        else:
            err, denom = abs(value - a), abs(a)
        rel = err / denom if denom > 1e-12 else err
        if best_err is None or err < best_err - 1e-15:
            best, best_err, best_rel = a, err, rel
    if best is None:
        return None, None, "fabricated"
    if best_rel <= rel_tol_exact:
        return best, best_rel, "exact"
    if best_rel <= rel_tol_round:
        return best, best_rel, "rounded"
    return best, best_rel, "fabricated"


def check(text: str, allowed: list, context_window: int = 18,
          auto_expand: bool = True) -> GuardResult:
    """校验文本中的所有数字。auto_expand=True 时自动做单位展开，避免万元表述误报。"""
    res = GuardResult()
    if auto_expand:
        allowed = expand_units(allowed)
    aset = set(allowed)
    clean = _strip_skippable(text or "")
    for m in _NUM_TOKEN.finditer(clean):
        tok = m.group(0)
        try:
            v, is_pct = _to_float(tok)
        except Exception:
            continue
        # 纯序号（1. / 2.）由 abs_tol 兜住，此处不过滤，保持判定可解释
        # 白名单里有 0~30 的通用值
        c = clean[max(0, m.start() - context_window): m.end() + context_window]
        ctx = c.replace("\n", " ")
        if is_pct:
            # 百分比按"百分数"口径匹配（白名单已含 ×100 与 ÷100 形式）
            pass
        near, rel, lvl = classify(v, allowed)
        hit = NumberHit(raw=tok, value=v, percent=is_pct,
                        closest=near, rel_err=rel, level=lvl, context=ctx)
        res.hits.append(hit)
        res.n_checked += 1
        if lvl == "exact":
            res.n_exact += 1
        elif lvl == "rounded":
            res.n_rounded += 1
        else:
            res.n_fabricated += 1
    res.ok = res.n_fabricated == 0
    return res


def build_retry_hint(res: GuardResult, max_items: int = 12) -> str:
    """把违规数字整理成给 LLM 的纠正指令（用于自动重试）。"""
    if not res.fabricated:
        return ""
    lines = ["以下数字在事实数据中找不到依据，请删除或改用事实中的准确数字："]
    for h in res.fabricated[:max_items]:
        lines.append(f'  - "{h.raw}"（上下文：{h.context}）')
    if len(res.fabricated) > max_items:
        lines.append(f"  ... 另有 {len(res.fabricated) - max_items} 处")
    lines.append("要求：只能引用提供的事实数据中的数字，不要自行计算或估算新数字。")
    return "\n".join(lines)


if __name__ == "__main__":
    allowed = [14.37, 54448644, 47606769, 13915308, 6.13, 13, 5, 0.6, 1.4, 2025]
    good = "优化后预期贡献从 4760.7 万元提升到 5444.9 万元，提升 14.37%。"
    bad = "优化后预期贡献提升到 6455 万元，提升 43.7%，共涉及 5 个渠道，窗口 13 周。"
    for tag, txt in [("正常", good), ("含幻觉", bad)]:
        r = check(txt, allowed)
        print(f"\n[{tag}] ok={r.ok} 检查={r.n_checked} 精确={r.n_exact} "
              f"舍入={r.n_rounded} 编造={r.n_fabricated}")
        for h in r.fabricated:
            print(f"   违规: {h.raw}  最接近={h.closest}  相对误差={h.rel_err}")
    print("\n" + build_retry_hint(check(bad, allowed)))
