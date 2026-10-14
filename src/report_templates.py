# -*- coding: utf-8 -*-
"""
阶段 4 · 模板降级渲染（Fallback Template）
===========================================

为什么必须有它
--------------
LLM 是不可靠依赖：可能没配密钥、可能超时、可能被限流、可能返回违反事实的数字。
如果报告页在 LLM 失败时白屏，业务方看到的就是"系统坏了"。

所以本模块用**纯代码**渲染一份结构与 LLM 版完全一致的报告。
它有三个用途：
  1. LLM 不可用时的兜底交付物（质量按"可直接发出去"的标准做）
  2. LLM 可用时的**对照基线**（人工可对比 LLM 是否真的更会表达）
  3. Streamlit "刷新立即出结果"的首屏内容（LLM 版生成后替换）

原则：模板只做**事实的搬运与排版**，不做任何推断性表述。
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass


def _as_dict(facts) -> dict:
    return asdict(facts) if is_dataclass(facts) else dict(facts)


def _wan(x, nd=1):
    """元 -> 万元字符串。"""
    if x is None:
        return "—"
    try:
        return f"{float(x) / 1e4:,.{nd}f}"
    except Exception:
        return "—"


def _n(x, nd=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _pct(x, nd=2, sign=True):
    if x is None:
        return "—"
    try:
        s = "+" if (sign and float(x) > 0) else ""
        return f"{s}{float(x):.{nd}f}%"
    except Exception:
        return "—"


SEV_LABEL = {"high": "高", "medium": "中", "low": "低", "info": "说明"}


def render_fallback(facts) -> str:
    """渲染完整的六节报告（纯模板，零推断）。"""
    d = _as_dict(facts)
    meta = d.get("meta") or {}
    q = d.get("model_quality") or {}
    opt = d.get("optimization") or {}
    alloc = d.get("allocation") or []
    marg = d.get("marginal") or []
    elast = d.get("elasticity") or []
    risks = d.get("risks") or []

    by_ch = {m.get("渠道"): m for m in marg}
    rank = sorted(marg, key=lambda m: (m.get("边际回报") is None, -(m.get("边际回报") or 0)))
    top = rank[0] if rank else {}
    bot = rank[-1] if rank else {}

    # ---- 一、结论 ----
    # 注意：弹性宽度 = 上限 − 下限。±20% 对应宽度 0.4，±40% 对应宽度 0.8。
    # （早期把"宽度 0.4"误当 ±40%，导致报告引用了错误的档位收益。）
    def _find_w(w):
        return next((e for e in elast
                     if abs((e.get("弹性宽度") or -1) - w) < 1e-9), None)

    rec = _find_w(0.8) or _find_w(0.4)
    rec20 = _find_w(0.4)
    high_ch = [m.get("渠道") for m in rank if (m.get("边际回报") or 0) >= 1]
    low_ch = [m.get("渠道") for m in rank if (m.get("边际回报") or 0) < 1]

    lines = []
    lines.append("## 一、一句话结论")
    lines.append("")
    _shift = (f"把预算从低边际回报渠道（{'、'.join(low_ch)}）"
              f"向高边际回报渠道（{'、'.join(high_ch)}）倾斜" if low_ch
              else f"把预算进一步向高边际回报渠道（{'、'.join(high_ch)}）集中")
    lines.append(
        f"在保持总预算 **{_wan(opt.get('窗口内总预算(元)'))} 万元**不变、"
        f"单渠道允许调整 **{opt.get('约束') or '—'}** 的前提下，"
        f"{_shift}，"
        f"模型估计可把窗口内媒体贡献从 **{_wan(opt.get('当前分配预期贡献(元)'))} 万元**"
        f"提升到 **{_wan(opt.get('优化分配预期贡献(元)'))} 万元**，"
        f"即 **{_pct(opt.get('提升幅度(%)'))}**。"
        f"{'该结论的落地幅度需要业务方在下一节的弹性档位中拍板。' if rec else ''}")
    lines.append("")
    lines.append(
        f"> 说明：上述金额为**模型内部口径**（仅用于同一模型内的相对比较），"
        f"不可直接与财务口径的营销收入等同。对外沟通请以**相对提升幅度**为准。"
        f"增量 94% 可信区间为 "
        f"{_wan(opt.get('增量94%CI下界(元)'))} ~ {_wan(opt.get('增量94%CI上界(元)'))} 万元。"
        f"本报告由模板生成（未启用大模型润色），所有数字均直接取自模型输出文件，"
        f"可用其中的事实指纹追溯。")
    lines.append("")

    # ---- 二、问题 ----
    lines.append("## 二、我们面对的问题")
    lines.append("")
    lines.append(
        f"公司 {meta.get('数据周期', '—')} 共 {meta.get('周数', '—')} 周的出海投放数据中，"
        f"五个渠道（{'、'.join(meta.get('渠道') or [])}）合计花费 "
        f"**{_wan(meta.get('总花费(元)'))} 万元**，对应总收入 "
        f"**{_wan(meta.get('总收入(元)'))} 万元**。")
    lines.append("")
    lines.append(
        "直观做法是算各渠道 ROI 然后砍低投高，但这样做有三个坑："
        "**一是**渠道之间高度同步投放（花同一笔钱、涨同一个大盘），单看 ROI 无法分离各自的真实贡献；"
        "**二是**广告效果有跨周残留（本周投放的效果会延续到之后几周），当周对比会低估长效渠道；"
        "**三是**每个渠道都会饱和——钱投到一定程度边际回报骤降，"
        "因此「把全部预算给 ROI 最高的渠道」在数学上最优、在业务上不可行。")
    lines.append("")
    lines.append(
        "本项目用贝叶斯营销组合模型（MMM）同时处理这三个问题："
        "用广告残留（Adstock）刻画跨周效果、用饱和曲线（Hill）刻画边际递减、"
        "用先验与后验分布量化不确定性，最终给出**带可信区间的预算重分配建议**。")
    lines.append("")

    # ---- 三、现状诊断 ----
    lines.append("## 三、现状诊断：钱花得怎么样")
    lines.append("")
    lines.append("| 渠道 | 当前周花费(万元) | 当前周贡献(万元) | 边际回报 | 平均 ROI |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in rank:
        lines.append(
            f"| {m.get('渠道')} | {_wan(m.get('当前周花费(元)'))} | "
            f"{_wan(m.get('当前周贡献(元)'))} | **{_n(m.get('边际回报'), 3)}** | "
            f"{_n(m.get('平均ROI'), 3)} |")
    lines.append("")
    if top and bot:
        lines.append(
            f"**怎么读这张表**：*边际回报* 指在当前花费水平上再多投 1 元能多拿回多少元。"
            f"最高的 **{top.get('渠道')}** 为 **{_n(top.get('边际回报'), 3)}**，"
            f"最低的 **{bot.get('渠道')}** 仅 **{_n(bot.get('边际回报'), 3)}**。")
        lines.append("")
        lines.append(
            f"这意味着：往 {top.get('渠道')} 多投 1 元的回报，明显高于往 {bot.get('渠道')} 多投 1 元"
            f"（**{_n(top.get('边际回报'), 3)} 元** vs **{_n(bot.get('边际回报'), 3)} 元**）。"
            f"{'其中 ' + str(bot.get('渠道')) + ' 的边际回报低于 1，意味着当前水平上每多投 1 元拿不回 1 元，属于该收缩的对象。' if (bot.get('边际回报') or 1) < 1 else ''}")
    lines.append("")

    # ---- 四、优化建议 ----
    lines.append("## 四、优化建议：钱该怎么花")
    lines.append("")
    lines.append("| 渠道 | 当前窗口预算(万元) | 建议窗口预算(万元) | 建议变化 | 说明 |")
    lines.append("|---|---:|---:|---:|---|")
    for a in sorted(alloc, key=lambda x: -(x.get("变化(%)") or 0)):
        chg = a.get("变化(%)")
        desc = "加投（顶到上限）" if (chg or 0) > 1 else ("减投（压到下限）" if (chg or 0) < -1 else "基本不变")
        lines.append(
            f"| {a.get('渠道')} | {_wan(a.get('当前窗口预算(元)'))} | "
            f"{_wan(a.get('优化窗口预算(元)'))} | {_pct(chg, 1)} | {desc} |")
    lines.append("")
    if opt.get("预期增量(元)"):
        lines.append(
            f"预期窗口内媒体贡献增量 **{_wan(opt.get('预期增量(元)'))} 万元**"
            f"（模型内部口径），"
            f"94% 可信区间 [{_wan(opt.get('增量94%CI下界(元)'))}, "
            f"{_wan(opt.get('增量94%CI上界(元)'))}] 万元，"
            f"相对提升 **{_pct(opt.get('提升幅度(%)'))}**。")
        lines.append("")
    lines.append(
        "> **为什么不干脆全押最高效渠道？** 模型在不受约束时的理论最优是把全部预算压给单一渠道"
        "（提升幅度远高于上表）。但真实业务中不允许这样做：渠道团队编制、广告账号权重、"
        "平台返点与商务关系、素材产能都需要维持基本投放量。"
        "因此本方案在「单渠道调整幅度受限」的约束下求最优——**约束越宽松，能拿的增量越多**，"
        "这正是下一节要业务方拍板的事。")
    lines.append("")

    # ---- 五、需要拍板 ----
    lines.append("## 五、需要业务方拍板的事")
    lines.append("")
    lines.append("真正决定收益的不是模型小数点后几位，而是**允许每个渠道调整多大幅度**。")
    lines.append("")
    lines.append("| 弹性档位（单渠道可调整范围） | 模型估计提升 | 数值优化是否收敛 |")
    lines.append("|---|---:|---|")
    for e in elast:
        label = e.get("说明") or "—"
        lines.append(f"| {label} | **{_pct(e.get('提升(%)'))}** | "
                     f"{'是' if e.get('收敛') else '否（已用兜底解）'} |")
    lines.append("")
    if rec:
        lines.append(
            f"**推荐档位**：±20% ~ ±40%。理由：±20% 内变动对渠道团队与平台关系冲击很小、"
            f"可当月执行，对应提升 **{_pct(rec20.get('提升(%)') if rec20 else None)}**；"
            f"若业务方能接受 ±40% 的调整（需要 1~2 个季度逐步到位），"
            f"可把提升空间放大到 **{_pct(rec.get('提升(%)'))}**。"
            f"建议分两步走：先按 ±20% 执行一个季度验证模型预测，再决定是否放开。")
        lines.append("")
        _e5 = _find_w(0.1)
        _einf = next((e for e in elast if e.get("弹性上限") is None), None)
        lines.append(
            f"> 读法：这张表的每一行都是一次完整的预算重分配求解（总预算不变，"
            f"只改『单渠道允许调整的幅度』）。可以看到**收益对弹性极为敏感**——"
            f"从 ±5% 的 {_pct(_e5.get('提升(%)') if _e5 else None)}"
            f" 到无约束的 {_pct(_einf.get('提升(%)') if _einf else None)}。"
            f"这意味着：**先和渠道团队谈弹性，比纠结模型精度更值钱**。")
    lines.append("")

    # ---- 六、边界 ----
    lines.append("## 六、这份分析的边界")
    lines.append("")
    if risks:
        rank_sev = {"high": 0, "medium": 1, "low": 2, "info": 3}
        for r in sorted(risks, key=lambda x: rank_sev.get(x.get("severity"), 9)):
            lines.append(f"- **[{SEV_LABEL.get(r.get('severity'), '?')}] {r.get('title')}**"
                         f"：{r.get('detail')}")
    lines.append("")
    lines.append(
        "**什么情况下上面的结论会失效**：如果未来几周的控制变量"
        "（版本更新、竞品活动、节假日、长期趋势）实际水平与历史分布明显偏离，"
        "或平台算法/竞价环境发生结构性变化，则需要重新拟合模型后再决策。")
    lines.append("")

    lines.append("---")
    lines.append(f"*生成时间：{d.get('generated_at', '—')}　|　"
                 f"事实指纹：{_hash_of(d)}　|　渲染方式：模板降级*")
    return "\n".join(lines)


def _hash_of(d: dict) -> str:
    import hashlib
    import json
    try:
        raw = json.dumps(d, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "n/a"


if __name__ == "__main__":
    from report_facts import build_facts
    md = render_fallback(build_facts())
    print(md[:1600])
    print("\n... 总长度:", len(md), "字符")
