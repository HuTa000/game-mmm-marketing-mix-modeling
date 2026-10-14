# -*- coding: utf-8 -*-
"""
阶段 4 · 报告事实层（Report Facts）
===================================

设计原则：**数字与叙事严格分离**。

    各阶段产物(CSV)  ──►  ReportFacts  ──►  LLM / 模板
                          唯一数字源         只负责组织语言
                              │
                              └──► allowed_numbers() ──► 校验 LLM 是否编造数字

为什么不直接把原始表丢给 LLM：
  1. LLM 会"顺着语境编数字"——让它算占比、做四则运算，错误率不可控；
  2. 编造出来的数字无法审计，业务/面试场景是硬伤；
  3. 正确分工：**确定性代码负责算数，LLM 负责表达**。

本模块产出结构化的 ReportFacts，并提供三个下游接口：
  - to_prompt_payload()  给 LLM 的紧凑 JSON（唯一允许引用的数字全集）
  - allowed_numbers()    数字白名单，供 report_guard 校验
  - RISKS                项目级局限声明（诚实披露 = 技术深度）

运行（自检）：
    .venv/Scripts/python.exe src/report_facts.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPDIR = os.path.join(ROOT, "outputs", "reports")
DATA_CSV = os.path.join(ROOT, "data", "processed", "game_ads_weekly.csv")
SCHEMA_VERSION = "1.0"

_NUMRE = re.compile(r"-?\d[\d,]*\.?\d*")


# ================================================================ 工具

def _num(x, default=None):
    """尽量把任意标量转成 float（容忍千分位、百分号、正号）。"""
    if x is None:
        return default
    if isinstance(x, (bool, np.bool_)):
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else default
    s = str(x).strip().replace(",", "").replace("%", "").replace("+", "")
    try:
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _num_first(s, default=None):
    """从复合文本里抽第一个数字，如 '2,865,997 (6.0%)' -> 2865997.0。"""
    m = _NUMRE.search(str(s) if s is not None else "")
    return _num(m.group(0), default) if m else default


def _clean(obj):
    """递归把 NaN/inf 变成 None，保证 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _summary(path) -> dict:
    """读『指标 / 值』两列汇总表。值可能是数字、百分比或复合文本。"""
    d = _csv(path)
    out = {}
    if d is None or d.shape[1] < 2:
        return out
    kc, vc = d.columns[0], d.columns[1]
    for _, r in d.iterrows():
        k = str(r[kc]).strip()
        out[k] = {"text": str(r[vc]), "num": _num_first(r[vc])}
    return out


def _sv(S: dict, key: str, default=None):
    """从 summary dict 取数值。"""
    item = S.get(key)
    return item["num"] if item and item["num"] is not None else default


def _st(S: dict, key: str, default=None):
    """从 summary dict 取文本。"""
    item = S.get(key)
    return item["text"] if item and item["text"] is not None else default


# ================================================================ 常量

CHANNELS_CN = ["Facebook", "Google", "TikTok", "YouTube", "KOL"]

GLOSSARY = {
    "ROI": "渠道总贡献 ÷ 渠道总花费。>1 表示该渠道带来的收入高于投入。",
    "边际回报": "在当前花费水平上，再多投 1 元能多拿回多少元。"
                "这是决定『钱该往哪挪』的指标；>1 值得加投，<1 应考虑缩减。",
    "饱和度": "当前花费在饱和曲线上的位置。接近 100% 表示继续加钱几乎没有回报。",
    "Adstock(广告残留)": "广告效果的跨期衰减——本周投放的效果会延续到之后几周。",
    "Hill 饱和曲线": "花费与效果之间呈 S 形，边际回报随投入先升后降。",
    "可信区间": "贝叶斯后验给出的区间估计，反映参数不确定性（非置信区间的频率学含义）。",
    "R-hat": "多链收敛诊断，<1.01 视为收敛良好。",
    "ESS": "有效样本量，越大说明后验估计越稳定。",
}

RISKS = [
    {
        "id": "R1", "severity": "info", "title": "数据为仿真数据（真值已知）",
        "detail": "全流程使用随机模拟的出海游戏投放数据（含已知真值），用于验证方法链路"
                  "与参数回收能力。接入真实投放数据时需重新设定先验并做校准。",
        "evidence": "data_generator.py / data/processed/game_ads_weekly.csv",
    },
    {
        "id": "R2", "severity": "high", "title": "基线 vs 媒体存在结构性不可识别",
        "detail": "媒体贡献的一大块是『与花费波动无关的恒定分量』（模型测算约占收入的 46.8%），"
                  "这部分与基线常数在数学上不可分离：y = (基线+α) + (媒体−α) + ε 恒成立。"
                  "后果是模型的『媒体/基线切分比例』对先验设定较敏感，"
                  "但**渠道之间的相对效率排序仍然稳健**（ROI 排序一致性 Spearman ρ = +1.000），"
                  "因此业务结论应建立在渠道排序与边际回报上，而不是绝对的媒体贡献金额。",
        "evidence": "阶段 2 可识别性分析（mmm_core.py 注释 #46.8%）/ "
                    "docs/阶段2_贝叶斯MMM建模与诊断.md",
    },
    {
        "id": "R3", "severity": "medium", "title": "渠道之间存在共线性（可识别性受限）",
        "detail": "真实投放计划中五个渠道的花费高度同步，导致 OLS/贝叶斯都难以分离各渠道独立效果。"
                  "本项目通过注入独立脉冲（渠道错峰投放）把最大渠道间相关从 0.87 降到 0.24，"
                  "显著改善可识别性。真实业务中，建议主动安排渠道错峰测试。",
        "evidence": "identifiability_experiment.csv（D0~D3 剂量实验）",
    },
    {
        "id": "R4", "severity": "medium", "title": "参数不可识别但 ROI 稳健",
        "detail": "饱和曲线的高度参数与半饱和点参数存在补偿关系，后验分布不一定回到真值，"
                  "但两者的组合（即 ROI）仍能准确恢复。结论：**参数解读要谨慎，ROI 与排序可直接用**。",
        "evidence": "stage2_recovery.csv / stage2_roi_ci.csv",
    },
    {
        "id": "R5", "severity": "medium", "title": "优化窗口的控制变量按历史值填充",
        "detail": "预算优化窗口内的控制变量（版本更新、竞品活动、节假日、趋势）按历史观测值保留，"
                  "而非预测值。若未来这些因素的实际水平与历史偏离较大，预期增量需要重新评估。",
        "evidence": "阶段 3 探针（create_zero_dataset preserve_observed=True）",
    },
    {
        "id": "R6", "severity": "low", "title": "优化窗口存在不可控的『常数底』",
        "detail": "优化窗口起点前若干周的既有投放会通过广告残留继续贡献收入，"
                  "这部分不随本次分配改变，构成一个常数底。评估增量时已扣除该底数。",
        "evidence": "stage3_summary.csv（常数底 carry-in 余温）",
    },
    {
        "id": "R7", "severity": "medium", "title": "可信区间仅反映参数不确定性",
        "detail": "优化结果的可信区间来自后验参数的不确定性，未叠加未来的市场波动与执行偏差，"
                  "因此实际落地的收益波动会大于报告区间。",
        "evidence": "evaluate_response_distribution / noise_level 设置",
    },
    {
        "id": "R8", "severity": "low", "title": "数值优化在过宽约束下可能不收敛",
        "detail": "当渠道预算上下限放得过宽时，序列二次规划（SLSQP）可能因数值条件变差而无法收敛。"
                  "工程上已加兜底：失败时返回最近可行解并显式标注收敛状态，报告不采用未收敛结果。",
        "evidence": "stage3_bounds_sweep.csv（收敛列）",
    },
    {
        "id": "R9", "severity": "high", "title": "缺少实验校准（geo-lift / pause test）",
        "detail": "营销组合模型本质上是观测数据的统计推断，无法单独证明因果。"
                  "业界做法是用地理实验或停投测试做校准。本项目未包含实验数据，"
                  "因此所有结论定位为『决策支持』而非『因果证明』。",
        "evidence": "方法论局限（业界共识）",
    },
    {
        "id": "R10", "severity": "medium", "title": "模型口径金额与财务口径不可直接等值",
        "detail": "报告中的『预期媒体贡献』『增量』来自模型标准化空间还原后的内部口径，"
                  "用于同一模型内的相对比较与排序；其绝对金额受模型缩放因子影响，"
                  "与财务报表的营销收入口径存在系统性差异，不可直接相减或对外引用。"
                  "对外沟通请使用三类安全量：预算规模（真实货币）、提升幅度（相对值）、"
                  "边际回报与 ROI（比值）。",
        "evidence": "阶段 3 探针（target_scale 与响应变量口径核对）",
    },
]


# ================================================================ 数据结构

@dataclass
class ReportFacts:
    """报告所需的全部事实。所有数字只能来自这里。"""

    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""
    meta: dict = field(default_factory=dict)
    model_quality: dict = field(default_factory=dict)
    optimization: dict = field(default_factory=dict)
    allocation: list = field(default_factory=list)
    marginal: list = field(default_factory=list)
    roi: list = field(default_factory=list)
    elasticity: list = field(default_factory=list)
    scenarios: list = field(default_factory=list)
    robustness: list = field(default_factory=list)
    identifiability: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    glossary: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)   # 缺失的输入表，供报告标注

    # ---------------- 下游接口 ----------------

    def to_prompt_payload(self) -> dict:
        """给 LLM 的载荷：去掉纯工程字段，保留全部可引用事实。"""
        d = _clean(asdict(self))
        d.pop("schema_version", None)
        return d

    def facts_hash(self) -> str:
        """事实指纹：内容变了才重新生成报告（缓存键）。"""
        raw = json.dumps(_clean(asdict(self)), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(_clean(asdict(self)), ensure_ascii=False, indent=2)

    def allowed_numbers(self) -> list:
        """
        数字白名单。包含：
          - 所有事实字段里的数值
          - 其万元 / 亿元 / 百分数换算形式（报告常用单位）
          - 文本字段里出现的数字（日期、区间等）
          - 一批通用安全值（序号、常见比例），避免序号被误判为编造
        """
        raw: list = []
        _walk_numbers(asdict(self), raw)
        out = set()
        for v in raw:
            for x in (v, -v, abs(v), v / 1e4, v / 1e8, v / 1e6, v / 100.0, v * 100.0):
                if isinstance(x, float) and np.isfinite(x):
                    out.add(round(float(x), 6))
        # 通用安全值：序号 / 常见比例 / 常见整数
        out.update(float(i) for i in range(0, 31))
        out.update({0.25, 0.5, 0.75, 1.5, 2.5, 3.5, 4.5, 10.0, 12.0, 20.0, 24.0,
                    30.0, 52.0, 100.0, 365.0, 1000.0})
        # 报告结构性常量：**置信水平本身**。
        # 报告模板固定会写「增量 94% 可信区间为 …」「94% 可信区间 [a, b]」——
        # 这个 94 是置信水平（报告格式的结构事实），不是数据值。
        # 不登记的话，校验器会把它与白名单里最近的 92.403 相比
        # （相对误差 1.73% > 1% 阈值）→ 判为「编造」→
        # **任何正确提到可信区间的报告都会被降级成模板**（静默的假阴性）。
        # 放在这里而不是 meta：meta 由磁盘/会话两个构建器各自整体赋值，
        # 写在 dataclass 默认值里会被覆盖掉（踩过）。
        out.update({94.0, 0.94, 97.0, 0.97, 0.03})
        return sorted(out)


def _walk_numbers(obj, out: list):
    """递归抽取所有数值；字符串里的数字也要抽（日期、百分比文本）。"""
    if isinstance(obj, bool):
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUMRE.findall(obj):
            v = _num(m)
            if v is not None:
                out.append(v)
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        v = float(obj)
        if np.isfinite(v):
            out.append(v)


# ================================================================ 构建

def _load_dataframe():
    """
    优先用数据生成器现算 —— 保证与阶段 2/3 使用**同一份数据**。
    注意：data/processed/*.csv 是阶段 1 的产物，可能早于数据生成器的后续修订，
    直接读盘会导致报告口径与建模口径不一致（实测总花费差 2 倍）。
    """
    try:
        from data_generator import generate_game_data
        df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
        return df, "generate_game_data(n_weeks=156, seed=2024)"
    except Exception:
        return _csv(DATA_CSV), "data/processed/game_ads_weekly.csv"


def _build_meta() -> tuple[dict, dict]:
    """元信息 + 数据派生的规模指标。"""
    meta = {
        "项目": "游戏出海广告效果分析 · 贝叶斯营销组合建模（MMM）",
        "分析方法": "贝叶斯 MMM（Adstock 广告残留 + Hill 饱和曲线，NUTS 采样）",
        "渠道": CHANNELS_CN,
        "控制变量": ["版本更新", "竞品活动", "节假日", "长期趋势"],
    }
    extra = {}
    dc, src = _load_dataframe()
    meta["数据来源"] = src
    if dc is not None:
        spend_cols = [c for c in dc.columns
                      if c.endswith("_spend") or c in
                      ("facebook", "google", "tiktok", "youtube", "kol")]
        spend_cols = [c for c in spend_cols if c not in ("revenue",)]
        tc = [c for c in dc.columns if c.startswith("_true_contrib_")]
        media_total = float(dc[tc].sum().sum()) if tc else None
        base_total = float(dc["_true_baseline"].sum()) if "_true_baseline" in dc else None
        rev_total = float(dc["revenue"].sum()) if "revenue" in dc else None
        total_spend = float(dc[spend_cols].sum().sum()) if spend_cols else None
        meta["数据周期"] = f"{str(dc['date'].iloc[0])[:10]} ~ {str(dc['date'].iloc[-1])[:10]}"
        meta["周数"] = int(len(dc))
        meta["总花费(元)"] = total_spend
        meta["总收入(元)"] = rev_total
        if media_total is not None and base_total is not None:
            _tot = media_total + base_total
            extra["真值口径·媒体贡献占比(%)"] = media_total / _tot * 100
            extra["真值口径·基线贡献占比(%)"] = base_total / _tot * 100
            extra["真值口径·全期媒体贡献(元)"] = media_total
            extra["真值口径·全期基线贡献(元)"] = base_total
        if media_total and total_spend:
            extra["真值口径·整体ROAS"] = media_total / total_spend
        # 口径提示：阶段 3 的贡献/增量金额来自模型标准化空间的内部口径，
        # 与财务口径的营销收入存在系统性差异，禁止直接等值比较。
        extra["口径提示"] = ("模型口径的贡献/增量金额仅用于同一模型内的相对比较，"
                           "不可与财务口径直接等值；对外沟通请使用提升幅度、边际回报与 ROI。")
    return meta, extra


def _build_quality() -> dict:
    """模型拟合质量（用于给报告的可信度背书）。"""
    q = {}
    S = _summary(os.path.join(REPDIR, "stage2_summary.csv"))
    ms = _csv(os.path.join(REPDIR, "stage2_summary.csv"))
    if ms is not None and len(ms):
        cols = [("param" if str(c).startswith("Unnamed") else c) for c in ms.columns]
        ms.columns = cols
        q["后验参数个数"] = int(len(ms))
        if "r_hat" in ms:
            q["最大R-hat"] = float(pd.to_numeric(ms["r_hat"], errors="coerce").max())
        if "ess_bulk" in ms:
            q["最小有效样本量ESS"] = float(pd.to_numeric(ms["ess_bulk"], errors="coerce").min())
        if "param" in ms and "mean" in ms:
            row = ms[ms["param"] == "intercept_contribution"]
            if len(row):
                q["截距贡献(标准化空间)"] = float(row["mean"].iloc[0])
    else:
        pass
    q["收敛判据"] = "R-hat < 1.01、ESS > 400、发散样本占比 < 2%"
    return q


def build_facts(repdir: str = REPDIR) -> ReportFacts:
    """从各阶段产物构建 ReportFacts（唯一数字来源）。"""
    f = ReportFacts(generated_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.glossary = GLOSSARY
    f.risks = RISKS

    meta, extra = _build_meta()
    meta.update(extra)
    f.meta = meta
    f.model_quality = _build_quality()

    # ---------- 优化汇总 ----------
    S = _summary(os.path.join(repdir, "stage3_summary.csv"))
    cur = _sv(S, "当前分配预期贡献(元)")
    optv = _sv(S, "优化分配预期贡献(元)")
    f.optimization = {
        "优化窗口": _st(S, "优化窗口"),
        "窗口内总预算(元)": _sv(S, "窗口内总预算(元)"),
        "约束": _st(S, "约束"),
        "当前分配预期贡献(元)": cur,
        "优化分配预期贡献(元)": optv,
        "预期增量(元)": _sv(S, "预期增量(元)"),
        "增量94%CI下界(元)": _sv(S, "增量94%CI下界(元)"),
        "增量94%CI上界(元)": _sv(S, "增量94%CI上界(元)"),
        "提升幅度(%)": _sv(S, "提升幅度"),
        "常数底(元)": _sv(S, "常数底(carry-in余温)"),
        "收敛状态": _st(S, "收敛状态"),
    }
    if cur:
        f.optimization["常数底占比(%)"] = (
            f.optimization["常数底(元)"] / cur * 100 if f.optimization["常数底(元)"] else None)
    if not S:
        f.missing.append("stage3_summary.csv")

    # ---------- 分渠道分配 ----------
    d = _csv(os.path.join(repdir, "stage3_allocation.csv"))
    if d is not None:
        for _, r in d.iterrows():
            f.allocation.append({
                "渠道": r["渠道"],
                "当前周均花费(元)": _num(r.get("当前周均花费(元)")),
                "当前窗口预算(元)": _num(r.get("当前窗口预算(元)")),
                "优化窗口预算(元)": _num(r.get("优化窗口预算(元)")),
                "变化(%)": _num(r.get("变化(%)")),
                "边际回报": _num(r.get("边际回报")),
                "平均ROI": _num(r.get("平均ROI")),
            })
    else:
        f.missing.append("stage3_allocation.csv")

    # ---------- 边际回报 ----------
    d = _csv(os.path.join(repdir, "stage3_marginal_roi.csv"))
    if d is not None:
        for _, r in d.iterrows():
            f.marginal.append({
                "渠道": r["渠道"],
                "当前周花费(元)": _num(r.get("当前周花费(元)")),
                "当前周贡献(元)": _num(r.get("当前周贡献(元)")),
                "边际回报": _num(r.get("边际回报")),
                "边际回报CI下": _num(r.get("边际回报_p5")),
                "边际回报CI上": _num(r.get("边际回报_p95")),
                "平均ROI": _num(r.get("平均ROI")),
            })
    else:
        f.missing.append("stage3_marginal_roi.csv")

    # ---------- 渠道 ROI（阶段 2） ----------
    d = _csv(os.path.join(repdir, "stage2_roi_ci.csv"))
    if d is not None:
        for _, r in d.iterrows():
            f.roi.append({
                "渠道": r["channel"],
                "ROI均值": _num(r.get("mean")),
                "ROI_94%CI下": _num(r.get("lo")),
                "ROI_94%CI上": _num(r.get("hi")),
                "真值ROI": _num(r.get("true")),
            })
    else:
        f.missing.append("stage2_roi_ci.csv")

    # ---------- 约束弹性 ----------
    d = _csv(os.path.join(repdir, "stage3_bounds_sweep.csv"))
    if d is not None:
        for _, r in d.iterrows():
            lo, hi = _num(r.get("弹性下限")), _num(r.get("弹性上限"))
            f.elasticity.append({
                "弹性下限": lo, "弹性上限": hi,
                "弹性宽度": _num(r.get("弹性宽度")),
                "预期贡献(元)": _num(r.get("预期贡献(元)")),
                "提升(%)": _num(r.get("提升(%)")),
                "收敛": bool(r.get("收敛")),
                "说明": "无约束（理论上界）" if (hi is None or hi == float("inf") or
                            not np.isfinite(hi)) else f"每渠道允许调整 ±{(hi - 1) * 100:.0f}%",
            })
    else:
        f.missing.append("stage3_bounds_sweep.csv")

    # ---------- 情景分析 ----------
    d = _csv(os.path.join(repdir, "stage3_scenarios.csv"))
    if d is not None:
        agg = {}
        for _, r in d.iterrows():
            k = str(r["情景"])
            a = agg.setdefault(k, {"情景": k, "预期增量(万元)": _num(r.get("预期增量(万元)")),
                                   "基线增量(万元)": _num(r.get("基线增量(万元)")),
                                   "分配": []})
            a["分配"].append({
                "渠道": r["渠道"],
                "优化预算(万元)": _num(r.get("优化预算(万元)")),
                "当前预算(万元)": _num(r.get("当前预算(万元)")),
                "变化(%)": _num(r.get("变化(%)")),
            })
        f.scenarios = list(agg.values())
    else:
        f.missing.append("stage3_scenarios.csv")

    # ---------- 稳健性 ----------
    d = _csv(os.path.join(repdir, "stage3_robustness.csv"))
    if d is not None:
        for _, r in d.iterrows():
            f.robustness.append({
                "渠道": r["渠道"],
                "点估计(元)": _num(r.get("点估计(元)")),
                "均值(元)": _num(r.get("均值(元)")),
                "中位数(元)": _num(r.get("中位数(元)")),
                "p5(元)": _num(r.get("p5(元)")),
                "p95(元)": _num(r.get("p95(元)")),
                "点估计分位(%)": _num(r.get("点估计分位")),
                "样本数": _num(r.get("样本数")),
            })
    else:
        f.missing.append("stage3_robustness.csv")

    # ---------- 可识别性剂量实验 ----------
    d = _csv(os.path.join(repdir, "identifiability_experiment.csv"))
    if d is not None:
        for _, r in d.iterrows():
            f.identifiability.append({
                "场景": r["场景"],
                "平均渠道相关系数": _num(r.get("平均|相关|")),
                "最大渠道相关系数": _num(r.get("最大|相关|")),
                "OLS判定系数R2": _num(r.get("OLS_R²")),
                "ROL排序一致性Spearman": _num(r.get("Spearman_ρ")),
            })
    else:
        f.missing.append("identifiability_experiment.csv")

    return f


# ================================================================ 自检

def main():
    f = build_facts()
    print("=" * 78)
    print("ReportFacts 自检")
    print("=" * 78)
    print(f"  facts_hash      = {f.facts_hash()}")
    print(f"  元信息          = {f.meta.get('数据周期')} / {f.meta.get('周数')} 周")
    print(f"  真值媒体占比     = {f.meta.get('真值口径·媒体贡献占比(%)')}")
    print(f"  模型质量        = {f.model_quality}")
    print(f"  优化            = 提升 {f.optimization.get('提升幅度(%)')}%  "
          f"增量 {f.optimization.get('预期增量(元)')}")
    for name, arr in [("分配", f.allocation), ("边际", f.marginal), ("ROI", f.roi),
                      ("弹性", f.elasticity), ("情景", f.scenarios),
                      ("稳健性", f.robustness), ("可识别性", f.identifiability)]:
        print(f"  {name:8s} 行数 = {len(arr)}")
    print(f"  风险条目        = {len(f.risks)}")
    print(f"  缺失输入        = {f.missing if f.missing else '无'}")
    allow = f.allowed_numbers()
    print(f"  数字白名单大小   = {len(allow)}  样例 = {allow[:8]}")
    payload = f.to_prompt_payload()
    print(f"  LLM 载荷 JSON 长度 = {len(json.dumps(payload, ensure_ascii=False))} 字符")
    print("=" * 78)


if __name__ == "__main__":
    main()
