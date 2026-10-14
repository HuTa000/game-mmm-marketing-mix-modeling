# -*- coding: utf-8 -*-
"""
会话内事实层（Report Facts · 运行时构建）
=========================================

为什么需要这一层（与 src/report_facts.py 的分工）
------------------------------------------------
`src/report_facts.py` 解决的是**离线交付**：读 outputs/reports/ 下的阶段产物 CSV，
构建一份固定的事实（对应内置模拟数据的阶段 2/3 结果）。

但应用要支持**用户上传自己的数据**：这时磁盘上根本没有对应的 CSV，
事实必须从当前会话的模型后验与优化结果**现算**。

    src/report_facts.py   ← 读盘，服务"内置数据 + 离线报告"
    app/core/facts.py     ← 现算，服务"当前会话 + 交互报告"

两者**共用同一个数据结构**（`ReportFacts`）、同一套风险声明（`RISKS`）、
同一套术语表（`GLOSSARY`），以及同一个数字白名单机制（`allowed_numbers()`）。
这样报告渲染与幻觉校验的代码只写一份。

最重要的设计约束：**风险条目的数字必须是现算的，不能写死。**
    `RISKS` 里原本写着"恒定分量约占收入的 46.8%""渠道间相关从 0.87 降到 0.24"
    —— 那是内置数据的数字。用户上传自己的数据后这些数字全不成立，
    如果照搬，报告就会**用错误的数字声明自己的局限**，比不写还糟。
    因此这里按当前数据重算每一项，并保留原有的判据框架。
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from . import paths, eda

paths.ensure_src_on_path()

import report_facts as rf                                          # noqa: E402

ReportFacts = rf.ReportFacts
CHANNELS_CN = rf.CHANNELS_CN
SEV_LABEL = {"high": "高", "medium": "中", "low": "低", "info": "说明"}


# ============================================================
# 小工具
# ============================================================
def _f(x, default=None):
    """安全转 float。"""
    if x is None:
        return default
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:                                              # noqa: BLE001
        return default


def _cn(ch: str) -> str:
    """英文渠道名 → 中文显示名（KOL / Facebook …），未知则原样返回。"""
    m = {"facebook": "Facebook", "google": "Google", "tiktok": "TikTok",
         "youtube": "YouTube", "kol": "KOL"}
    return m.get(str(ch).lower(), str(ch))


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → 干净的记录列表（NaN → None，保证 JSON 可序列化）。"""
    if df is None or not len(df):
        return []
    out = []
    for _, r in df.iterrows():
        rec = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                rec[str(k)] = int(v)
            elif isinstance(v, (np.floating, float)):
                rec[str(k)] = _f(v)
            elif isinstance(v, (np.bool_, bool)):
                rec[str(k)] = bool(v)
            else:
                rec[str(k)] = None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v
        out.append(rec)
    return out


# ============================================================
# 元信息（从当前数据现算）
# ============================================================
def _meta(ds, cfg) -> dict:
    df = ds.df
    spend_cols = [c for c in cfg.channel_cols if c in df.columns]
    total_spend = float(pd.to_numeric(df[spend_cols].stack(), errors="coerce").sum()) \
        if spend_cols else None
    rev = pd.to_numeric(df[cfg.target_col], errors="coerce")
    dt = pd.to_datetime(df[cfg.date_col], errors="coerce")
    m = {
        "项目": "游戏出海广告效果分析 · 贝叶斯营销组合建模（MMM）",
        "分析方法": "贝叶斯 MMM（Adstock 广告残留 + Hill 饱和曲线，NUTS 采样）",
        "渠道": [_cn(c) for c in cfg.channel_cols],
        "控制变量": [_cn(c) for c in cfg.control_cols],
        "数据来源": getattr(ds, "source_label", "当前会话数据"),
        "数据周期": f"{str(dt.min())[:10]} ~ {str(dt.max())[:10]}" if dt.notna().any() else "—",
        "周数": int(len(df)),
        "总花费(元)": total_spend,
        "总收入(元)": _f(rev.sum()),
        "口径提示": ("模型口径的贡献/增量金额仅用于同一模型内的相对比较，"
                  "不可与财务口径直接等值；对外沟通请使用提升幅度、边际回报与 ROI。"),
    }
    return m


def _quality(art) -> dict:
    d = art.get("diag") or {}
    q = {}
    if d.get("ok"):
        q["后验参数个数"] = int(d.get("n_params", 0))
        q["最大R-hat"] = _f(d.get("max_rhat"))
        q["最小有效样本量ESS"] = _f(d.get("min_ess"))
        q["发散样本占比(%)"] = _f(d.get("divergence_pct"))
        q["后验样本量"] = int(d.get("n_samples", 0))
        q["截距贡献(标准化空间)"] = _f(d.get("intercept"))
        q["收敛结论"] = d.get("verdict")
    q["收敛判据"] = "R-hat < 1.01、ESS > 400、发散样本占比 < 2%"
    return q


# ============================================================
# 风险声明（数字全部现算）
# ============================================================
def build_risks(ds, art, opt_result, readiness=None, max_corr=None) -> list[dict]:
    """
    按当前数据重算风险条目。

    判据框架沿用项目既定的一套（数据来源 / 可识别性 / 共线性 / 参数补偿 /
    外推假设 / 常数底 / 区间含义 / 数值收敛 / 缺少实验校准 / 口径差异），
    但**每条里的数字都换成当前会话的真实值**。
    """
    risks: list[dict] = []
    df = ds.df
    is_builtin = getattr(ds, "source", "") == "builtin"
    n_weeks = int(len(df))

    # ---- R1 数据来源 ----
    if is_builtin:
        risks.append({
            "id": "R1", "severity": "info", "title": "数据为仿真数据（真值已知）",
            "detail": "全流程使用随机模拟的出海游戏投放数据（含已知真值），"
                      "用于验证方法链路与参数回收能力。接入真实投放数据时需重新设定先验并做校准。",
            "evidence": "src/data_generator.py",
        })
    else:
        risks.append({
            "id": "R1", "severity": "info", "title": "数据来自用户上传（无真值可比）",
            "detail": f"本次分析使用上传数据（{n_weeks} 期，"
                      f"{len(ds.channel_cols)} 个渠道）。真实世界没有标准答案，"
                      "因此无法用『参数是否回收真值』来判断可信度，"
                      "只能依赖收敛诊断、贡献守恒与业务合理性三个替代判据。",
            "evidence": f"上传文件：{getattr(ds, 'source_label', '—')}",
        })

    # ---- R2 结构不可识别（数字现算） ----
    ident = art.get("ident") or {}
    if ident.get("ok"):
        share = _f(ident.get("const_share"))
        ratio = _f(ident.get("confusable_ratio"))
        if share is not None:
            sev = "high" if share > 0.35 else ("medium" if share > 0.20 else "low")
            risks.append({
                "id": "R2", "severity": sev,
                "title": "基线 vs 媒体存在结构性不可识别",
                "detail": f"媒体贡献里有 {share:.1%} 是『与花费波动无关的恒定分量』"
                          f"（混淆量约 {ratio:.2f} 倍）——这部分与基线常数在数学上不可分离："
                          f"y = (基线+α) + (媒体−α) + ε 恒成立。"
                          "后果是模型的『媒体/基线切分比例』对先验设定较敏感，"
                          "但渠道之间的**相对效率排序**仍可依赖。"
                          "业务结论应建立在渠道排序、边际回报与增量上，而不是绝对的媒体贡献金额。",
                "evidence": "app/core/model.py::identifiability（本次现算）",
            })

    # ---- R3 共线性（数字现算） ----
    if max_corr is not None and np.isfinite(max_corr):
        sev = "high" if max_corr > 0.8 else ("medium" if max_corr > 0.6 else "low")
        risks.append({
            "id": "R3", "severity": sev,
            "title": "渠道之间存在共线性（可识别性受限）",
            "detail": f"本次数据中渠道间最大 |相关系数| = {max_corr:.3f}。"
                      + ("共动程度较高，各渠道的独立效果难以干净拆分 —— "
                         "建议在实际投放中主动安排渠道错峰测试，或引入实验数据做外部锚定。"
                         if max_corr > 0.6 else
                         "共动程度可控，渠道效果基本可以分开估计。"),
            "evidence": "app/core/eda.py::collinearity（本次现算）",
        })

    # ---- R4 参数补偿 ----
    rec = art.get("recovery") or {}
    if rec.get("n_tested"):
        risks.append({
            "id": "R4", "severity": "medium",
            "title": "部分参数不可识别，但 ROI 与排序稳健",
            "detail": "饱和曲线的高度参数与半饱和点参数存在补偿关系，"
                      "后验分布不一定回到各自的真值；但两者的组合（即 ROI）仍能恢复，"
                      f"本次参数区间命中真值 {rec.get('n_hit')}/{rec.get('n_tested')} 项。"
                      "结论：**参数单独解读要谨慎，ROI 与渠道排序可直接用**。",
            "evidence": "app/core/model.py::recovery_table",
        })
    else:
        risks.append({
            "id": "R4", "severity": "medium",
            "title": "参数不可识别性未能用真值校验",
            "detail": "饱和曲线的两个参数存在补偿关系（同时放大/缩小可给出几乎相同的响应），"
                      "因此**单个参数的后验分布不应被直接解读**；"
                      "可靠的是两者组合出来的 ROI 与渠道排序。",
            "evidence": "方法论局限",
        })

    # ---- R5 外推与控制变量 ----
    risks.append({
        "id": "R5", "severity": "medium", "title": "优化窗口的控制变量按历史值填充",
        "detail": "预算优化窗口内的控制变量（版本更新、竞品活动、节假日、长期趋势）"
                  "按历史观测值保留，而非未来预测值。若未来这些因素的实际水平与历史"
                  "偏离较大，预期增量需要重新评估。",
        "evidence": "app/core/optimize.py::build_optimizer",
    })

    # ---- R6 常数底 ----
    main = (opt_result or {}).get("main") or {}
    floor_pct = _f(main.get("constant_floor_pct"))
    if floor_pct is not None:
        risks.append({
            "id": "R6", "severity": "low", "title": "优化窗口存在不可控的『常数底』",
            "detail": f"优化窗口起点前既有投放会通过广告残留继续贡献收入，"
                      f"本次测算该底约占当前贡献的 {floor_pct:.1f}%。"
                      "这部分不随本次分配改变，评估增量时已扣除。"
                      "因此应关注『优化 vs 维持现状的差值』，而非绝对总量。",
            "evidence": "app/core/optimize.py::optimize_budget（zero_utility）",
        })

    # ---- R7 区间含义 ----
    risks.append({
        "id": "R7", "severity": "medium", "title": "可信区间只反映参数不确定性",
        "detail": "优化结果的可信区间来自后验参数的不确定性，"
                  "**未叠加未来市场波动与执行偏差**，因此实际落地的收益波动会大于报告区间。",
        "evidence": "evaluate_response_distribution",
    })

    # ---- R8 数值收敛 ----
    tab = ((opt_result or {}).get("sweep") or {}).get("table")
    n_nc = 0
    if tab is not None and len(tab) and "收敛" in tab.columns:
        n_nc = int((~tab["收敛"].astype(bool)).sum())
    risks.append({
        "id": "R8", "severity": "medium" if n_nc else "low",
        "title": "数值优化在过宽约束下可能不收敛",
        "detail": ("当渠道预算上下限放得过宽时，序列二次规划（SLSQP）可能因数值条件"
                   "变差而无法收敛。已加兜底：失败时返回最近可行解并显式标注收敛状态，"
                   "报告不采用未收敛结果。"
                   + (f"本次弹性扫描中有 {n_nc} 档未收敛。" if n_nc else
                      "本次弹性扫描全部收敛。")),
        "evidence": "app/core/optimize.py::optimize_budget",
    })

    # ---- R9 缺少实验校准 ----
    risks.append({
        "id": "R9", "severity": "high", "title": "缺少实验校准（geo-lift / pause test）",
        "detail": "营销组合模型本质上是观测数据的统计推断，无法单独证明因果。"
                  "业界做法是用地理实验或停投测试做校准。本次分析未包含实验数据，"
                  "因此所有结论定位为『决策支持』而非『因果证明』。",
        "evidence": "方法论局限（业界共识）",
    })

    # ---- R10 口径差异 ----
    risks.append({
        "id": "R10", "severity": "medium",
        "title": "模型口径金额与财务口径不可直接等值",
        "detail": "报告中的『预期媒体贡献』『增量』来自模型缩放空间还原后的内部口径，"
                  "用于同一模型内的相对比较与排序；其绝对金额受缩放因子影响，"
                  "与财务报表的营销收入口径存在系统性差异，不可直接相减或对外引用。"
                  "对外沟通请使用三类安全量：预算规模（真实货币）、提升幅度（相对值）、"
                  "边际回报与 ROI（比值）。",
        "evidence": "app/core/model.py::get_target_scale",
    })
    return risks


# ============================================================
# 主构建
# ============================================================
def build_facts(art: dict, opt_result: dict, ds, *, include_readiness: bool = True) -> ReportFacts:
    """
    从**当前会话**的模型结果（art）与优化结果（opt_result）构建事实层。

    参数
    ----
      art        : app/core/model.py 的返回值（含 diag / roi / ident / recovery / mix）
      opt_result : app/core/optimize.py::analyze 的返回值
      ds         : data_io.Dataset

    返回 ReportFacts（与 src/report_facts.py 同构，可直接喂给
    report_templates.render_fallback 与 report_guard.check）。
    """
    cfg = art.get("cfg")
    if cfg is None:
        raise ValueError("art 里没有模型配置（cfg），无法构建事实层")

    f = ReportFacts(generated_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.glossary = dict(rf.GLOSSARY)
    f.meta = _meta(ds, cfg)
    f.model_quality = _quality(art)

    # ---------- 优化汇总 ----------
    main = (opt_result or {}).get("main") or {}
    dist = (opt_result or {}).get("dist") or {}
    if main.get("ok"):
        f.optimization = {
            "优化窗口": (f"{(opt_result.get('window') or {}).get('start', '—')} ~ "
                     f"{(opt_result.get('window') or {}).get('end', '—')} "
                     f"({(opt_result.get('opt_meta') or {}).get('num_periods', '—')} 期)"),
            "窗口内总预算(元)": _f(main.get("total")),
            "约束": f"单渠道允许调整 ±{_f(opt_result.get('elasticity'), 0) * 100:.0f}%",
            "当前分配预期贡献(元)": _f(main.get("hist_utility")),
            "优化分配预期贡献(元)": _f(main.get("utility")),
            "预期增量(元)": _f(main.get("utility") - main.get("hist_utility") if main.get("utility") is not None else None),
            "增量94%CI下界(元)": _f(dist.get("net_lo")),
            "增量94%CI上界(元)": _f(dist.get("net_hi")),
            "提升幅度(%)": _f(main.get("lift_pct")),
            "常数底(元)": _f(main.get("zero_utility")),
            "常数底占比(%)": _f(main.get("constant_floor_pct")),
            "收敛状态": f"success={main.get('success')}, nit={main.get('nit')}",
            "卡角点渠道数": int(main.get("corner_hit", 0)),
        }
    else:
        f.missing.append("预算优化结果（未能求解）")

    # ---------- 分渠道分配 ----------
    alloc = (opt_result or {}).get("alloc")
    if alloc is not None and len(alloc):
        a = alloc.rename(columns={"边际回报": "边际回报", "平均ROI": "平均ROI"})
        f.allocation = [
            {**r, "渠道": _cn(r.get("渠道"))}
            for r in _records(a)
        ]
    else:
        f.missing.append("渠道分配表")

    # ---------- 边际回报 ----------
    marg = (opt_result or {}).get("marginal")
    if marg is not None and len(marg):
        f.marginal = [
            {**r, "渠道": _cn(r.get("渠道"))}
            for r in _records(marg)
        ]
    else:
        f.missing.append("边际回报表")

    # ---------- 渠道 ROI ----------
    roi = art.get("roi")
    if roi is not None and len(roi):
        for _, r in roi.iterrows():
            f.roi.append({
                "渠道": _cn(r.get("渠道", r.get("channel", "?"))),
                "ROI均值": _f(r.get("ROI均值")),
                "ROI_94%CI下": _f(r.get("ROI下界")),
                "ROI_94%CI上": _f(r.get("ROI上界")),
                "P(ROI>1)": _f(r.get("P(ROI>1)")),
                "结论": r.get("结论") if isinstance(r.get("结论"), str) else None,
            })
    else:
        f.missing.append("渠道 ROI 表")

    # ---------- 弹性-收益（核心交付物） ----------
    sweep = (opt_result or {}).get("sweep") or {}
    tab = sweep.get("table")
    if tab is not None and len(tab):
        for _, r in tab.iterrows():
            e = _f(r.get("弹性(%)"))
            hi = _f(r.get("弹性上限"))
            f.elasticity.append({
                "弹性下限": _f(r.get("弹性下限")),
                "弹性上限": hi,
                "弹性宽度": (None if hi is None or not np.isfinite(hi) or e is None or not np.isfinite(e)
                          else round(2 * e / 100.0, 4)),
                "弹性档位(%)": e if (e is not None and np.isfinite(e)) else None,
                "预期贡献(元)": _f(r.get("预期贡献(元)")),
                "提升(%)": _f(r.get("提升(%)")),
                "收敛": bool(r.get("收敛")) if r.get("收敛") is not None else None,
                "角点命中": int(r.get("角点命中")) if pd.notna(r.get("角点命中")) else None,
                "说明": (r.get("说明") if isinstance(r.get("说明"), str) else "—"),
            })
    else:
        f.missing.append("弹性-收益曲线")

    # ---------- 情景分析 ----------
    scen = (opt_result or {}).get("scenarios") or {}
    summ, detail = scen.get("summary"), scen.get("detail")
    if summ is not None and len(summ):
        for _, s in summ.iterrows():
            rows = []
            if detail is not None and len(detail):
                sub = detail[detail["情景"] == s["情景"]]
                for _, r in sub.iterrows():
                    rows.append({"渠道": _cn(r.get("渠道")),
                                 "优化预算(万元)": (lambda v: None if v is None else v / 1e4)(_f(r.get("优化预算(元)"))),
                                 "当前预算(万元)": (lambda v: None if v is None else v / 1e4)(_f(r.get("当前预算(元)"))),
                                 "变化(%)": _f(r.get("变化(%)"))})
            f.scenarios.append({
                "情景": s.get("情景"), "预算倍数": _f(s.get("预算倍数")),
                "总预算(元)": _f(s.get("总预算(元)")),
                "优化贡献(元)": _f(s.get("优化贡献(元)")),
                "基线贡献(元)": _f(s.get("基线贡献(元)")),
                "提升(%)": _f(s.get("提升(%)")),
                "收敛": bool(s.get("收敛")) if s.get("收敛") is not None else None,
                "分配": rows,
            })
    else:
        f.missing.append("情景分析")

    # ---------- 稳健性 ----------
    rob = (opt_result or {}).get("robustness") or {}
    rtab = rob.get("table")
    if rtab is not None and len(rtab):
        f.robustness = [
            {**r, "渠道": _cn(r.get("渠道"))}
            for r in _records(rtab)
        ]

    # ---------- 可识别性剂量实验（仅内置数据有） ----------
    if getattr(ds, "source", "") == "builtin":
        d = rf._csv(f"{rf.REPDIR}/identifiability_experiment.csv")     # noqa: SLF001
        if d is not None:
            for _, r in d.iterrows():
                f.identifiability.append({
                    "场景": r.get("场景"),
                    "平均渠道相关系数": _f(r.get("平均|相关|")),
                    "最大渠道相关系数": _f(r.get("最大|相关|")),
                    "OLS判定系数R2": _f(r.get("OLS_R²")),
                    "ROL排序一致性Spearman": _f(r.get("Spearman_ρ")),
                })

    # ---------- 风险（数字现算） ----------
    max_corr = None
    try:
        if include_readiness:
            # 注意签名：collinearity(df, channel_cols) —— 只吃渠道列
            coll = eda.collinearity(ds.df, ds.channel_cols)
            if coll.get("ok"):
                max_corr = _f(coll.get("max_corr"))
    except Exception:                                              # noqa: BLE001
        pass
    f.risks = build_risks(ds, art, opt_result, max_corr=max_corr)
    return f


# ============================================================
# 与磁盘事实层的选择器
# ============================================================
def get_facts(art=None, opt_result=None, ds=None) -> tuple[ReportFacts, str]:
    """
    决定用哪一份事实：
      · 会话里已有模型 + 优化结果 → **现算**（上传数据唯一可行的路径；
        内置数据也优先，保证"页面上看到什么，报告里就说什么"）
      · 否则回退到磁盘产物事实层（离线场景 / 页面被直接打开）

    返回 (facts, source)，source ∈ {"session", "disk"}。
    """
    if art and art.get("ok") and opt_result and opt_result.get("ok") and ds is not None:
        f = build_facts(art, opt_result, ds)
        f.meta["事实来源"] = "当前会话（模型与优化结果现算）"
        return f, "session"
    return rf.build_facts(), "disk"
