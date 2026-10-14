"""
概览与数据（阶段 6 交付的核心页面）

本页职责：
    1. 作为整个应用的入口，说明项目在做什么、怎么用；
    2. 提供**两条数据路径**：内置模拟数据 / 上传自己的数据；
    3. 把数据校验的全部结论（error/warning/info）如实呈现给用户；
    4. 载入后展示规模 KPI、数据预览、数据字典，为后续页面备好输入。

设计上刻意把"校验问题清单"放在最显眼处：上传失败时用户最需要的不是
"出错了"，而是"哪一列有问题、为什么、怎么改"。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from core import cache, paths, schema, state, ui       # noqa: E402

paths.ensure_src_on_path()

ui.inject()
ui.page_header(
    "游戏出海广告效果分析 · 贝叶斯营销组合建模（MMM）",
    "用贝叶斯 MMM 回答两个问题：<b>过去的钱花得值不值</b>（渠道贡献与 ROI），"
    "<b>未来的钱该怎么花</b>（约束下的预算最优分配）。"
    "建模流程：Adstock 广告残留 → Hill 饱和曲线 → NUTS 采样 → 贡献分解 → 预算优化 → 自动报告。",
    "🎯",
)

ui.steps(["载入数据", "探索性分析", "建模与诊断", "预算优化", "决策报告"], current=0)

st.divider()

# ============================================================
# 辅助函数
# ============================================================
def _template_csv() -> str:
    """生成一份格式正确的模板 CSV，让用户有样可循。"""
    try:
        from data_generator import generate_game_data
        df = generate_game_data(n_weeks=8, seed=7, with_controls=True).head(8)
        keep = [c for c in df.columns if not c.startswith("_")]
        keep = [c for c in keep if "impression" not in c and "click" not in c
                and c not in ("region", "new_users")]
        return df[keep].to_csv(index=False)
    except Exception:
        base = ["date"] + [f"{c}_spend" for c in schema.CHANNELS] + \
               ["revenue"] + list(schema.CONTROLS)
        return ",".join(base) + "\n" + "\n".join(
            [",".join([f"2023-01-{i*7+1:02d}"] + ["10000"] * (len(base) - 1))
             for i in range(8)])


def _quality_table(ds) -> pd.DataFrame:
    """逐渠道的数据质量体检表。"""
    rows = []
    n = max(1, ds.n_rows)
    for c in ds.channel_cols:
        s = pd.to_numeric(ds.df[c], errors="coerce").fillna(0.0)
        mean, std = float(s.mean()), float(s.std(ddof=0))
        rows.append({
            "渠道": schema.CHANNEL_CN.get(c, c),
            "周均花费(万元)": round(mean / 1e4, 2),
            "最大单周(万元)": round(float(s.max()) / 1e4, 2),
            "零投放周占比": f"{float((s == 0).mean()) * 100:.1f}%",
            "波动系数(CV)": round(std / mean, 3) if mean > 0 else np.nan,
        })
    t = pd.to_numeric(ds.df[ds.target_col], errors="coerce")
    rows.append({
        "渠道": "【收入】",
        "周均花费(万元)": round(float(t.mean()) / 1e4, 2),
        "最大单周(万元)": round(float(t.max()) / 1e4, 2),
        "零投放周占比": f"{float((t <= 0).mean()) * 100:.1f}%",
        "波动系数(CV)": round(float(t.std(ddof=0) / t.mean()), 3),
    })
    return pd.DataFrame(rows)


def _dict_md(ds) -> str:
    lines = ["**标准字段**（下载模板即按此格式）：", ""]
    lines.append("| 字段 | 含义 | 是否必需 |")
    lines.append("|---|---|---|")
    lines.append(f"| `{ds.date_col}` | 周日期（YYYY-MM-DD 或 YYYY/M/D） | ✅ 必需 |")
    lines.append(f"| `{ds.target_col}` | 该周收入 / 流水（用于建模的目标） | ✅ 必需 |")
    for c in ds.channel_cols:
        lines.append(f"| `{c}` | {schema.CHANNEL_CN.get(c, c)} 当周广告花费 | ✅ 必需（至少 2 个）|")
    if ds.control_cols:
        for c in ds.control_cols:
            lines.append(f"| `{c}` | {schema.CONTROL_CN.get(c, c)}（用于控制混杂因素） | 可选 |")
    else:
        lines.append("| `holiday` / `version_update` / `competitor_activity` / `trend` "
                     "| 控制变量 | 可选（建议提供）|")
    lines += [
        "",
        "**列名容错**：渠道列可写成 `facebook_spend`、`Facebook Spend`、`FB花费`、`KOL 投放 (元)`；"
        "日期列可写成 `date` / `日期` / `week`；收入列可写成 `revenue` / `收入` / `流水` / `GMV`。",
        "",
        "**金额单位**：请用「元」，支持带千分位逗号与货币符号（如 `1,234,500` 或 `¥1234500`）。",
        "",
        "**控制变量为什么重要**：版本更新、节假日、竞品动作都会同时影响收入和投放节奏。"
        "不控制它们，模型可能把「因为大促才加投」误读成「加投带来了收入」，"
        "这就是内生性（endogeneity）——MMM 最需要警惕的偏差来源。",
    ]
    return "\n".join(lines)


# ============================================================
# 一、选择数据源
# ============================================================
st.subheader("第 1 步 · 选择数据源")

tab_builtin, tab_upload = st.tabs(["🎲 内置模拟数据（推荐先试这个）", "📤 上传我的数据"])

with tab_builtin:
    st.markdown(
        "内置数据由 `src/data_generator.py` 生成，**与阶段 2/3 的建模结果同源**，"
        "并且带有真值标注（真实 β、衰减率、各渠道真实贡献）——"
        "这意味着模型估得准不准可以被客观检验，不是自说自话。")
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        n_weeks = st.slider("周数", 52, 208, 156, 26,
                            help="156 周 = 3 年，能覆盖 3 个完整年度周期。少于 104 周季节项不稳定。")
    with c2:
        seed = st.number_input("随机种子", 1, 99999, 2024, 1,
                               help="固定种子 = 数据可复现。换种子可检验结论的稳健性。")
    with c3:
        with_ctl = st.checkbox("包含控制变量", value=True,
                               help="版本更新 / 竞品活动 / 节假日 / 长期趋势。"
                                    "关掉可以直观看到'遗漏变量'的后果。")
    if st.button("载入内置数据", type="primary", **ui.FULL_WIDTH):
        with st.spinner("正在生成并校验数据…"):
            ds = cache.cached_builtin_dataset(int(n_weeks), int(seed), bool(with_ctl))
        state.set_dataset(ds, source="builtin")
        st.rerun()

with tab_upload:
    st.markdown(
        "支持 **CSV / Excel**。列名不必严格一致——"
        "`facebook_spend`、`Facebook Spend`、`FB花费` 都能被识别；"
        "日期列可叫 `date` / `日期` / `week`，收入列可叫 `revenue` / `收入` / `流水`。")
    up = st.file_uploader("上传周度数据文件", type=["csv", "txt", "xlsx", "xls"],
                          accept_multiple_files=False)

    # ---- 模板下载：降低"不知道要交什么格式"的门槛 ----
    with st.expander("没有数据？先下载一个模板看看格式", expanded=False):
        tpl = _template_csv()
        st.download_button("⬇️ 下载数据模板（CSV）", data=tpl.encode("utf-8-sig"),
                           file_name="mmm_data_template.csv", mime="text/csv",
                           **ui.FULL_WIDTH)
        st.caption("模板含 5 个渠道花费、收入、4 个控制变量，共 8 周示例。"
                   "把内容替换成你的真实数据即可。")

    if up is not None:
        payload = up.getvalue()
        with st.spinner("正在解析并校验上传文件…"):
            ds = cache.cached_uploaded(payload, up.name)
        rep = ds.report

        st.markdown("#### 校验结果")
        ui.issues_panel(rep)

        if rep.ok and ds.n_rows > 0:
            st.success(f"✅ 数据可用：{ds.n_rows} 行 · {len(ds.channel_cols)} 个渠道 · "
                       f"{len(ds.control_cols)} 个控制变量。已载入，可继续下一步。")
            state.set_dataset(ds, source="upload")
        else:
            st.error("⛔ 这份数据暂时无法使用，请按上面的提示修正后重新上传。"
                     "（应用不会猜测或补造缺失的列，以免给你错误结论。）")

st.divider()

# ============================================================
# 二、当前数据概览
# ============================================================
ds = state.get_dataset()
if not state.has_dataset():
    ui.empty_state("还没有载入数据",
                   "在上方选择「内置模拟数据」并点击载入，或上传你自己的 CSV。"
                   "载入后这里会显示规模指标、数据预览与数据字典。",
                   "📂")
    st.stop()

st.subheader("第 2 步 · 数据概览")
sm = ds.summary()
ui.kpi_row([
    {"label": "观测周期", "value": f"{sm['行数(周)']} 周", "delta": sm["时间范围"],
     "tone": "brand"},
    {"label": "总花费", "value": f"{sm['总花费(万元)']:,.0f} 万元",
     "delta": f"周均 {sm['总花费(万元)']/max(1,sm['行数(周)']):,.1f} 万元"},
    {"label": "总收入", "value": f"{sm['总收入(万元)']:,.0f} 万元",
     "delta": f"周均 {sm['总收入(万元)']/max(1,sm['行数(周)']):,.1f} 万元"},
    {"label": "整体 ROAS", "value": f"{sm['整体 ROAS']:.2f}",
     "delta": "收入 ÷ 总花费（含基线，非媒体 ROI）"},
    {"label": "渠道 / 控制变量", "value": f"{sm['渠道数']} / {sm['控制变量数']}",
     "delta": f"粒度约 {sm['数据粒度']}"},
], per_row=5)

st.caption(f"数据来源：{ds.source_label}　|　数据指纹 `{ds.fingerprint}`"
           "（指纹变化会自动清空模型/优化/报告，避免用旧结论配新数据）")

tab_prev, tab_qual, tab_dict = st.tabs(["数据预览", "质量速览", "数据字典"])

with tab_prev:
    ui.dataframe(ds.to_preview(12),
                 caption=f"仅显示前 12 行（共 {ds.n_rows} 行）。列名已统一为标准字段。")

with tab_qual:
    ui.dataframe(_quality_table(ds),
                 caption="零值占比高说明该渠道投放稀疏，会加大其 Adstock / 饱和参数的估计难度；"
                         "波动系数（标准差/均值）过低则缺少可识别性。")

with tab_dict:
    st.markdown(_dict_md(ds))

# ---- 后续步骤引导 ----
st.divider()
st.subheader("第 3 步 · 继续往下")
ready_model = cache.pretrained_available()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**📊 探索性分析**\n\n趋势、季节性、渠道共线性、滞后效应——"
                "先判断数据支不支持做归因。")
with c2:
    st.markdown(f"**🔬 建模与诊断**\n\n"
                f"{'已检测到阶段 2 预训练后验，可直接复用（秒级）。' if ready_model else '应用内拟合贝叶斯 MMM（约 10~20 分钟）。'}")
with c3:
    st.markdown("**💰 预算优化 / 📝 决策报告**\n\n约束下的最优分配、"
                "弹性-收益曲线、自动生成业务报告。")
st.caption("提示：左侧边栏的「阶段产物体检」可以看到哪些阶段能力已经就绪。")

