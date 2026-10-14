"""
路径解析与产物发现（纯逻辑，不依赖 streamlit）。

为什么需要这个模块：
    Streamlit 的启动目录取决于用户在哪执行 `streamlit run`。
    如果代码里写死相对路径（"../outputs/..."），换个目录启动就会崩。
    这里统一向上查找项目根（以 src/ + requirements.txt 同时存在为标志），
    保证应用在任意工作目录下都能正确定位资源。
"""
from __future__ import annotations

import os
import sys

# ------------------------------------------------------------
# 项目根定位
# ------------------------------------------------------------
_MARKERS = ("src", "requirements.txt")


def find_project_root(start: str | None = None) -> str:
    """从 start 向上逐级查找项目根目录。"""
    cur = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(10):
        if all(os.path.exists(os.path.join(cur, m)) for m in _MARKERS):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:          # 已到盘符根
            break
        cur = parent
    # 兜底：app/core/paths.py → app/core → app → ROOT（上溯三级）
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT: str = find_project_root()
SRC: str = os.path.join(ROOT, "src")
APP: str = os.path.join(ROOT, "app")
DOCS: str = os.path.join(ROOT, "docs")
DATADIR: str = os.path.join(ROOT, "data")
OUTPUTS: str = os.path.join(ROOT, "outputs")
FIGDIR: str = os.path.join(OUTPUTS, "figures")
REPDIR: str = os.path.join(OUTPUTS, "reports")
ARCHIVE: str = os.path.join(ROOT, "archive")

# 阶段 2 后验的两条路径：精简版优先（详见 pretrained_pkl 的说明）
LITE_PKL: str = "stage2_mmm_idata_lite.pkl"
FULL_PKL: str = "stage2_mmm_idata.pkl"


def ensure_src_on_path() -> str:
    """把 src/ 加入 sys.path，使 app 层能直接 import mmm_core / response_curve 等。"""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    return SRC


# ------------------------------------------------------------
# 阶段产物清单（应用据此决定哪些功能可用）
# ------------------------------------------------------------
# key -> (相对路径, 中文名, 所属阶段)
ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "stage2_idata":  ("outputs/reports/stage2_mmm_idata.pkl", "阶段2 后验样本存档", "阶段2"),
    "stage2_lite":   ("outputs/reports/stage2_mmm_idata_lite.pkl",
                      "阶段2 后验（精简版 · 部署用）", "阶段2"),
    "stage2_roi_ci": ("outputs/reports/stage2_roi_ci.csv",    "渠道 ROI 与可信区间", "阶段2"),
    "stage2_recovery": ("outputs/reports/stage2_recovery.csv", "参数回收检验", "阶段2"),
    "stage2_contrib": ("outputs/reports/stage2_contributions.csv", "贡献分解明细", "阶段2"),
    "stage3_allocation": ("outputs/reports/stage3_allocation.csv", "最优分配方案", "阶段3"),
    "stage3_marginal":   ("outputs/reports/stage3_marginal_roi.csv", "边际回报", "阶段3"),
    "stage3_sweep":      ("outputs/reports/stage3_bounds_sweep.csv", "弹性-收益曲线", "阶段3"),
    "stage3_scenarios":  ("outputs/reports/stage3_scenarios.csv", "情景分析", "阶段3"),
    "stage3_summary":    ("outputs/reports/stage3_summary.csv", "优化汇总", "阶段3"),
    "stage4_report":     ("outputs/reports/stage4_report.md", "自动生成的决策报告", "阶段4"),
    "stage5_html":       ("outputs/reports/stage5_report.html", "单文件 HTML 综合报告", "阶段5"),
    "identifiability":   ("outputs/reports/identifiability_experiment.csv", "可识别性实验", "实验"),
}

# 图表清单：key -> (文件名, 中文标题)
FIGURES: dict[str, tuple[str, str]] = {
    "eda1": ("eda1_trend.png", "收入与花费趋势"),
    "eda2": ("eda2_decompose.png", "STL 季节分解"),
    "eda3": ("eda3_collinearity.png", "渠道共线性诊断"),
    "eda4": ("eda4_lag.png", "滞后相关性"),
    "eda5": ("eda5_saturation.png", "花费-收入饱和曲线"),
    "s2_trace": ("stage2_1_trace.png", "MCMC 轨迹图"),
    "s2_post": ("stage2_2_posteriors.png", "后验分布与真值对照"),
    "s2_contrib": ("stage2_3_contributions.png", "贡献分解"),
    "s2_roi": ("stage2_4_roi_forest.png", "ROI 森林图"),
    "s3_curve": ("stage3_1_response_curves.png", "渠道响应曲线"),
    "s3_marg": ("stage3_2_marginal_roi.png", "边际回报"),
    "s3_alloc": ("stage3_3_allocation_compare.png", "优化前后分配对比"),
    "s3_scen": ("stage3_4_scenarios.png", "预算情景分析"),
    "s3_rob": ("stage3_5_robustness.png", "稳健性检验"),
}


def pretrained_pkl() -> str | None:
    """
    返回**实际可用**的阶段 2 后验存档路径，优先精简版。

    为什么要有这个函数：全量存档 113 MB，其中 99% 是逐期贡献分解数组，
    而优化与诊断真正需要的参数后验只占约 1 MB。`src/export_lite_posterior.py`
    把逐期数组按链抽稀，产出约 15 MB 的精简版 —— 这才推得上 GitHub
    （单文件上限 100 MB）与云端托管。两条路径的消费方完全一致，
    因此这里统一成「谁在就用谁」，应用层不必关心差异。
    """
    for name in (LITE_PKL, FULL_PKL):
        p = os.path.join(REPDIR, name)
        if os.path.isfile(p):
            return p
    return None


def artifact_path(key: str) -> str:
    """产物绝对路径（不检查是否存在）。"""
    rel = ARTIFACTS[key][0]
    return os.path.join(ROOT, rel.replace("/", os.sep))


def artifact_status() -> list[dict]:
    """扫描全部阶段产物，返回存在性与体积，供界面的"环境体检"面板使用。"""
    rows = []
    for key, (rel, name, stage) in ARTIFACTS.items():
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        exists = os.path.isfile(p)
        rows.append({
            "key": key,
            "name": name,
            "stage": stage,
            "path": p,
            "exists": exists,
            "size_mb": (os.path.getsize(p) / 1e6) if exists else 0.0,
        })
    return rows


def figure_path(key: str) -> str:
    return os.path.join(FIGDIR, FIGURES[key][0])


def ensure_dirs() -> None:
    for d in (FIGDIR, REPDIR):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":       # 便于排查路径问题：python app/core/paths.py
    print(f"ROOT    = {ROOT}")
    print(f"SRC     = {SRC}  (存在: {os.path.isdir(SRC)})")
    print(f"REPDIR  = {REPDIR}")
    print("\n产物状态:")
    for r in artifact_status():
        flag = "OK " if r["exists"] else "-- "
        print(f"  {flag}{r['stage']:5s} {r['name']:22s} {r['size_mb']:8.2f} MB")
