"""
阶段 2 主脚本：贝叶斯 MMM 完整建模与诊断
=========================================
输出：
  1. 完整拟合诊断（R-hat / ESS / divergences / trace）
  2. 后验参数分布图（含真值对照）
  3. 各渠道贡献分解（堆叠图）
  4. 各渠道 ROI + 94% 可信区间（森林图）
  5. 参数回收检验
  6. 保存模型（供阶段 3 预算优化复用）

运行：python src/stage2_model.py            # 完整采样
      python src/stage2_model.py --quick    # 快速模式
"""
from __future__ import annotations

import os
import sys
import warnings
import argparse

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from data_generator import (generate_game_data, CHANNELS, TRUE_BETA, TRUE_DECAY,
                            TRUE_KAPPA, TRUE_GAMMA_FRAC, get_true_contribution)
from mmm_core import (MMMConfig, normalize_columns, prepare_data, build_mmm_safe,
                      fit_mmm, diagnose, check_model_sanity, compute_contributions,
                      compute_roi, check_recovery, get_target_scale,
                      get_spend_series)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "outputs", "figures")
REPDIR = os.path.join(ROOT, "outputs", "reports")
MODELDIR = os.path.join(ROOT, "outputs", "reports")   # 与 CSV 同目录，避开沙箱目录限制
for d in (FIGDIR, REPDIR, MODELDIR):
    os.makedirs(d, exist_ok=True)


def plot_trace(idata, path):
    """诊断图 1：trace（判断收敛）"""
    import arviz as az
    var = [v for v in ["intercept_contribution", "saturation_beta", "adstock_alpha",
                       "saturation_kappa", "saturation_slope", "y_sigma"]
           if v in idata.posterior.data_vars]
    try:
        # 兼容 arviz_plots 新 API
        pc = az.plot_trace(idata, var_names=var)
        plt.gcf().set_size_inches(14, 12)
    except Exception:
        # 退回老 API
        az.plot_trace(idata, var_names=var, compact=True, figsize=(14, 12))
    plt.suptitle("Trace 图：4 条链是否充分混合（像'毛毛虫'= 好）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=105, bbox_inches="tight")
    plt.close()


def plot_posteriors(idata, path):
    """诊断图 2：关键参数后验 + 真值对照"""
    pt = idata.posterior
    fig, axes = plt.subplots(4, 5, figsize=(20, 13))
    specs = [("saturation_beta", TRUE_BETA, "beta (ROI尺度)"),
             ("adstock_alpha", TRUE_DECAY, "decay (衰减率)"),
             ("saturation_kappa", None, "kappa (半饱和点)"),
             ("saturation_slope", TRUE_KAPPA, "slope (陡峭度)")]
    for r, (pname, truth, label) in enumerate(specs):
        for i, c in enumerate(CHANNELS):
            ax = axes[r, i]
            if pname not in pt:
                continue
            arr = pt[pname].values[:, :, i].ravel()
            ax.hist(arr, bins=40, color="#2E86AB", alpha=.75, density=True)
            lo, hi = np.percentile(arr, [3, 97])
            ax.axvspan(lo, hi, color="#2E86AB", alpha=.15)
            if truth is not None:
                ax.axvline(truth[c], color="#C0392B", ls="--", lw=2)
            ax.set_title(f"{label}\n{c}", fontsize=9)
    plt.suptitle("后验分布（红线 = 真实值；蓝带 = 94% 可信区间）", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=105, bbox_inches="tight")
    plt.close()


def plot_contributions(contrib, y, df, path):
    """诊断图 3：贡献分解堆叠图"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), height_ratios=[2, 1])
    dates = pd.to_datetime(contrib["date"])
    ch_cols = [c for c in CHANNELS if c in contrib.columns]

    # 上：堆叠面积
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B8EA5"]
    ax = axes[0]
    ax.stackplot(dates, *[contrib[c].values for c in ch_cols],
                 labels=ch_cols, colors=colors[:len(ch_cols)], alpha=.85)
    if "_baseline" in contrib.columns:
        ax.stackplot(dates, contrib["_baseline"].values, labels=["基线"],
                     colors=["#95A5A6"], alpha=.6)
    ax.plot(dates, y.values, color="black", lw=1.5, label="实际收入")
    ax.set_ylabel("收入（元）")
    ax.set_title("收入贡献分解（堆叠图）：各渠道 + 基线，合计 = 预测收入")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.grid(alpha=.25)

    # 下：占比随时间变化
    ax = axes[1]
    total = contrib[ch_cols].sum(axis=1).replace(0, np.nan)
    for i, c in enumerate(ch_cols):
        ax.plot(dates, contrib[c] / total, lw=1.3, label=c, color=colors[i])
    ax.set_ylabel("占媒体总贡献比例")
    ax.set_xlabel("日期")
    ax.set_title("各渠道贡献占比随时间变化（看是否有结构性漂移）")
    ax.legend(loc="upper left", ncol=5, fontsize=9)
    ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(path, dpi=105, bbox_inches="tight")
    plt.close()


def plot_roi_forest(idata, contrib, df, path, target_scale=1.0):
    """诊断图 4：ROI 森林图（含 94% 可信区间）"""
    pt = idata.posterior
    fig, ax = plt.subplots(figsize=(11, 6))

    # 用后验逐样本算 ROI 区间
    # 【注意】channel_contribution 是"标准化空间"的量，
    #   还原到原始尺度需 × target_scale（见 get_target_scale 的说明）
    rows = []
    for i, c in enumerate(CHANNELS):
        ch_contrib = pt["channel_contribution"].values[:, :, :, i]  # (chain,draw,date)
        sx = ch_contrib.sum(axis=2).ravel() * target_scale           # (sample,) 元
        spend = get_spend_series(df, c).sum()
        roi_s = sx / spend
        rows.append({
            "channel": c, "mean": float(np.mean(roi_s)),
            "lo": float(np.percentile(roi_s, 3)),
            "hi": float(np.percentile(roi_s, 97)),
            "true": TRUE_BETA[c],
        })
    rd = pd.DataFrame(rows).sort_values("mean")

    ypos = np.arange(len(rd))
    ax.hlines(ypos, rd["lo"], rd["hi"], color="#2E86AB", lw=3, alpha=.7)
    ax.plot(rd["mean"], ypos, "o", color="#1B4F72", ms=9, label="估计 ROI（均值）")
    ax.plot(rd["true"], ypos, "D", color="#C0392B", ms=8, label="真实 ROI")
    ax.set_yticks(ypos)
    ax.set_yticklabels(rd["channel"])
    ax.axvline(1.0, color="gray", ls=":", lw=1.5, label="盈亏平衡 ROI=1")
    ax.set_xlabel("ROI（每 1 元花费带来的收入）")
    ax.set_title("各渠道 ROI 森林图（横线 = 94% 可信区间）")
    ax.legend()
    ax.grid(alpha=.25, axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=105, bbox_inches="tight")
    plt.close()
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速模式（少采样）")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("阶段 2：贝叶斯 MMM 完整建模与诊断")
    print("=" * 74)

    # ---------- 1. 数据 ----------
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=args.quick)
    if args.quick:
        cfg.apply_quick_mode()
    df = normalize_columns(df, cfg)

    X, y, meta, used_ctrl = prepare_data(df, cfg)
    print(f"\n[1] 数据就绪")
    print(f"    样本: {X.shape[0]} 周 × {len(cfg.channel_cols)} 渠道")
    print(f"    控制变量: {used_ctrl}")
    print(f"    目标 y: 均值 {y.mean():,.0f}  最大 {y.max():,.0f}")

    # ---------- 2. 建模 ----------
    print(f"\n[2] 构建模型")
    mmm, note = build_mmm_safe(cfg, control_cols=used_ctrl)
    print(f"    {note}")
    print(f"    缩放策略: target = {cfg.target_scaling}")
    print(f"    Adstock: GeometricAdstock(l_max={cfg.adstock_l_max})")
    print(f"    饱和: HillSaturation")
    print(f"    年度季节性: {cfg.yearly_seasonality} 阶 Fourier")

    # ---------- 3. 采样 ----------
    print(f"\n[3] NUTS 采样 (numpyro 后端)")
    print(f"    draws={cfg.draws} tune={cfg.tune} chains={cfg.chains} "
          f"target_accept={cfg.target_accept}")
    idata, elapsed = fit_mmm(mmm, X, y, cfg, progressbar=False)
    print(f"    完成，耗时 {elapsed:.1f}s   target_scale = {get_target_scale(mmm):,.0f}")

    # ---------- 4. 诊断 ----------
    print(f"\n[4] 拟合诊断")
    diag = diagnose(idata, cfg, quiet=False)

    print(f"\n[5] 模型自洽性检查")
    sanity = check_model_sanity(mmm, y)

    # ---------- 6. 贡献分解 ----------
    print(f"\n[6] 贡献分解（官方守恒分解）")
    contrib, cinfo = compute_contributions(mmm, df, cfg, meta, method="library")
    print(f"    {cinfo['note']}")
    print(f"    媒体贡献合计 = {cinfo['media_sum']/1e4:,.0f} 万元")
    print(f"    历史总收入   = {cinfo['true_y_sum']/1e4:,.0f} 万元")
    print(f"    媒体占比     = {cinfo['media_share']:.1%}   "
          f"(真实设计值 53.9%)")

    # ---------- 7. ROI ----------
    print(f"\n[7] 各渠道 ROI")
    roi = compute_roi(contrib, df, cfg)
    print(roi.round(3).to_string(index=False))

    # ---------- 8. 参数回收 ----------
    print(f"\n[8] 参数回收检验（估计 vs 真实）")
    rec = check_recovery(diag, TRUE_BETA, TRUE_DECAY, cfg)
    print(rec.to_string(index=False))

    # ---------- 9. 图示 ----------
    roi_ci = None
    if not args.no_plot:
        print(f"\n[9] 生成诊断图")
        plot_trace(idata, os.path.join(FIGDIR, "stage2_1_trace.png"))
        print(f"    ✓ stage2_1_trace.png")
        plot_posteriors(idata, os.path.join(FIGDIR, "stage2_2_posteriors.png"))
        print(f"    ✓ stage2_2_posteriors.png")
        plot_contributions(contrib, y, df, os.path.join(FIGDIR, "stage2_3_contributions.png"))
        print(f"    ✓ stage2_3_contributions.png")
        roi_ci = plot_roi_forest(idata, contrib, df,
                                 os.path.join(FIGDIR, "stage2_4_roi_forest.png"),
                                 target_scale=get_target_scale(mmm))
        print(f"    ✓ stage2_4_roi_forest.png")
        print("\n    ROI 及 94% 可信区间：")
        print(roi_ci.round(3).to_string(index=False))

    # ---------- 10. 保存 ----------
    print(f"\n[10] 保存结果")
    # 保存模型（阶段 3 预算优化会复用）
    # 【踩坑】Windows 沙箱下 outputs/models 目录写入受限，改用 h5netcdf 引擎
    import pickle
    saved = False
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            kw = {"engine": eng} if eng else {}
            mmm.save(os.path.join(MODELDIR, "stage2_mmm.nc"), **kw)
            print(f"    ✓ 模型已保存: outputs/reports/stage2_mmm.nc (engine={eng})")
            saved = True
            break
        except Exception:
            continue
    if not saved:
        # 兜底：pickle 保存 idata + 关键参数
        try:
            with open(os.path.join(MODELDIR, "stage2_mmm_idata.pkl"), "wb") as f:
                pickle.dump({"idata": idata, "config": cfg,
                             "channel_cols": cfg.channel_cols,
                             "control_cols": used_ctrl,
                             "y_mean": float(y.mean()), "y_max": float(y.max())}, f)
            print(f"    ✓ 模型已保存(兜底 pickle): outputs/reports/stage2_mmm_idata.pkl")
        except Exception as e:
            print(f"    [提示] 模型保存失败({type(e).__name__}: {str(e)[:80]})")

    contrib.to_csv(os.path.join(REPDIR, "stage2_contributions.csv"),
                   index=False, encoding="utf-8-sig")
    roi.to_csv(os.path.join(REPDIR, "stage2_roi.csv"),
               index=False, encoding="utf-8-sig")
    rec.to_csv(os.path.join(REPDIR, "stage2_recovery.csv"),
               index=False, encoding="utf-8-sig")
    if roi_ci is not None:
        roi_ci.to_csv(os.path.join(REPDIR, "stage2_roi_ci.csv"),
                      index=False, encoding="utf-8-sig")
    diag["summary"].to_csv(os.path.join(REPDIR, "stage2_summary.csv"),
                           encoding="utf-8-sig")
    print(f"    ✓ 结果 CSV 已保存至 outputs/reports/")

    # ---------- 11. 总结 ----------
    print("\n" + "=" * 74)
    print("阶段 2 交付总结")
    print("=" * 74)
    print(f"  收敛: R-hat>1.01 的参数 {diag['n_bad_rhat']}/{diag['n_params']}, "
          f"divergences {diag['divergence_pct']:.2f}%")
    print(f"  截距: {diag['intercept']:+.3f} {'✓ 健康' if diag['intercept_healthy'] else '✗ 异常'}")
    print(f"  媒体占比: {cinfo['media_share']:.1%} (真值 53.9%)")
    print(f"  ROI 排序: {' > '.join(roi['渠道'].tolist())}")
    print(f"  真实排序: {' > '.join(sorted(CHANNELS, key=lambda k: -TRUE_BETA[k]))}")

    from scipy.stats import spearmanr
    rho, p = spearmanr(roi["ROI"].values,
                       [TRUE_BETA[c] for c in roi["渠道"]])
    print(f"  Spearman ρ = {rho:+.3f} (p={p:.4f})")
    print()


if __name__ == "__main__":
    main()
