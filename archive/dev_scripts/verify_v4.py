"""
验证 v4：破解 saturation_beta 的语义 + 1.5 倍虚高的来源
=======================================================
三组实验的 media 占比都是 146%~156%，beta 回收结果也一致，
说明问题**不在数据缩放**，而在**模型结构本身**。

本脚本要回答三个问题：
  Q1. saturation_beta 的语义到底是什么？（是"每单位花费的贡献"还是"每单位饱和值的贡献"？）
  Q2. 1.5 倍虚高从哪来？
  Q3. Hill 饱和函数在我们的参数下，对标准化后（均值=1）的渠道输入
      实际输出是多少？（关键：如果饱和输出恒在 0.9 以上，就几乎没有饱和）

方法：直接从后验里取参数，手工重算 Hill 输出，看它的量级。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, TRUE_BETA, TRUE_DECAY, TRUE_KAPPA, TRUE_GAMMA_FRAC
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series
from pymc_marketing.mmm import GeometricAdstock, HillSaturation


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior

    print("=" * 78)
    print("Q1/Q3: Hill 饱和函数在标准化输入下的实际输出")
    print("=" * 78)

    adstock = GeometricAdstock(l_max=cfg.adstock_l_max)
    sat = HillSaturation()

    rows = []
    for i, c in enumerate(cfg.channel_cols):
        # 标准化后的渠道花费（均值≈1）
        x_std = X[c].values
        # 用后验均值参数手工算 adstock
        d_est = float(pt["adstock_alpha"].values[:, :, i].mean())
        ad = adstock.function(x=x_std, alpha=d_est)
        k_est = float(pt["saturation_kappa"].values[:, :, i].mean())
        # Hill 的 gamma 参数名在 1.x 里是 saturation_slope（1/gamma）
        try:
            gamma_est = float(pt["saturation_slope"].values[:, :, i].mean())
        except Exception:
            gamma_est = float(pt["saturation_gamma"].values[:, :, i].mean()) \
                if "saturation_gamma" in pt else np.nan
        b_est = float(pt["saturation_beta"].values[:, :, i].mean())

        # Hill: x^k / (x^k + gamma^k)  —— 注意 1.x 里 saturation_slope 可能就是 k
        sat_out = (ad ** k_est) / (ad ** k_est + gamma_est ** k_est)
        contrib_manual = b_est * sat_out
        rows.append({
            "渠道": c,
            "est_decay": d_est, "真decay": TRUE_DECAY[c],
            "est_k": k_est, "真k": TRUE_KAPPA[c],
            "est_slope": gamma_est,
            "adstock范围": f"[{ad.min():.2f},{ad.max():.2f}]",
            "sat输出范围": f"[{sat_out.min():.3f},{sat_out.max():.3f}]",
            "sat均值": float(sat_out.mean()),
            "est_beta": b_est, "真beta": TRUE_BETA[c],
            "手工贡献万": float(contrib_manual.sum()) / 1e4,
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 手工贡献合计 vs 真实 ----
    manual_total = sum(r["手工贡献万"] for r in rows) * 1e4
    true_y = float(df["revenue"].sum())
    print(f"\n手工重算的媒体贡献合计 = {manual_total/1e4:,.0f} 万元 "
          f"({manual_total/true_y:.1%} of 真实收入)")
    print(f"真实的媒体贡献合计     = 11,178 万元 (46.7%)")

    # ---- Q2: 后验里 channel_contribution 的量级 ----
    print("\n" + "=" * 78)
    print("Q2: 后验内置 channel_contribution 的量级（看模型内部怎么算的）")
    print("=" * 78)
    avail = list(pt.data_vars)
    print("后验变量:", avail)
    for v in ["channel_contribution", "total_media_contribution",
              "total_media_contribution_original_scale"]:
        if v in pt:
            arr = pt[v].values
            print(f"  {v}: shape={arr.shape}, mean={arr.mean():.2f}, sum={arr.sum():.1f}")
    ts = float(np.squeeze(np.asarray(mmm.idata.constant_data["target_scale"].values)))
    print(f"  target_scale = {ts:,.0f}")
    if "channel_contribution" in pt:
        cc = pt["channel_contribution"].values  # (chain,draw,date,channel)
        print("\n  各渠道 channel_contribution（标准化空间）:")
        for i, c in enumerate(cfg.channel_cols):
            print(f"    {c:10s} mean={cc[:,:,:,i].mean():.3f}  "
                  f"sum={cc[:,:,:,i].sum():.1f}")
        # 乘以 scale 还原
        print("\n  channel_contribution × target_scale 还原:")
        for i, c in enumerate(cfg.channel_cols):
            g = cc[:, :, :, i].sum() * ts
            s = float(get_spend_series(df, c).sum())
            print(f"    {c:10s} 贡献={g/1e4:>10,.0f} 万元  ROI={g/s:.3f}")
        print(f"\n  还原后媒体合计 = "
              f"{cc.sum()*ts/1e4:,.0f} 万元 ({cc.sum()*ts/true_y:.1%})")


if __name__ == "__main__":
    main()
