"""
验证 v9：验证 channel_scale 假设 —— 最终破解
==============================================
v8 发现：
  · Σ channel_contribution = 150.07
  · mu = predict(X) = 100.10
  · 比值 100.10/150.07 = 0.6670
  · channel_scale = [?, ?, ?, ?, ?]  (5个)

假设：channel_contribution 是"乘法空间"的量，
     真实加性贡献 = channel_contribution / channel_scale
     类似 predict 内部 channel_data_ = channel_data / channel_scale

验证：如果 channel_contribution/channel_scale 之和 = mu，
     那就找到了正确的还原方式。
"""
import os
import sys
import warnings

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data_generator import generate_game_data, TRUE_BETA
from mmm_core import MMMConfig, normalize_columns, prepare_data, build_mmm_safe, fit_mmm, get_spend_series


def main():
    df = generate_game_data(n_weeks=156, seed=2024)
    cfg = MMMConfig(quick_mode=True).apply_quick_mode()
    df = normalize_columns(df, cfg)
    X, y, meta, _ = prepare_data(df, cfg)

    mmm, _ = build_mmm_safe(cfg)
    idata, _ = fit_mmm(mmm, X, y, cfg, progressbar=False)
    pt = idata.posterior
    cdata = mmm.idata.constant_data

    ts = float(np.squeeze(np.asarray(cdata["target_scale"].values)))
    ch_scale = np.asarray(cdata["channel_scale"].values).ravel()
    print("channel_scale =", ch_scale)
    print("target_scale =", f"{ts:,.2f}")

    cc = pt["channel_contribution"].values  # (chain,draw,date,channel)
    mu = pt["mu"].values                    # (chain,draw,date)

    # 假设 A：Σchannel_contribution = mu （可加）
    cc_sum = cc.mean(axis=(0, 1)).sum(axis=1)   # (date,)
    mu_mean = mu.mean(axis=(0, 1))              # (date,)
    print(f"\n[假设A] Σcc = {cc_sum.sum():.4f}  vs  mu = {mu_mean.sum():.4f}")

    # 假设 B：Σ(cc/channel_scale) = mu
    cc_divided = cc.mean(axis=(0, 1)) / ch_scale[None, :]   # (date,channel)
    print(f"[假设B] Σ(cc/ch_scale) = {cc_divided.sum():.4f}  vs  mu = {mu_mean.sum():.4f}")

    # 假设 C：Σ(cc * channel_scale) ?
    cc_mul = cc.mean(axis=(0, 1)) * ch_scale[None, :]
    print(f"[假设C] Σ(cc×ch_scale) = {cc_mul.sum():.4f}  vs  mu = {mu_mean.sum():.4f}")

    # 假设 D：channel_contribution 本身已含 scale（因为内部又存了 original_scale 版本）
    if "total_media_contribution_original_scale" in pt:
        tm = pt["total_media_contribution_original_scale"].values
        print(f"\n[对照] total_media_contribution_original_scale:")
        print(f"   shape={tm.shape}  mean={tm.mean():,.2f}  sum={tm.sum():,.2f}")
        print(f"   真实 y sum = {float(df['revenue'].sum()):,.2f}")

    # ---- 用假设 B 算 ROI ----
    print("\n" + "=" * 88)
    print("用假设B（cc / channel_scale）还原各渠道贡献")
    print("=" * 88)
    print(f"{'渠道':10s} {'ch_scale':>10s} {'原始cc':>12s} {'cc/ch_scale':>13s} "
          f"{'×ts万元':>14s} {'估ROI':>8s} {'真ROI':>7s}")
    print("-" * 82)
    for i, c in enumerate(cfg.channel_cols):
        raw = cc[:, :, :, i].sum() / (2 * 300)
        div = raw / ch_scale[i]
        orig = div * ts
        s = float(get_spend_series(df, c).sum())
        print(f"{c:10s} {ch_scale[i]:>10.3f} {raw:>12.4f} {div:>13.4f} "
              f"{orig/1e4:>14,.0f} {orig/s:>8.3f} {TRUE_BETA[c]:>7.2f}")

    total_div = sum((cc[:, :, :, i].sum() / 600 / ch_scale[i]) * ts
                    for i in range(5))
    true_y = float(df["revenue"].sum())
    print(f"\n假设B 媒体合计 = {total_div/1e4:,.0f} 万元 "
          f"({total_div/true_y:.1%} of 收入, 真实 46.7%)")

    # ---- 直接读官方 API 再确认 ----
    print("\n" + "=" * 88)
    print("官方 API 最终确认")
    print("=" * 88)
    dsA = mmm.data.get_contributions(original_scale=True)
    print("  变量:", list(dsA.data_vars))
    chA = dsA["channels"].mean(("chain", "draw"))
    baseA = dsA["baseline"].mean(("chain", "draw"))
    seaA = dsA["seasonality"].mean(("chain", "draw"))
    chA_sum = float(np.asarray(chA.values).sum())
    base_sum = float(np.asarray(baseA.values).sum())
    sea_sum = float(np.asarray(seaA.values).sum())
    print(f"  channels 合计 = {chA_sum/1e4:,.0f} 万元 ({chA_sum/true_y:.1%})")
    print(f"  baseline 合计 = {base_sum/1e4:,.0f} 万元 ({base_sum/true_y:.1%})")
    print(f"  seasonality   = {sea_sum/1e4:,.0f} 万元")
    print(f"  三者合计      = {(chA_sum+base_sum+sea_sum)/1e4:,.0f} 万元 "
          f"(真实 y = {true_y/1e4:,.0f} 万元)")
    print(f"\n  各渠道 ROI（官方值）:")
    for i, c in enumerate(cfg.channel_cols):
        g = float(np.asarray(chA.sel(channel=c).values).sum())
        s = float(get_spend_series(df, c).sum())
        print(f"    {c:10s} ROI={g/s:.3f}  真={TRUE_BETA[c]:.2f}")


if __name__ == "__main__":
    main()
