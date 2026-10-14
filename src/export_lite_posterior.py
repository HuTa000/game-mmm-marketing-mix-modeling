# -*- coding: utf-8 -*-
"""
导出「精简版后验」—— 让应用可以在零采样、无大文件的前提下部署
================================================================

背景（实测数据）
----------------
`outputs/reports/stage2_mmm_idata.pkl` 共 **113.3 MB**。拆开看：

    [posterior]              113.10 MB
        ├─ 参数类（无 date 维度）       0.77 MB   ← 真正驱动优化的量
        └─ 时变类（含 date=156 维度） 112.32 MB   ← 占 99.2%
             channel_contribution   37.44 MB
             control_contribution   29.95 MB
             fourier_contribution   29.95 MB
             yearly_seasonality      7.49 MB
             mu                      7.49 MB
    [sample_stats]             0.19 MB   ← 收敛诊断（divergences）要用
    [observed/constant/fit]    0.02 MB

结论：**99% 的体积来自「逐期贡献分解」，而优化器只需要其中 0.7 MB 的参数后验。**

为什么必须瘦身
--------------
1. GitHub 单文件上限 100 MB —— 113 MB 的 pkl 直接推不上去；
2. 托管环境（如 Streamlit Cloud）对仓库体积与冷启动时间都有约束；
3. 部署演示并不需要 6000 个抽样 —— 点估计求解实测 500 抽样就完全够
   （见 docs/阶段9：最优分配偏差 0 元）。

做法
----
对**所有含 draw 维度的组**按链等间隔抽稀。posterior 与 sample_stats 使用
同一套索引，保证「链 / draw」结构一致（否则下游按 draw 对齐会错位）；
不带 draw 维度的组（观测值、常量数据、原始数据）原样保留。

用法
----
    .venv/Scripts/python.exe src/export_lite_posterior.py              # 默认每链 300 抽
    .venv/Scripts/python.exe src/export_lite_posterior.py --draws 600  # 更精细

产物
----
`outputs/reports/stage2_mmm_idata_lite.pkl` —— 精简版后验本体
`outputs/reports/stage2_mmm_idata_lite.json` —— 元信息旁挂文件（链数 / 每链抽样数 / 体积）

为什么要写那个 json
-------------------
应用需要在页面上如实说明「档里有几个后验样本」。如果为了拿这个数字去
`pickle.load` 一次 pkl，就等于**为了显示一行字加载 20 MB 数据**。
所以把元信息单独写出来，应用读 json（毫秒级）即可。

`app/core/model.py` 与 `src/stage3_optimize.py` 会**优先**加载精简版 pkl；
若不存在则自动回退到全量存档 —— 因此这个脚本是纯增量、可选的。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPDIR = os.path.join(ROOT, "outputs", "reports")
FULL = os.path.join(REPDIR, "stage2_mmm_idata.pkl")
LITE = os.path.join(REPDIR, "stage2_mmm_idata_lite.pkl")

# 这些顶层键直接原样搬运（模型配置 / 元信息，体积极小）
KEEP_TOP = ("config", "channel_cols", "control_cols", "y_mean", "y_max")


def slim_draws(ds, keep: int):
    """
    按链等间隔抽稀 `draw` 维度。

    为什么用等间隔而不是取前 k 个：NUTS 去掉 warmup 后的样本近似 iid，
    两种取法在统计上都成立；但等间隔覆盖了整个链的时间跨度，
    对存在残余自相关的变量（如 mu）更稳妥。
    """
    if keep <= 0 or "draw" not in ds.sizes:
        return ds
    n = int(ds.sizes["draw"])
    if keep >= n:
        return ds
    idx = np.linspace(0, n - 1, keep).round().astype(int)
    out = ds.isel(draw=idx)
    # 抽稀后 draw 坐标仍是原索引值（如 0,5,10,…）。重置为 0..keep-1，
    # 避免下游 isel(draw=[i]) / slice(0,k) 按位置取值时静默错位 ——
    # 这类错位不会报错，只会让区间算错，属于最难查的一类 bug。
    return out.assign_coords(draw=np.arange(keep))


def slim_tree(idata, keep: int):
    """重建一棵 DataTree，各子组按同一规则抽稀。"""
    import xarray as xr

    # 先做出一份「子节点 → 数据集」的映射，再一次性构造，避免中途失败留下半成品
    parts: dict[str, object] = {}
    for gname, node in idata.children.items():
        ds = getattr(node, "dataset", None)
        if ds is None:
            continue
        parts[gname] = slim_draws(ds, keep)

    try:
        return xr.DataTree.from_dict(parts)
    except Exception:                                          # noqa: BLE001
        # 老版本 xarray 的兜底写法（不同版本 DataTree 构造签名有差异）
        tree = xr.DataTree()
        for gname, ds in parts.items():
            tree[gname] = xr.DataTree(dataset=ds)
        return tree


def _write_meta_only(lite_path: str) -> int:
    """
    为已存在的精简版 pkl 补写 .json 元信息（不重新抽稀）。

    用途：精简版是早期版本导出的、当时还没有旁挂文件 —— 这时不必重跑
    一遍 109 MB 的读取与抽稀，直接读精简版本身补齐即可（0.0 秒）。
    """
    if not os.path.isfile(lite_path):
        print(f"✗ 找不到精简版后验：{lite_path}")
        return 2
    t0 = time.perf_counter()
    with open(lite_path, "rb") as f:
        bundle = pickle.load(f)
    idata = bundle["idata"]
    n_chain = int(idata.posterior.sizes.get("chain", 1))
    n_draw = int(idata.posterior.sizes.get("draw", 0))
    out_mb = os.path.getsize(lite_path) / 1e6
    meta = {
        "lite": True,
        "source_file": os.path.basename(FULL),
        "source_size_mb": round(os.path.getsize(FULL) / 1e6, 1)
        if os.path.isfile(FULL) else None,
        "lite_size_mb": round(out_mb, 1),
        "chains": n_chain,
        "draws_per_chain": n_draw,
        "total_draws": int(n_chain * n_draw),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "由 --meta-only 补写（未重新抽稀）",
    }
    meta_path = os.path.splitext(lite_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 已补写元信息：{meta_path}")
    print(f"  {n_chain} 链 × {n_draw} 抽 = {n_chain * n_draw} 个样本"
          f"（读取 {time.perf_counter() - t0:.1f}s，体积 {out_mb:.1f} MB）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="把阶段 2 后验导出为部署友好的精简版")
    ap.add_argument("--draws", type=int, default=300,
                    help="每条链保留的抽样数（默认 300 → 总 1200 抽）")
    ap.add_argument("--full", default=FULL, help="全量后验路径")
    ap.add_argument("--out", default=LITE, help="精简版输出路径")
    ap.add_argument("--meta-only", action="store_true",
                    help="只为**已存在**的精简版补写 .json 元信息，不做抽稀")
    args = ap.parse_args()

    if args.meta_only:
        return _write_meta_only(args.out)

    if not os.path.isfile(args.full):
        print(f"✗ 找不到全量后验：{args.full}")
        print("  请先运行 src/stage2_model.py 完成阶段 2 拟合。")
        return 2

    full_mb = os.path.getsize(args.full) / 1e6
    print("=" * 74)
    print("导出精简版后验")
    print("=" * 74)
    print(f"  输入 : {args.full}")
    print(f"         {full_mb:.1f} MB")
    print(f"  每链保留 : {args.draws} 抽")

    t0 = time.perf_counter()
    with open(args.full, "rb") as f:
        bundle = pickle.load(f)
    idata = bundle["idata"]
    n_chain = int(idata.posterior.sizes.get("chain", 1))
    n_draw = int(idata.posterior.sizes.get("draw", 0))
    print(f"  原始结构 : {n_chain} 链 × {n_draw} 抽 = {n_chain * n_draw} 样本"
          f"（读取 {time.perf_counter() - t0:.1f}s）")

    t0 = time.perf_counter()
    lite_tree = slim_tree(idata, args.draws)
    print(f"  抽稀完成 : {time.perf_counter() - t0:.1f}s")

    lite = {"idata": lite_tree}
    for k in KEEP_TOP:
        if k in bundle:
            lite[k] = bundle[k]
    # 其余未列出的顶层键也一并带上，避免丢信息
    for k, v in bundle.items():
        lite.setdefault(k, v)

    t0 = time.perf_counter()
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(lite, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.out)
    out_mb = os.path.getsize(args.out) / 1e6
    print(f"  写出完成 : {time.perf_counter() - t0:.1f}s")

    # ---- 元信息旁挂文件 ----
    # 应用要在页面上如实报「档里有多少样本」，而不该为了这一行字去 load 一个
    # 20 MB 的 pkl。所以把关键计数单独写出来，读它只要毫秒级。
    meta = {
        "lite": True,
        "source_file": os.path.basename(args.full),
        "source_size_mb": round(full_mb, 1),
        "lite_size_mb": round(out_mb, 1),
        "ratio": round(out_mb / full_mb, 4),
        "chains": n_chain,
        "draws_per_chain": int(min(args.draws, n_draw)),
        "total_draws": int(n_chain * min(args.draws, n_draw)),
        "source_draws_per_chain": n_draw,
        "source_total_draws": int(n_chain * n_draw),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元信息   : {meta_path}")
    print("-" * 74)
    print(f"  全量 {full_mb:7.1f} MB  →  精简 {out_mb:7.1f} MB"
          f"   （压缩到 {out_mb / full_mb:.1%}，省 {full_mb - out_mb:.1f} MB）")
    print(f"  每链 {n_draw} → {min(args.draws, n_draw)} 抽"
          f"（总 {n_chain * min(args.draws, n_draw)} 个样本）")
    print(f"  输出 : {args.out}")
    print("=" * 74)
    print("提示：应用会优先加载精简版；删掉该文件即自动回退全量存档。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
