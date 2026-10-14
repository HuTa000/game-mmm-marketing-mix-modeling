# -*- coding: utf-8 -*-
"""
阶段 3 探针 v7：解剖优化模型的响应变量图结构
目标：搞清 total_media_contribution_original_scale 到底由哪些项组成
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, xarray as xr
from data_generator import generate_game_data
from mmm_core import MMMConfig, prepare_data, build_mmm_safe, get_spend_series

SEP = "=" * 78
def sec(t): print("\n" + SEP + "\n" + t + "\n" + SEP)

def main():
    df = generate_game_data(n_weeks=156, seed=2024, with_controls=True)
    cfg = MMMConfig()
    X, y, meta, used_ctrl = prepare_data(df, cfg)
    with open(os.path.join(ROOT, "outputs/reports/stage2_mmm_idata.pkl"), "rb") as f:
        idata = pickle.load(f)["idata"]
    mmm, _ = build_mmm_safe(cfg, used_ctrl)
    mmm.build_model(X, y); mmm.idata = idata
    spend_max = np.array([float(get_spend_series(df, c).max()) for c in cfg.channel_cols])
    mmm._channel_scales = spend_max
    opt = mmm.budget_optimizer("2025-06-08", "2025-08-31")
    m = opt.model

    sec("A. 优化模型全部变量清单")
    for name in m.named_vars:
        node = m.named_vars[name]
        dims = m.named_vars_to_dims.get(name)
        t = type(node).__name__
        print(f"  {name:52s} dims={str(dims):32s} type={t}")

    sec("B. deterministic 的头节点（看响应变量怎么算的）")
    import pytensor
    for rv in m.deterministics:
        print(f"  {rv.name:52s} = {rv.owner.op}  <- inputs: {[str(i) for i in (rv.owner.inputs if rv.owner else [])][:6]}")

    sec("C. 逐个求值：看每项的量级")
    hist_weekly = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    plan = xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",))
    # 用优化器的 pack 把 plan 变成决策向量并写入模型
    x = opt._pack_decision_vector(plan)
    fn = opt._objective_and_grad
    val, grad = fn(x)
    print(f"  objective = {val:,.4f}  utility = {-val:,.4f}")
    print(f"  gradient  = {np.asarray(grad)}")

    sec("D. 用 set_posterior 后的模型计算各 deterministic")
    # XTensorVariable 要用 eval()；先给决策向量设一个历史值
    import pytensor
    hist_weekly = np.array([float(get_spend_series(df, c).mean()) for c in cfg.channel_cols])
    plan0 = xr.DataArray(hist_weekly, coords={"channel": cfg.channel_cols}, dims=("channel",))
    x = opt._pack_decision_vector(plan0)
    # 把决策向量写进共享变量（用优化器内部机制）
    try:
        opt._variables.unpack_into(x) if hasattr(opt._variables, "unpack_into") else None
    except Exception:
        pass
    for det in ("channel_contribution", "total_media_contribution_original_scale",
                "control_contribution", "intercept_contribution", "mu",
                "saturation_beta", "adstock_alpha", "saturation_kappa", "saturation_slope"):
        if det in m.named_vars:
            try:
                vv = m.named_vars[det].eval() if hasattr(m.named_vars[det], "eval") else m.named_vars[det].get_value()
                arr = np.asarray(vv)
                print(f"  {det:50s} shape={str(arr.shape):18s} sum={arr.sum():>18,.4f} mean={arr.mean():>16,.4f}")
            except Exception as e:
                print(f"  {det:50s} 取值失败: {type(e).__name__}: {str(e)[:90]}")

    sec("E. 关键：channel_contribution 与 target_scale 的关系")
    try:
        cc = np.asarray(m.named_vars["channel_contribution"].eval())
        tot = np.asarray(m.named_vars["total_media_contribution_original_scale"].eval())
        print(f"  channel_contribution.sum() = {cc.sum():,.4f}")
        print(f"  total_media_contrib_orig   = {float(tot):,.4f}")
        print(f"  比值 (target_scale?)       = {float(tot)/cc.sum() if cc.sum()!=0 else float('nan'):,.4f}")
        print(f"  模型 target_scale          = {np.asarray(m.named_vars['target_scale'].get_value()).ravel()[0]:,.4f}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")

    sec("F. 把决策向量设成 0 看响应（纯基线检验）")
    x0 = np.zeros_like(x)
    v0, g0 = fn(x0)
    print(f"  全零花费: utility = {-v0:,.4f}")
    print(f"  全零花费: grad    = {np.asarray(g0)}")
    x_hist = x
    v1, g1 = fn(x_hist)
    print(f"  历史花费: utility = {-v1:,.4f}")

    sec("G. 极端：只把 tiktok 设成 10 倍")
    idx = cfg.channel_cols.index("tiktok")
    v_t = np.zeros(5); v_t[idx] = hist_weekly[idx] * 10
    xt = opt._pack_decision_vector(xr.DataArray(v_t, coords={"channel": cfg.channel_cols}, dims=("channel",)))
    vt, gt = fn(xt)
    print(f"  tiktok×10: utility = {-vt:,.4f}")
    print(f"  grad = {np.asarray(gt)}")

    sec("H. 各渠道单独投（用 pack 走正规通道）")
    for i, c in enumerate(cfg.channel_cols):
        vv = np.zeros(5); vv[i] = hist_weekly[i]
        xi = opt._pack_decision_vector(xr.DataArray(vv, coords={"channel": cfg.channel_cols}, dims=("channel",)))
        vi, gi = fn(xi)
        print(f"  {c:10s} 单独投 {vv[i]:>12,.0f}  utility={-vi:>16,.4f}  grad_i={np.asarray(gi)[i]:.6e}")


if __name__ == "__main__":
    main()
