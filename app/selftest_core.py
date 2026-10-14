"""
数据模块离线自测（不启动浏览器）。

为什么要有这个脚本：
    "上传必须处理格式错误、字段缺失"是硬性要求，但靠人肉在浏览器里
    一一点击验证既慢又容易漏。这里把每一类异常都固化成断言，
    一跑就知道兜底逻辑有没有退化。

运行：
    .venv/Scripts/python.exe app/selftest_core.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import data_io, eda, model as M, paths, schema       # noqa: E402
from core import facts as FACTS, optimize as OPT, report as REP  # noqa: E402

paths.ensure_src_on_path()

from mmm_core import get_spend_series                          # noqa: E402

# ------------------------------------------------------------
# 极简断言框架
# ------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"    ← {detail}" if detail and not cond else ""))


def section(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def codes(rep) -> set:
    return {i.code for i in rep.issues}


def upload(df: pd.DataFrame, name: str = "t.csv", **kw):
    payload = df.to_csv(index=False, **kw).encode("utf-8")
    return data_io.load_uploaded(payload, name)


def up_bytes(payload: bytes, name: str):
    return data_io.load_uploaded(payload, name)


# ============================================================
# 基准数据（标准化后的规范列）
# ============================================================
BASE = data_io.load_builtin().df.copy()
N = len(BASE)
print(f"基准数据：{N} 行 × {BASE.shape[1]} 列")
print(f"列：{list(BASE.columns)}")

# ------------------------------------------------------------
section("T01 内置数据")
ds = data_io.load_builtin()
check("无 error", ds.report.ok, f"errors={[i.code for i in ds.report.errors]}")
check("识别 5 个渠道", len(ds.channel_cols) == 5, str(ds.channel_cols))
check("识别 4 个控制变量", len(ds.control_cols) == 4, str(ds.control_cols))
check("行数正确", ds.n_rows == 156, str(ds.n_rows))
check("指纹非空", len(ds.fingerprint) == 16, ds.fingerprint)
check("总花费与阶段2/3同源(约1.67亿)",
      abs(ds.total_spend() / 1e8 - 1.67) < 0.05, f"{ds.total_spend()/1e8:.3f} 亿")

# ------------------------------------------------------------
section("T02 正常上传（往返一致）")
ds2 = upload(BASE)
check("无 error", ds2.report.ok, str([i.code for i in ds2.report.errors]))
check("无 warning（干净数据）", len(ds2.report.warnings) == 0,
      str([i.code for i in ds2.report.warnings]))
check("列结构一致", (ds2.channel_cols == ds.channel_cols)
      and (ds2.control_cols == ds.control_cols))
check("数值无损", np.allclose(ds2.df["facebook"].values, ds.df["facebook"].values))

# ------------------------------------------------------------
section("T03 GBK 编码（Excel 直接另存 CSV 的常见情况）")
# 注意：必须让文件里含中文，否则 GBK 与 UTF-8 的字节序列完全相同（纯 ASCII 子集），
# 编码嗅探会正确地识别为 UTF-8 而不给提示 —— 这是正确行为，不是漏报。
gbk_df = BASE.rename(columns={"date": "日期", "revenue": "收入"})
payload = gbk_df.to_csv(index=False).encode("gbk")
ds3 = up_bytes(payload, "gbk.csv")
check("可读取", ds3.report.ok, str([i.code for i in ds3.report.errors]))
check("给出非 UTF-8 提示", "READ_NON_UTF8" in codes(ds3.report),
      f"encoding={ds3.report.encoding!r}")
check("数据完整", ds3.n_rows == N, str(ds3.n_rows))
check("中文列名正确映射", ds3.report.mapping.get("date") == "日期"
      and ds3.report.mapping.get("revenue") == "收入",
      str({k: ds3.report.mapping.get(k) for k in ("date", "revenue")}))

# ------------------------------------------------------------
section("T04 缺日期列 → 必须报 error（不许瞎猜）")
ds4 = upload(BASE.drop(columns=["date"]))
check("判定为不可用", not ds4.report.ok)
check("错误码为 NO_DATE_COL", "NO_DATE_COL" in codes(ds4.report),
      str([i.code for i in ds4.report.errors]))
check("返回空数据而不猜", ds4.n_rows == 0, str(ds4.n_rows))

# ------------------------------------------------------------
section("T05 缺收入列")
ds5 = upload(BASE.drop(columns=["revenue"]))
check("判定为不可用", not ds5.report.ok)
check("错误码为 NO_TARGET_COL", "NO_TARGET_COL" in codes(ds5.report),
      str([i.code for i in ds5.report.errors]))

# ------------------------------------------------------------
section("T06 只剩 1 个渠道")
ds6 = upload(BASE.drop(columns=["google", "tiktok", "youtube", "kol"]))
check("判定为不可用", not ds6.report.ok)
check("错误码为 TOO_FEW_CHANNELS", "TOO_FEW_CHANNELS" in codes(ds6.report),
      str([i.code for i in ds6.report.errors]))

# ------------------------------------------------------------
section("T07 缺部分渠道（可继续，但必须提示）")
ds7 = upload(BASE.drop(columns=["youtube", "kol"]))
check("可用", ds7.report.ok)
check("给出缺渠道警告", "MISSING_CHANNELS" in codes(ds7.report))
check("只保留识别到的渠道", len(ds7.channel_cols) == 3, str(ds7.channel_cols))

# ------------------------------------------------------------
section("T08 非数值脏数据（'暂无数据' 之类的占位）")
dirty = BASE.copy()
# pandas 3.0 不允许把字符串直接写进 float64 列，先转 object（真实读 CSV 时本就是 object）
dirty["facebook"] = dirty["facebook"].astype(object)
dirty["google"] = dirty["google"].astype(object)
dirty.loc[0, "facebook"] = "暂无数据"
dirty.loc[1, "google"] = "-"
ds8 = upload(dirty)
check("可用", ds8.report.ok)
check("给出强制转换警告", "COERCION" in codes(ds8.report))
check("脏值已置 0", float(ds8.df.loc[0, "facebook"]) == 0.0,
      str(ds8.df.loc[0, "facebook"]))

# ------------------------------------------------------------
section("T09 空文件 / 只有表头")
ds9a = up_bytes(b"", "empty.csv")
check("空文件报 READ_EMPTY", "READ_EMPTY" in codes(ds9a.report))
ds9b = up_bytes(b"date,facebook,google,revenue\n", "header_only.csv")
check("只有表头判为不可用", not ds9b.report.ok,
      str([i.code for i in ds9b.report.errors]))

# ------------------------------------------------------------
section("T10 重复日期")
dup = pd.concat([BASE, BASE.head(3)], ignore_index=True)
ds10 = upload(dup)
check("可用", ds10.report.ok)
check("给出重复日期警告", "DUP_DATES" in codes(ds10.report))
check("已按日期合并回 156 行", ds10.n_rows == N, str(ds10.n_rows))

# ------------------------------------------------------------
section("T11 中文/混合别名列名")
alias = BASE.rename(columns={
    "date": "日期", "revenue": "收入",
    "facebook": "Facebook Spend", "google": "google_cost",
    "tiktok": "TikTok花费", "youtube": "油管花费",
    "kol": "KOL 投放 (元)",
    "holiday": "节假日", "version_update": "版本更新",
})
ds11 = upload(alias)
check("可用", ds11.report.ok, str([i.code for i in ds11.report.errors]))
check("5 个渠道全部识别", len(ds11.channel_cols) == 5, str(ds11.channel_cols))
check("控制变量识别 4 个（competitor_activity 与 trend 保持英文名也被识别）",
      len(ds11.control_cols) == 4, str(ds11.control_cols))
check("数值对齐（'KOL 投放 (元)' 正确映射）",
      np.allclose(ds11.df["kol"].values, BASE["kol"].values))

# ------------------------------------------------------------
section("T12 分号分隔 + 制表符分隔")
ds12a = upload(BASE, sep=";")
check("分号可识别", ds12a.report.ok and ds12a.n_rows == N)
check("给出分隔符提示", "READ_SEP" in codes(ds12a.report))
ds12b = upload(BASE, sep="\t")
check("制表符可识别", ds12b.report.ok and ds12b.n_rows == N)

# ------------------------------------------------------------
section("T13 只有周次序号、没有日期列")
wk = BASE.drop(columns=["date"]).copy()
wk.insert(0, "week", np.arange(1, N + 1))
ds13 = upload(wk)
check("可用", ds13.report.ok, str([i.code for i in ds13.report.errors]))
check("触发序号合成兜底", "DATE_FROM_WEEK_INDEX" in codes(ds13.report))
check("合成了 156 行日期", ds13.n_rows == N, str(ds13.n_rows))
check("日期为周一（2023-01-02 起）",
      pd.Timestamp(ds13.df["date"].iloc[0]).dayofweek == 0,
      str(ds13.df["date"].iloc[0]))

# ------------------------------------------------------------
section("T14 行数过少")
ds14 = upload(BASE.head(5))
check("判定为不可用", not ds14.report.ok)
check("错误码为 TOO_FEW_ROWS", "TOO_FEW_ROWS" in codes(ds14.report),
      str([i.code for i in ds14.report.errors]))

# ------------------------------------------------------------
section("T15 时间轴间断（中间缺 5 周）")
gap = BASE.drop(index=range(10, 15)).reset_index(drop=True)
ds15 = upload(gap)
check("可用", ds15.report.ok)
check("给出间断警告", "TIME_GAP" in codes(ds15.report))
check("已补齐回 156 行", ds15.n_rows == N, str(ds15.n_rows))

# ------------------------------------------------------------
section("T16 负花费（退款冲销）")
neg = BASE.copy()
neg.loc[0, "facebook"] = -50000.0
ds16 = upload(neg)
check("可用", ds16.report.ok)
check("给出负花费警告", "SPEND_NEGATIVE" in codes(ds16.report))
check("负值已归零", float(ds16.df["facebook"].min()) >= 0.0,
      str(ds16.df["facebook"].min()))

# ------------------------------------------------------------
section("T17 某渠道全零（无投放）")
zero = BASE.copy()
zero["kol"] = 0.0
ds17 = upload(zero)
check("可用", ds17.report.ok)
check("给出全零渠道警告", "CHANNEL_ALL_ZERO" in codes(ds17.report))

# ------------------------------------------------------------
section("T18 完全没有控制变量")
ds18 = upload(BASE[["date"] + schema.CHANNELS + ["revenue"]])
check("可用", ds18.report.ok)
check("提示缺少控制变量", "NO_CONTROL_COL" in codes(ds18.report))
check("控制变量数为 0", len(ds18.control_cols) == 0)

# ------------------------------------------------------------
section("T19 千分位 / 货币符号 / 中文数量级 / 全角数字")
m = 60
cur = pd.DataFrame({
    "date": pd.date_range("2023-01-02", periods=m, freq="7D"),
    "facebook": np.full(m, "¥12,345.6"),
    "google": np.full(m, "$2,000.00"),
    "tiktok": np.full(m, "1.5万"),
    "youtube": np.full(m, "２３４５"),          # 全角数字
    "kol": np.full(m, "(1,200)"),               # 会计负数写法
    "revenue": np.full(m, "1,000,000"),
})
ds19 = upload(cur)
check("可用", ds19.report.ok, str([i.code for i in ds19.report.errors]))
check("人民币+千分位解析正确",
      abs(float(ds19.df["facebook"].iloc[0]) - 12345.6) < 1e-6,
      str(ds19.df["facebook"].iloc[0]))
check("美元符号解析正确",
      abs(float(ds19.df["google"].iloc[0]) - 2000.0) < 1e-6,
      str(ds19.df["google"].iloc[0]))
check("中文数量级 '1.5万' → 15000",
      abs(float(ds19.df["tiktok"].iloc[0]) - 15000.0) < 1e-6,
      str(ds19.df["tiktok"].iloc[0]))
check("全角数字解析正确",
      abs(float(ds19.df["youtube"].iloc[0]) - 2345.0) < 1e-6,
      str(ds19.df["youtube"].iloc[0]))
check("会计负数 '(1,200)' → -1200（随后按负花费归零）",
      float(ds19.df["kol"].iloc[0]) == 0.0 and "SPEND_NEGATIVE" in codes(ds19.report),
      str(ds19.df["kol"].iloc[0]))

# ------------------------------------------------------------
section("T20 历史过短（不足 1 年）")
ds20 = upload(BASE.head(40))
check("可用", ds20.report.ok)
check("给出短历史警告", "SHORT_HISTORY" in codes(ds20.report),
      str(sorted(codes(ds20.report))))

# ------------------------------------------------------------
section("T21 极端输入不崩溃（健壮性）")
try:
    r1 = data_io.load_uploaded(b"\x00\x01\x02\x03\xff\xfe", "binary.bin")
    ok1 = not r1.report.ok
except Exception as e:
    ok1 = False
    print("   异常:", type(e).__name__, e)
check("二进制乱码 → 友好报错而非抛异常", ok1)

try:
    r2 = up_bytes("这不是CSV只是一句话".encode("utf-8"), "text.csv")
    ok2 = not r2.report.ok
except Exception as e:
    ok2 = False
    print("   异常:", type(e).__name__, e)
check("纯文本 → 友好报错", ok2)

try:
    single = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=60, freq="7D"),
                           "revenue": np.random.default_rng(0).uniform(9e5, 1.1e6, 60)})
    r3 = upload(single)
    ok3 = not r3.report.ok and "NO_CHANNEL_COL" in codes(r3.report)
except Exception as e:
    ok3 = False
    print("   异常:", type(e).__name__, e)
check("无渠道列 → NO_CHANNEL_COL", ok3)

# ============================================================
# 阶段 7：探索性分析算法（app/core/eda.py）
# 全部用**人造数据 + 已知答案**断言 —— 这是把"诊断结论"变成"可回归的代码"的唯一办法。
# ============================================================
section("T22 EDA · VIF 正确性（与 statsmodels 对照 + 退化处理）")
rng = np.random.default_rng(7)
Xr = rng.normal(size=(240, 4))
Xr[:, 3] = 0.6 * Xr[:, 0] + 0.5 * Xr[:, 1] + rng.normal(0, 0.5, 240)   # 人为制造相关
mine = eda._vif(Xr)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    Xc = add_constant(Xr)
    theirs = [float(variance_inflation_factor(Xc, i + 1)) for i in range(4)]
    ok_eq = all(abs(a - b) < 1e-6 for a, b in zip(mine, theirs))
except Exception as e:                                     # noqa: BLE001
    ok_eq = False
    print("   statsmodels 对照异常:", type(e).__name__, e)
check("自研 VIF 与 statsmodels 数值等价", ok_eq,
      f"mine={[round(v, 4) for v in mine]}")

Xcol = np.column_stack([rng.normal(size=150), rng.normal(size=150)])
Xcol = np.column_stack([Xcol, Xcol[:, 0] + Xcol[:, 1]])    # 第三列完全共线
vd = eda._vif(Xcol)
check("完全共线列 → VIF = ∞（而不是被忽略）", vd[2] == float("inf"), str(vd))

Xcon = np.column_stack([rng.normal(size=150), np.full(150, 5.0)])
vc = eda._vif(Xcon)
check("常数列 → VIF = NaN", bool(np.isnan(vc[1])) and np.isfinite(vc[0]), str(vc))

# ------------------------------------------------------------
section("T23 EDA · 已知滞后期 / 已知饱和形态")
r5 = np.random.default_rng(5)
n_l = 130
x_l = r5.uniform(1, 10, n_l)
y_l = np.zeros(n_l)
y_l[3:] = x_l[:-3]                                         # 构造 y_t = x_{t-3}
d_lag = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=n_l, freq="7D"),
                      "facebook": x_l, "revenue": y_l})
lg = eda.lag_correlation(d_lag, "date", "revenue", ["facebook"], max_lag=8)
check("已知滞后 3 周 → 峰值定位准确", lg["peaks"]["facebook"]["lag"] == 3,
      str(lg["peaks"]["facebook"]))

r6 = np.random.default_rng(6)
n_s = 140
xs = r6.uniform(2e4, 8e5, n_s)                             # 随机顺序，避免与时间共线
concave = 3e6 * (xs / 8e5) ** 0.4 + r6.normal(0, 3e4, n_s)
convex = 3e6 * (xs / 8e5) ** 2.2 + r6.normal(0, 3e4, n_s)
d_cc = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=n_s, freq="7D"),
                     "facebook": xs, "revenue": concave})
d_cv = d_cc.assign(revenue=convex)
s_cc = eda.saturation(d_cc, ["facebook"], "revenue")
s_cv = eda.saturation(d_cv, ["facebook"], "revenue")
check("凹（幂 0.4）数据 → 判定为递减",
      s_cc["channels"]["facebook"]["shape"] == "递减",
      f"{s_cc['channels']['facebook']['shape']} flat={s_cc['channels']['facebook']['flatness']:.3f}")
check("凸（幂 2.2）数据 → 判定为递增",
      s_cv["channels"]["facebook"]["shape"] == "递增",
      f"{s_cv['channels']['facebook']['shape']} flat={s_cv['channels']['facebook']['flatness']:.3f}")

# ------------------------------------------------------------
section("T24 EDA · 季节分解与周期自适应")
r7 = np.random.default_rng(8)
n_d = 156
t_d = np.arange(n_d)
d_seas = pd.DataFrame({
    "date": pd.date_range("2023-01-02", periods=n_d, freq="7D"),
    "facebook": 1e5,
    "revenue": 1e6 + 3e5 * np.sin(2 * np.pi * t_d / 52) + r7.normal(0, 1e4, n_d)})
dec = eda.decompose(d_seas, "date", "revenue", period=52)
check("纯季节数据 → 季节项解释 > 85% 方差",
      dec["ok"] and dec["var_season_pct"] > 85.0,
      f"season={dec.get('var_season_pct', float('nan')):.1f}%")
dec_short = eda.decompose(d_seas.head(40), "date", "revenue", period=52)
check("40 周数据 → 周期自动缩短并回报 period_used",
      dec_short["ok"] and dec_short["period_used"] < 52
      and "自动缩短" in dec_short.get("note", ""),
      f"period_used={dec_short.get('period_used')}")

# ------------------------------------------------------------
section("T25 EDA · 退化输入必须优雅降级（不抛异常）")
try:
    ok_flat = eda.decompose(d_seas.assign(revenue=1e6), "date", "revenue")["ok"] is False
    ok_zero = eda.saturation(d_seas.assign(facebook=0.0), ["facebook"],
                             "revenue")["channels"]["facebook"]["ok"] is False
    ok_lag = eda.lag_correlation(d_seas.head(6), "date", "revenue",
                                 ["facebook"])["ok"] is False
    ok_tr = eda.trend(d_seas.head(2), "date", "revenue", ["facebook"])["ok"] is False
    ok_col = eda.collinearity(d_seas, ["facebook"])["ok"] is False
    ok_all = ok_flat and ok_zero and ok_lag and ok_tr and ok_col
except Exception as e:                                     # noqa: BLE001
    ok_all = False
    print("   异常:", type(e).__name__, e)
check("收入无波动 / 全零渠道 / 观测过少 / 单渠道 → 均返回 ok=False 而非崩溃", ok_all)


def _synth_clean(n: int = 180, seed: int = 3) -> pd.DataFrame:
    """纯净人造数据：渠道独立均匀分布（无共线、无季节）+ 已知真实系数。"""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=n, freq="7D")})
    d["facebook"] = r.uniform(5e4, 2.0e5, n)
    d["google"] = r.uniform(8e4, 3.2e5, n)
    d["tiktok"] = r.uniform(3e4, 1.5e5, n)
    d["version_update"] = (r.random(n) < 0.08).astype(float)
    d["competitor_activity"] = (r.random(n) < 0.15).astype(float)
    d["holiday"] = (r.random(n) < 0.12).astype(float)
    d["trend"] = np.linspace(0.0, 1.0, n)
    d["revenue"] = (4e5
                    + 1.5 * d["facebook"] + 1.2 * d["google"] + 0.8 * d["tiktok"]
                    + 3e4 * d["version_update"] - 2e4 * d["competitor_activity"]
                    + 2e4 * d["holiday"] + 2e5 * d["trend"]
                    + r.normal(0, 1.2e4, n))
    return d


# ------------------------------------------------------------
section("T26 EDA · 块自助法 + 结论卡片")
d_clean = _synth_clean()
TRUE_B = {"facebook": 1.5, "google": 1.2, "tiktok": 0.8}
idt = eda.identifiability(d_clean, list(TRUE_B), "revenue", list(schema.CONTROLS))
got = {r["列"]: r["系数"] for _, r in idt["table"].iterrows()}
max_err = max(abs(got[c] - TRUE_B[c]) for c in TRUE_B)
check("控制了遗漏变量 → OLS 精确回收已知系数（误差 < 0.05）",
      max_err < 0.05, f"max_err={max_err:.4f}")

# 反例：故意不控制 trend —— 趋势项系数很大（2e5），即使它与渠道只有
# 有限样本的微小相关（~1/√n），也会把渠道系数带偏。这就是遗漏变量偏差。
idt_naive = eda.identifiability(d_clean, list(TRUE_B), "revenue")
naive_err = max(abs({r["列"]: r["系数"] for _, r in idt_naive["table"].iterrows()}[c]
                    - TRUE_B[c]) for c in TRUE_B)
check("故意遗漏控制变量 → 系数明显被带偏（偏差实证，而非算法缺陷）",
      naive_err > max_err + 0.05 and naive_err > 0.05,
      f"控制后 err={max_err:.4f} vs 未控制 err={naive_err:.4f}")

check("块自助法区间全部不跨 0", idt["n_cross_zero"] == 0,
      str(idt["channel_table"][["列", "CI下", "CI上"]].to_dict("records")))
check("P(系数>0) 全部 > 0.99",
      bool((idt["channel_table"]["P(系数>0)"] > 0.99).all()))

idx_b = eda._block_indices(100, 6, np.random.default_rng(0))
check("块自助索引：长度正确且不越界",
      len(idx_b) == 100 and int(idx_b.min()) >= 0 and int(idx_b.max()) < 100)

rr_clean = eda.readiness(d_clean, "date", "revenue",
                         list(TRUE_B), list(schema.CONTROLS))
check("干净数据 → 结论卡片无硬伤", rr_clean["n_fail"] == 0,
      str([c["key"] for c in rr_clean["checks"] if c["status"] == "fail"]))
check("干净数据 → 判定为『支持做归因』",
      rr_clean["level"] == "pass", f"{rr_clean['level']} score={rr_clean['score']:.0f}")

d_bad = _synth_clean()
d_bad["tiktok"] = d_bad["facebook"] + d_bad["google"]        # 诱导完全共线
col_bad = eda.collinearity(d_bad, ["facebook", "google", "tiktok"])
check("完全共线 → VIF = ∞ 且判定 fail",
      col_bad["max_vif"] == float("inf") and col_bad["level"] == "fail",
      f"max_vif={col_bad['max_vif']} level={col_bad['level']}")
rr_bad = eda.readiness(d_bad, "date", "revenue",
                       ["facebook", "google", "tiktok"], list(schema.CONTROLS))
check("完全共线 → 结论卡片出现硬伤（不再误判为健康）", rr_bad["n_fail"] >= 1,
      f"n_fail={rr_bad['n_fail']} verdict={rr_bad['verdict']}")

# ------------------------------------------------------------
section("T27 EDA · 内置数据端到端（与阶段 1/2/3 的已知结论对齐）")
res_real = eda.analyze_all(BASE, "date", "revenue", schema.CHANNELS, schema.CONTROLS)
check("内置数据 → 结论为『支持做归因』",
      res_real["readiness"]["level"] == "pass",
      f"{res_real['readiness']['level']} score={res_real['readiness']['score']:.0f}")
check("收入与总花费相关 ≈ 0.899（与阶段 1 一致）",
      abs(res_real["trend"]["corr_rev_spend"] - 0.899) < 0.01,
      f"{res_real['trend']['corr_rev_spend']:.3f}")
check("最大 VIF ≈ 1.10（与阶段 1 一致）",
      abs(res_real["collinearity"]["max_vif"] - 1.10) < 0.05,
      f"{res_real['collinearity']['max_vif']:.3f}")
check("季节项方差占比 ≈ 51%（与阶段 1 一致）",
      abs(res_real["decompose"]["var_season_pct"] - 51.0) < 2.0,
      f"{res_real['decompose']['var_season_pct']:.1f}%")
check("建议 Adstock 窗口 l_max = 6（与阶段 2 的建模设定一致）",
      res_real["lag"]["suggested_l_max"] == 6,
      str(res_real["lag"]["suggested_l_max"]))
check("渠道净末段斜率排序与阶段 3 边际回报排序高度一致（tiktok 第一、google 第二）",
      sorted(res_real["saturation"]["channels"],
             key=lambda c: -res_real["saturation"]["channels"][c].get("slope_high", -9))[:2]
      == ["tiktok", "google"],
      str({c: round(r.get("slope_high", float("nan")), 2)
           for c, r in res_real["saturation"]["channels"].items()}))


# ============================================================
# 阶段 8：建模与诊断逻辑（app/core/model.py）
# 原则：**不重新采样**。所有断言都建立在"复用阶段 2 后验"或"提前退出的兜底路径"上，
#       否则自测本身就要跑十几分钟，没人会去跑它。
# ============================================================
section("T28 建模 · 配置桥与上传兜底（不采样）")
cfg_b, notes_b = M.build_config(ds, None)
check("内置数据 → 5 渠道 / 4 控制变量 / 季节性 2 阶",
      len(cfg_b.channel_cols) == 5 and len(cfg_b.control_cols) == 4
      and cfg_b.yearly_seasonality == 2,
      f"ch={cfg_b.channel_cols} ctl={len(cfg_b.control_cols)} yearly={cfg_b.yearly_seasonality}")

cfg_nc, notes_nc = M.build_config(ds18, None)
check("无控制变量 → cfg.control_cols 为空（稍后按『不传』处理）",
      len(cfg_nc.control_cols) == 0, str(cfg_nc.control_cols))

cfg_sh, notes_sh = M.build_config(ds20, None)                    # ds20 = 40 周
check("40 周历史 → 季节性自动降为 1 阶并给出警告",
      cfg_sh.yearly_seasonality == 1
      and any(n["level"] == "warn" and "季节性" in n["title"] for n in notes_sh),
      f"yearly={cfg_sh.yearly_seasonality} notes={[n['title'] for n in notes_sh]}")

cfg_z, notes_z = M.build_config(ds17, None)                      # ds17 = kol 全零
check("全零渠道被自动剔除（4 个可建模渠道）",
      len(cfg_z.channel_cols) == 4 and "kol" not in cfg_z.channel_cols,
      str(cfg_z.channel_cols))

zy = BASE.copy()
for _c in schema.CHANNELS:
    zy[_c] = 0.0
ds_zy = upload(zy)
cfg_zy, notes_zy = M.build_config(ds_zy, None)
check("所有渠道全零 → 给出 fail 级提示",
      any(n["level"] == "fail" for n in notes_zy),
      str([n["title"] for n in notes_zy]))
art_zy = M.fit(ds_zy, None)
check("该情形下 fit 提前退出、不进入采样",
      art_zy.get("ok") is False and "notes" in art_zy,
      f"ok={art_zy.get('ok')} reason={art_zy.get('reason')}")

check("采样量提示含样本总数与耗时档位",
      "次后验抽样" in M.sample_size_note(cfg_b)
      and "分钟" in M.sample_size_note(cfg_b), M.sample_size_note(cfg_b))

import json as _json                                        # noqa: E402

snap = M.config_snapshot(cfg_b)
snap_again = M.config_snapshot(M.build_config(ds, None)[0])
h1 = _json.dumps(snap, sort_keys=True, default=str)
h2 = _json.dumps(snap_again, sort_keys=True, default=str)
check("配置快照可序列化且稳定（同一配置两次得到同一缓存键）",
      h1 == h2 and "channels" in snap and snap["channels"] == cfg_b.channel_cols, "")
check("不同配置 → 快照不同（换了超参/数据绝不会命中旧缓存）",
      _json.dumps(M.config_snapshot(cfg_z), sort_keys=True, default=str) != h1, "")

# ------------------------------------------------------------
section("T29 建模 · 复用阶段 2 后验（不重新采样）")
art = M.load_pretrained(data_io.load_builtin())
check("复用成功", art.get("ok") is True, str(art.get("reason")))
diag = art["diag"]
check("标记为预训练来源、耗时为 0（确实没重采样）",
      art["source"] == "pretrained" and float(art["elapsed"]) == 0.0,
      f"source={art['source']} elapsed={art['elapsed']}")
check("复用结果与阶段 2 报告一致：R-hat 全部达标",
      diag["n_bad_rhat"] == 0 and diag["max_rhat"] < 1.01, f"max_rhat={diag['max_rhat']:.4f}")
check("发散样本为 0（与阶段 2 的诊断结论一致）",
      diag["n_divergences"] == 0, str(diag["n_divergences"]))
check("最小 ESS > 400", diag["min_ess"] > 400, f"{diag['min_ess']:.0f}")
check("截距为正 → 基线健康",
      diag["intercept"] is not None and diag["intercept"] > 0,
      f"{diag['intercept']:.3f}")
# 样本数不写死：应用可能加载「精简版后验」（部署友好，逐期数组按链抽稀），
# 此时样本数会少于 6000。要检的是「= 链数 × 每链抽样数」这个不变量。
_n_ch_pt = int(art["idata"].posterior.sizes["chain"])
_n_dr_pt = int(art["idata"].posterior.sizes["draw"])
check(f"后验样本数 = 链数 × 每链抽样数（{_n_ch_pt} × {_n_dr_pt} = {_n_ch_pt * _n_dr_pt}）",
      diag["n_samples"] == _n_ch_pt * _n_dr_pt, str(diag["n_samples"]))

# ---- 页面文案里的样本数必须与事实一致 ----
# 这一条是补的：`views/modeling.py` 曾经硬编码「含 6,000 个后验样本」，
# 换成精简版后页面照样显示 6,000，而实际只有 1,200 —— 不报错、只是数字错。
# 面试场景下这种错误最致命。现在页面的数字来自 model.pretrained_info()，
# 它读的是导出脚本写的同名 .json 元信息，所以这里校验「元信息 = 事实」。
_pv = M.pretrained_info()
if _pv.get("available") and _pv.get("total_draws"):
    check(f"页面文案报的样本数与实际一致（元信息 {_pv['total_draws']:,} "
          f"= 实际 {_n_ch_pt * _n_dr_pt:,}）",
          int(_pv["total_draws"]) == _n_ch_pt * _n_dr_pt,
          f"samples_note={_pv.get('samples_note')!r}")
    check("链数 / 每链抽样数也与实际一致",
          int(_pv["chains"]) == _n_ch_pt and int(_pv["draws_per_chain"]) == _n_dr_pt,
          f"meta={_pv['chains']}×{_pv['draws_per_chain']}")
else:
    check("页面文案的样本数走「不报具体数字」降级路径（元信息缺失时不许瞎报）",
          "samples_note" not in _pv, str(_pv))

san = art["sanity"]
labels = [c["key"] for c in san.get("checks", [])]
check("自洽性三项齐全（可加性 / 截距 / 预测精度）",
      {"additive", "intercept", "pred"} <= set(labels), str(labels))
check("预测总量偏差 < 5%（复用路径下也不会因 MaskedArray 而静默丢失）",
      san.get("pred_ok") is True, f"pred_err={san.get('pred_err')}")

# ------------------------------------------------------------
section("T30 建模 · ROI 与阶段 2 产物的口径一致性")
roi_nv = art["roi"]
check("ROI 五渠道齐全且降序",
      len(roi_nv) == 5 and list(roi_nv["渠道"]) ==
      ["tiktok", "google", "kol", "facebook", "youtube"],
      str(list(roi_nv["渠道"])))
check("区间满足 下界 < 均值 < 上界",
      bool(((roi_nv["ROI下界"] < roi_nv["ROI均值"])
            & (roi_nv["ROI均值"] < roi_nv["ROI上界"])).all()), "")
check("P(ROI>1) 落在 [0,1]",
      bool(roi_nv["P(ROI>1)"].between(0, 1).all()), "")

roi_csv = M.roi_from_csv()
if roi_csv is None:
    check("阶段 2 的 ROI 区间表可用（用于交叉核对）", False, "缺少 stage2_roi_ci.csv")
else:
    mine = roi_nv.set_index("渠道")
    worst = 0.0
    for _, r in roi_csv.iterrows():
        c = r["渠道"]
        if c not in mine.index:
            worst = 9.9
            continue
        for col in ("ROI均值", "ROI下界", "ROI上界"):
            worst = max(worst, abs(float(r[col]) - float(mine.loc[c, col])))
    # 【口径说明】stage2_roi_ci.csv 由阶段的 `channel_contribution × target_scale` 得到；
    # 本页统一改用官方 `mmm.data.get_contributions(original_scale=True)`。
    # 全量后验（6000 抽样）下两者差 4.7e-4（相对 0.04%）；
    # 若加载的是**精简版后验**（1200 抽样，逐期数组按链抽稀），蒙特卡洛误差会
    # 放大到约 4e-3（相对 ~0.13%）—— 量级仍远小于任何业务判断阈值，
    # 且渠道排序与「该撤的结论」都不受影响（见 docs/阶段9 的 lite/full 对照）。
    check("现算 ROI 与 stage2_roi_ci.csv 口径一致（差异 < 1%）",
          worst < 1e-2, f"max_diff={worst:.2e}")
    check("youtube 的 94% 区间整体 < 1（该撤的结论稳健）",
          float(mine.loc["youtube", "ROI上界"]) < 1.0,
          f"上界={mine.loc['youtube', 'ROI上界']:.3f}")

# ------------------------------------------------------------
section("T31 建模 · 贡献守恒与收入构成的两张账")
cinfo = art["cinfo"]
check("守恒分解残差 < 0.1%", cinfo["conservative"] and cinfo["residual_pct"] < 0.001,
      f"residual={cinfo['residual_pct']:.5%}")
check("媒体占收入比 ≈ 57%（与阶段 2 的 56.9% 同口径）",
      abs(cinfo["media_share"] - 0.570) < 0.01, f"{cinfo['media_share']:.4f}")

mix = art["mix"]
total = mix["total"]
y_sum = float(BASE["revenue"].sum())
check("总量账：基线 + 媒体 ≈ 历史总收入（误差 < 0.5%）",
      abs((total["金额(万元)"].sum() * 1e4 - y_sum) / y_sum) < 0.005,
      f"和={total['金额(万元)'].sum()*1e4:,.0f} vs 收入={y_sum:,.0f}")
check("总量账只含基线与渠道两类（季节/控制不混进来）",
      set(total["类别"]) <= {"基线", "媒体"}, str(set(total["类别"])))

wave = mix["wave"]
ctl_row = wave[wave["分量"] == "控制变量"]
check("波动账：控制变量的年度净额约为 0（不是被误当成总量）",
      len(ctl_row) == 0 or abs(float(ctl_row["年度净额(万元)"].iloc[0])) < 1.0,
      str(wave.to_dict("records")))
check("波动账给出了振幅（季节项振幅 > 0）",
      float(wave[wave["分量"].str.contains("季节")]["逐周振幅 ±(万元)"].iloc[0]) > 0, "")

# ------------------------------------------------------------
section("T32 建模 · 参数回收与排序（唯一能对答案的场景）")
rec = art["recovery"]
check("回收表被测参数 10 项（5 渠道 × ROI/衰减）", rec["n_tested"] == 10, str(rec["n_tested"]))
_tab = rec["table"].set_index("渠道")
_rv = roi_nv.set_index("渠道")
_roi_diff = max(abs(float(_tab.loc[c, "估_ROI"]) - float(_rv.loc[c, "ROI均值"]))
                for c in _rv.index if c in _tab.index)
check("ROI 用后验区间比真值（口径未被 saturation_beta 污染）",
      "估_ROI" in rec["table"].columns and _roi_diff < 1e-9,
      f"max_diff={_roi_diff:.2e} cols={list(rec['table'].columns)}")
check("★ ROI 排序与真值完全一致",
      rec["rank_info"].get("order_match") is True,
      f"估={rec['rank_info'].get('est_order')} 真={rec['rank_info'].get('true_order')}")
check("衰减 λ 的区间命中率 ≥ 3/5",
      int(sum(1 for v in rec["table"]["衰减λ命中"] if v == "✓")) >= 3,
      str(list(rec["table"]["衰减λ命中"])))

# ------------------------------------------------------------
section("T33 建模 · 缓存载荷往返与轻量视图")
light = M.light(art)
check("light() 剥掉了不可序列化的大对象",
      not ({"mmm", "idata", "df"} & set(light.keys())), str(sorted(light.keys())))
check("light() 保留了展示所需的全部字段",
      {"diag", "roi", "cinfo", "sanity", "mix", "recovery", "ident"} <= set(light),
      "")

payload = M.to_cache_payload(art)
check("缓存载荷只含可落盘字段（idata + 配置 + 元信息）",
      "idata" in payload and "cfg" in payload and "mmm" not in payload,
      str(sorted(payload.keys())))
art2 = M.from_cache_payload(payload, BASE)
check("从载荷还原 → 成功且标记为缓存来源（同时保留原始来源）",
      art2.get("ok") is True and art2.get("source") == "cache"
      and art2.get("origin_source") == "pretrained",
      f"source={art2.get('source')} origin={art2.get('origin_source')} "
      f"reason={art2.get('reason')}")
if art2.get("ok"):
    d2 = (art2["roi"].set_index("渠道")["ROI均值"]
          - roi_nv.set_index("渠道")["ROI均值"]).abs().max()
    check("还原后的 ROI 与原始逐位一致（重建结构没有改变数值）",
          float(d2) < 1e-9, f"max_diff={d2:.2e}")

# ------------------------------------------------------------
section("T34 建模 · 后验抽样与绘图数据")
dr = M.draws_for_plot(art["idata"], "adstock_alpha", channels=cfg_b.channel_cols)
check("draws_for_plot 返回每渠道一个样本数组",
      dr["ok"] and set(dr["values"]) == set(cfg_b.channel_cols)
      and all(len(v) > 100 for v in dr["values"].values()),
      str({k: len(v) for k, v in dr["values"].items()}))
tr = M.trace_for_plot(art["idata"], "adstock_alpha", channel="tiktok")
check("trace_for_plot 返回 4 条链", tr["ok"] and len(tr["chains"]) == 4,
      str(len(tr.get("chains", []))))
check("不存在的变量 → 明确返回 ok=False 而不是抛异常",
      M.draws_for_plot(art["idata"], "不存在的变量")["ok"] is False, "")

# ------------------------------------------------------------
section("T35 建模 · 结构性不可识别的量化")
ident = art["ident"]
check("可识别性计算成功", ident.get("ok") is True, str(ident.get("reason")))
check("恒定分量 + 波动分量 = 媒体总贡献（分解自洽）",
      abs((ident["const_part"] + ident["wave_part"]) - cinfo["media_sum"]) < 1.0,
      f"{ident['const_part']:,.0f}+{ident['wave_part']:,.0f} vs {cinfo['media_sum']:,.0f}")
check("恒定分量占比 > 35% → 判定为 fail（正是不该忽略的那个风险）",
      ident["const_share"] > 0.35 and ident["level"] == "fail",
      f"share={ident['const_share']:.3f} level={ident['level']}")
check("结论里明确给出缓解手段（实验校准）",
      "geo-lift" in ident["verdict"] or "pause test" in ident["verdict"], "")


# ============================================================
# 阶段 9：预算优化 + 决策报告
# ============================================================
section("T36 优化 · 窗口推断与退化保护")
_w = OPT.infer_window(art["df"], art["cfg"].date_col)
check("内置数据窗口推断成功", _w["ok"] is True, str(_w))
check("决策期 = 13 期（156 周的 25%）", _w["n_decision"] == 13, str(_w.get("n_decision")))
check("窗口末端 = 数据末端", _w["end"] == str(pd.to_datetime(art["df"]["date"]).max().date()),
      f"{_w['end']} vs {pd.to_datetime(art['df']['date']).max().date()}")
check("周期识别为周（7 天）", _w["freq_days"] == 7, str(_w["freq_days"]))
_w30 = OPT.infer_window(BASE.head(30), "date")
check("30 期数据：决策期被压缩到 ≤24 期（必须给预热留位置）",
      _w30["ok"] and _w30["n_decision"] <= 30 - OPT.DEFAULT_CARRY_IN,
      str(_w30.get("n_decision")))
_w10 = OPT.infer_window(BASE.head(10), "date")
check("10 期数据 → 明确拒绝并说明差多少（不许硬算）",
      _w10["ok"] is False and "至少需要" in _w10.get("reason", ""), str(_w10.get("reason")))

section("T37 优化 · channel_scales 假优化防护（本轮最关键的坑）")
_mmm, _cfg, _df = art["mmm"], art["cfg"], art["df"]
_sc = OPT.ensure_channel_scales(_mmm, _df, _cfg)
_h = OPT.channel_scales_health(_mmm, _cfg)
check("挂载后体检通过", _h["ok"] is True, _h["verdict"])
check("channel_scales = 各渠道训练期最大花费（与库内换算自洽）",
      bool(np.allclose(_sc["scales"],
                       [float(get_spend_series(_df, c).max()) for c in _cfg.channel_cols])),
      str(np.round(_sc["scales"], 1)))

_opt_ok, _meta_ok = OPT.build_optimizer(_mmm, _df, _cfg, _w)
_hw = OPT.hist_weekly_spend(_df, _cfg)
_ht = _hw * _meta_ok["num_periods"]
check("窗口内总预算 = 周均花费 × 决策期数",
      abs(_ht.sum() - _hw.sum() * _meta_ok["num_periods"]) < 1e-6, f"{_ht.sum():,.0f}")

_r_ok = OPT.optimize_budget(_opt_ok, _cfg, _ht, low=0.6, high=1.4)
check("挂载态：求解成功", _r_ok["ok"] is True, str(_r_ok.get("reason")))
check("挂载态：迭代次数 > 1（优化器真的在搜索）", _r_ok["nit"] > 1, f"nit={_r_ok['nit']}")
check("挂载态：分配不是均匀分配（不是假优化）",
      float(np.std(_r_ok["budgets"])) > 1.0,
      str(np.round(_r_ok["budgets"] / 1e4, 1)))

# ★ 对照实验：故意删掉 channel_scales，让库回退到 1.0
_saved = _mmm._channel_scales
try:
    delattr(_mmm, "_channel_scales")
    check("删掉属性后体检立刻判为不可信（能识别出坑）",
          OPT.channel_scales_health(_mmm, _cfg)["ok"] is False, "")
    _opt_bad = _mmm.budget_optimizer(_w["start"], _w["end"])
    _r_bad = OPT.optimize_budget(_opt_bad, _cfg, _ht, low=0.6, high=1.4)
finally:
    _mmm._channel_scales = _saved          # 必须恢复，否则污染后续断言

check("未挂载态：求解仍返回 ok，但分配退化为『并列解』",
      bool(_r_bad.get("ok")) and _r_bad.get("degenerate") is True,
      str(np.round(np.asarray(_r_bad.get("budgets", [])) / 1e4, 1)))
check("★ 未挂载态：提升数值上为 0（跑完一圈一个增量都没找到）",
      bool(_r_bad.get("ok")) and abs(float(_r_bad["lift_pct"])) < 1e-4,
      f"lift={_r_bad.get('lift_pct')!r}")
check("未挂载态：nit 崩塌（对比挂载态差了 20 倍以上）",
      int(_r_bad.get("nit", 99)) <= 2 and int(_r_ok["nit"]) > 20,
      f"未挂载 nit={_r_bad.get('nit')} vs 挂载 nit={_r_ok['nit']}")
check("★ 未挂载态 success 仍为 True —— 这才是『静默』的含义",
      _r_bad.get("success") is True, str(_r_bad.get("success")))
check("挂载态：不出现并列解（真实最优会把渠道按斜率/边界分开）",
      _r_ok.get("degenerate") is False, str(_r_ok.get("dup_pairs")))
check("挂载态提升显著高于未挂载态（> 10 个百分点）",
      float(_r_ok["lift_pct"]) - float(_r_bad.get("lift_pct") or 0) > 10,
      f"{_r_ok['lift_pct']:+.3f}% vs {_r_bad.get('lift_pct'):+.3f}%")
check("并列检测确实点名了那组渠道（可读的诊断，不是布尔值）",
      len(_r_bad.get("dup_pairs") or []) >= 1,
      str(_r_bad.get("dup_pairs")))
check("假优化时 message 里给出可操作提示（指向 channel_scales）",
      "channel_scales" in (_r_bad.get("message") or ""),
      (_r_bad.get("message") or "")[:60])
check("恢复后仍然健康（对照实验不污染状态）",
      OPT.channel_scales_health(_mmm, _cfg)["ok"] is True, "")

section("T38 优化 · 约束优化与守恒性")
check("总预算守恒（各渠道分配之和 = 给定总预算）",
      abs(float(_r_ok["budgets"].sum()) - _ht.sum()) < 1e-6 * float(_ht.sum()),
      f"{_r_ok['budgets'].sum():,.2f} vs {_ht.sum():,.2f}")
check("所有渠道都落在允许区间内",
      bool(np.all(_r_ok["budgets"] >= _r_ok["bounds_lo"] - 1e-6)
           and np.all(_r_ok["budgets"] <= _r_ok["bounds_hi"] + 1e-6)), "")
check("优化后贡献不低于维持现状（否则数学上不成立）",
      _r_ok["utility"] >= _r_ok["hist_utility"] - 1e-9,
      f"{_r_ok['utility']:,.0f} vs {_r_ok['hist_utility']:,.0f}")
check("存在正的『常数底』（carry-in 余温），且小于当前贡献",
      0 < _r_ok["zero_utility"] < _r_ok["hist_utility"],
      f"底={_r_ok['zero_utility']:,.0f} 当前={_r_ok['hist_utility']:,.0f}")
check("硬边界下最优解至少有一个卡在角点上（本页核心洞察）",
      _r_ok["corner_hit"] >= 1, f"角点 {_r_ok['corner_hit']}/{len(_cfg.channel_cols)}")
check("±40% 下提升为正（钱确实能挪出增量）",
      _r_ok["lift_pct"] > 0, f"{_r_ok['lift_pct']:+.2f}%")

_dist = OPT.response_distribution(_opt_ok, _cfg, _r_ok["budgets"])
check("后验不确定性传导成功（拿到区间）", _dist["ok"] is True, str(_dist.get("reason")))
check("净增量区间满足 下界 < 均值 < 上界",
      _dist["net_lo"] < _dist["net_mean"] < _dist["net_hi"],
      f"[{_dist['net_lo']:,.0f}, {_dist['net_hi']:,.0f}]")
check("区间样本数 = 后验样本数（不写死，兼容精简版后验）",
      _dist["samples"] == (int(_opt_ok.idata.posterior.sizes["chain"])
                           * int(_opt_ok.idata.posterior.sizes["draw"])),
      str(_dist["samples"]))

section("T39 优化 · 弹性-收益单调性")
_r10 = OPT.optimize_budget(_opt_ok, _cfg, _ht, low=0.9, high=1.1)
_r60 = OPT.optimize_budget(_opt_ok, _cfg, _ht, low=0.4, high=1.6)
check("±60% 的提升 ≥ ±10%（放宽约束不应让收益变差）",
      _r60["lift_pct"] >= _r10["lift_pct"] - 1e-9,
      f"±10%={_r10['lift_pct']:.3f}%  ±60%={_r60['lift_pct']:.3f}%")
check("±10% 的提升 ≥ 0", _r10["lift_pct"] >= -1e-9, f"{_r10['lift_pct']:+.3f}%")
check("两档都在允许区间内（约束被真正执行）",
      bool(np.all(_r10["budgets"] >= _r10["bounds_lo"] - 1e-6)
           and np.all(_r10["budgets"] <= _r10["bounds_hi"] + 1e-6)), "")

section("T40 报告 · 会话事实层（现算）与白名单")
_rt = OPT.response_tables(_mmm, _cfg, _df, art["idata"])
_marg = _rt["marginal"]
_alloc = pd.DataFrame({
    "渠道": list(_cfg.channel_cols),
    "当前窗口预算(元)": _ht, "优化窗口预算(元)": _r_ok["budgets"],
    "变化(%)": (_r_ok["budgets"] / _ht - 1) * 100,
    "边际回报": [float(_marg.loc[_marg["渠道"] == c, "边际回报"].iloc[0])
              for c in _cfg.channel_cols],
    "平均ROI": [float(_marg.loc[_marg["渠道"] == c, "平均ROI"].iloc[0])
             for c in _cfg.channel_cols],
})
_min_opt = {
    "ok": True, "window": _w, "opt_meta": _meta_ok,
    "base_total": float(_ht.sum()), "elasticity": 0.4, "hist_total": _ht,
    "main": _r_ok, "dist": _dist, "alloc": _alloc, "marginal": _marg,
    "sweep": {"table": pd.DataFrame(), "inf_row": None},     # 空 → 走缺失分支
    "scenarios": {}, "robustness": {"ok": False, "reason": "本次未运行"},
    "recommend": OPT.recommend(_cfg, _alloc, _marg,
                               {"lift_at_20": None, "lift_at_40": None}, _r_ok),
}
_f = FACTS.build_facts(art, _min_opt, ds)
check("会话事实层可构建且指纹非空", bool(_f.facts_hash()), _f.facts_hash())
check("提升幅度与求解结果逐位一致（无二次加工）",
      abs(float(_f.optimization["提升幅度(%)"]) - _r_ok["lift_pct"]) < 1e-9,
      f"{_f.optimization['提升幅度(%)']} vs {_r_ok['lift_pct']}")
check("窗口内总预算与 hist_total 一致",
      _f.optimization.get("窗口内总预算(元)") is not None
      and abs(float(_f.optimization["窗口内总预算(元)"]) - float(_ht.sum())) < 1.0,
      str(_f.optimization.get("窗口内总预算(元)")))
check("分渠道分配表行数 = 渠道数", len(_f.allocation) == 5, str(len(_f.allocation)))
check("边际回报表行数 = 渠道数", len(_f.marginal) == 5, str(len(_f.marginal)))
check("渠道 ROI 表非空（来自模型后验）", len(_f.roi) == 5, str(len(_f.roi)))
check("空输入的板块被如实标进 missing（不假装有）",
      "弹性-收益曲线" in _f.missing and "情景分析" in _f.missing, str(_f.missing))
check("风险条目的共线性数字是现算的（不是写死的 0.87/0.24）",
      any("最大 |相关系数|" in (r.get("detail") or "") for r in _f.risks),
      str([r["id"] for r in _f.risks]))
check("风险条目数量 = 10（框架完整）", len(_f.risks) == 10, str(len(_f.risks)))
_allow = _f.allowed_numbers()
check("白名单规模合理（>500 个可引用值）", len(_allow) > 500, str(len(_allow)))
check("白名单含提升幅度的百分数形式（模板会用）",
      any(abs(a - float(_r_ok["lift_pct"])) < 1e-6 for a in _allow), "")

section("T41 报告 · 三路径与幻觉拦截")
_r_tpl = REP.generate(_f, use_llm=False)
check("模板路径：来源正确且内容充实",
      _r_tpl["source"] == "template" and _r_tpl["chars"] > 1000, str(_r_tpl["chars"]))
check("模板路径不做数字校验（无幻觉风险，无需校验）",
      _r_tpl["guard"] is None, "")
check("模板报告含事实指纹（可追溯）", _f.facts_hash() in _r_tpl["text"], "")

_r_mock = REP.generate(_f, use_llm=True, force_mock=True, mock_fabricate=False)
check("模拟正常路径：采用大模型稿且校验通过",
      _r_mock["source"] == "llm" and _r_mock["guard"].ok is True,
      f"source={_r_mock['source']} fab={_r_mock['guard'].n_fabricated}")
check("校验器确实检查了大量数字（不是空转）",
      _r_mock["guard"].n_checked > 50, str(_r_mock["guard"].n_checked))

_r_bad = REP.generate(_f, use_llm=True, force_mock=True, mock_fabricate=True)
check("★ 注入幻觉 → 被拦截并降级为模板",
      _r_bad["source"] == "llm_rejected" and _r_bad["guard"].ok is False,
      f"source={_r_bad['source']} fab={_r_bad['guard'].n_fabricated}")
check("拦截条数 > 0 且给出可核对明细",
      _r_bad["guard"].n_fabricated >= 1 and len(_r_bad["audit_rows"]) >= 1,
      f"fab={_r_bad['guard'].n_fabricated} rows={len(_r_bad['audit_rows'])}")
check("被拒稿件留档（可审计『为什么被拦』）",
      bool(_r_bad.get("rejected_doc")) and len(_r_bad["rejected_doc"]) > 500,
      str(len(_r_bad.get("rejected_doc") or "")))
check("被拦截时页面展示的是模板稿（正确但朴素的优先）",
      _r_bad["text"].startswith("> **生成方式**：大模型稿被拒"), _r_bad["text"][:40])
check("降级来源标记与审计块都写入文档",
      "数字校验审计" in _r_bad["text"] and "无依据" in _r_bad["text"], "")


# ------------------------------------------------------------
section("T42 优化 · 后验子集加速的正确性与可恢复性（新增）")
# 动态读取当前后验规模。应用可能加载「精简版后验」（逐期数组按链抽稀，
# 便于提交仓库 / 云端部署），此时样本数会少于 6000 —— 断言要检的是
# 「行为是否正确」，而不是「样本数等于某个写死的魔法值」。
_N_CHAIN = int(art["idata"].posterior.sizes["chain"])
_N_DRAW = int(art["idata"].posterior.sizes["draw"])
_N_SAMPLES = _N_CHAIN * _N_DRAW
_FAST_OK = _N_DRAW >= OPT.FAST_MIN_FULL          # 样本够多才值得截子集
_sub, _si = OPT.fast_posterior(art["idata"])
if _FAST_OK:
    check(f"能从 {_N_SAMPLES} 抽样中抽出 {OPT.FAST_DRAWS} 子集",
          _si["applied"] is True and _sub is not None, str(_si))
    check("子集抽样数 = FAST_DRAWS",
          int(_sub.posterior.sizes["draw"]) == OPT.FAST_DRAWS,
          str(dict(_sub.posterior.sizes)))
else:
    check(f"后验规模较小（{_N_SAMPLES} 抽样）→ 如实不启用子集加速（宁慢不错）",
          _si["applied"] is False and _sub is None, str(_si))
    _sub = art["idata"]          # 后续 A/B 退化为「自己比自己」，偏差应为 0
_small = OPT.fast_posterior(art["idata"].isel(chain=[0], draw=slice(0, 400)))
check("后验本来就小 → 不截取（宁慢不错）", _small[0] is None, str(_small[1]))
check("没有后验 → 不截取且说明原因",
      OPT.fast_posterior(None)[0] is None, str(OPT.fast_posterior(None)[1]))

# A/B：子集 vs 全量，同一约束下的最优分配应几乎一致
_opt_ab, _meta_ab = OPT.build_optimizer(_mmm, _df, _cfg, _w)
_ht_ab = OPT.hist_weekly_spend(_df, _cfg) * _meta_ab["num_periods"]
_opt_ab.set_posterior(art["idata"])
_r_full2 = OPT.optimize_budget(_opt_ab, _cfg, _ht_ab, low=0.6, high=1.4)
_opt_ab.set_posterior(_sub)
_r_fast2 = OPT.optimize_budget(_opt_ab, _cfg, _ht_ab, low=0.6, high=1.4)
_opt_ab.set_posterior(art["idata"])
_ab_dev = float(np.abs(np.asarray(_r_fast2["budgets"])
                       - np.asarray(_r_full2["budgets"])).max())
_ab_rel = _ab_dev / max(1.0, float(np.max(np.asarray(_r_full2["budgets"]))))
check(f"子集与全量的最优分配偏差 < 1%（实测 {_ab_rel:.5%}）", _ab_rel < 0.01,
      f"dev={_ab_dev:,.0f} 元")
check("两者提升幅度差 < 0.1 个百分点",
      abs(_r_fast2["lift_pct"] - _r_full2["lift_pct"]) < 0.1,
      f"{_r_fast2['lift_pct']:.4f}% vs {_r_full2['lift_pct']:.4f}%")

# ★ 进出必须对称：这是「加速」最危险的副作用 —— 忘了恢复，
#    后续所有不确定性分析都会静默地用子集算，且完全不报错。
with OPT.fast_solve(_opt_ab, art["idata"]) as _fs:
    _draws_in = int(_opt_ab.idata.posterior.sizes["draw"])
    check("块内：优化器后验按需换成子集（样本不足时保持原样）",
          _draws_in == (OPT.FAST_DRAWS if _FAST_OK else _N_DRAW), f"draw={_draws_in}")
check("★ 块外：后验已自动恢复为原状（否则区间会静默失真）",
      int(_opt_ab.idata.posterior.sizes["draw"]) == _N_DRAW,
      f"draw={int(_opt_ab.idata.posterior.sizes['draw'])}")
check("fast_solve 状态自洽（启用则必须恢复；未启用则无需恢复、也不谎报）",
      (bool(_fs.get("applied")) and _fs.get("restored") is True)
      or (not _fs.get("applied") and not _fs.get("restored")),
      str(_fs))

try:
    with OPT.fast_solve(_opt_ab, art["idata"]):
        raise RuntimeError("故意抛异常，检验 finally 是否生效")
except RuntimeError:
    pass
check("★ 块内抛异常时仍然恢复原状后验（finally 生效）",
      int(_opt_ab.idata.posterior.sizes["draw"]) == _N_DRAW,
      f"draw={int(_opt_ab.idata.posterior.sizes['draw'])}")

# 残差渠道诊断：唯一不卡边界的渠道，其取值是预算守恒的算术结果
_res_ch = _r_ok.get("residual_channel")
check("残差渠道被识别出来（唯一不卡边界 → 取值是算术残差）",
      _res_ch is not None and len(_r_ok.get("interior_channels", [])) == 1,
      f"residual={_res_ch} interior={_r_ok.get('interior_channels')}")
if _res_ch is not None:
    _ri = list(_cfg.channel_cols).index(_res_ch)
    _others_corner = all(bool(_r_ok["at_lo"][i] or _r_ok["at_hi"][i])
                         for i in range(len(_cfg.channel_cols)) if i != _ri)
    check("除残差渠道外其余渠道全卡边界（这才是残差成立的前提）",
          _others_corner, str(np.round(_r_ok["budgets"] / 1e4, 1)))
    check("残差提示文字已生成且点名了该渠道",
          _res_ch in _r_ok.get("residual_note", ""), _r_ok.get("residual_note", "")[:60])
    # 真正可证伪的检验：总预算 +5%，若「残差」成立，其余四个渠道会原位不动，
    # 增量全部落在残差渠道身上 —— 证明它的取值来自算术，而非模型测算。
    _r_up = OPT.optimize_budget(_opt_ok, _cfg, _ht_ab, low=0.6, high=1.4,
                                total_budget=float(_ht_ab.sum()) * 1.05)
    _b0 = np.asarray(_r_ok["budgets"], dtype=float)
    _b1 = np.asarray(_r_up["budgets"], dtype=float)
    _d_others = max(abs(_b1[i] - _b0[i]) for i in range(len(_cfg.channel_cols)) if i != _ri)
    _d_res = _b1[_ri] - _b0[_ri]
    check("★ 总预算 +5%：其余渠道原位不动、增量全被残差渠道吸收（确证算术性）",
          _d_others < 1.0 and abs(_d_res - float(_ht_ab.sum()) * 0.05) < 1.0,
          f"Δothers_max={_d_others:,.2f} 元  Δ残差={_d_res:,.0f} 元")

# 结构被钉死 → 换后验样本分配几乎不变（这是「零波动」的真实原因）
_rb = OPT.robustness(_mmm, _cfg, _df, art["idata"], _w, _ht_ab,
                     low=0.6, high=1.4, n_draws=6, use_existing_opt=_opt_ab)
check("稳健性可运行且全部样本求解成功",
      _rb.get("ok") is True and _rb["n_ok"] >= 1, str(_rb.get("n_ok")))
if _rb.get("ok"):
    check("稳健性产出「逐渠道边界方向」而不只是标准差",
          {"stable_hi", "stable_lo", "unstable", "always_corner"} <= set(_rb), "")
    check("★ 至少有一个渠道在所有样本里方向一致（结构性结论，可直接进决策）",
          len(_rb.get("stable_hi") or []) + len(_rb.get("stable_lo") or []) >= 1,
          f"加投={_rb.get('stable_hi')} 减投={_rb.get('stable_lo')} "
          f"方向不稳={_rb.get('unstable')}")
    if _res_ch is not None:
        _bd = _rb["table"].set_index("渠道")
        check("残差渠道在所有样本里都不卡边界（与残差诊断自洽）",
              int(_bd.loc[_res_ch, "未卡边界"]) == int(_rb["n_ok"]),
              str(_bd["未卡边界"].to_dict()))
    check("方向列的取值都在预期集合内（没有未定义状态）",
          set(_rb["table"]["方向"]) <= {"结构性加投", "结构性减投",
                                       "方向不稳定", "方向偏弱", "算术残差"},
          str(set(_rb["table"]["方向"])))
check("稳健性结束后优化器后验已恢复原状",
      int(_opt_ab.idata.posterior.sizes["draw"]) == _N_DRAW,
      f"draw={int(_opt_ab.idata.posterior.sizes['draw'])}")


# ============================================================
# 汇总
# ============================================================
section("结果汇总")
n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
n_fail = len(_RESULTS) - n_pass
print(f"  通过 {n_pass} / {len(_RESULTS)}")
if n_fail:
    print("\n  失败项：")
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"    ✗ {name}   {detail}")
print(f"\n  结论：{'全部通过 ✅' if n_fail == 0 else f'{n_fail} 项失败 ❌'}")
sys.exit(1 if n_fail else 0)
