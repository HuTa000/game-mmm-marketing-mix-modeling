"""
数据模块（纯逻辑，不依赖 streamlit）—— 应用的数据入口。

两条数据路径：
    ① 内置模拟数据 : load_builtin()   —— 与阶段 2/3 完全同源（generate_game_data）
    ② 上传文件     : load_uploaded()  —— 用户自己的 CSV / Excel

设计要点（对应"上传必须兜底"的硬性要求）：
    - 读取阶段：编码嗅探（utf-8 / gbk / utf-16 / big5…）、分隔符嗅探、Excel 支持
    - 识别阶段：别名匹配（见 schema.py）→ 失败则扫描列内容推断日期列
    - 校验阶段：所有问题收敛为 ValidationIssue（error / warning / info 三级），
               绝不抛未捕获异常把界面搞成白屏
    - 标准化阶段：列名统一为 date / revenue / facebook… / 控制变量，
               使下游 mmm_core、response_curve 无需任何适配即可处理上传数据

关键约定：本模块**只报告、只修复可安全修复的问题**。
遇到 error 级问题（如没有目标列）不猜、不编，返回 report 让界面明确告知用户。
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from . import schema                       # 作为包导入（Streamlit 应用内）
except ImportError:                            # 直接运行：python app/core/data_io.py
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema

# ------------------------------------------------------------
# 常量
# ------------------------------------------------------------
MIN_ROWS_HARD = 8          # 少于这个行数无法建模 → error
MIN_ROWS_SOFT = 52         # 少于一年 → 季节性不可靠 → warning
ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "gbk", "gb18030", "utf-16", "big5", "latin-1")
SEP_CANDIDATES = (",", ";", "\t", "|")

# 全角数字/符号 → 半角（用于数值解析）
_FW_TABLE = str.maketrans("０１２３４５６７８９．－", "0123456789.-")


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Issue:
    """单条校验发现。level 决定界面用颜色/图标，hint 告诉用户怎么修。"""
    level: str          # "error" | "warning" | "info"
    code: str           # 机器可读的编号，便于自测断言
    title: str          # 一句话说明
    detail: str = ""    # 为什么
    hint: str = ""      # 怎么办

    def to_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "title": self.title,
                "detail": self.detail, "hint": self.hint}


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    mapping: dict = field(default_factory=dict)     # 规范名 -> 原始列名
    n_rows: int = 0
    n_cols: int = 0
    encoding: str = ""
    separator: str = ""
    ignored_columns: list[str] = field(default_factory=list)

    def add(self, level: str, code: str, title: str, detail: str = "", hint: str = "") -> None:
        self.issues.append(Issue(level, code, title, detail, hint))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "info"]

    @property
    def ok(self) -> bool:
        """是否可继续（无 error）。warning 不阻断。"""
        return not self.errors

    def counts(self) -> dict:
        return {"error": len(self.errors), "warning": len(self.warnings),
                "info": len(self.infos)}


@dataclass
class Dataset:
    """应用内统一的数据载体。df 的列名已标准化。"""
    df: pd.DataFrame
    date_col: str
    target_col: str
    channel_cols: list[str]
    control_cols: list[str]
    source: str                       # "builtin" | "upload"
    source_label: str                 # 展示用（文件名或"内置模拟数据"）
    report: ValidationReport
    fingerprint: str = ""
    granularity_days: float = 7.0

    # ---- 便捷属性 ----
    @property
    def n_rows(self) -> int:
        return int(len(self.df))

    @property
    def spend_cols(self) -> list[str]:
        return list(self.channel_cols)

    def date_range(self) -> tuple[str, str]:
        if self.n_rows == 0:
            return ("—", "—")
        d = pd.to_datetime(self.df[self.date_col])
        return (str(d.iloc[0])[:10], str(d.iloc[-1])[:10])

    def total_spend(self) -> float:
        return float(self.df[self.channel_cols].sum().sum())

    def total_revenue(self) -> float:
        return float(self.df[self.target_col].sum())

    def to_preview(self, n: int = 8) -> pd.DataFrame:
        """给界面展示用：列名中文化 + 金额转万元。"""
        cols = [self.date_col] + self.channel_cols + [self.target_col] + self.control_cols
        cols = [c for c in cols if c in self.df.columns]
        out = self.df[cols].head(n).copy()
        ren = {self.date_col: "日期", self.target_col: "收入(元)"}
        for c in self.channel_cols:
            ren[c] = f"{schema.CHANNEL_CN.get(c, c)}花费(元)"
        for c in self.control_cols:
            ren[c] = schema.CONTROL_CN.get(c, c)
        return out.rename(columns=ren)

    def summary(self) -> dict:
        """单行汇总指标，供 KPI 卡片使用。"""
        return {
            "行数(周)": self.n_rows,
            "时间范围": " ~ ".join(self.date_range()),
            "渠道数": len(self.channel_cols),
            "控制变量数": len(self.control_cols),
            "总花费(万元)": self.total_spend() / 1e4,
            "总收入(万元)": self.total_revenue() / 1e4,
            "整体 ROAS": (self.total_revenue() / self.total_spend()
                        if self.total_spend() > 0 else float("nan")),
            "数据粒度": f"{self.granularity_days:.0f} 天",
        }


# ============================================================
# 工具
# ============================================================
def _clean_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    清理列名：去首尾空白、剔除空列与无名列。
    返回 (清理后的 df, 被忽略的列名列表)
    """
    df = df.copy()
    kv = {}
    for c in df.columns:
        s = str(c).strip()
        if s.startswith("Unnamed:") or s == "" or s.lower() in ("nan", "none"):
            kv[c] = f"__unnamed_{len(kv)}__"
        else:
            kv[c] = s
    df = df.rename(columns=kv)

    # 全空列直接丢掉
    drop = [c for c in df.columns
            if c.startswith("__unnamed_") or df[c].isna().all()]
    ignored = [c for c in drop]
    df = df.drop(columns=drop)
    return df, ignored


def _try_parse_dates(s: pd.Series) -> tuple[pd.Series, float, bool]:
    """尝试把一列解析为日期。返回 (解析结果, 成功率, 是否使用了 dayfirst)。"""
    ser = s.astype(str).str.strip()
    if len(ser) == 0:
        return pd.to_datetime(ser, errors="coerce"), 0.0, False

    best_parsed, best_rate, best_dayfirst = None, -1.0, False
    for dayfirst in (False, True):
        parsed = pd.to_datetime(ser, errors="coerce", dayfirst=dayfirst)
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_parsed, best_rate, best_dayfirst = parsed, rate, dayfirst
        if rate >= 0.999:
            break
    return best_parsed, best_rate, best_dayfirst


def _to_numeric(s: pd.Series) -> tuple[pd.Series, int]:
    """
    转数值：兼容千分位逗号、货币符号、百分号、全角数字、会计负数写法 (1,234)，
    以及中文数量级（1.5万 → 15000、2.3亿 → 230000000）。
    返回 (数值列, 无法转换的非空单元格个数)

    注意：早期版本用 `Series.translate`（并不存在该方法）导致字符串列一转换就抛
    AttributeError —— 因为数值列会走上面的提前返回分支，所以只有"混入文字的列"
    才会暴露这个 bug，属于典型的"只在脏数据下崩溃"。
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce"), int(s.isna().sum())

    raw = s.astype(str).str.strip()
    t = raw.map(lambda v: v.translate(_FW_TABLE))          # 全角数字/符号 → 半角

    neg = t.str.startswith("(") & t.str.endswith(")")      # 会计负数 (1,234)
    # 必须先剥掉括号，否则 to_numeric("(1200)") 会得到 NaN 而不是 -1200
    t = t.str.replace(r"[()（）]", "", regex=True)

    scale = pd.Series(1.0, index=t.index)
    has_wan = t.str.contains(r"万元|万|萬", regex=True, na=False)
    has_yi = t.str.contains(r"亿元|亿", regex=True, na=False)
    scale = scale.mask(has_wan, 1e4).mask(has_yi, 1e8)

    t = t.str.replace(r"[¥$€£￥]|人民币|RMB|CNY|USD|亿元|万元|亿|万|萬|元",
                      "", regex=True)
    t = t.str.replace(r"[,\s　]", "", regex=True)     # 半角逗号/空白 + 全角空格(U+3000)
    t = t.str.replace("%", "", regex=False)

    num = pd.to_numeric(t, errors="coerce") * scale
    num = num.mask(neg & num.notna(), -num)
    n_bad = int((num.isna() & s.notna() & (raw != "")).sum())
    return num, n_bad


def _looks_like_week_index(s: pd.Series) -> bool:
    """
    判断一列是否是"周次序号"（1,2,3,…,N 的连续整数）。
    按**值**判断而不是按列名 —— 这样列名叫 period / 期数 / t 也能被识别。
    """
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 3:
        return False
    arr = v.to_numpy(dtype=float)
    if not np.all(np.isclose(np.mod(arr, 1), 0.0)):     # 必须是整数
        return False
    iv = np.sort(arr.astype("int64"))
    if iv[0] < -1 or iv[-1] > 5000:
        return False
    return bool(np.array_equal(iv, np.arange(iv[0], iv[0] + len(iv))))   # 必须严格连续


def _infer_frequency(dates: pd.Series) -> tuple[float, bool]:
    """推断粒度（天）与是否存在不规则间隔。"""
    d = pd.to_datetime(dates).sort_values().dropna().unique()
    if len(d) < 3:
        return 7.0, False
    diffs = np.diff(np.asarray(d, dtype="datetime64[D]").astype("int64"))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 7.0, False
    med = float(np.median(diffs))
    irregular = bool((diffs > med * 1.5).any())
    return med, irregular


# ============================================================
# 文件读取（编码 + 分隔符嗅探）
# ============================================================
def read_table(data: bytes | str, filename: str = "") -> tuple[pd.DataFrame, str, str, list[Issue]]:
    """
    把上传的字节流读成 DataFrame。
    返回 (df, encoding, separator, issues)。读不出来时 df 为空且 issues 含 error。
    """
    issues: list[Issue] = []
    name = (filename or "").lower()

    # ---- Excel ----
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            engine = "openpyxl" if not name.endswith(".xls") else "xlrd"
            df = pd.read_excel(io.BytesIO(data if isinstance(data, bytes) else data.encode()),
                               engine=engine)
            issues.append(Issue("info", "READ_EXCEL", f"已按 Excel 读取（engine={engine}）"))
            return df, f"excel:{engine}", "-", issues
        except ImportError as e:
            issues.append(Issue(
                "error", "READ_EXCEL_NO_ENGINE",
                "缺少 Excel 解析库，无法读取该文件",
                f"需要 openpyxl（.xlsx）或 xlrd（.xls）：{e}",
                "改为另存为 CSV 后上传，或在环境中安装 openpyxl"))
            return pd.DataFrame(), "", "", issues
        except Exception as e:
            issues.append(Issue(
                "error", "READ_EXCEL_FAIL", "Excel 文件解析失败",
                f"{type(e).__name__}: {e}",
                "确认文件未加密、未损坏，且第一行是表头"))
            return pd.DataFrame(), "", "", issues

    # ---- 文本类 ----
    if isinstance(data, bytes):
        text, enc = None, ""
        for e in ENCODING_CANDIDATES:
            try:
                text = data.decode(e)
                enc = e
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            issues.append(Issue(
                "error", "READ_ENCODING", "无法识别文件编码",
                "尝试过 utf-8 / gbk / utf-16 / big5 / latin-1 均失败",
                "请用 Excel 另存为「CSV UTF-8」后重试"))
            return pd.DataFrame(), "", "", issues
    else:
        text, enc = str(data), "str"

    if text.strip() == "":
        issues.append(Issue("error", "READ_EMPTY", "文件内容为空",
                            hint="请确认导出的文件里确实有数据"))
        return pd.DataFrame(), enc, "", issues

    if enc not in ("utf-8-sig", "utf-8", "str"):
        issues.append(Issue(
            "warning", "READ_NON_UTF8", f"检测到非 UTF-8 编码（{enc}），已自动按该编码读取",
            "中文环境下常见于 Excel 直接另存 CSV（GBK/GB18030）",
            "如需更稳妥，可在 Excel 中另存为「CSV UTF-8」"))

    # 分隔符嗅探：谁切出来的列最多就用谁
    best_df, best_sep, best_ncol = None, ",", 0
    for sep in SEP_CANDIDATES:
        try:
            d = pd.read_csv(io.StringIO(text), sep=sep, nrows=200, engine="python",
                            skip_blank_lines=True)
        except Exception:
            continue
        if d.shape[1] > best_ncol:
            best_df, best_sep, best_ncol = d, sep, d.shape[1]
    if best_df is None or best_ncol <= 1:
        # 可能是单列文本或纯乱码
        try:
            best_df = pd.read_csv(io.StringIO(text), sep=",", engine="python")
            best_sep = ","
        except Exception as e:
            issues.append(Issue("error", "READ_PARSE", "CSV 解析失败",
                                f"{type(e).__name__}: {e}",
                                "确认是标准 CSV（逗号分隔、首行表头）"))
            return pd.DataFrame(), enc, "", issues
        if best_df.shape[1] <= 1:
            issues.append(Issue(
                "warning", "READ_SINGLE_COL", "只解析出 1 列，可能分隔符不是逗号/分号/制表符",
                hint="检查是否用了其它分隔符（如空格、中文逗号「，」）"))
    else:
        # 用确定的 sep 完整读取
        best_df = pd.read_csv(io.StringIO(text), sep=best_sep, engine="python",
                              skip_blank_lines=True)

    if best_sep != ",":
        issues.append(Issue("info", "READ_SEP", f"检测到分隔符为 {best_sep!r}"))
    return best_df, enc, best_sep, issues


# ============================================================
# 主流程：校验 + 标准化
# ============================================================
def validate_and_standardize(
    df_raw: pd.DataFrame,
    *,
    source: str,
    source_label: str,
    report: ValidationReport | None = None,
    regularize: bool = True,
) -> Dataset:
    """
    把任意来源的表标准化为应用内部契约：
        列名 = date / revenue / facebook…kol / version_update…trend
    所有问题写进 report，绝不抛异常。
    """
    rep = report or ValidationReport()
    rep.n_rows, rep.n_cols = int(df_raw.shape[0]), int(df_raw.shape[1])

    # ---------- 0. 基本形状 ----------
    if df_raw.shape[0] == 0:
        rep.add("error", "NO_ROWS", "表中没有数据行",
                "只读到表头或空文件", "确认导出时包含数据行")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], source, source_label, rep)
    if df_raw.shape[1] < 2:
        rep.add("error", "TOO_FEW_COLS", "有效列太少",
                f"仅解析出 {df_raw.shape[1]} 列，无法识别日期/渠道/收入",
                "确认分隔符正确，且表头与数据在同一张表里")

    df, ignored = _clean_columns(df_raw)
    rep.ignored_columns = ignored
    if ignored:
        rep.add("info", "IGNORED_COLS", f"已忽略 {len(ignored)} 个空列/无名列")

    # ---------- 1. 日期列 ----------
    # 顺序很重要：先按列名匹配并解析 → 解析不动再按内容另找 → 都不行才用"连续整数序号"
    # 合成周日期。早期版本把序号兜底放在别名匹配之外，导致「列名 week、内容是 1,2,3」
    # 的表被误判为"没有日期列" —— 这是实测出来的顺序问题。
    date_src = schema.match_date_column(df.columns)
    how = "alias" if date_src else None
    parsed, rate, dayfirst = None, -1.0, False

    if date_src is not None:
        parsed, rate, dayfirst = _try_parse_dates(df[date_src])

    # 别名命中的列解析不动（例：列名 week，内容是 1,2,3…）→ 换一列按内容找
    if date_src is not None and rate < 0.8:
        alt = None
        for c in df.columns:
            if c == date_src or pd.api.types.is_numeric_dtype(df[c]):
                continue
            p2, r2, d2 = _try_parse_dates(df[c])
            if alt is None or r2 > alt[1]:
                alt = (c, r2, p2, d2)
        if alt is not None and alt[1] > rate:
            rep.add("info", "DATE_ALT_COL",
                    f"列「{date_src}」无法解析为日期，已改用「{alt[0]}」",
                    f"新列的日期解析成功率 {alt[1] * 100:.0f}%")
            date_src, rate, parsed, dayfirst, how = alt[0], alt[1], alt[2], alt[3], "content"

    # 仍无可用日期 → 找"连续整数序号"列，合成周日期
    if date_src is None or rate < 0.8:
        wk = None
        order = ([date_src] if date_src is not None else []) + \
                [c for c in df.columns if c != date_src]
        for c in order:
            if _looks_like_week_index(df[c]):
                wk = c
                break
        if wk is None:
            rep.add("error", "NO_DATE_COL", "找不到可用的日期列",
                    "既没有列名像日期，也没有哪一列的内容能解析成日期"
                    + (f"（「{date_src}」解析成功率仅 {max(0, rate) * 100:.0f}%）"
                       if date_src else ""),
                    "请提供形如 2023-01-02 的周日期列（列名可用 date / 日期 / week）；"
                    "若只有周次序号（1、2、3…）请补充起始日期")
            return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                           [], [], source, source_label, rep)
        n = pd.to_numeric(df[wk], errors="coerce")
        start = pd.Timestamp("2023-01-02")
        df[schema.DATE_COL] = start + pd.to_timedelta((n - n.min()) * 7, unit="D")
        if wk != schema.DATE_COL:
            df = df.drop(columns=[wk])
        date_src, how = schema.DATE_COL, "week_index"
        rep.add("warning", "DATE_FROM_WEEK_INDEX",
                f"未找到日期列，已按「{wk}」的连续序号合成周日期",
                "合成起点假定为 2023-01-02（周一）。这不影响媒体贡献与 ROI 的相对结论，"
                "但会让季节项与真实日历错位",
                "建议补充真实日期列后重新上传，否则节假日/季节性效应会对应错")
    else:
        if how == "alias":
            rep.add("info", "DATE_BY_ALIAS",
                    f"日期列识别为「{date_src}」（按列名匹配）")
        else:
            rep.add("info", "DATE_BY_CONTENT",
                    f"日期列识别为「{date_src}」（按列内容推断，成功率 {rate * 100:.0f}%）")
        if dayfirst and rate < 0.999:
            rep.add("info", "DATE_DAYFIRST", "日期按「日/月/年」解析",
                    hint="若结果不对，请把日期统一改成 YYYY-MM-DD 格式")

    parsed_dates, rate, dayfirst = _try_parse_dates(df[date_src])
    if rate < 0.9:
        bad = int(parsed_dates.isna().sum())
        rep.add("warning", "DATE_PARTIAL", f"有 {bad} 行日期无法解析，已丢弃这些行",
                hint="检查这些行的日期格式是否与其它行不一致")
    df = df.loc[parsed_dates.notna()].copy()
    df[schema.DATE_COL] = pd.to_datetime(parsed_dates.dropna()).values
    if date_src != schema.DATE_COL:
        df = df.drop(columns=[date_src])
    rep.mapping["date"] = date_src

    if df.shape[0] == 0:
        rep.add("error", "NO_VALID_ROWS", "所有行的日期都解析失败",
                hint="请把日期列统一为 YYYY-MM-DD 或 YYYY/M/D")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], source, source_label, rep)

    # ---------- 2. 目标列 ----------
    target_src = schema.match_target_column(df.columns)
    if target_src is None:
        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and schema.norm_key(c) != schema.norm_key(schema.DATE_COL)]
        non_spend = [c for c in numeric_cols if not schema.looks_like_spend_col(c)]
        rep.add("error", "NO_TARGET_COL", "找不到收入/目标列",
                f"按别名表未匹配到收入列；数值列候选："
                f"{'、'.join(map(str, non_spend[:6])) or '（无）'}",
                "请提供收入列，列名可用 revenue / 收入 / 流水 / GMV")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], source, source_label, rep)
    rep.mapping["revenue"] = target_src

    # ---------- 3. 渠道列 ----------
    ch_map = schema.match_channel_columns([c for c in df.columns if c != target_src])
    if len(ch_map) == 0:
        rep.add("error", "NO_CHANNEL_COL", "找不到任何渠道花费列",
                "需要至少 2 个渠道的花费列（如 facebook_spend、google_cost）",
                "列名可用 facebook/google/tiktok/youtube/kol 加 spend/cost/花费 后缀")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], source, source_label, rep)
    if len(ch_map) == 1:
        rep.add("error", "TOO_FEW_CHANNELS", "只识别到 1 个渠道",
                f"识别到：{list(ch_map.values())[0]}",
                "MMM 至少需要 2 个渠道才能做预算分配")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], source, source_label, rep)
    missing_ch = [c for c in schema.CHANNELS if c not in ch_map]
    if missing_ch:
        rep.add("warning", "MISSING_CHANNELS",
                f"缺少 {len(missing_ch)} 个渠道，将只用识别到的 {len(ch_map)} 个建模",
                f"缺失：{'、'.join(schema.CHANNEL_CN[c] for c in missing_ch)}；"
                f"本次使用：{'、'.join(schema.CHANNEL_CN[c] for c in ch_map)}",
                "若这些渠道确实有投放，请补齐花费列后重新上传")
    rep.mapping.update({c: v for c, v in ch_map.items()})

    # ---------- 4. 控制变量（可选） ----------
    ctl_map = schema.match_control_columns(
        [c for c in df.columns if c != target_src and c not in ch_map.values()])
    rep.mapping.update(ctl_map)
    if len(ctl_map) == 0:
        rep.add("info", "NO_CONTROL_COL",
                "未识别到控制变量，将仅用 媒体 + 季节 + 基线 建模",
                "缺少控制变量会削弱模型对内生性/遗漏变量的防护",
                "若能提供，列名可用 holiday、version_update、trend 等")
    elif len(ctl_map) < len(schema.CONTROLS):
        miss = [schema.CONTROL_CN[c] for c in schema.CONTROLS if c not in ctl_map]
        rep.add("info", "PARTIAL_CONTROL",
                f"识别到 {len(ctl_map)}/{len(schema.CONTROLS)} 个控制变量",
                f"未识别：{'、'.join(miss)}")

    # ---------- 5. 组装 + 数值化 ----------
    build = {schema.DATE_COL: df[schema.DATE_COL]}
    for ch, src in ch_map.items():
        build[ch] = df[src]
    build[schema.TARGET_COL] = df[target_src]
    for ctl, src in ctl_map.items():
        build[ctl] = df[src]
    out = pd.DataFrame(build)

    total_bad = 0
    for c in out.columns:
        if c == schema.DATE_COL:
            continue
        num, n_bad = _to_numeric(out[c])
        out[c] = num
        total_bad += n_bad
    if total_bad > 0:
        rep.add("warning", "COERCION", f"有 {total_bad} 个单元格无法转为数值，已置为空值",
                "常见原因：单元格里混入了文字、货币符号或说明性备注",
                "这些位置随后按规则填补（花费补 0、收入与趋势类线性插值）")

    # ---------- 6. 排序 + 去重 + 规整 ----------
    out = out.sort_values(schema.DATE_COL).reset_index(drop=True)
    dup = int(out[schema.DATE_COL].duplicated().sum())
    if dup > 0:
        agg = {c: "sum" for c in out.columns if c != schema.DATE_COL}
        out = out.groupby(schema.DATE_COL, as_index=False).agg(agg)
        rep.add("warning", "DUP_DATES", f"存在 {dup} 个重复日期，已按日期合并（数值列求和）",
                hint="若重复行代表不同区域/产品线，请先拆分再上传")

    freq, irregular = _infer_frequency(out[schema.DATE_COL])
    gran = float(freq) if freq > 0 else 7.0
    rep.add("info", "GRANULARITY", f"推断数据粒度为约 {gran:.0f} 天",
            hint="MMM 按周建模；日粒度数据建议先聚合成周再上传")

    if regularize and irregular and out.shape[0] >= 3:
        full = pd.date_range(out[schema.DATE_COL].min(), out[schema.DATE_COL].max(),
                             freq=f"{int(round(gran))}D")
        before = out.shape[0]
        out = (out.set_index(schema.DATE_COL)
                  .reindex(full)
                  .rename_axis(schema.DATE_COL)
                  .reset_index())
        ch_cols = [c for c in ch_map if c in out.columns]
        ctl_cols = [c for c in ctl_map if c in out.columns]
        for c in ch_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        for c in [schema.TARGET_COL] + ctl_cols:
            out[c] = (pd.to_numeric(out[c], errors="coerce")
                        .interpolate(limit_direction="both"))
        added = out.shape[0] - before
        if added > 0:
            rep.add("warning", "TIME_GAP", f"时间轴存在间断，已补齐 {added} 个缺失周期",
                    f"补齐规则：花费补 0；收入与控制变量线性插值（补齐后共 {out.shape[0]} 行）",
                    "若这些周期实际有数据，建议在源表里补全后再上传")

    # ---------- 7. 数据质量检查 ----------
    n = out.shape[0]
    if n < MIN_ROWS_HARD:
        rep.add("error", "TOO_FEW_ROWS", f"数据只有 {n} 行，无法建模",
                f"至少需要 {MIN_ROWS_HARD} 行，建议不少于 {MIN_ROWS_SOFT} 周",
                "请补充更长周期的数据后重试")
    elif n < MIN_ROWS_SOFT:
        rep.add("warning", "SHORT_HISTORY", f"数据只有 {n} 周（少于 1 年）",
                "年度季节性无法被可靠估计，季节项会主要靠先验约束",
                "建议补充到至少 104 周（2 年）以获得稳定结果")

    tgt = out[schema.TARGET_COL]
    n_na_t = int(tgt.isna().sum())
    if n_na_t > 0:
        out[schema.TARGET_COL] = tgt.interpolate(limit_direction="both")
        rep.add("warning", "TARGET_NA", f"收入列有 {n_na_t} 个缺失值，已线性插值填补")
    if (out[schema.TARGET_COL] <= 0).any():
        n_bad = int((out[schema.TARGET_COL] <= 0).sum())
        rep.add("warning", "TARGET_NONPOS", f"收入列有 {n_bad} 个非正值（0 或负数）",
                "MMM 的收益曲线假设收入为正；非正值会被模型当作异常",
                "请核查是否为缺数占位（如用 0 表示未统计）")

    ch_cols_final = [c for c in ch_map if c in out.columns]
    for c in ch_cols_final:
        col = out[c]
        n_na = int(col.isna().sum())
        if n_na > 0:
            out[c] = col.fillna(0.0)
        n_neg = int((col < 0).sum())
        if n_neg > 0:
            out.loc[out[c] < 0, c] = 0.0
            rep.add("warning", "SPEND_NEGATIVE", f"「{schema.CHANNEL_CN.get(c, c)}」有 {n_neg} 个负花费，已置 0",
                    hint="负花费通常来自退款冲销，建议在源表处理")
        if float(col.fillna(0).sum()) == 0.0:
            rep.add("warning", "CHANNEL_ALL_ZERO", f"「{schema.CHANNEL_CN.get(c, c)}」花费全为 0",
                    "该渠道无投放，其贡献与 ROI 无法估计",
                    "建议在设置里剔除该渠道，或补齐数据")
        n_zero_ratio = float((col.fillna(0) == 0).mean())
        if 0 < n_zero_ratio and n_zero_ratio > 0.6:
            rep.add("info", "CHANNEL_SPARSE",
                    f"「{schema.CHANNEL_CN.get(c, c)}」有 {n_zero_ratio*100:.0f}% 的周期花费为 0",
                    "稀疏投放会加大该渠道 Adstock/饱和参数的估计难度")

    for c in ctl_map:
        if c in out.columns and out[c].isna().any():
            out[c] = pd.to_numeric(out[c], errors="coerce").interpolate(limit_direction="both").fillna(0.0)

    out = out.reset_index(drop=True)
    ds = Dataset(df=out,
                 date_col=schema.DATE_COL,
                 target_col=schema.TARGET_COL,
                 channel_cols=ch_cols_final,
                 control_cols=[c for c in ctl_map if c in out.columns],
                 source=source,
                 source_label=source_label,
                 report=rep,
                 granularity_days=gran)
    ds.fingerprint = dataset_fingerprint(ds)
    return ds


def dataset_fingerprint(ds: Dataset) -> str:
    """
    数据指纹：用于缓存键。
    取标准化后核心列的数值（保留 4 位）参与哈希 —— 数值一变指纹就变，
    保证缓存不会把旧模型套到新数据上。
    """
    try:
        cols = [ds.date_col] + ds.channel_cols + [ds.target_col] + ds.control_cols
        cols = [c for c in cols if c in ds.df.columns]
        sub = ds.df[cols].copy()
        sub[ds.date_col] = pd.to_datetime(sub[ds.date_col]).astype("int64")
        for c in cols:
            if c != ds.date_col:
                sub[c] = pd.to_numeric(sub[c], errors="coerce").round(4)
        payload = sub.to_csv(index=False, lineterminator="\n")
        meta = f"|{ds.source}|{len(ds.channel_cols)}|{len(ds.control_cols)}"
        return hashlib.sha1((payload + meta).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return hashlib.sha1(str(len(ds.df)).encode()).hexdigest()[:16]


# ============================================================
# 两条数据入口
# ============================================================
def load_builtin(n_weeks: int = 156, seed: int = 2024,
                 with_controls: bool = True) -> Dataset:
    """内置模拟数据 —— 与阶段 2/3 完全同源，保证应用结果与文档一致。"""
    import sys
    if os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "src") not in sys.path:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "src"))
    from data_generator import generate_game_data

    rep = ValidationReport()
    df = generate_game_data(n_weeks=n_weeks, seed=seed, with_controls=with_controls)
    rep.add("info", "BUILTIN", "已载入内置模拟数据（带真值标注）",
            f"generate_game_data(n_weeks={n_weeks}, seed={seed})；"
            f"含 _true_contrib_* 与 _true_baseline 真值列，可用于验证模型是否估对",
            "内置数据仅用于演示与验证；换成真实业务数据时请走「上传数据」")
    return validate_and_standardize(
        df, source="builtin", source_label=f"内置模拟数据（{n_weeks} 周 · seed {seed}）",
        report=rep)


def load_uploaded(data: bytes, filename: str) -> Dataset:
    """
    上传文件入口。**任何异常都会转成 error 级 issue**，不会向上抛出，
    保证界面永远能给出可读的提示而不是白屏。
    """
    rep = ValidationReport()
    label = os.path.basename(filename or "上传文件")
    try:
        df, enc, sep, read_issues = read_table(data, filename)
    except Exception as e:      # 兜底：连 read_table 都炸了的极端情况
        rep.add("error", "READ_UNKNOWN", "文件读取出现未预料的错误",
                f"{type(e).__name__}: {e}")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], "upload", label, rep)

    for it in read_issues:
        rep.add(it.level, it.code, it.title, it.detail, it.hint)
        if it.level == "error":
            return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                           [], [], "upload", label, rep)
    rep.encoding, rep.separator = enc, sep

    if df is None or df.shape[0] == 0:
        rep.add("error", "READ_NO_TABLE", "没能从文件中读出任何表格内容",
                hint="确认文件是标准 CSV/Excel，且第一行是列名")
        return Dataset(pd.DataFrame(), schema.DATE_COL, schema.TARGET_COL,
                       [], [], "upload", label, rep)

    return validate_and_standardize(df, source="upload", source_label=label, report=rep)


# ============================================================
# 自测用：从磁盘路径加载（模拟"上传"）
# ============================================================
def load_path(path: str) -> Dataset:
    with open(path, "rb") as fh:
        return load_uploaded(fh.read(), os.path.basename(path))


if __name__ == "__main__":
    ds = load_builtin()
    print(f"内置数据: {ds.n_rows} 行 × {len(ds.channel_cols)} 渠道 | 指纹 {ds.fingerprint}")
    print("汇总:", {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in ds.summary().items()})
    print("问题清单:", ds.report.counts())
