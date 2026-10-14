"""
字段规范与别名表（纯逻辑，不依赖 streamlit）。

业务背景：
    分析师拿到的投放数据表，列名没有统一标准。同一个"Facebook 周花费"，
    可能是 facebook_spend / Facebook Spend / fb_cost / FB花费 / 脸书广告费……
    如果应用只认一种写法，用户上传就会立刻报错，体验极差。

设计：
    为每个规范字段维护一组别名（全部小写、去空格/下划线后比较），
    匹配时做归一化，命中即可。别名表可以随时扩充，不影响其它模块。
"""
from __future__ import annotations

import re

# ------------------------------------------------------------
# 规范化字段名
# ------------------------------------------------------------
DATE_COL = "date"
TARGET_COL = "revenue"

# 5 个付费渠道（与 src/data_generator.py 的 CHANNELS 保持一致）
CHANNELS: list[str] = ["facebook", "google", "tiktok", "youtube", "kol"]

CHANNEL_CN: dict[str, str] = {
    "facebook": "Facebook",
    "google": "Google",
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "kol": "KOL",
}

# 控制变量（可选）
CONTROLS: list[str] = ["version_update", "competitor_activity", "holiday", "trend"]

CONTROL_CN: dict[str, str] = {
    "version_update": "版本更新",
    "competitor_activity": "竞品活动",
    "holiday": "节假日",
    "trend": "长期趋势",
}

# ------------------------------------------------------------
# 别名表：key = 规范名，value = 可接受的写法（已归一化的小写形式）
# ------------------------------------------------------------
DATE_ALIASES: list[str] = [
    "date", "日期", "时间", "周", "周次", "周序", "周序号", "week", "weekno",
    "weeknum", "weeknumber", "weekstart", "week_start", "weekending",
    "week_ending", "ds", "startdate", "start_date", "month", "月份", "period",
    "统计周期", "数据日期", "投放周",
]

TARGET_ALIASES: list[str] = [
    "revenue", "收入", "营收", "流水", "销售额", "销售收入", "总收入",
    "gmv", "sales", "income", "totalrevenue", "total_revenue", "revenuecny",
    "收入元", "流水元", "iap", "内购收入", "充值金额",
]

# 每个渠道的别名。注意：渠道名后面常带 spend / cost / 花费 等后缀，
# 这里显式列出常见组合；同时 data_io 还会做"后缀剥离"的二次匹配。
CHANNEL_ALIASES: dict[str, list[str]] = {
    "facebook": ["facebook", "facebookspend", "facebookcost", "facebook广告",
                 "facebook花费", "facebook投放", "fb", "fbspend", "fbcost",
                 "fb广告", "fb花费", "meta", "metaspend", "metacost",
                 "脸书", "脸书花费", "面书"],
    "google":   ["google", "googlespend", "googlecost", "googleads", "google广告",
                 "google花费", "google投放", "adwords", "googleadwords", "sem",
                 "谷歌", "谷歌花费", "谷歌广告", "adwords花费"],
    "tiktok":   ["tiktok", "tiktokspend", "tiktokcost", "tiktokads", "tiktok广告",
                 "tiktok花费", "tiktok投放", "tt", "ttspend", "douyin",
                 "抖音", "抖音花费", "tiktok国际版"],
    "youtube":  ["youtube", "youtubespend", "youtubecost", "youtubeads",
                 "youtube广告", "youtube花费", "youtube投放", "yt", "ytspend",
                 "油管", "油管花费", "yt广告"],
    "kol":      ["kol", "kolspend", "kolcost", "kol广告", "kol花费", "kol投放",
                 "kol合作", "达人", "达人花费", "网红", "网红营销", "influencer",
                 "influencerspend", "influencercost", "红人", "红人营销",
                 "cps", "网红推广"],
}

CONTROL_ALIASES: dict[str, list[str]] = {
    "version_update": ["versionupdate", "version_update", "版本更新", "版本",
                       "更新", "appupdate", "版本发布", "资料片"],
    "competitor_activity": ["competitoractivity", "competitor_activity", "竞品活动",
                            "竞品", "竞争对手活动", "竞品投放", "竞争活动"],
    "holiday": ["holiday", "节假日", "节日", "假期", "假日", "法定假期"],
    "trend": ["trend", "趋势", "长期趋势", "时间趋势", "线性趋势"],
}

# 数值后缀：匹配渠道列时允许出现（用于二次匹配兜底）
SPEND_SUFFIXES: tuple[str, ...] = (
    "spend", "spends", "cost", "costs", "花费", "投放", "投入", "费用",
    "支出", "广告费", "广告花费", "营销费用", "预算",
)


def norm_key(name: str) -> str:
    """
    列名归一化：用于别名匹配。
        小写 → 去掉空格/下划线/连字符/括号/单位符号
    例：'Facebook_Spend (元)' → 'facebookspend元' → 进一步剥离单位 → 'facebookspend'
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-\.\(\)\[\]（）【】]", "", s)
    s = re.sub(r"[（(]?(元|万元|rmb|cny|usd|\$|¥)[）)]?$", "", s)
    return s


def strip_spend_suffix(nkey: str) -> str:
    """剥离花费类后缀：'facebook花费' → 'facebook'。匹配不到则原样返回。"""
    for suf in SPEND_SUFFIXES:
        if nkey.endswith(suf) and len(nkey) > len(suf):
            return nkey[: -len(suf)]
    return nkey


# 预计算归一化后的别名集合，加速匹配
_NORM_DATE = {norm_key(a) for a in DATE_ALIASES}
_NORM_TARGET = {norm_key(a) for a in TARGET_ALIASES}
_NORM_CHANNEL: dict[str, set[str]] = {
    c: {norm_key(a) for a in al} | {c} for c, al in CHANNEL_ALIASES.items()}
_NORM_CONTROL: dict[str, set[str]] = {
    c: {norm_key(a) for a in al} | {norm_key(c)} for c, al in CONTROL_ALIASES.items()}

# 疑似可作为目标的"非花费"数值列（用于给出更友好的错误提示）
SPEND_LIKE_HINTS: tuple[str, ...] = ("spend", "cost", "花费", "投放", "广告费", "预算")


def match_date_column(columns) -> str | None:
    for c in columns:
        if norm_key(c) in _NORM_DATE:
            return c
    return None


def match_target_column(columns) -> str | None:
    for c in columns:
        if norm_key(c) in _NORM_TARGET:
            return c
    return None


def match_channel_columns(columns) -> dict[str, str]:
    """
    返回 {规范渠道名: 原始列名}。
    两轮匹配：① 别名表精确命中；② 剥离花费后缀后与渠道名比对。
    """
    found: dict[str, str] = {}
    used: set[str] = set()
    normed = {c: norm_key(c) for c in columns}

    # 第一轮：别名表
    for ch, aliases in _NORM_CHANNEL.items():
        for c in columns:
            if c in used:
                continue
            if normed[c] in aliases:
                found[ch] = c
                used.add(c)
                break

    # 第二轮：后缀剥离（例：'facebook花费' → 'facebook'）
    for ch in CHANNELS:
        if ch in found:
            continue
        for c in columns:
            if c in used:
                continue
            if strip_spend_suffix(normed[c]) == ch:
                found[ch] = c
                used.add(c)
                break
    return found


def match_control_columns(columns) -> dict[str, str]:
    found: dict[str, str] = {}
    used: set[str] = set()
    normed = {c: norm_key(c) for c in columns}
    for ctl, aliases in _NORM_CONTROL.items():
        for c in columns:
            if c in used:
                continue
            if normed[c] in aliases:
                found[ctl] = c
                used.add(c)
                break
    return found


def looks_like_spend_col(name: str) -> bool:
    nk = norm_key(name)
    return any(h in nk for h in SPEND_LIKE_HINTS)


if __name__ == "__main__":       # 快速验性别名匹配：python app/core/schema.py
    demo_cols = ["日期", "Facebook Spend", "google_cost", "TikTok花费", "youtube",
                 "KOL 投放 (元)", "收入", "版本更新", "holiday", "备注说明"]
    print("列名                 → 匹配结果")
    print(f"  日期列     : {match_date_column(demo_cols)}")
    print(f"  目标列     : {match_target_column(demo_cols)}")
    print(f"  渠道列     : {match_channel_columns(demo_cols)}")
    print(f"  控制列     : {match_control_columns(demo_cols)}")
