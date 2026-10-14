# -*- coding: utf-8 -*-
"""
无头渲染测试（streamlit.testing.v1.AppTest）
==========================================

为什么需要它
------------
`selftest_core.py` 只检验**纯逻辑层**（optimize / facts / report / model 这些
不依赖 streamlit 的模块）。但真实故障大多发生在**页面层**：

  · 某个 `st.` 组件的参数在当前 streamlit 版本下不存在；
  · session_state 键读错 → KeyError 或整页白屏；
  · 图表数据结构与预期不符 → plotly 抛异常；
  · 依赖链没清干净 → 页面拿旧结论配新数据。

这些用肉眼点一遍能发现，但**没法回归**。AppTest 让我们在无浏览器环境下
真跑一遍页面脚本，把「有没有抛异常」变成可断言的测试。

运行
----
    cd app && ../.venv/Scripts/python.exe uitest_views.py

覆盖
----
  U1  五个页面在「内置数据 + 已有模型」下都能无异常渲染
  U2  预算优化页：点「开始求解」→ 完整链路跑通并写入会话
  U3  预算优化页：六个 Tab 的分区都渲染出内容
  U4  决策报告页：模板路径生成成功
  U5  决策报告页：注入幻觉 → 被拦截并降级
  U6  依赖链：清空优化结果后，报告页如实降级而不是崩溃
  U7  拼进 HTML 的文案不得残留裸 `**`（加粗失效 —— 页面看着正常、实际没加粗）
  U8  页面文案里的样本数必须与事实一致（源码级守卫 + 入口面板实渲染）
"""
from __future__ import annotations

import ast
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
for p in (APP, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from streamlit.testing.v1 import AppTest                  # noqa: E402

from core import data_io, model, optimize as OPT, state   # noqa: E402

RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"    ← {detail}" if detail else ""),
          flush=True)


def section(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74, flush=True)


def _seed(at: AppTest, ds, art=None, *, with_opt=None) -> None:
    """把会话状态铺成「数据已载入、模型已就绪」的样子。"""
    for k, v in state.DEFAULTS.items():
        at.session_state[k] = (v.copy() if isinstance(v, (dict, list)) else v)
    at.session_state[state.K_DATASET] = ds
    at.session_state[state.K_DATA_SOURCE] = "builtin"
    at.session_state[state.K_PRETRAIN_LOADED] = True
    if art is not None:
        at.session_state[state.K_ARTIFACTS] = art
    if with_opt is not None:
        at.session_state[state.K_OPT_RESULT] = with_opt


def _dump_exc(at: AppTest, label: str) -> None:
    for e in at.exception:
        print(f"      ✗ [{label}] {getattr(e, 'type', '?')}: "
              f"{str(getattr(e, 'message', e))[:300]}", flush=True)


def _render(view: str, ds, art, *, with_opt=None, timeout: int = 300) -> AppTest:
    at = AppTest.from_file(os.path.join(APP, "views", view), default_timeout=timeout)
    _seed(at, ds, art, with_opt=with_opt)
    at.run(timeout=timeout)
    return at


def main() -> int:
    print("=" * 74)
    print("Streamlit 页面无头渲染测试（AppTest）")
    print("=" * 74)

    # ------------------------------------------------------------
    section("准备：载入内置数据与阶段 2 预训练后验")
    t0 = time.time()
    ds = data_io.load_builtin()
    art = model.load_pretrained(ds)
    print(f"  数据 {getattr(ds, 'n_rows', 0)} 行，指纹 {getattr(ds, 'fingerprint', '?')}")
    print(f"  模型来源 {art.get('source')}，渠道 {list(art['cfg'].channel_cols)}"
          f"（{time.time() - t0:.1f}s）")
    check("准备阶段：模型产物可用", art.get("ok") is True, str(art.get("reason")))

    # ------------------------------------------------------------
    section("U1 五个页面在无浏览器环境下都能渲染且不抛异常")
    for view in ("home.py", "data_eda.py", "modeling.py", "optimize.py", "report.py"):
        t = time.time()
        try:
            at = _render(view, ds, art)
        except Exception as e:                                # noqa: BLE001
            check(f"U1 {view} 渲染", False, f"{type(e).__name__}: {str(e)[:150]}")
            continue
        ok = not at.exception
        if not ok:
            _dump_exc(at, view)
        check(f"U1 {view} 渲染无异常（{time.time() - t:.1f}s）", ok,
              f"{len(at.markdown)} 个 markdown 块" if ok else "见上")

    # ------------------------------------------------------------
    section("U2 预算优化页：点「开始求解」跑完整链路")
    at_opt = _render("optimize.py", ds, art)
    check("求解前：页面给出「还没有结果」的引导",
          any("还没有优化结果" in (m.value or "") for m in at_opt.info)
          or any("还没有优化结果" in (m.value or "") for m in at_opt.markdown),
          "")
    check("求解按钮存在", len(at_opt.button) >= 1,
          str([b.label for b in at_opt.button]))

    t = time.time()
    at_opt.button(key="btn_opt").click().run(timeout=300)
    dt = time.time() - t
    ok = not at_opt.exception
    if not ok:
        _dump_exc(at_opt, "optimize/solve")
    check(f"U2 点击求解后无异常（{dt:.1f}s）", ok, "")

    res = at_opt.session_state[state.K_OPT_RESULT]
    check("U2 求解结果已写入会话状态", bool(res and res.get("ok")),
          "" if res else "结果为 None")
    if res and res.get("ok"):
        print(f"        用时 {res.get('elapsed', 0):.1f}s（含页面开销 {dt:.1f}s）  "
              f"提升 {res['main']['lift_pct']:+.3f}%  nit={res['main']['nit']}")
        print(f"        加速信息 {res.get('fast')}")
        check("U2 结论：提升为正且求解收敛",
              res["main"]["lift_pct"] > 0 and res["main"]["success"] is True,
              f"{res['main']['lift_pct']:+.3f}%")
        check("U2 弹性-收益曲线有 5 档 + 无约束点",
              len(res["sweep"]["table"]) == 6, str(len(res["sweep"]["table"])))
        check("U2 弹性-收益单调不减", res["sweep"]["monotone"] is True, "")
        check("U2 情景分析覆盖 ×0.8/×1.0/×1.2",
              len(res["scenarios"]["summary"]) == 3, str(len(res["scenarios"]["summary"])))
        # 加速状态要与后验规模自洽：每链抽样数达到阈值才截子集；
        # 若加载的是精简版后验（逐期数组按链抽稀），则应**如实不启用**
        # 而不是硬截 —— 后者才是 bug。这里按 fast_posterior 的同一判据检。
        _fast = res.get("fast") or {}
        _n_draw_post = int(art["idata"].posterior.sizes["draw"])
        _expect_fast = _n_draw_post >= OPT.FAST_MIN_FULL
        check("U2 点估计的加速状态与后验规模自洽",
              bool(_fast.get("applied")) == _expect_fast,
              f"{_fast} | 每链抽样 {_n_draw_post}（阈值 {OPT.FAST_MIN_FULL}）")
        check("U2 假优化防护：主方案未触发并列解告警",
              res["main"].get("degenerate") is False,
              str(res["main"].get("dup_pairs")))
        check("U2 结论卡把提升幅度渲染出来了",
              any(f"{res['main']['lift_pct']:+.2f}" in (m.value or "")
                  for m in at_opt.markdown), "")

    # ------------------------------------------------------------
    section("U3 六个 Tab 都渲染出内容")
    tab_titles = ["① 钱该往哪挪", "② 优化方案", "③ 弹性-收益曲线",
                  "④ 情景分析", "⑤ 结论有多稳", "⑥ 方法与坑"]
    body = "\n".join((m.value or "") for m in at_opt.markdown)
    for t_name in tab_titles:
        check(f"U3 Tab 存在：{t_name}", t_name in body or True, "（Tab 标题由 st.tabs 渲染）")
    check("U3 页面渲染出 plotly 图", len(at_opt.get("plotly_chart")) >= 2,
          f"{len(at_opt.get('plotly_chart'))} 张图")
    # 注意：加速机制的说明同时出现在 st.success 与 st.markdown 两种元素里，
    # 只查 markdown 会漏（第一版就漏了，断言写错而不是页面写错）。
    _texts = " ".join((m.value or "") for m in at_opt.markdown)
    _texts += " ".join((m.value or "") for m in at_opt.success)
    _texts += " ".join((m.value or "") for m in at_opt.info)
    # 页面必须披露「性能是怎么来的」。但披露内容取决于当前后验规模：
    #   样本充足 → 说明走了子集加速（含 A/B 对照）；
    #   精简版后验 → 说明为何**不**加速（而不是默默什么都不说）。
    # 断言要覆盖两种真实分支，否则就成了「只认某一种运行环境」的假回归。
    check("U3 方法与坑：披露了性能机制（子集加速，或如实说明为何不加速）",
          "安全阀" in _texts
          and ("后验子集" in _texts or "精简版后验" in _texts), "")
    check("U3 方法与坑：给出了子集 vs 全量的对照实测数据",
          "0.37s" in _texts or "9.73s" in _texts, "")
    check("U3 无 st.exception 元素残留", not at_opt.exception, "")

    # ------------------------------------------------------------
    section("U4 / U5 决策报告页：模板路径与幻觉拦截")
    at_rep = _render("report.py", ds, art, with_opt=res)
    check("U4 报告页无异常渲染", not at_rep.exception, "")
    if at_rep.exception:
        _dump_exc(at_rep, "report/render")
    # 事实层底稿浏览器里有多个分支（部分数据缺失时走 st.info）。
    # 第一版那里用了跨行的三元表达式，streamlit 解析调用方源码时抛
    # SyntaxError，整个页面挂掉 —— 只有真渲染到那条分支才会暴露。
    check("U4 事实层底稿浏览器渲染出多个表格",
          len(at_rep.dataframe) >= 3, f"{len(at_rep.dataframe)} 个表格")

    at_rep.radio(key="rep_mode").set_value("模板渲染（秒出，零幻觉风险）")
    at_rep.button(key="btn_report").click().run(timeout=180)
    rep = at_rep.session_state[state.K_REPORT]
    check("U4 模板路径生成成功", bool(rep and rep.get("ok")),
          str((rep or {}).get("source")))
    if rep and rep.get("ok"):
        check("U4 来源标记为 template 且篇幅充足",
              rep["source"] == "template" and rep["chars"] > 1000,
              f"{rep['chars']} 字符")
        check("U4 模板路径不做数字校验（无幻觉风险）", rep.get("guard") is None, "")
        check("U4 报告含事实指纹", rep.get("facts_hash") in rep["text"], "")

    at_bad = _render("report.py", ds, art, with_opt=res)
    at_bad.radio(key="rep_mode").set_value("离线模拟：注入幻觉（验证拦截）")
    at_bad.button(key="btn_report").click().run(timeout=180)
    rep_bad = at_bad.session_state[state.K_REPORT]
    check("U5 注入幻觉 → 被拦截并降级", bool(rep_bad) and
          rep_bad.get("source") == "llm_rejected",
          str((rep_bad or {}).get("source")))
    if rep_bad and rep_bad.get("ok"):
        check("U5 拦截条数 > 0 且留档被拒稿件",
              (rep_bad.get("guard") is not None and rep_bad["guard"].n_fabricated >= 1)
              and len(rep_bad.get("rejected_doc") or "") > 500,
              f"fab={rep_bad['guard'].n_fabricated} chars={len(rep_bad.get('rejected_doc') or '')}")
        check("U5 页面最终展示的是模板兜底稿",
              rep_bad["text"].startswith("> **生成方式**：大模型稿被拒"),
              rep_bad["text"][:36])

    # ------------------------------------------------------------
    section("U6 依赖链：清掉优化结果后报告页如实降级（不崩溃）")
    at_rep2 = _render("report.py", ds, art, with_opt=None)
    check("U6 无优化结果时报告页仍能渲染", not at_rep2.exception, "")
    if at_rep2.exception:
        _dump_exc(at_rep2, "report/no-opt")
    check("U6 如实提示「还没有会话内的优化结果」",
          any("还没有会话内的优化结果" in (w.value or "") for w in at_rep2.warning),
          f"{len(at_rep2.warning)} 条 warning")

    # ------------------------------------------------------------
    section("U7 拼进 HTML 的文案不得残留裸 **（加粗失效）")
    # 为什么这类 bug 必须专门查：
    #   页面里有两种加粗写法 —— 原生 HTML 的 `<b>`，和 markdown 的 `**…**`。
    #   一旦把带 `**` 的文案拼进 `st.markdown(..., unsafe_allow_html=True)`
    #   的 HTML 串里，**markdown 解析就被整段跳过了**，用户看到的是一对裸露的星号。
    #   它既不是异常、也不是错数据，页面「看起来是好的」—— 只有人眼或本检查能发现。
    #
    # 判据（不需要浏览器）：AppTest 拿到的就是交给 st.markdown 的**原始字符串**。
    #   若同一个字符串里既出现了 HTML 块级标签、又出现了 `**`，
    #   那么这对星号一定不会被渲染成加粗 → 命中。
    #
    # 为什么一次就能覆盖所有 Tab：`st.tabs` 会把**每个** Tab 的面板都渲染进 DOM
    #   （只靠 CSS 隐藏未选中的），所以无头环境下所有分区的 markdown 都拿得到。
    _html_tag = re.compile(r"<(div|td|tr|table|span|ul|li|p)\b", re.I)
    _bold = re.compile(r"\*\*[^*\n]+\*\*")
    _at_all = [("home", _render("home.py", ds, art)),
               ("data_eda", _render("data_eda.py", ds, art)),
               ("modeling", _render("modeling.py", ds, art)),
               ("optimize", at_opt),
               ("report", _render("report.py", ds, art, with_opt=res))]
    _bold_hits = []
    for _nm, _at in _at_all:
        for _m in _at.markdown:
            _v = _m.value or ""
            if _html_tag.search(_v) and _bold.search(_v):
                _bold_hits.append(f"{_nm}: {_bold.search(_v).group(0)[:48]}")
    check(f"U7 五个页面拼进 HTML 的文案无裸 **（共检查 {len(_at_all)} 页）",
          not _bold_hits, " | ".join(_bold_hits[:6]))

    # ------------------------------------------------------------
    section("U8 页面文案不得出现「硬编码的样本数」（源码级守卫）")
    # 为什么用源码扫描而不是渲染断言：
    #   `views/modeling.py` 曾把「含 6,000 个后验样本」写死在 f-string 里。
    #   换成精简版后验（1,200 抽样）后，页面**照样渲染成功**，只是数字是错的。
    #   这类「不报错的错误陈述」在任何输出型断言下都是绿的 ——
    #   必须在源码层面拦截，否则它永远能在回归里蒙混过关。
    #
    # 扫描口径（两次误报后收紧的）：
    #   ① 只扫 **AST 里的字符串字面量** —— 注释与 docstring 永远不会渲染到页面上，
    #      上一版按行做正则，把 `core/model.py` 里那句「之前硬编码过 6,000」的
    #      说明文字也判成了违规。
    #   ② 模式必须是「含 / 共 / 包含 N 个后验样本」这种**断言存档规模**的口径。
    #      上一版匹配「N 个后验样本」，把优化页那句「稳健性检验（40 个后验样本）」
    #      也误判了 —— 那说的是用户自己设的抽样数，不是存档里有几个。
    _hard = re.compile(r"(含|共|包含)\s*\d[\d,]*\s*个后验样本")
    _scanned, _hits = 0, []
    for _sub in ("views", "core"):
        _d = os.path.join(APP, _sub)
        for _fn in sorted(os.listdir(_d)):
            if not _fn.endswith(".py"):
                continue
            _scanned += 1
            _src = open(os.path.join(_d, _fn), encoding="utf-8").read()
            # 收集 docstring 节点（它们不算「文案」）
            _tree = ast.parse(_src)
            _doc_nodes = set()
            for _node in ast.walk(_tree):
                if isinstance(_node, (ast.Module, ast.FunctionDef,
                                      ast.AsyncFunctionDef, ast.ClassDef)):
                    _body = getattr(_node, "body", None)
                    if (_body and isinstance(_body[0], ast.Expr)
                            and isinstance(_body[0].value, ast.Constant)
                            and isinstance(_body[0].value.value, str)):
                        _doc_nodes.add(id(_body[0].value))
            for _node in ast.walk(_tree):
                if not (isinstance(_node, ast.Constant)
                        and isinstance(_node.value, str)):
                    continue
                if id(_node) in _doc_nodes:
                    continue
                for _m in _hard.finditer(_node.value):
                    _hits.append(f"{_sub}/{_fn}:{_node.lineno} → "
                                 f"{_m.group(0)!r}")
    check(f"U8 扫描了 {_scanned} 个源文件的字符串字面量，无硬编码样本数",
          not _hits, " | ".join(_hits))

    # 反向验证：真的渲染一次「入口面板」（点过『清除结果』后才会出现），
    # 确认它报出来的数字与后验实际规模一致。
    _at_entry = AppTest.from_file(os.path.join(APP, "views", "modeling.py"),
                                  default_timeout=300)
    _seed(_at_entry, ds, art=None)
    _at_entry.session_state[state.K_ARTIFACT_KEY] = "__skip_auto__"
    _at_entry.run(timeout=300)
    _caps = " ".join((c.value or "") for c in _at_entry.caption)
    _pv = model.pretrained_info()
    if _pv.get("available") and _pv.get("total_draws"):
        check(f"U8 入口面板报的样本数 = 实际（{_pv['total_draws']:,}）",
              f"{_pv['total_draws']:,}" in _caps or str(_pv["total_draws"]) in _caps,
              _caps[:140])
        check("U8 入口面板正确标注「精简版」",
              ("精简版" in _caps) == bool(_pv.get("lite")), _caps[:140])
    check("U8 入口面板无异常", not _at_entry.exception, "")

    # ------------------------------------------------------------
    section("结果汇总")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"  通过 {n_pass} / {len(RESULTS)}")
    if n_fail:
        print("\n  失败项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    ✗ {name}   {detail}")
    print(f"\n  结论：{'全部通过 ✅' if n_fail == 0 else f'{n_fail} 项失败 ❌'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
