# -*- coding: utf-8 -*-
"""
全应用视觉体检（Playwright 驱动真实浏览器）
==========================================

为什么不只用无头 AppTest
-----------------------
`app/uitest_views.py` 的 U7 已经能在无浏览器环境下抓住「裸 `**`」，
那为什么还要这个脚本？两条理由：

1. **它验证的是「渲染结果」，不是「交给 st.markdown 的字符串」。**
   U7 的判据是静态的（同一个字符串里既有 HTML 标签又有 `**`）。
   而这里读的是浏览器 DOM 里的 `innerText` —— 属于**独立的第二条证据链**。
   一个静态判据 + 一个真实渲染判据，比其中一个的「更严格版本」可靠。

2. **它同时体检「页面上有没有异常痕迹」**（`AttributeError` / `Traceback` 等）。
   这是 AppTest 覆盖不到的：AppTest 只看得到它自己渲染的那个脚本，
   而真实浏览器看到的是**整条链路**（入口 → 路由 → 页面）。
   实测中 `streamlit_app.py` 调用了 `core.ui` 里当时不存在的函数，
   首页整页抛 `AttributeError`，而 `curl /_stcore/health` 照样返回 **HTTP 200**。

前置条件
--------
需要 Playwright（**不在项目 `.venv` 里**，故意不装 —— 它只是开发期的体检工具，
不该进生产依赖）：

    pip install playwright          # 用任意隔离环境
    # 无需 `playwright install`：脚本用系统自带的 Edge（channel="msedge"）

用法
----
    # 先起服务
    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.port 8501

    # 再跑体检（换成本机装了 playwright 的那个 python）
    python src/experiments/visual_audit.py
    python src/experiments/visual_audit.py --url http://localhost:8899 --shot

输出
----
  逐页面 / 逐 Tab 报告；发现裸 `**` 时打印**命中的原文片段**（便于直接定位代码）。
  退出码：0 = 全绿；1 = 有命中。

覆盖范围（5 个页面 + 16 个 Tab）
--------------------------------
  概览与数据      （默认）
  探索性分析      （默认）+ 5 个 Tab
  建模与诊断      （默认）+ 6 个 Tab
  预算优化        （默认）+ 6 个 Tab   ← 需要先点过「开始求解」
  决策报告        （默认）

踩坑备忘（写脚本时踩过的，改脚本前先看）
--------------------------------------
  · **Tab 定位**：`data-baseweb="tab"` 在当前 Streamlit 版本**不存在**。
    真实结构是 `div[data-testid="stTab"][role="tab"]`（文本在内部 `<p>` 里，
    所以 `get_by_text` 会命中被 CSS 隐藏的 `<p>` → 点击超时）。
  · **导航方式**：页面间跳转要用**侧栏链接**做客户端路由。
    用 `page.goto()` 会重开会话，把「已载入数据」的状态清空。
  · **优化页的 Tab 不是一开始就有的**：必须先拿到模型（访问建模页触发自动复用后验）
    并点过「开始求解」，六个 Tab 才会出现。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 每个页面对应的 Tab（不含 Tab 的页面给空列表）
PAGES: list[tuple[str, list[str]]] = [
    ("概览与数据", []),
    ("探索性分析", ["📈 趋势与节律", "🧩 渠道可分离性", "⏳ 广告滞后效应",
                    "📉 花费-收入与饱和", "🧪 可识别性与数据质量"]),
    ("建模与诊断", ["① 结果可信吗", "② 渠道 ROI", "③ 收入怎么分",
                    "④ 参数与后验", "⑤ 参数回收", "⑥ 局限与缓解"]),
    ("预算优化", ["① 钱该往哪挪", "② 优化方案", "③ 弹性-收益曲线",
                  "④ 情景分析", "⑤ 结论有多稳", "⑥ 方法与坑"]),
    ("决策报告", []),
]

# 判定「裸 **」的脚本：只看叶子节点，排除 <pre>/<code>（代码里出现 ** 是正常的）
SCAN_BOLD = """() => {
    const bad = [];
    document.querySelectorAll('p,span,div,li,td,th,h1,h2,h3,h4,em,strong').forEach(e => {
        if (e.closest('pre') || e.closest('code')) return;
        if (e.children.length > 0) return;
        const t = (e.innerText || '');
        if (/\\*\\*/.test(t)) {
            bad.push({tag: e.tagName, cls: e.className || '',
                      txt: t.replace(/\\s+/g, ' ').slice(0, 110)});
        }
    });
    return bad;
}"""

# 异常痕迹（页面把错误堆栈直接渲染出来了）
SCAN_ERR = r"text=/(AttributeError|Traceback|NameError|TypeError|KeyError|ValueError):/"


def click_tab(pg, text: str) -> None:
    """点击 Tab：优先点可见的那个候选（未选中的面板可能留在 DOM 里）。"""
    loc = pg.locator('div[data-testid="stTab"][role="tab"]', has_text=text)
    n = loc.count()
    if n == 0:
        raise RuntimeError(f"找不到 Tab：{text}")
    for i in range(n):
        if loc.nth(i).is_visible():
            loc.nth(i).click()
            pg.wait_for_timeout(1500)
            return
    raise RuntimeError(f"Tab 存在但不可见（{n} 个候选）：{text}")


def main() -> int:
    ap = argparse.ArgumentParser(description="全应用视觉体检（裸 ** / 异常痕迹）")
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--shot", action="store_true", help="同时把每个状态截图到 outputs/screenshots/_audit/")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("✗ 需要 playwright：pip install playwright")
        print("  （脚本用系统 Edge，无需 playwright install）")
        return 2

    shotdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs", "screenshots", "_audit")
    if args.shot:
        os.makedirs(shotdir, exist_ok=True)

    bad_bold: list[tuple[str, str, list]] = []
    bad_err: list[tuple[str, str, int]] = []

    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        ctx = br.new_context(viewport={"width": 1440, "height": 1000},
                             device_scale_factor=1.5)
        pg = ctx.new_page()
        pg.set_default_timeout(120000)

        print("→ 打开首页并载入内置数据")
        pg.goto(args.url, wait_until="domcontentloaded")
        pg.wait_for_selector('button:has-text("载入内置数据")', timeout=90000)
        pg.locator('button:has-text("载入内置数据")').first.click()
        pg.wait_for_timeout(8000)

        # 必须先访问建模页：后验是在该页「首次进入时自动复用」的。
        # 直接去优化页会拿不到模型产物 → 没有求解按钮 → 也就没有那六个 Tab。
        print("→ 访问建模页（触发自动复用后验）")
        pg.get_by_role("link", name="建模与诊断").first.click()
        pg.wait_for_timeout(12000)

        print("→ 预算优化页：点「开始求解」（让结果区与六个 Tab 出现）")
        pg.get_by_role("link", name="预算优化").first.click()
        pg.wait_for_timeout(5000)
        if pg.locator('button:has-text("开始求解")').count():
            pg.locator('button:has-text("开始求解")').first.click()
            for _ in range(120):
                pg.wait_for_timeout(4000)
                if pg.locator('button:has-text("重新求解")').count():
                    break
            print("  求解完成")
        else:
            print("  ⚠️ 未找到求解按钮（模型产物可能缺失）")
        pg.wait_for_timeout(3000)

        for pname, tabs in PAGES:
            pg.get_by_role("link", name=pname).first.click()
            pg.wait_for_timeout(4000)
            for label, tab in [("（默认）", None)] + [(t, t) for t in tabs]:
                if tab:
                    try:
                        click_tab(pg, tab)
                    except Exception as e:                       # noqa: BLE001
                        print(f"  [skip] {pname} / {label}: {type(e).__name__}: {e}")
                        continue
                hits = pg.evaluate(SCAN_BOLD)
                n_err = pg.locator(SCAN_ERR).count()
                if args.shot:
                    fn = f"{pname}_{label}".replace("/", "_").replace("（", "").replace("）", "") \
                        + ".png"
                    pg.screenshot(path=os.path.join(shotdir, fn), full_page=True)
                if hits:
                    bad_bold.append((pname, label, hits))
                    print(f"  ✗ {pname} / {label}: 裸 ** × {len(hits)}")
                    for b in hits[:5]:
                        print(f"       <{b['tag']} class='{b['cls'][:26]}'> {b['txt']}")
                if n_err:
                    bad_err.append((pname, label, n_err))
                    print(f"  ✗ {pname} / {label}: 异常痕迹 × {n_err}")
                if not hits and not n_err:
                    print(f"  ✓ {pname} / {label}")

        br.close()

    print("\n" + "=" * 70)
    n_bold = sum(len(h) for _, _, h in bad_bold)
    print(f"裸 ** 命中：{n_bold} 处" + ("  ✅" if not n_bold else "  ❌"))
    print(f"异常痕迹  ：{sum(n for _, _, n in bad_err)} 处"
          + ("  ✅" if not bad_err else "  ❌"))
    if args.shot:
        print(f"截图已存：{shotdir}")
    if bad_bold or bad_err:
        print("\n排查提示：")
        print("  · 裸 **  → 检查该文案是否用了 `_esc()` 而没有用 `_inline()`/`ui.md_esc()`")
        print("            （`_inline` = 先转义 HTML、再放行 `**`）")
        print("  · 异常痕迹 → 先重启 Streamlit 进程：热重载**不覆盖** import 进来的")
        print("            `core/*.py`，旧模块会留在内存里。")
    return 1 if (bad_bold or bad_err) else 0


if __name__ == "__main__":
    sys.exit(main())
