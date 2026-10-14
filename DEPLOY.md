# 部署说明

本项目有三种部署方式。**先看这张表选路径**，再看对应小节。

| 路径 | 适用场景 | 需要的东西 | 首次可用耗时 |
|---|---|---|---|
| **A 本地运行** | 面试演示（最稳） | Python 3.10+ | 装依赖约 5 分钟 |
| **B Docker** | 交给别人跑 / 服务器常驻 | Docker | 构建约 10 分钟 |
| **C Streamlit Cloud** | 简历里放一个可点的链接 | GitHub 账号 | 约 15 分钟 |

三种路径跑的是**同一个应用**，差别只在「怎么把它起起来」。下面先讲一个所有路径都要理解的前提。

---

## 0. 一个必须先理解的前提：应用依赖什么

应用**不重采样**也能完整演示 —— 这是它能被轻松部署的关键。但前提是磁盘上有阶段 2 的后验存档：

```
outputs/reports/stage2_mmm_idata_lite.pkl   ← 精简版，约 23 MB（推荐，见第 4 节）
outputs/reports/stage2_mmm_idata.pkl        ← 全量版，114 MB（本地开发用）
```

**两者有任意一个就行**（应用优先用精简版）。若都没有：

- 「概览与数据」「探索性分析」两页**照常可用**（它们只依赖数据）；
- 点「建模与诊断」会触发**真正的 NUTS 采样**，首次约 10~20 分钟 —— 本地演示可以接受，云端按量计费的实例要留意。

其余阶段产物（`outputs/reports/*.csv`、`outputs/figures/*.png`）都是**可选**的：有就展示，没有页面会给出降级提示而不是报错。

> 这就是为什么 `.gitignore` 里排除了 114MB 的全量 pkl、却显式放行了 23MB 的精简版 —— 见第 4 节。

---

## A. 本地运行（演示首选）

```bash
cd "游戏出海广告效果 MMM"

# 1. 建虚拟环境（Python 3.12 推荐，3.10+ 可用）
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS / Linux

# 2. 装依赖
pip install -r requirements.txt

# 3. 起服务
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
# 浏览器打开 http://localhost:8501
```

**Windows 上不需要 C 编译器**：项目默认用 `numpyro`（JAX 后端，纯 Python）。若控制台出现
`g++ not available`，那是 pytensor 的探测提示，**不影响采样**，可以忽略。

冒烟自检（不启动浏览器，约 1 分钟）：

```bash
.venv/Scripts/python.exe app/selftest_core.py    # 逻辑层 208+ 条断言
.venv/Scripts/python.exe app/uitest_views.py     # 页面层 38 条断言（AppTest 无头渲染）
```

---

## B. Docker

```bash
# 构建（.dockerignore 已排除 .venv / 全量后验 / 开发脚本）
docker build -t mmm-app .

# 运行
docker run --rm -p 8501:8501 mmm-app
# 浏览器打开 http://localhost:8501
```

带 LLM 密钥运行（可选，只有「决策报告」页的大模型润色需要）：

```bash
docker run --rm -p 8501:8501 \
  -e LLM_API_KEY=sk-xxx \
  -e LLM_BASE_URL=https://api.deepseek.com/v1 \
  -e LLM_MODEL=deepseek-chat \
  mmm-app
```

**关于镜像体积**：基础镜像 `python:3.12-slim`，依赖装完约 1.2~1.5 GB（PyMC + JAX 占大头）。
镜像里**不含** 114MB 的全量后验；精简版（23MB）会被打进去。

健康检查已内置（`HEALTHCHECK` 探 `/_stcore/health`），对接 k8s / compose 时可直接用。

### docker compose（可选）

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      LLM_API_KEY: ${LLM_API_KEY:-}
      LLM_BASE_URL: ${LLM_BASE_URL:-}
      LLM_MODEL: ${LLM_MODEL:-}
    restart: unless-stopped
```

---

## C. Streamlit Community Cloud（简历链接）

1. 把仓库推到 GitHub；
2. 打开 <https://share.streamlit.io> → New app；
3. Main file path 填 **`app/streamlit_app.py`**；
4. （可选）在 Settings → Secrets 粘贴 `.streamlit/secrets.toml.example` 的内容并填入密钥；
5. Deploy。

### 这一条路上有两个必须提前知道的约束

**约束 1：GitHub 单文件上限 100 MB。**
全量后验 114 MB，推不上去。**必须**先用第 4 节的脚本导出精简版（23 MB）。
`.gitignore` 已经放行了精简版、排除了全量版，所以照第 4 节操作即可，不必改忽略规则。

**约束 2：免费实例内存约 1 GB。**
应用冷启动很轻（只加载后验与数据），但**点「建模与诊断」在本页拟合**会编译 PyTensor 计算图 + 跑 NUTS，
很可能超内存被杀。规避方式：

- 演示时**只用内置数据**，走「复用阶段 2 后验」路径（秒级、不采样）；
- 上传数据要看模型时，本地演示，或把实例升级到付费档。

页面本身对这种情况有兜底：拟合失败会如实报错并给出原因，不会白屏。

---

## 4. 后验存档瘦身：114 MB → 23 MB

这是让路径 C 可行的关键一步。

```bash
# 默认每链保留 300 抽（4 链 × 300 = 1200 样本）
.venv/Scripts/python.exe src/export_lite_posterior.py

# 想要更精细（体积约 45 MB）
.venv/Scripts/python.exe src/export_lite_posterior.py --draws 600
```

### 为什么可以抽稀

拆开 114 MB 的存档，体积分布是这样的（脚本会自己打印这张表）：

| 组 / 变量 | 体积 | 说明 |
|---|---|---|
| `posterior` · 参数类（adstock / kappa / beta / slope / gamma…） | **0.77 MB** | 真正驱动优化求解的量 |
| `posterior` · `channel_contribution` | 37.44 MB | 逐期贡献分解（156 期 × 5 渠道） |
| `posterior` · `control_contribution` | 29.95 MB | 逐期 |
| `posterior` · `fourier_contribution` | 29.95 MB | 逐期 |
| `posterior` · `yearly_seasonality` / `mu` | 14.98 MB | 逐期 |
| `sample_stats`（divergences 等） | 0.19 MB | 收敛诊断 |
| `observed_data` / `constant_data` / `fit_data` | 0.02 MB | 原始数据 |

**99% 的体积来自「逐期数组」，而优化只需要其中 0.7 MB 的参数后验。**
抽稀对时变数组的影响极小（画图与区间用的样本仍上千）。

### 抽稀前后的实测对照（本机）

| | 全量 | 精简版 | 结论 |
|---|---|---|---|
| 文件体积 | 114.2 MB | **22.7 MB** | 压到 19.9% |
| 读取耗时 | 1.3 s | **0.0 s** | 更快 |
| `draw` / 总样本 | 1500 / 6000 | 300 / 1200 | — |
| max R-hat | 1.0021 | 1.0039 | 均 < 1.01 ✅ |
| min ESS | 2815 | **998** | 均 > 400 ✅ |
| divergences | 0 | 0 | ✅ |
| 收敛结论 | 收敛良好 | 收敛良好 | **完全一致** |
| 最优分配（万元） | 157.6 / 518.4 / 517.2 / 119.5 / 78.9 | **逐位相同** | 无损 |
| 预期提升 | +12.830% | +12.851% | 差 0.02 pp |

唯一可见的差异：ROI 的重算值与阶段 2 CSV 的偏差从 0.04% 放大到约 0.13%（蒙特卡洛误差），
远小于任何业务判断阈值。自测里有对应断言。

### 抽稀是「可逆」的

应用通过 `app/core/paths.py::pretrained_pkl()` 决定加载哪个文件：**有精简版用精简版，没有就回退全量**。
删掉 `stage2_mmm_idata_lite.pkl` 即自动回到全量，不需要改任何代码。

另外，后验的**逐期数组按链等间隔抽稀**，同时 `posterior` 与 `sample_stats` 使用同一套索引 ——
这是刻意的：两组样本长度若不一致，下游 `isel(draw=[i])` 这类按位置取值的代码会**静默错位**（不报错、但区间算错）。

---

## 5. 常见问题

**Q：页面能打开，但「建模与诊断」提示缺少模型结果？**
A：磁盘上没有阶段 2 后验。跑 `src/stage2_model.py`（约 10~20 分钟），或从别处拷一份后验存档进 `outputs/reports/`。

**Q：部署后「大模型润色」总是降级成模板？**
A：先看报告页的**生成日志**。`src/llm_client.py` 只读环境变量，托管环境要写在 secrets 里 ——
应用启动时会自动桥接（`app/core/report.py::bridge_secrets`），桥接后的键名会打印在日志里。
若日志显示「大模型不可用」，检查密钥/端点/模型名三项。

**Q：端口冲突 / 想指定端口？**
A：
```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.port 8899
```

**Q：想对外演示但不想暴露错误堆栈？**
A：把 `.streamlit/config.toml` 里的 `showErrorDetails` 从 `"full"` 改为 `"none"`。

**Q：Docker 构建很慢？**
A：依赖层在 `COPY . .` 之前，改代码不会重装依赖。首次构建慢是正常的（PyMC + JAX 体积大）。

**Q：我刚改了 `app/core/` 下的文件，页面却报 `AttributeError: module 'core.xxx' has no attribute ...`（或者行为根本没变）？**
A：**重启 Streamlit 进程。** Streamlit 的热重载只对「页面脚本」生效，
`import` 进来的模块（`core/*.py`）会一直留在进程内存里不重新加载。
本次开发中确实踩到了：`core/ui.py` 新增了 `reset_button()`，
但**已经在跑的进程**还持着旧模块 → 首页直接抛 `AttributeError`，
而 `curl /_stcore/health` 依然返回 **HTTP 200**。

```bash
# 找到并干掉占用 8501 的进程，然后重启
netstat -ano | grep ":8501" | grep LISTENING
taskkill //PID <pid> //F            # Windows
# kill <pid>                        # macOS / Linux
```

> **推论（很重要）**：`HTTP 200` 只能证明服务进程活着，**不能证明页面渲染成功**。
> Streamlit 的应用外壳是静态的，脚本报错也照样返回 200。
> 真正的「页面是否正常」要靠 `app/uitest_views.py` 或至少打开一次浏览器确认。

**Q：为什么 `outputs/reports/` 里有个 `stage2_mmm_idata_lite.json`？**
A：那是精简版后验的**元信息旁挂文件**（链数 / 每链抽样数 / 体积）。
应用要在页面上如实说明「档里有几个后验样本」，如果为此 `pickle.load` 一次 pkl，
就等于**为了显示一行字加载 20 MB 数据**。所以读 json（毫秒级）。
它缺失时页面会退化为「不报具体数字」，不会崩。可用
`python src/export_lite_posterior.py --meta-only` 重新生成。

---

## 6. 演示前自检清单

```bash
# ① 逻辑层与页面层回归（无需浏览器，约 3~5 分钟）
.venv/Scripts/python.exe app/selftest_core.py     # 应为 212/212
.venv/Scripts/python.exe app/uitest_views.py      # 应为 43/43

# ② 起服务并确认 HTTP 200
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.port 8899 &
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8899/

# ③ 确认后验存档在位（决定「建模」是秒级还是 10 分钟）
ls -lh outputs/reports/stage2_mmm_idata*.pkl outputs/reports/stage2_mmm_idata_lite.json

# ④（可选，需要 playwright）全应用视觉体检：真浏览器走 5 页 × 16 Tab
#    扫「裸 **（加粗失效）」与「页面上有没有异常痕迹」
python src/experiments/visual_audit.py
```

> 第 ④ 步与第 ① 步的 U7 断言是**两条独立的证据链**：U7 是静态判据
> （检查交给 `st.markdown` 的字符串），第 ④ 步读的是浏览器 DOM 里的**实际渲染结果**，
> 并且能看到 U7 看不到的东西 —— 整条链路的异常痕迹
> （`streamlit_app.py` 抛 `AttributeError` 就是靠它发现的，而 health 仍返回 200）。

> ⚠️ 第 ② 步的 `HTTP 200` **只说明服务进程活着**，不说明页面渲染成功
> —— Streamlit 的外壳是静态的，页面脚本抛异常也返回 200（见第 5 节那条 FAQ）。
> 所以在真正上台前，**请务必打开一次浏览器**，按第 7 节的手动清单走一遍。

### 上台前的手动清单（2 分钟）

| # | 操作 | 期望看到 |
|---|---|---|
| 1 | 打开首页 | 顶部有标题与副标题，**页面上没有任何 `**`** |
| 2 | 点「载入内置数据」 | 3 秒内出现「数据已载入」，并出现 数据预览 / 质量速览 / 数据字典 三个 Tab |
| 3 | 切到「建模与诊断」 | 自动复用后验（秒级），显示 **共 1,200 个后验样本**；R-hat / ESS / 发散 全绿 |
| 4 | 切到「预算优化」→ 点「开始求解」 | 约 40~120 秒；完成后出现 6 个 Tab |
| 5 | 逐个点开 6 个 Tab | 每页都有图；**没有裸 `**`**；没有红色报错框 |
| 6 | 切到「决策报告」 | 能出稿（模板路径秒出） |
| 7 | 把窗口拖到约 800px 宽 | 表格可横向滚动、图表自适应，不出现内容被裁掉 |

> 第 1 / 5 步的「裸 `**`」不是吹毛求疵：它是**唯一一类「页面看起来正常、内容却是错的」**缺陷，
> 且已经被 `uitest_views.py` 的 U7 断言自动守住（判据做过负向验证，见 `docs/阶段10`）。

演示脚本与话术见 `docs/阶段10_美化部署与演示.md`。
