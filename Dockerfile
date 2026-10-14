# ============================================================
# 游戏出海广告效果 · 贝叶斯 MMM —— 应用镜像
# ------------------------------------------------------------
# 设计要点
#   · 基础镜像 python:3.12-slim：与 README 推荐的 Python 版本一致；
#   · 采样后端 numpyro 走 **CPU 版 JAX**（PyPI 默认即 CPU 版），
#     因此镜像里不需要 CUDA，也不需要 g++ 编译器 —— 这是本项目
#     能在 Windows / 轻量服务器上跑起来的前提；
#   · 应用的核心演示路径（内置数据 + 复用阶段 2 后验）**完全不重新采样**，
#     所以镜像可以很轻、冷启动很快；
#   · 依赖单独一层先行安装：改代码时不会重装依赖。
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAX_PLATFORMS=cpu

WORKDIR /app

# 先只拷依赖清单，让这一层能被 Docker 缓存复用
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 再拷项目本体（.dockerignore 已排除虚拟环境、全量后验 114MB、开发脚本等）
COPY . .

# 依赖自检：装不上就构建失败，而不是等到运行时才报错
RUN python -c "import streamlit; print('streamlit', streamlit.__version__)"

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py"]
