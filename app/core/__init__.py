"""
app.core —— Streamlit 应用的共享基础设施。

分层原则（这是本阶段最重要的架构决策）：
    ├── 纯逻辑层（不 import streamlit）: paths / schema / data_io
    │      → 可脱离浏览器运行，能被 selftest_core.py 直接单测
    └── 运行时层（依赖 streamlit）      : state / cache / ui
           → 只在 Streamlit 脚本环境里使用

把"能不能读对数据"与"界面长什么样"解耦，好处是数据模块的
正确性可以自动化验证，不需要靠人肉点界面试。
"""

__all__ = ["paths", "schema", "data_io"]
