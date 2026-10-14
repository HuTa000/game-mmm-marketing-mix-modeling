"""
缓存策略（依赖 streamlit）。

三级缓存，各司其职（用错会直接导致报错或性能崩塌）：

┌──────────────┬────────────────────┬──────────────────────────────────────┐
│ 层级          │ 手段                │ 存什么 / 为什么                        │
├──────────────┼────────────────────┼──────────────────────────────────────┤
│ L1 会话状态   │ st.session_state   │ 当前会话的重结果（模型、优化、报告）      │
│              │                    │ 跨页面切换保留；切页不重算 ← 体验关键    │
│ L2 进程缓存   │ cache_data         │ 可序列化的小结果（数据、事实层）          │
│              │ cache_resource     │ 不可序列化的重对象（PyMC 模型对象）       │
│ L3 磁盘缓存   │ outputs/reports/   │ 拟合出的后验（idata），跨重启可用         │
│              │ app_fit_*.pkl      │ 避免重启应用后又要等 10~20 分钟          │
└──────────────┴────────────────────┴──────────────────────────────────────┘

为什么模型对象必须走 cache_resource：
    `st.cache_data` 会对返回值做**序列化拷贝**，PyMC 模型（含 PyTensor 图与
    编译函数）无法被 pickle，会直接报错；`cache_resource` 只存引用，不拷贝。

为什么缓存键里必须带数据指纹：
    否则用户换了一份数据，却命中上一次的缓存，拿到旧模型的结论 —— 这是正确性
    问题，比报错更危险。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

import streamlit as st

from . import paths
from . import ui

paths.ensure_src_on_path()

FIT_CACHE_PREFIX = "app_fit_"
FIT_CACHE_MANIFEST = os.path.join(paths.REPDIR, "app_fit_manifest.json")


# ============================================================
# L2-a：可序列化结果缓存
# ============================================================
@st.cache_data(show_spinner=False, max_entries=8)
def cached_builtin_dataset(n_weeks: int = 156, seed: int = 2024,
                           with_controls: bool = True):
    """内置模拟数据。虽然生成本身很快，但缓存掉可以避免每次交互都重算 EDA 输入。"""
    from . import data_io
    return data_io.load_builtin(n_weeks=n_weeks, seed=seed, with_controls=with_controls)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_uploaded(payload: bytes, name: str):
    """
    解析 + 校验上传文件。
    用文件字节本身做缓存键 —— 用户每次交互都会重跑脚本，
    没有这层缓存就会反复解析同一个文件（大文件下明显卡顿）。
    """
    from . import data_io
    return data_io.load_uploaded(payload, name)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_facts(repdir: str, _stamp: float):
    """
    报告事实层（唯一数字源）。_stamp 传产物目录的最新修改时间，
    产物一变缓存自动失效 —— 这是"缓存键包含数据版本"的落地做法。
    """
    import report_facts
    return report_facts.build_facts(repdir=repdir)


def repdir_stamp() -> float:
    """报告目录内最新文件时间戳，作为事实层缓存版本号。"""
    try:
        files = [os.path.join(paths.REPDIR, f) for f in os.listdir(paths.REPDIR)]
        mtimes = [os.path.getmtime(f) for f in files if os.path.isfile(f)]
        return max(mtimes) if mtimes else 0.0
    except Exception:
        return 0.0


def get_facts():
    """取事实层（带版本化缓存）。阶段 9 的报告页直接用这个。"""
    return cached_facts(paths.REPDIR, repdir_stamp())


@st.cache_data(show_spinner=False, max_entries=3)
def cached_session_facts(fingerprint: str, opt_key: str, _art, _opt, _ds):
    """
    会话内事实层（阶段 9）。

    为什么要缓存：报告页每次交互都会重跑脚本，而构建事实层要遍历
    模型诊断 + 优化结果 + 一次共线性计算。虽然不慢，但每次交互都重算会累积成卡顿。

    为什么键是 (数据指纹, 优化结果键)：这两个决定了事实的**全部内容**。
    用 `_art / _opt / _ds` 前缀让 Streamlit 跳过对它们的哈希
    （`_art` 里含 PyMC 模型对象，哈希/序列化都会炸），
    但只要指纹或结果键变了，缓存一定失效 —— 正确性优先。
    """
    from . import facts as F
    return F.build_facts(_art, _opt, _ds)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_eda(fingerprint: str, _df, date_col: str, target_col: str,
               channels: tuple, controls: tuple,
               max_lag: int = 8, period: int = 52):
    """
    探索性分析的全量结果（阶段 7）。

    关键点：`_df` 带下划线前缀 → Streamlit **不对它做哈希**，缓存键只由
    指纹 + 列名 + 参数组成。这样每次页面重跑不会重新 pickle 整张表，
    又因为键里含数据指纹，换了数据一定不会命中旧结果（正确性优先）。
    """
    from . import eda
    return eda.analyze_all(_df, date_col, target_col,
                           list(channels), list(controls),
                           max_lag=int(max_lag), period=int(period))


# ============================================================
# L2-b：重对象缓存（不序列化）
# ============================================================
def pretrained_stamp() -> float:
    # 用「实际可用存档」的修改时间做缓存键：生成/替换精简版后缓存自动失效。
    p = paths.pretrained_pkl()
    return os.path.getmtime(p) if p else 0.0


def pretrained_available() -> bool:
    return paths.pretrained_pkl() is not None


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_pretrained(stamp: float):
    """
    载入阶段 2 的后验，重建 MMM 对象（不重采样）。

    【为什么必须走 cache_resource 而不是 cache_data】
    返回值里的 `mmm` 含 PyTensor 计算图与编译好的函数，无法 pickle，
    cache_data 会在返回时尝试深拷贝并直接抛错。cache_resource 只存引用。

    【为什么 stamp 不加下划线】
    加了下划线 Streamlit 就不对它做哈希 → 缓存永远命中第一次的结果。
    这里让 pkl 的修改时间参与键，后验一旦更新，缓存自动失效。

    复用 src/stage3_optimize.load_model_and_data —— 它已经把手动挂载
    channel_scales 的坑修好了，应用层直接调用，不重复实现。
    """
    if not pretrained_available():
        return None
    try:
        import stage3_optimize as s3
        df, cfg, mmm, X, y, meta, used_ctrl, idata = s3.load_model_and_data()
        return {"df": df, "cfg": cfg, "mmm": mmm, "X": X, "y": y,
                "meta": meta, "used_ctrl": used_ctrl, "idata": idata,
                "source": "pretrained"}
    except Exception as e:                                    # noqa: BLE001
        return {"__error__": f"{type(e).__name__}: {e}"}


def get_pretrained_bundle():
    """复用阶段 2 后验（同一会话内只重建一次，约 5~10 秒；命中后毫秒级）。"""
    return cached_pretrained(pretrained_stamp())


# ============================================================
# 阶段 8：拟合结果的跨重启缓存（L3）
# ============================================================
def fit_key_for(ds, user_cfg: dict) -> str:
    """
    拟合缓存键 = 数据指纹 × 超参哈希。

    【这是正确性问题，不是性能问题】
    少了数据指纹，用户换了一份数据却命中上一次的模型，
    页面会展示与当前数据毫无关系的 ROI —— 而且不报错。
    """
    return fit_cache_key(getattr(ds, "fingerprint", "") or "nofp", user_cfg or {})


def get_disk_fit(ds, user_cfg: dict):
    """从磁盘还原一次历史拟合（重启浏览器后免去重跑 10 分钟）。"""
    from . import model as M
    payload = load_fit(fit_key_for(ds, user_cfg))
    if not payload:
        return None
    return M.from_cache_payload(payload, ds.df)


def save_disk_fit(ds, user_cfg: dict, art: dict):
    from . import model as M
    return save_fit(fit_key_for(ds, user_cfg), M.to_cache_payload(art))


# ============================================================
# 缓存键：数据指纹 × 模型配置
# ============================================================
def config_hash(cfg: dict) -> str:
    """把用户可调超参折叠成短哈希（键排序保证顺序无关）。"""
    payload = json.dumps(cfg or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def fit_cache_key(dataset_fingerprint: str, cfg: dict) -> str:
    return f"{dataset_fingerprint}-{config_hash(cfg)}"


# ============================================================
# L3：磁盘缓存（跨重启）
# ============================================================
def _read_manifest() -> dict:
    try:
        with open(FIT_CACHE_MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_manifest(m: dict) -> None:
    try:
        os.makedirs(paths.REPDIR, exist_ok=True)
        with open(FIT_CACHE_MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass


def fit_cache_path(key: str) -> str:
    return os.path.join(paths.REPDIR, f"{FIT_CACHE_PREFIX}{key}.pkl")


def save_fit(key: str, payload: dict) -> str | None:
    """
    把一次拟合的结果落盘。**只存可序列化部分**（idata + 配置 + 缩放元信息），
    模型结构在读取时用同样的 X/y 重建 —— 这也是阶段 2 已验证可行的做法。
    """
    safe = {k: v for k, v in payload.items()
            if k in ("idata", "cfg", "used_ctrl", "y_mean", "y_max",
                     "elapsed", "diagnostics", "config_snapshot", "created_at")}
    p = fit_cache_path(key)
    try:
        os.makedirs(paths.REPDIR, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(safe, fh, protocol=pickle.HIGHEST_PROTOCOL)
        m = _read_manifest()
        m[key] = {"path": p, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "size_mb": round(os.path.getsize(p) / 1e6, 1)}
        # 只保留最近 3 份，避免磁盘无限增长
        if len(m) > 3:
            for k in sorted(m, key=lambda x: m[x].get("saved_at", ""))[:-3]:
                try:
                    os.remove(m[k]["path"])
                except Exception:
                    pass
                m.pop(k, None)
        _write_manifest(m)
        return p
    except Exception:
        return None


def load_fit(key: str) -> dict | None:
    p = fit_cache_path(key)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def clear_fit_cache(delete_files: bool = True) -> int:
    """清空磁盘拟合缓存，返回清理条数。"""
    m = _read_manifest()
    n = 0
    for k, v in list(m.items()):
        if delete_files:
            try:
                os.remove(v.get("path", ""))
            except Exception:
                pass
        m.pop(k, None)
        n += 1
    _write_manifest(m)
    return n


def cache_status() -> dict:
    """给界面展示缓存状态（面试演示时可以直观说明"结果被缓存住了"）。"""
    m = _read_manifest()
    total = sum(v.get("size_mb", 0) for v in m.values())
    return {"entries": len(m), "total_mb": round(total, 1),
            "pretrained": pretrained_available(),
            "pretrained_mb": round(
                os.path.getsize(paths.artifact_path("stage2_idata")) / 1e6, 1
            ) if pretrained_available() else 0.0,
            "detail": m}


# ============================================================
# 进度反馈：长时间任务必须让用户看到"在动"
# ============================================================
class TaskStatus(ui.StatusBox):
    """
    统一的长期任务进度反馈。

    为什么不能只靠 spinner：拟合耗时 10~20 分钟，用户看不到阶段与剩余感，
    会以为卡死而关掉页面。这里把过程拆成可读的阶段，并追加耗时。

    实现已**合并**到 core/ui.py 的 StatusBox（本类只是保留旧名字的别名）。
    合并的原因：这里原本写对了（用 `update(state=...)`），
    而阶段 9 新写的优化页/报告页各自直接用 `st.status(...)` 并调用了
    并不存在的 `.step()/.done()/.fail()`，于是点按钮就抛异常。
    同一件事有两份实现，就一定会有一份是错的 —— 所以只留一份。
    """

    def done(self, msg: str = "完成") -> None:
        self.close(f"{msg}（耗时 {self.elapsed:.1f} 秒）", "complete")

    def fail(self, msg: str) -> None:
        self.close(msg, "error")


def make_numpyro_progress(bar, total_hint: int = 3000):
    """
    给 PyMC 的 numpyro 后端做一个进度回调适配器。
    numpyro 的 progress_bar 接口在不同版本间有差异，这里做成**尽力而为**：
    回调失败不影响采样本身（绝不能让进度显示把主流程搞崩）。
    """
    state = {"n": 0}

    def _cb(*a, **kw):
        try:
            state["n"] += 1
            frac = min(0.95, state["n"] / max(1, total_hint))
            if callable(bar):
                bar(frac, f"采样中… 已推进 {state['n']} 步")
        except Exception:
            pass
    return _cb
