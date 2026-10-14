# -*- coding: utf-8 -*-
"""
阶段 4 · LLM 客户端（OpenAI 兼容）
===================================

特点
----
1. **零第三方依赖**：直接用 requests 调 /chat/completions，不强制安装 openai SDK；
2. **多供应商**：任何 OpenAI 兼容端点（DeepSeek / OpenAI / 通义 / 智谱 / 本地 vLLM）；
3. **可配置**：全部走环境变量，不写死密钥；
4. **可降级**：未配置或调用失败时返回明确错误，由上层切模板（保证 Streamlit 不白屏）；
5. **可离线验证**：LLM_MOCK=1 时用注入文本模拟回复，用于测试完整链路。

环境变量
--------
    LLM_API_KEY      首选密钥（也可用 DEEPSEEK_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY）
    LLM_BASE_URL     端点，如 https://api.deepseek.com/v1
    LLM_MODEL        模型名，如 deepseek-chat / gpt-4o-mini
    LLM_TIMEOUT      超时秒数（默认 60）
    LLM_MOCK         =1 时启用离线模拟

    LLM_PROMPT_PRICE / LLM_COMPLETION_PRICE   可选，每百万 token 单价（元），用于成本估算
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import requests

# 供应商默认端点推断
_DEFAULT_BASE = {
    "DEEPSEEK_API_KEY": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "DASHSCOPE_API_KEY": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "MOONSHOT_API_KEY": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
}


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 60
    max_retries: int = 2
    temperature: float = 0.3
    max_tokens: int = 2200
    mock: bool = False
    prompt_price: float = 0.0        # 元 / 百万 token
    completion_price: float = 0.0

    def describe(self) -> str:
        if self.mock:
            return "离线模拟模式（LLM_MOCK=1）"
        if not self.api_key:
            return "未配置（将使用模板降级）"
        return f"{self.model} @ {self.base_url}"


@dataclass
class LLMResponse:
    ok: bool = False
    text: str = ""
    model: str = ""
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    error: str = ""
    finish_reason: str = ""

    def usage_line(self) -> str:
        if not self.ok:
            return f"失败: {self.error}"
        cost = f"  估算成本 ¥{self.cost:.4f}" if self.cost else ""
        return (f"{self.model}  {self.latency:.1f}s  "
                f"tokens={self.prompt_tokens}+{self.completion_tokens}{cost}")


def from_env(env: dict | None = None) -> LLMConfig:
    e = os.environ if env is None else env
    cfg = LLMConfig()
    cfg.mock = str(e.get("LLM_MOCK", "")).strip() in ("1", "true", "True", "yes")
    cfg.timeout = int(e.get("LLM_TIMEOUT", "60") or "60")

    key, base, model = "", "", ""
    if e.get("LLM_API_KEY"):
        key = e["LLM_API_KEY"].strip()
    else:
        for k, (ab, am) in _DEFAULT_BASE.items():
            if e.get(k):
                key, base, model = e[k].strip(), ab, am
                break
    cfg.api_key = key
    cfg.base_url = (e.get("LLM_BASE_URL") or base).strip()
    cfg.model = (e.get("LLM_MODEL") or model).strip()
    if cfg.api_key and not cfg.base_url:
        cfg.base_url = "https://api.openai.com/v1"
    if cfg.api_key and not cfg.model:
        cfg.model = "gpt-4o-mini"
    try:
        cfg.prompt_price = float(e.get("LLM_PROMPT_PRICE", "0") or 0)
        cfg.completion_price = float(e.get("LLM_COMPLETION_PRICE", "0") or 0)
    except Exception:
        pass
    return cfg


def is_configured(cfg: LLMConfig) -> tuple[bool, str]:
    """静态检查：能否发起真实调用。"""
    if cfg.mock:
        return True, "离线模拟模式"
    missing = []
    if not cfg.api_key:
        missing.append("LLM_API_KEY")
    if not cfg.base_url:
        missing.append("LLM_BASE_URL")
    if not cfg.model:
        missing.append("LLM_MODEL")
    if missing:
        return False, "缺少配置：" + ", ".join(missing)
    return True, cfg.describe()


def chat(cfg: LLMConfig, system: str, user: str,
         mock_response: str | None = None,
         history: list | None = None) -> LLMResponse:
    """
    调用 chat/completions。失败不抛异常，通过 LLMResponse.ok 表达。

    history: 可选的多轮消息（不含 system），用于"数字校验失败后重试"。
    """
    resp = LLMResponse(model=cfg.model)
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    if user:
        messages.append({"role": "user", "content": user})

    # ---------- 离线模拟 ----------
    if cfg.mock:
        t0 = time.time()
        resp.ok = True
        resp.text = mock_response if mock_response is not None else "（离线模拟：未注入回复）"
        resp.latency = time.time() - t0
        resp.model = "mock"
        resp.finish_reason = "stop"
        resp.prompt_tokens = max(1, (len(system) + len(user)) // 4)
        resp.completion_tokens = max(1, len(resp.text) // 4)
        return resp

    # ---------- 配置检查 ----------
    ok, why = is_configured(cfg)
    if not ok:
        resp.error = why
        return resp

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {cfg.api_key}"}

    last_err = ""
    t0 = time.time()
    for attempt in range(cfg.max_retries + 1):
        try:
            r = requests.post(url, headers=headers,
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              timeout=cfg.timeout)
            if r.status_code == 200:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                resp.text = (choice.get("message") or {}).get("content", "") or ""
                resp.finish_reason = choice.get("finish_reason", "")
                u = data.get("usage") or {}
                resp.prompt_tokens = int(u.get("prompt_tokens", 0) or 0)
                resp.completion_tokens = int(u.get("completion_tokens", 0) or 0)
                if cfg.prompt_price or cfg.completion_price:
                    resp.cost = (resp.prompt_tokens / 1e6 * cfg.prompt_price +
                                 resp.completion_tokens / 1e6 * cfg.completion_price)
                resp.latency = time.time() - t0
                resp.ok = bool(resp.text.strip())
                if not resp.ok:
                    resp.error = "返回内容为空"
                return resp
            # 4xx（除 429）不重试
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code < 500 and r.status_code != 429:
                break
        except requests.Timeout:
            last_err = f"请求超时（{cfg.timeout}s）"
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {str(ex)[:160]}"
        if attempt < cfg.max_retries:
            time.sleep(1.5 * (attempt + 1))   # 指数退避

    resp.latency = time.time() - t0
    resp.error = last_err or "未知错误"
    return resp


if __name__ == "__main__":
    cfg = from_env()
    ok, why = is_configured(cfg)
    print("配置状态:", ok, "|", why)
    r = chat(cfg, "你是助手", "用一句话说明什么是营销组合模型。",
             mock_response="营销组合模型是一种用统计方法量化各广告渠道对销量贡献的分析框架。")
    print("调用结果:", r.ok, "|", r.usage_line())
    if r.ok:
        print("返回:", r.text[:120])
