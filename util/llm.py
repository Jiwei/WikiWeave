"""LLM 客户端:默认本机 vLLM qwen(openai 兼容),关 thinking,带重试与批处理并发。

参数全部来自 config/llm.json,凭据用环境变量覆盖。LLM 不可用/失败不抛异常、
由上层降级到关键词兜底,pipeline 永不中断。文件末尾提供兼容层,承接参考项目
indexing/llm.py 的旧接口(LlmConfig / _call_llm / call_llm),让搬移代码零改动。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from .paths import CONFIG_DIR


def _load_cfg() -> dict:
    path = CONFIG_DIR / "llm.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return {}


_CFG = _load_cfg()


def _api_key() -> str:
    env = _CFG.get("api_key_env", "LLM_API_KEY")
    return os.environ.get(env) or _CFG.get("api_key_default", "EMPTY")


def available() -> bool:
    return bool(_CFG.get("base_url"))


def phase_settings(phase: str) -> dict:
    ph = _CFG.get("phases", {}).get(phase, {})
    return {"batch_size": ph.get("batch_size", 20), "max_workers": ph.get("max_workers", 4)}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile(r"^.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = _THINK_TAIL_RE.sub("", text)
    return text.strip()


def _post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("x-api-key", _api_key())
    req.add_header("anthropic-version", "2023-06-01")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_payload(system: str, user: str) -> tuple[str, dict]:
    base = _CFG.get("base_url", "").rstrip("/")
    model = _CFG.get("model", "")
    temperature = _CFG.get("temperature", 0.0)
    max_tokens = _CFG.get("max_tokens", 4096)
    protocol = _CFG.get("protocol", "openai")

    if protocol == "anthropic":
        url = f"{base}/messages"
        payload = {
            "model": model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        }
        if _CFG.get("disable_thinking", True):
            payload["thinking"] = {"type": "disabled"}
        return url, payload

    url = f"{base}/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if _CFG.get("disable_thinking", True):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return url, payload


def _extract_text(resp: dict) -> str:
    if "choices" in resp:
        return resp["choices"][0]["message"]["content"] or ""
    if "content" in resp and isinstance(resp["content"], list):
        return "".join(b.get("text", "") for b in resp["content"])
    return ""


def chat(user: str, system: str = "") -> str:
    """一次对话补全。失败(重试耗尽)返回空串,不抛异常。"""
    if not available():
        return ""
    url, payload = _build_payload(system, user)
    timeout = _CFG.get("timeout", 120)
    retry = _CFG.get("retry", {})
    attempts = int(retry.get("max_attempts", 3))
    backoff = float(retry.get("backoff_seconds", 2))
    for i in range(attempts):
        try:
            resp = _post(url, payload, timeout)
            return _strip_think(_extract_text(resp))
        except Exception as e:
            if i == attempts - 1:
                import logging
                logging.getLogger(__name__).warning("LLM 调用失败(降级): %s", e)
                return ""
            time.sleep(backoff * (2 ** i))
    return ""


def extract_json(text: str) -> Any:
    """从 LLM 回复里抠出 JSON(数组优先,其次对象)。失败返回 None。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def batch_map(items: list, worker: Callable[[list], list], phase: str = "classify",
              fallback: Callable[[list], list] | None = None) -> list:
    """把 items 切成 batch,线程池并发跑 worker(batch)->results,按序对齐返回。"""
    ps = phase_settings(phase)
    bs, mw = ps["batch_size"], ps["max_workers"]
    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    results: list[Any] = [None] * len(batches)

    def _run(idx: int, batch: list):
        try:
            return idx, worker(batch)
        except Exception:
            return idx, (fallback(batch) if fallback else [None] * len(batch))

    with ThreadPoolExecutor(max_workers=mw) as ex:
        futs = [ex.submit(_run, i, b) for i, b in enumerate(batches)]
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res

    out: list = []
    for res in results:
        out.extend(res or [])
    return out


# ── 兼容层:承接参考项目 indexing/llm.py 的旧接口(搬移代码零改动) ──

@dataclass
class LlmConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    available: bool = True

    @classmethod
    def from_env(cls) -> "LlmConfig":
        return cls(
            api_key=_api_key(),
            base_url=_CFG.get("base_url", ""),
            model=_CFG.get("model", ""),
            max_tokens=_CFG.get("max_tokens", 4096),
            available=available(),
        )


def _call_llm(prompt: str, cfg: "LlmConfig | None" = None, *, system: str = "") -> str:
    """参考 indexing/llm.py 的真实签名 (prompt, cfg) → 返回文本响应(失败返回空串)。

    全仓库调用点均为 ``_call_llm(prompt, cfg)`` 形式;之前的兼容层误写成
    ``(cfg, system, user)``,导致 hook/naming-key/artifact 等处 LLM 增强调用报
    “missing 1 required positional argument: 'user'” 而降级。此处纠正签名。
    """
    if cfg is not None and getattr(cfg, "available", True) is False:
        return ""
    return chat(prompt, system=system)


def call_llm(prompt: str, cfg: "LlmConfig | None" = None, **kwargs) -> str:
    return _call_llm(prompt, cfg)
