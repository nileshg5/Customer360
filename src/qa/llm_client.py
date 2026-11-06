"""Groq chat client with model fallback. The API key is read from the environment only."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

import groq

from src.config import Settings

log = logging.getLogger("qa.llm")


class LlmError(RuntimeError):
    """The LLM could not produce a reply (all models failed or the key was rejected)."""


@dataclass(frozen=True)
class LlmReply:
    text: str
    model: str


StatusCallback = Optional[Callable[[str], None]]


class ChatClient(Protocol):
    def complete(self, messages: List[Dict[str, str]], on_status: StatusCallback = None) -> LlmReply: ...


def _should_fall_back(exc: Exception) -> bool:
    """Model/availability problems move on to the fallback model; auth and request errors do not."""
    if isinstance(exc, (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError,
                        groq.InternalServerError, groq.NotFoundError)):
        return True
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code >= 500:
            return True
        text = str(exc).lower()
        return "model_decommissioned" in text or "model_not_found" in text or "decommissioned" in text
    return False


def _retry_after_seconds(exc: Exception, cap: float) -> float:
    """Seconds from the Retry-After header (default 2), capped."""
    try:
        value = float(exc.response.headers.get("retry-after", 2))      # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        value = 2.0
    return max(0.0, min(value, cap))


class _AllRateLimited(Exception):
    def __init__(self, last: Exception) -> None:
        super().__init__(type(last).__name__)
        self.last = last


class GroqChatClient:
    def __init__(self, api_key: str, models: List[str], temperature: float, timeout: float,
                 reasoning_effort: Optional[str] = None, sdk_client=None,
                 max_retry_wait: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.models = models
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_retry_wait = max_retry_wait
        self._sleep = sleep
        self._client = sdk_client or groq.Groq(api_key=api_key, timeout=timeout, max_retries=0)

    def complete(self, messages: List[Dict[str, str]], on_status: StatusCallback = None) -> LlmReply:
        """Try each model in order. If every model is rate limited, wait once (Retry-After,
        capped) and try the whole list again."""
        try:
            return self._try_models(messages)
        except _AllRateLimited as rl:
            wait = _retry_after_seconds(rl.last, self.max_retry_wait)
            log.warning("all models rate limited; retrying once in %.1fs", wait)
            if on_status:
                on_status(f"Rate limited by the AI service, retrying in {wait:.0f}s...")
            self._sleep(wait)
            try:
                return self._try_models(messages)
            except _AllRateLimited as again:
                raise LlmError("all models are rate limited (RateLimitError)") from again.last

    def _try_models(self, messages: List[Dict[str, str]]) -> LlmReply:
        last: Optional[Exception] = None
        all_rate_limited = True
        for model in self.models:
            kwargs = dict(model=model, messages=messages, temperature=self.temperature,
                          response_format={"type": "json_object"})
            if self.reasoning_effort and model.startswith("openai/gpt-oss"):
                kwargs.update(reasoning_effort=self.reasoning_effort, include_reasoning=False)
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return LlmReply(text=resp.choices[0].message.content or "", model=model)
            except groq.AuthenticationError as exc:
                raise LlmError("the Groq API key was rejected") from exc
            except Exception as exc:                       # noqa: BLE001 - classified below
                if not _should_fall_back(exc):
                    raise LlmError(f"LLM request failed ({type(exc).__name__})") from exc
                log.warning("model %s unavailable (%s); trying next model", model, type(exc).__name__)
                all_rate_limited = all_rate_limited and isinstance(exc, groq.RateLimitError)
                last = exc
        if all_rate_limited and last is not None:
            raise _AllRateLimited(last)
        raise LlmError(f"all models failed (last error: {type(last).__name__})")


def client_from_env(cfg: Settings) -> Optional[GroqChatClient]:
    """Return a client if the API key env var is set, else None (the app falls back to presets)."""
    key = (os.environ.get(cfg.llm.api_key_env) or "").strip()
    if not key:
        return None
    models = [cfg.llm.model] + ([cfg.llm.fallback_model] if cfg.llm.fallback_model else [])
    return GroqChatClient(key, models, cfg.llm.temperature, cfg.llm.timeout_seconds,
                          cfg.llm.reasoning_effort, max_retry_wait=cfg.llm.max_retry_wait_seconds)
