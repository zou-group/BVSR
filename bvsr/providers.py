from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import env, models

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

_pace_lock = threading.Lock()
_last_call = [0.0]


def pace() -> None:
    interval = float(env.get("TOGETHER_MIN_INTERVAL", "0"))
    if interval <= 0:
        return
    with _pace_lock:
        dt = time.time() - _last_call[0]
        if dt < interval:
            time.sleep(interval - dt)
        _last_call[0] = time.time()


def together_client(timeout: float):
    from openai import OpenAI
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=env.together_api_key(),
                  timeout=timeout, max_retries=0)


def _backoff(attempt: int, msg: str, timeouts: list[int], max_retries: int) -> None:
    m = msg.lower()
    if attempt >= max_retries - 1:
        raise _GiveUp
    if "429" in m or "rate" in m:
        time.sleep(min(60, 4 * 2 ** attempt) + random.uniform(0, 3))
    elif "timed out" in m or "timeout" in m or "read operation" in m:
        timeouts[0] += 1
        if timeouts[0] >= 2:
            raise _GiveUp
        time.sleep(1)
    elif any(c in m for c in ("500", "502", "503", "504", "overload", "connection")):
        time.sleep(min(30, 3 * 2 ** attempt) + random.uniform(0, 2))
    else:
        time.sleep(2)


class _GiveUp(Exception):
    pass


@dataclass
class ChatResult:
    text: str
    reasoning: str | None = None
    finish_reason: str | None = None


class Provider(ABC):
    model: str

    def __init__(self, model: str, *, timeout: float = 900, max_retries: int = 6,
                 reasoning_effort: str | None = None):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    @abstractmethod
    def _create(self, prompt: str, temperature: float, max_tokens: int):
        pass

    def _before_call(self) -> None:
        pass

    def complete(self, prompt: str, temperature: float = 0.8, max_tokens: int = 65536) -> ChatResult:
        timeouts = [0]
        for attempt in range(self.max_retries):
            try:
                self._before_call()
                r = self._create(prompt, temperature, max_tokens)
                choice = r.choices[0]
                msg = choice.message
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                return ChatResult(text=msg.content or "", reasoning=reasoning or None,
                                  finish_reason=getattr(choice, "finish_reason", None))
            except Exception as e:
                try:
                    _backoff(attempt, str(e), timeouts, self.max_retries)
                except _GiveUp:
                    raise e from None

    def chat(self, prompt: str, temperature: float = 0.8, max_tokens: int = 65536) -> str:
        r = self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        return f"<think>\n{r.reasoning}\n</think>\n{r.text}" if r.reasoning else r.text


class TogetherProvider(Provider):
    def __init__(self, model: str, *, timeout: float = 900, max_retries: int = 6,
                 reasoning_effort: str | None = None):
        super().__init__(model, timeout=timeout, max_retries=max_retries, reasoning_effort=reasoning_effort)
        self.client = together_client(timeout)

    def _before_call(self) -> None:
        pace()

    def _create(self, prompt, temperature, max_tokens):
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        return self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}], **extra)


class OpenAIProvider(Provider):
    def __init__(self, model: str, *, timeout: float = 900, max_retries: int = 6,
                 reasoning_effort: str | None = None):
        super().__init__(model, timeout=timeout, max_retries=max_retries, reasoning_effort=reasoning_effort)
        from openai import OpenAI
        self.client = OpenAI(api_key=env.openai_api_key(), timeout=timeout, max_retries=0)

    def _create(self, prompt, temperature, max_tokens):
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        return self.client.chat.completions.create(
            model=self.model, max_completion_tokens=int(max_tokens),
            messages=[{"role": "user", "content": prompt}], **extra)


def make_provider(model: str, **kw) -> TogetherProvider:
    model_id, default_effort = models.resolve(model)
    if kw.get("reasoning_effort", "auto") == "auto":
        kw["reasoning_effort"] = default_effort
    return TogetherProvider(model_id, **kw)
