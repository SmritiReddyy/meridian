"""Provider-agnostic chat client.

Every agent talks to an `LLM`: give it a system prompt and messages, get text
(or parsed JSON) back. `OpenAICompatibleLLM` covers Groq and any other
OpenAI-style endpoint; the provider is chosen in configs/default.yaml, so moving
to another model is a config change, not a code change.

Responses are cached on disk keyed by the full request, so re-running an
evaluation costs nothing, and rate-limit responses (429) are retried after the
delay the provider asks for. Token usage is counted per role for the
evaluation's cost and latency figures.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

Message = dict[str, str]


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.cached_calls += other.cached_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds


@dataclass
class UsageMeter:
    by_role: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))

    def record(self, role: str, u: Usage) -> None:
        self.by_role[role].add(u)

    def total(self) -> Usage:
        t = Usage()
        for u in self.by_role.values():
            t.add(u)
        return t


class LLM(Protocol):
    model: str

    async def complete(self, system: str, messages: list[Message], *, role: str = "",
                       json_mode: bool = False, max_tokens: int = 1024,
                       temperature: float = 0.1) -> str: ...


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> Any:
    """JSON from a model reply, tolerating code fences or prose around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start:end + 1])


class ResponseCache:
    """SQLite cache of completions keyed by a hash of the whole request."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, body TEXT)")

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT body FROM completions WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, body: dict[str, Any]) -> None:
        self._db.execute("INSERT OR REPLACE INTO completions VALUES (?, ?)", (key, json.dumps(body)))
        self._db.commit()


class OpenAICompatibleLLM:
    """Chat completions against an OpenAI-style endpoint (Groq by default)."""

    def __init__(self, model: str, *, base_url: str, api_key: str, meter: UsageMeter | None = None,
                 cache: ResponseCache | None = None, timeout_s: float = 120,
                 max_retries: int = 8, requests_per_minute: int = 0,
                 params: dict[str, Any] | None = None) -> None:
        self.model = model
        params = dict(params or {})
        self._max_tokens_floor = int(params.pop("max_tokens_floor", 0))
        self._params = params   # extra request fields, e.g. reasoning_effort
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._meter = meter or UsageMeter()
        self._cache = cache
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._rpm = requests_per_minute
        self._sent: collections.deque[float] = collections.deque()   # send times in the last minute
        self._pace = asyncio.Lock()

    @property
    def meter(self) -> UsageMeter:
        return self._meter

    async def _paced(self) -> None:
        """Sliding-window limit: bursts are fine, only the 60 s total is capped."""
        if not self._rpm:
            return
        async with self._pace:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= 60:
                self._sent.popleft()
            if len(self._sent) >= self._rpm:
                await asyncio.sleep(60 - (now - self._sent[0]))
                self._sent.popleft()
            self._sent.append(time.monotonic())

    async def complete(self, system: str, messages: list[Message], *, role: str = "",
                       json_mode: bool = False, max_tokens: int = 1024,
                       temperature: float = 0.1) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max(max_tokens, self._max_tokens_floor),
            "temperature": temperature,
            **self._params,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        key = ResponseCache.key(payload)
        if self._cache and (hit := self._cache.get(key)):
            self._meter.record(role, Usage(calls=1, cached_calls=1))
            return hit["content"]

        t0 = time.monotonic()
        body = await self._post(payload)
        elapsed = time.monotonic() - t0
        content = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage", {})
        self._meter.record(role, Usage(calls=1, prompt_tokens=usage.get("prompt_tokens", 0),
                                       completion_tokens=usage.get("completion_tokens", 0),
                                       seconds=elapsed))
        if self._cache:
            self._cache.put(key, {"content": content, "usage": usage})
        return content

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(self._max_retries):
                await self._paced()
                try:
                    r = await client.post(self._url, headers=self._headers, json=payload)
                except httpx.HTTPError as e:
                    delay = min(2 ** attempt, 30)
                    log.warning("LLM request failed (%s); retrying in %ss", e, delay)
                    await asyncio.sleep(delay)
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    delay = _retry_after(r) or min(2 ** attempt, 60)
                    try:
                        reason = r.json().get("error", {}).get("message", "")[:160]
                    except ValueError:
                        reason = ""
                    log.warning("LLM %s from provider (%s: %s); retrying in %.1fs", r.status_code, self.model,
                                reason, delay)
                    await asyncio.sleep(delay)
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"{r.status_code}: {r.text[:300]}")
                return r.json()
        raise LLMError(f"gave up after {self._max_retries} attempts")


def _retry_after(r: httpx.Response) -> float | None:
    """Seconds to wait, from Retry-After or Groq's x-ratelimit-reset-* headers ('1m2.5s', '450ms')."""
    for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = r.headers.get(h)
        if not v:
            continue
        try:
            return float(v)
        except ValueError:
            total = 0.0
            for amount, unit in re.findall(r"([\d.]+)(ms|s|m|h)", v):
                total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
            if total:
                return total + 0.5
    return None


class ScriptedLLM:
    """Test double: returns canned replies chosen by the role that asks."""

    def __init__(self, replies: dict[str, list[str] | str], model: str = "scripted") -> None:
        self.model = model
        self._replies = {k: ([v] if isinstance(v, str) else list(v)) for k, v in replies.items()}
        self.calls: list[tuple[str, str, list[Message]]] = []
        self.meter = UsageMeter()

    async def complete(self, system: str, messages: list[Message], *, role: str = "",
                       json_mode: bool = False, max_tokens: int = 1024,
                       temperature: float = 0.1) -> str:
        self.calls.append((role, system, messages))
        self.meter.record(role, Usage(calls=1))
        queue = self._replies.get(role)
        if not queue:
            raise LLMError(f"no scripted reply for role {role!r}")
        return queue.pop(0) if len(queue) > 1 else queue[0]
