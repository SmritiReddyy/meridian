"""Web search with a hard daily quota.

Backends, in order of preference:
  * Tavily, when TAVILY_API_KEY is set;
  * You.com's free MCP endpoint (`you-search`), which needs no key.

Either way, at most `daily_limit` searches run per calendar day (UTC),
counted in SQLite so the cap survives restarts. Once the day's quota is
spent the tool reports itself unavailable and the researcher stops offering
it; nothing over the limit is ever sent.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from agents.state import Finding
from tools.base import ToolError, http_client

YOU_MCP_URL = "https://api.you.com/mcp?profile=free"


class DailyQuota:
    def __init__(self, path: str, name: str, limit: int) -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS quota (name TEXT, day TEXT, used INTEGER, PRIMARY KEY (name, day))")
        self._lock = threading.Lock()
        self.name = name
        self.limit = limit

    @staticmethod
    def today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def used(self) -> int:
        row = self._db.execute("SELECT used FROM quota WHERE name = ? AND day = ?", (self.name, self.today())).fetchone()
        return row[0] if row else 0

    def remaining(self) -> int:
        return max(self.limit - self.used(), 0)

    def take(self) -> bool:
        """Reserve one call; False if the day's quota is spent. Counted before the call is sent."""
        with self._lock:
            if self.used() >= self.limit:
                return False
            self._db.execute("INSERT INTO quota VALUES (?, ?, 1) ON CONFLICT(name, day) DO UPDATE SET used = used + 1",
                             (self.name, self.today()))
            self._db.commit()
            return True


class WebSearchTool:
    name = "web_search"
    description = ("General web search. Good for current events and topics outside encyclopedias and papers. "
                   "Limited daily budget: use only when wikipedia and arxiv will not have it.")

    def __init__(self, quota: DailyQuota, max_results: int = 5, timeout_s: float = 20) -> None:
        self.quota = quota
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.tavily_key = os.environ.get("TAVILY_API_KEY", "")
        self.backend = "tavily" if self.tavily_key else "you.com"

    @property
    def available(self) -> bool:
        return self.quota.remaining() > 0

    async def run(self, query: str) -> list[Finding]:
        if not self.quota.take():
            raise ToolError(f"web search daily limit of {self.quota.limit} reached")
        items = await (self._tavily(query) if self.tavily_key else self._you(query))
        seen, out = set(), []
        for title, url, content in items:
            if not url or url in seen or not content:
                continue
            seen.add(url)
            out.append(Finding(title=title or url, url=url, content=content, tool=self.name, query=query))
        return out[: self.max_results]

    async def _tavily(self, query: str) -> list[tuple[str, str, str]]:
        async with http_client(self.timeout_s) as c:
            r = await c.post("https://api.tavily.com/search",
                             json={"api_key": self.tavily_key, "query": query, "max_results": self.max_results})
        if r.status_code != 200:
            raise ToolError(f"tavily {r.status_code}")
        return [(i.get("title", ""), i.get("url", ""), i.get("content", "")) for i in r.json().get("results", [])]

    async def _you(self, query: str) -> list[tuple[str, str, str]]:
        """One `you-search` tool call over MCP's streamable HTTP transport."""
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        async with http_client(self.timeout_s) as c:
            init = await c.post(YOU_MCP_URL, headers=headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "meridian", "version": "0.2"}}})
            if init.status_code != 200:
                raise ToolError(f"you.com initialize {init.status_code}")
            if sid := init.headers.get("mcp-session-id"):
                headers["Mcp-Session-Id"] = sid
            await c.post(YOU_MCP_URL, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            r = await c.post(YOU_MCP_URL, headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "you-search", "arguments": {"query": query, "count": self.max_results}}})
        if r.status_code != 200:
            raise ToolError(f"you.com search {r.status_code}")
        result = mcp_result(r.text, r.headers.get("content-type", ""), request_id=2)
        if result.get("isError"):
            raise ToolError("you.com: " + " ".join(c.get("text", "") for c in result.get("content", []))[:300])
        return parse_you_results(result)


def mcp_result(body: str, content_type: str, request_id: int) -> dict:
    """The JSON-RPC result for request_id, from a JSON body or an SSE stream of several messages."""
    messages = []
    if "event-stream" in content_type:
        messages = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:") and line[5:].strip()]
    elif body.strip():
        parsed = json.loads(body)
        messages = parsed if isinstance(parsed, list) else [parsed]
    for m in messages:
        if m.get("id") == request_id:
            if "error" in m:
                raise ToolError(f"you.com: {m['error'].get('message', m['error'])}")
            return m.get("result", {})
    raise ToolError("you.com: no result in response")


def parse_you_results(result: dict) -> list[tuple[str, str, str]]:
    """(title, url, text) from you-search output: structured JSON if present, else its text content."""
    payload = result.get("structuredContent")
    if not payload:
        for c in result.get("content", []):
            if c.get("type") == "text":
                try:
                    payload = json.loads(c["text"])
                    break
                except (ValueError, KeyError):
                    continue
    if not isinstance(payload, dict):
        return []
    hits = payload.get("results", payload)
    rows = []
    for bucket in ("web", "news"):
        for h in (hits.get(bucket, []) if isinstance(hits, dict) else []):
            highlights = (h.get("contents") or {}).get("highlights") or []
            text = "\n".join(highlights) or " ".join(h.get("snippets") or []) or h.get("description", "")
            rows.append((h.get("title", ""), h.get("url", ""), text[:1500]))
    return rows
