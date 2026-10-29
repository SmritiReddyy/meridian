"""Tool interface. A tool takes one input string and returns findings."""
from __future__ import annotations

from typing import Protocol

import httpx

from agents.state import Finding

USER_AGENT = "Meridian/0.2 (research assistant; https://github.com/SmritiReddyy/meridian)"


class Tool(Protocol):
    name: str
    description: str   # shown to the researcher so it can choose

    async def run(self, query: str) -> list[Finding]: ...


class ToolError(RuntimeError):
    pass


def http_client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def chunk(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into pieces of at most max_chars."""
    out, cur = [], ""
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if len(para) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(para[i:i + max_chars] for i in range(0, len(para), max_chars))
            continue
        if len(cur) + len(para) + 1 > max_chars and cur:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out
