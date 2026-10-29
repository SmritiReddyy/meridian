"""Wikipedia: full-text search, then the plain-text article of the top hits, in passages."""
from __future__ import annotations

import asyncio

from agents.state import Finding
from tools.base import ToolError, chunk, http_client

API = "https://en.wikipedia.org/w/api.php"


class WikipediaTool:
    name = "wikipedia"
    description = "Encyclopedia articles. Good for definitions, history, people, established facts."

    def __init__(self, pages: int = 2, chunk_chars: int = 1200, timeout_s: float = 15) -> None:
        self.pages = pages
        self.chunk_chars = chunk_chars
        self.timeout_s = timeout_s

    async def run(self, query: str) -> list[Finding]:
        async with http_client(self.timeout_s) as c:
            r = await c.get(API, params={"action": "query", "list": "search", "srsearch": query,
                                         "srlimit": self.pages, "format": "json"})
            if r.status_code != 200:
                raise ToolError(f"wikipedia search {r.status_code}")
            titles = [h["title"] for h in r.json().get("query", {}).get("search", [])]
            # The extracts API returns full text for only one page per request,
            # so fetch each hit separately (and keep the search order).
            texts = await asyncio.gather(*(self._extract(c, t) for t in titles))
        findings = []
        for title, text in zip(titles, texts):
            # Drop the reference sections at the end of the article.
            for marker in ("\n== See also ==", "\n== References ==", "\n== Notes ==", "\n== External links =="):
                text = text.split(marker)[0]
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            for piece in chunk(text, self.chunk_chars):
                if piece.startswith("=="):
                    continue
                findings.append(Finding(title=title, url=url, content=piece, tool=self.name, query=query))
        return findings

    async def _extract(self, c, title: str) -> str:
        r = await c.get(API, params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                     "format": "json", "redirects": 1})
        if r.status_code != 200:
            raise ToolError(f"wikipedia extract {r.status_code}")
        pages = list(r.json().get("query", {}).get("pages", {}).values())
        return (pages[0].get("extract") or "") if pages else ""
