"""arXiv: search the export API and return each paper's abstract."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from agents.state import Finding
from tools.base import ToolError, http_client

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "does", "do", "for", "from", "how", "in", "is", "it", "of",
             "on", "or", "the", "to", "vs", "versus", "what", "which", "who", "why", "with", "between", "compare",
             "comparison", "difference", "differences"}


class ArxivTool:
    name = "arxiv"
    description = "Research paper abstracts. Good for methods, recent results, who proposed what."

    def __init__(self, max_results: int = 4, timeout_s: float = 20) -> None:
        self.max_results = max_results
        self.timeout_s = timeout_s

    @staticmethod
    def strict_query(query: str) -> str:
        """Every keyword required: `all:a AND all:b`. A bare `all:a b` matches papers with any of them."""
        # Years and other bare numbers are left out: arXiv does not match them as keywords,
        # so requiring one ("Lewis 2020") would make the strict search find nothing.
        terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query)
                 if t.lower() not in STOPWORDS and not t.isdigit()][:6]
        return " AND ".join(f"all:{t}" for t in terms) or f"all:{query}"

    async def run(self, query: str) -> list[Finding]:
        async with http_client(self.timeout_s) as c:
            entries = []
            # Strict first; fall back to the loose search if nothing has every term.
            for q in (self.strict_query(query), f"all:{query}"):
                r = await c.get(API, params={"search_query": q, "start": 0,
                                             "max_results": self.max_results, "sortBy": "relevance"})
                if r.status_code != 200:
                    raise ToolError(f"arxiv {r.status_code}")
                entries = ET.fromstring(r.text).findall("a:entry", NS)
                if entries:
                    break
        return self._parse(entries, query)

    def _parse(self, entries: list, query: str) -> list[Finding]:
        findings = []
        for e in entries:
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
            url = e.findtext("a:id", "", NS) or ""
            year = (e.findtext("a:published", "", NS) or "")[:4]
            authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)][:3]
            byline = ", ".join(authors) + (" et al." if len(e.findall("a:author", NS)) > 3 else "")
            if title and summary:
                findings.append(Finding(title=f"{title} ({byline}, {year})", url=url, content=summary,
                                        tool=self.name, query=query))
        return findings
