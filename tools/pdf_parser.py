"""Read a PDF from a URL (e.g. an arXiv paper) and return its text in passages."""
from __future__ import annotations

import io

from pypdf import PdfReader

from agents.state import Finding
from tools.base import ToolError, chunk, http_client

MAX_BYTES = 25 * 1024 * 1024


class PdfTool:
    name = "read_pdf"
    description = "Full text of a PDF given its URL. Use for a paper already found, to read past the abstract."

    def __init__(self, max_pages: int = 12, chunk_chars: int = 1200, timeout_s: float = 30) -> None:
        self.max_pages = max_pages
        self.chunk_chars = chunk_chars
        self.timeout_s = timeout_s

    async def run(self, query: str) -> list[Finding]:
        url = query.strip()
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/")
        if not url.startswith("http"):
            raise ToolError("read_pdf needs a URL")
        async with http_client(self.timeout_s) as c:
            r = await c.get(url)
        if r.status_code != 200:
            raise ToolError(f"pdf download {r.status_code}")
        if len(r.content) > MAX_BYTES:
            raise ToolError("pdf too large")
        return self.parse(r.content, url)

    def parse(self, data: bytes, url: str) -> list[Finding]:
        reader = PdfReader(io.BytesIO(data))
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else url)
        text = "\n".join((p.extract_text() or "") for p in reader.pages[: self.max_pages])
        return [Finding(title=str(title), url=url, content=piece, tool=self.name, query=url)
                for piece in chunk(text, self.chunk_chars)]
