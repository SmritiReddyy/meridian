from tools.arxiv_tool import ArxivTool
from tools.base import Tool, ToolError
from tools.code_executor import PythonTool
from tools.pdf_parser import PdfTool
from tools.web_search import DailyQuota, WebSearchTool
from tools.wikipedia_tool import WikipediaTool


def build_tools(enabled: list[str], timeout_s: float = 15, chunk_chars: int = 1200,
                quota_path: str = ":memory:", web_daily_limit: int = 100) -> dict[str, Tool]:
    """The tools named in config. Web search carries its own daily quota."""
    candidates: dict[str, Tool] = {
        "wikipedia": WikipediaTool(pages=3, chunk_chars=chunk_chars, timeout_s=timeout_s),
        "arxiv": ArxivTool(timeout_s=timeout_s),
        "web_search": WebSearchTool(DailyQuota(quota_path, "web_search", web_daily_limit), timeout_s=timeout_s),
        "read_pdf": PdfTool(chunk_chars=chunk_chars, timeout_s=timeout_s * 2),
        "python": PythonTool(timeout_s=timeout_s),
    }
    return {name: t for name, t in candidates.items() if name in enabled}


def available(tools: dict[str, Tool]) -> list[str]:
    """Names of tools that can run right now (web search drops out once its daily quota is spent)."""
    return [n for n, t in tools.items() if getattr(t, "available", True)]


__all__ = ["ArxivTool", "DailyQuota", "PdfTool", "PythonTool", "Tool", "ToolError", "WebSearchTool",
           "WikipediaTool", "available", "build_tools"]
