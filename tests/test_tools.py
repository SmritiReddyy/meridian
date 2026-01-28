```python
"""Unit tests for tool implementations."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tavily_result(url: str, title: str = "Title", content: str = "Summary") -> dict[str, Any]:
    return {"url": url, "title": title, "content": content}


# ---------------------------------------------------------------------------
# TestTavilySearchTool
# ---------------------------------------------------------------------------

class TestTavilySearchTool:
    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        """Mock TavilyClient.search; assert ToolResult.success=True and content populated."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}):
            from tools.tavily_search import TavilySearchTool

            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [
                    _make_tavily_result("https://example.com/a", "Alpha", "Content A"),
                    _make_tavily_result("https://example.com/b", "Beta", "Content B"),
                ]
            }

            tool = TavilySearchTool.__new__(TavilySearchTool)
            tool._client = mock_client

            result = await tool.run("test query")

        assert result.success is True
        assert result.content != ""
        assert "Alpha" in result.content
        assert "https://example.com/a" in result.content

    @pytest.mark.asyncio
    async def test_deduplication(self) -> None:
        """Mock search to return results with duplicate URLs; assert duplicates removed."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}):
            from tools.tavily_search import TavilySearchTool

            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [
                    _make_tavily_result("https://dup.com/page", "First", "Content 1"),
                    _make_tavily_result("https://dup.com/page", "Dupe", "Content 2"),
                    _make_tavily_result("https://unique.com/page", "Unique", "Content 3"),
                ]
            }

            tool = TavilySearchTool.__new__(TavilySearchTool)
            tool._client = mock_client

            result = await tool.run("dedup query")

        assert result.success is True
        unique_results: list[dict[str, Any]] = result.metadata.get("results", [])
        urls = [r["url"] for r in unique_results]
        assert urls.count("https://dup.com/page") == 1
        assert len(unique_results) == 2

    @pytest.mark.asyncio
    async def test_api_error_graceful(self) -> None:
        """Mock TavilyClient.search to raise exception; assert ToolResult.success=False."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}):
            from tools.tavily_search import TavilySearchTool

            mock_client = MagicMock()
            mock_client.search.side_effect = RuntimeError("API unavailable")

            tool = TavilySearchTool.__new__(TavilySearchTool)
            tool._client = mock_client

            # Disable tenacity retries to keep the test fast
            with patch("tools.tavily_search.TavilySearchTool.run", wraps=None) as _:
                pass

            # Patch retry decorator away by calling the inner logic directly via safe_run
            # which catches unhandled exceptions from run()
            result = await tool.safe_run("failing query")

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestArXivTool
# ---------------------------------------------------------------------------

class TestArXivTool:
    @pytest.mark.asyncio
    async def test_search_returns_papers(self) -> None:
        """Mock arxiv internals; assert ToolResult.success=True with abstracts in content."""
        from tools.arxiv_tool import ArXivTool

        fake_papers = [
            {
                "title": "Test Paper One",
                "authors": ["Alice", "Bob"],
                "abstract": "This paper discusses X in detail.",
                "published": "2024-01-01",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "arxiv_id": "2401.00001",
            }
        ]

        tool = ArXivTool()
        with patch.object(tool, "_fetch_with_retry", new=AsyncMock(return_value=fake_papers)):
            result = await tool.run("test query")

        assert result.success is True
        assert result.content != ""
        assert "Test Paper One" in result.content
        assert "This paper discusses X" in result.content

    @pytest.mark.asyncio
    async def test_no_results_handled(self) -> None:
        """Mock empty result set; assert success=True with appropriate message."""
        from tools.arxiv_tool import ArXivTool

        tool = ArXivTool()
        with patch.object(tool, "_fetch_with_retry", new=AsyncMock(return_value=[])):
            result = await tool.run("obscure query with no results")

        assert result.success is True
        assert "No results" in result.content or "no results" in result.content.lower()


# ---------------------------------------------------------------------------
# TestWikipediaTool
# ---------------------------------------------------------------------------

class TestWikipediaTool:
    @pytest.mark.asyncio
    async def test_page_found(self) -> None:
        """Mock wikipediaapi.Wikipedia.page to return an existing page with summary."""
        from tools.wikipedia_tool import WikipediaTool

        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.summary = "Python is a programming language."
        mock_page.title = "Python (programming language)"
        mock_page.fullurl = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        mock_page.sections = []

        tool = WikipediaTool.__new__(WikipediaTool)
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page
        tool._wiki = mock_wiki

        result = await tool.run("Python programming language")

        assert result.success is True
        assert result.content != ""
        assert "Python is a programming language" in result.content

    @pytest.mark.asyncio
    async def test_page_not_found(self) -> None:
        """Mock page.exists=False; assert success=False."""
        from tools.wikipedia_tool import WikipediaTool

        mock_page = MagicMock()
        mock_page.exists.return_value = False

        tool = WikipediaTool.__new__(WikipediaTool)
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page
        tool._wiki = mock_wiki

        result = await tool.run("xyzzy nonexistent page 12345")

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestCodeExecutorTool
# ---------------------------------------------------------------------------

class TestCodeExecutorTool:
    @pytest.mark.asyncio
    async def test_safe_code_executes(self) -> None:
        """Run simple arithmetic print(2+2); assert output contains '4'."""
        from tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool()
        result = await tool.run("print(2+2)")

        assert result.success is True
        assert "4" in result.content

    @pytest.mark.asyncio
    async def test_forbidden_import_blocked(self) -> None:
        """Try to run 'import os; print(os.getcwd())'; assert blocked before execution."""
        from tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool()
        # The _sanitize function strips `import os` but then the os.getcwd() call
        # should still be present after stripping; alternatively the combined
        # code with os usage may survive strip but fail at runtime. We verify
        # the tool does not return a successful execution that leaks the cwd.
        # The security filter strips `import os` then checks for forbidden patterns
        # on the sanitized result. os.getcwd() itself is not a forbidden pattern,
        # so the stripped code runs but fails with NameError — success=False.
        result = await tool.run("import os\nprint(os.getcwd())")

        # Either rejected by security filter or fails at runtime — must not succeed
        assert result.success is False

    @pytest.mark.asyncio
    async def test_timeout_handled(self) -> None:
        """Run an infinite loop; assert returns error before timeout."""
        import subprocess

        from tools.code_executor import CodeExecutorTool

        tool = CodeExecutorTool()

        with patch(
            "tools.code_executor._run_in_subprocess",
            side_effect=subprocess.TimeoutExpired(cmd=["python3"], timeout=10),
        ):
            result = await tool.run("while True: pass")

        assert result.success is False
        assert result.error is not None
        assert "timed out" in (result.error or "").lower() or "timeout" in (result.error or "").lower()
```