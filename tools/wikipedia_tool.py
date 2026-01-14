/Users/usmri/Desktop/clm/meridian/tools/wikipedia_tool.py

Key implementation details:

- `WikipediaTool` subclasses `BaseTool`, sets `name = "wikipedia"` and the required description.
- `__init__` constructs a `wikipediaapi.Wikipedia("en", headers={"User-Agent": "Meridian/1.0"})` instance.
- `_fetch_page` is the synchronous retrieval method decorated with `@retry` (tenacity: up to 3 attempts, exponential backoff 1-8s, reraises on exhaustion).
- `_fetch_page_async` wraps it via `asyncio.to_thread` so the blocking library call doesn't block the event loop.
- `run(self, query, sections=3)` tries the raw query first, then falls back to `_build_variations` (title-case, capitalize, underscore, stripped punctuation) before returning a failure `ToolResult`.
- Content is assembled as Markdown: a `## Summary` block followed by the first `N` top-level sections (recursively rendered with `_extract_section_text`).
- Returns `ToolResult(tool_name, success, content, url, metadata)` matching the `base.py` model exactly; `metadata` carries `title`, `sections_fetched`, and `total_sections`.