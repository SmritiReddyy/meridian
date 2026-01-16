/Users/usmri/Desktop/clm/meridian/agents/researcher.py

```python
"""Researcher agent node for the Meridian LangGraph pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import anthropic

from agents.state import AgentMessage, Finding, ResearchState
from memory.semantic import SemanticMemory, check_semantic_cache
from tools import ArXivTool, TavilySearchTool, WikipediaTool
```

Key design decisions:

- **Semantic cache first** — calls `check_semantic_cache(semantic_memory, sub_task)` before touching any external API; on a hit it short-circuits and returns immediately.
- **Tool execution** — iterates over all `tool_use` blocks Claude returns. Each tool call is wrapped in `_call_with_retry` (3 attempts, exponential backoff at 1s/2s/4s). If a tool exhausts retries, its error is logged and execution continues with the remaining tools (graceful degradation).
- **Normalisation** — tool results may be `str`, `dict`, or `list[dict]`; a small normaliser converts all three into `Finding` objects with `content`, `source`, `title`, `tool`, `sub_task`, and `retrieved_at`.
- **Deduplication** — SHA-256 of whitespace-normalised lowercased content; duplicates across the accumulated `all_findings` list are dropped before returning.
- **State update** — returns `all_findings` (merged+deduped), `current_task_idx` incremented by 1, and `agent_log` with timestamped `AgentMessage` entries covering cache hits, Claude invocation failures, per-tool success/failure, and the final summary.