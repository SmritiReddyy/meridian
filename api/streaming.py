```python
from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import AsyncIterator, Any

from pydantic import BaseModel, Field
from typing import Literal

from agents.state import ResearchState


class StreamEvent(BaseModel):
    type: Literal["agent_update", "finding", "thinking", "final", "error"]
    agent: str
    content: str
    data: dict = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


async def format_sse(event: StreamEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def stream_graph_events(
    graph: Any,
    state: ResearchState,
    session_id: str,
) -> AsyncIterator[str]:
    try:
        async for update in graph.astream(state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue

            for node_name, node_output in update.items():
                if not isinstance(node_output, dict):
                    continue

                # Yield agent_update events from agent_log entries
                agent_log: list[dict] = node_output.get("agent_log", [])
                for log_entry in agent_log:
                    if not isinstance(log_entry, dict):
                        continue

                    agent_name = log_entry.get("agent", node_name)
                    content = log_entry.get("content", "")
                    entry_type = log_entry.get("type", "agent_update")
                    extra_data = {
                        k: v
                        for k, v in log_entry.items()
                        if k not in {"agent", "content", "type"}
                    }
                    extra_data["session_id"] = session_id
                    extra_data["node"] = node_name

                    if entry_type not in {
                        "agent_update",
                        "finding",
                        "thinking",
                        "final",
                        "error",
                    }:
                        entry_type = "agent_update"

                    event = StreamEvent(
                        type=entry_type,  # type: ignore[arg-type]
                        agent=agent_name,
                        content=content,
                        data=extra_data,
                    )
                    yield await format_sse(event)

                # Yield a general agent_update for the node itself when there
                # is no agent_log but there is some visible state change.
                if not agent_log:
                    summary = node_output.get("summary") or node_output.get("message")
                    if summary:
                        event = StreamEvent(
                            type="agent_update",
                            agent=node_name,
                            content=str(summary),
                            data={"session_id": session_id, "node": node_name},
                        )
                        yield await format_sse(event)

                # Yield final event when the graph produces a final_answer
                final_answer = node_output.get("final_answer")
                if final_answer:
                    event = StreamEvent(
                        type="final",
                        agent=node_name,
                        content=str(final_answer),
                        data={
                            "session_id": session_id,
                            "node": node_name,
                            "sources": node_output.get("sources", []),
                        },
                    )
                    yield await format_sse(event)

                # Yield individual findings when present
                findings: list[Any] = node_output.get("findings", [])
                for finding in findings:
                    if isinstance(finding, dict):
                        finding_content = finding.get("content", str(finding))
                        finding_agent = finding.get("agent", node_name)
                        finding_data = {
                            k: v
                            for k, v in finding.items()
                            if k not in {"content", "agent"}
                        }
                        finding_data["session_id"] = session_id
                        finding_data["node"] = node_name
                    else:
                        finding_content = str(finding)
                        finding_agent = node_name
                        finding_data = {
                            "session_id": session_id,
                            "node": node_name,
                        }

                    event = StreamEvent(
                        type="finding",
                        agent=finding_agent,
                        content=finding_content,
                        data=finding_data,
                    )
                    yield await format_sse(event)

                # Yield thinking events when present
                thinking: str | None = node_output.get("thinking")
                if thinking:
                    event = StreamEvent(
                        type="thinking",
                        agent=node_name,
                        content=str(thinking),
                        data={"session_id": session_id, "node": node_name},
                    )
                    yield await format_sse(event)

                # Allow other coroutines to run between node updates
                await asyncio.sleep(0)

    except asyncio.CancelledError:
        # Client disconnected; re-raise so FastAPI can clean up
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        error_event = StreamEvent(
            type="error",
            agent="system",
            content=str(exc),
            data={
                "session_id": session_id,
                "traceback": tb,
            },
        )
        yield await format_sse(error_event)
```