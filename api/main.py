"""HTTP API for the frontend.

  POST /api/research                {question} → {session_id, stream_url}
  GET  /api/research/{id}/stream    server-sent events: agent_update …, then final or error
  GET  /api/sessions                recent sessions
  GET  /api/sessions/{id}           one session's events (replay)
  GET  /api/health

Research runs in a background task. Each agent step is written to episodic
memory as it happens, and the stream tails those rows, so a client that
connects late (or reconnects) still gets every event from the start.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import run_research
from api.service import Services, build_services

log = logging.getLogger("meridian.api")
services: Services | None = None
_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global services
    if services is None:
        services = build_services()
    yield


app = FastAPI(title="Meridian", lifespan=lifespan)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


async def _run(session_id: str, question: str) -> None:
    svc = services
    assert svc is not None

    async def emit(agent: str, content: str, data: dict) -> None:
        svc.episodic.event(session_id, "agent_update", agent, content, {k: v for k, v in data.items() if k != "contexts"})

    before = svc.meter.total()
    try:
        state = await run_research(question, svc.runtime(emit=emit), session_id)
        after = svc.meter.total()
        usage = {"llm_calls": after.calls - before.calls,
                 "tokens": (after.prompt_tokens + after.completion_tokens) - (before.prompt_tokens + before.completion_tokens)}
        citations = [c.model_dump() for c in state.citations]
        svc.episodic.event(session_id, "final", "synthesizer", state.answer,
                           {"answer": state.answer, "citations": citations, "confidence_scores": state.confidence,
                            "usage": usage})
        svc.episodic.finish(session_id, "done", state.answer, citations, state.confidence, usage)
    except Exception as e:  # noqa: BLE001 - report any failure to the client
        log.exception("research failed")
        svc.episodic.event(session_id, "error", "system", f"{type(e).__name__}: {e}", {})
        svc.episodic.finish(session_id, "error")


@app.post("/api/research")
async def start_research(req: ResearchRequest) -> dict:
    assert services is not None
    session_id = uuid.uuid4().hex
    services.episodic.start(session_id, req.question)
    task = asyncio.create_task(_run(session_id, req.question))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"session_id": session_id, "stream_url": f"/api/research/{session_id}/stream"}


@app.get("/api/research/{session_id}/stream")
async def stream(session_id: str) -> StreamingResponse:
    assert services is not None
    if services.episodic.status(session_id) is None:
        raise HTTPException(404, "unknown session")

    async def events():
        last = -1
        while True:
            for e in services.episodic.events(session_id, after=last):
                last = e["seq"]
                yield f"event: {e['type']}\ndata: {json.dumps(e, default=str)}\n\n"
                if e["type"] in ("final", "error"):
                    return
            if services.episodic.status(session_id) != "running" and not services.episodic.events(session_id, last):
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions")
async def sessions() -> dict:
    assert services is not None
    return {"sessions": services.episodic.sessions()}


@app.get("/api/sessions/{session_id}")
async def session(session_id: str) -> dict:
    assert services is not None
    if services.episodic.status(session_id) is None:
        raise HTTPException(404, "unknown session")
    return {"session_id": session_id, "status": services.episodic.status(session_id),
            "events": services.episodic.events(session_id)}


@app.get("/api/health")
async def health() -> dict:
    assert services is not None
    return {"status": "ok", "tools": sorted(services.tools),
            "models": {r: c.model for r, c in services.llms.items()},
            "tool_stats": services.procedural.stats()}
