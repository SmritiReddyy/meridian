"""Researcher: gathers evidence for each pending sub-task.

For each sub-task: reuse earlier findings if semantic memory has a close enough
match; otherwise ask the model which tools to call with what input (it sees
each tool's track record), run those calls concurrently, and keep the passages
most similar to the sub-question. A tool that fails or times out is recorded
and skipped; the others still count.
"""
from __future__ import annotations

import asyncio
import time

from agents.runtime import Runtime
from agents.state import AgentEvent, Finding, ResearchState, SubTask
from llm import parse_json
from tools import ToolError, available

SYSTEM = """You choose research tools. For the sub-question, pick 1 to 3 tool calls that will find
evidence to answer it. Tools:
{tools}

Track record so far:
{record}

Rules: write inputs as search queries (keywords, not full sentences), except python (code that
prints its result) and read_pdf (a PDF or arXiv URL). Prefer different tools or different
angles over repeating one query.

Reply with JSON only: {{"calls": [{{"tool": "<name>", "input": "<query>"}}, ...]}}"""


async def _choose_calls(task: SubTask, rt: Runtime, context: str) -> list[tuple[str, str]]:
    names = rt.procedural.rank(available(rt.tools))
    tools_text = "\n".join(f"- {n}: {rt.tools[n].description}" for n in names)
    prompt = f"Overall question: {context}\nSub-question: {task.question}"
    reply = await rt.llms["researcher"].complete(
        SYSTEM.format(tools=tools_text, record=rt.procedural.describe(names)),
        [{"role": "user", "content": prompt}], role="researcher", json_mode=True, max_tokens=300)
    calls: list[tuple[str, str]] = []
    try:
        for c in parse_json(reply).get("calls", [])[:3]:
            if c.get("tool") in names and str(c.get("input", "")).strip():
                calls.append((c["tool"], str(c["input"]).strip()))
    except (ValueError, AttributeError):
        pass
    # Never come back empty-handed because of a bad reply: fall back to the best-ranked search tool.
    return calls or [(next(n for n in names if n not in ("python", "read_pdf")), task.question)]


async def _run_call(tool: str, query: str, rt: Runtime) -> tuple[list[Finding], str | None]:
    """Each distinct (tool, input) runs once per session; a duplicate waits for the first."""
    key = (tool, " ".join(query.lower().split()))
    if key not in rt.tool_calls:
        rt.tool_calls[key] = asyncio.ensure_future(_call(tool, query, rt))
    return await asyncio.shield(rt.tool_calls[key])


async def _call(tool: str, query: str, rt: Runtime) -> tuple[list[Finding], str | None]:
    t0 = time.monotonic()
    try:
        found = await asyncio.wait_for(rt.tools[tool].run(query), rt.cfg("tools.timeout_s", 15) + 5)
        rt.procedural.record(tool, ok=True, empty=not found, ms=(time.monotonic() - t0) * 1000)
        return found, None
    except (ToolError, asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - one tool failing must not stop the rest
        rt.procedural.record(tool, ok=False, empty=True, ms=(time.monotonic() - t0) * 1000)
        return [], f"{tool}({query!r}) failed: {type(e).__name__}: {e}"


async def _research(task: SubTask, rt: Runtime, question: str) -> tuple[list[Finding], list[str], dict]:
    keep = rt.cfg("researcher.findings_per_subtask", 4)
    cached = rt.semantic.lookup(task.question, rt.cfg("researcher.semantic_cache_threshold", 0.92))
    if cached:
        findings, score = cached
        found = [f.model_copy(update={"subtask_id": task.id}) for f in findings]
        return found, [], {"cache_hit": True, "similarity": round(score, 3), "calls": []}

    calls = await _choose_calls(task, rt, question)
    results = await asyncio.gather(*(_run_call(t, q, rt) for t, q in calls))
    gathered, errors = [], []
    for found, err in results:
        gathered.extend(found)
        if err:
            errors.append(err)
    # The same passage can come back from two queries.
    seen, unique = set(), []
    for f in gathered:
        k = (f.url, f.content[:200])
        if k not in seen:
            seen.add(k)
            unique.append(f)
    # Embedding is CPU work: off the event loop, so other sub-tasks' tool calls keep running.
    ranked = await asyncio.to_thread(rt.semantic.rank, task.question, unique, keep)
    best = [f.model_copy(update={"subtask_id": task.id}) for f in ranked]
    rt.semantic.remember(task.question, best)
    return best, errors, {"cache_hit": False, "calls": [{"tool": t, "input": q} for t, q in calls]}


async def research(state: ResearchState, rt: Runtime) -> dict:
    pending = state.pending()
    outcomes = await asyncio.gather(*(_research(t, rt, state.question) for t in pending))
    findings, errors, events, done = [], [], [], []
    for task, (found, errs, info) in zip(pending, outcomes):
        findings.extend(found)
        errors.extend(errs)
        tried = [f"{c['tool']}: {c['input']}" for c in info["calls"]]
        done.append(task.model_copy(update={"done": True, "queries": tried}))
        how = (f"reused {len(found)} findings from memory (similarity {info['similarity']})" if info["cache_hit"]
               else f"{len(found)} findings via " + ", ".join(f"{c['tool']}: {c['input']}" for c in info["calls"]))
        sources = sorted({f.title for f in found})
        content = f"{task.id}. {task.question}\n{how}" + (f"\nSources: {'; '.join(sources)}" if sources else "")
        await rt.emit("researcher", content, {"subtask": task.id, **info, "findings": len(found),
                                              "errors": errs})
        events.append(AgentEvent(agent="researcher", content=content, data={"subtask": task.id, **info}))
    return {"subtasks": done, "findings": findings, "errors": errors, "events": events}
