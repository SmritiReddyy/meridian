"""Run the pipeline over the evaluation questions, once per configuration.

Each question gets a fresh semantic memory, so neither configuration can reuse
the other's findings and every answer is researched from scratch. Tools are
Wikipedia, arXiv, the PDF reader and Python (no web search; see below). Results go
to results/runs/<config>.jsonl, one line per question; questions already in
the file are skipped, so an interrupted run (e.g. on a daily rate limit)
resumes where it stopped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agents import run_research
from api.service import Services
from memory import SemanticMemory

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evaluation" / "questions.jsonl"
RUNS = ROOT / "results" / "runs"
CONFIGS = {"critic": True, "no_critic": False}


def load_questions(limit: int | None = None) -> list[dict]:
    qs = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    return qs[:limit] if limit else qs


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


async def run_config(svc: Services, config: str, questions: list[dict]) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{config}.jsonl"
    finished = done_ids(out)
    for q in questions:
        if q["id"] in finished:
            continue
        contexts: list[str] = []

        async def emit(agent: str, content: str, data: dict) -> None:
            if agent == "synthesizer":
                contexts.extend(data.get("contexts", []))

        semantic = SemanticMemory(svc.semantic.embedder)          # fresh, in memory only
        rt = svc.runtime(emit=emit, critic=CONFIGS[config], semantic=semantic)
        # Web search is left out: its free tier allows 100 searches a day, and Wikipedia +
        # arXiv keep the evaluation reproducible by anyone who re-runs it.
        rt.tools = {k: v for k, v in rt.tools.items() if k != "web_search"}
        before = {r: (u.calls, u.cached_calls, u.prompt_tokens, u.completion_tokens)
                  for r, u in svc.meter.by_role.items()}
        t0 = time.monotonic()
        state = await run_research(q["question"], rt)
        seconds = time.monotonic() - t0
        usage = {}
        for role, u in svc.meter.by_role.items():
            b = before.get(role, (0, 0, 0, 0))
            usage[role] = {"calls": u.calls - b[0], "cached": u.cached_calls - b[1],
                           "prompt_tokens": u.prompt_tokens - b[2], "completion_tokens": u.completion_tokens - b[3]}
        record = {
            "id": q["id"], "config": config, "question": q["question"], "reference": q["reference"],
            "answer": state.answer, "contexts": contexts,
            "citations": [c.model_dump() for c in state.citations],
            "invalid_citations": next((e.data.get("invalid_citations", []) for e in state.events
                                       if e.agent == "synthesizer"), []),
            "subtasks": [t.question for t in {t.id: t for t in state.subtasks}.values()],
            "critic_iterations": state.iterations, "critic_satisfied": state.sufficient,
            "findings": len(state.findings), "tool_errors": state.errors,
            "seconds": round(seconds, 2), "usage": usage,
        }
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{config}] {q['id']} {seconds:5.1f}s  {len(state.citations)} citations  "
              f"{state.iterations} critic passes  {len(state.errors)} tool errors", flush=True)
