"""Planner: splits the question into a few focused, independently researchable sub-questions."""
from __future__ import annotations

from agents.runtime import Runtime
from agents.state import AgentEvent, ResearchState, SubTask
from llm import parse_json

SYSTEM = """You plan research. Break the user's question into {n} or fewer focused sub-questions that,
answered together, fully answer it. Each sub-question must be answerable on its own from
an encyclopedia, research papers or the web. Do not answer anything yourself.

Reply with JSON only:
{{"plan": "<one sentence on the approach>", "subtasks": ["<sub-question>", ...]}}"""


async def plan(state: ResearchState, rt: Runtime) -> dict:
    n = rt.cfg("agent.max_subtasks", 4)
    reply = await rt.llms["planner"].complete(SYSTEM.format(n=n), [{"role": "user", "content": state.question}],
                                              role="planner", json_mode=True, max_tokens=400)
    try:
        parsed = parse_json(reply)
        questions = [q.strip() for q in parsed.get("subtasks", []) if isinstance(q, str) and q.strip()][:n]
        plan_text = str(parsed.get("plan", ""))
    except (ValueError, AttributeError):
        questions, plan_text = [], ""
    if not questions:  # a plan we cannot parse still lets research go ahead on the question itself
        questions, plan_text = [state.question], "Research the question directly."
    subtasks = [SubTask(id=f"t{i + 1}", question=q) for i, q in enumerate(questions)]
    content = plan_text + "\n" + "\n".join(f"{t.id}. {t.question}" for t in subtasks)
    await rt.emit("planner", content, {"subtasks": [t.question for t in subtasks]})
    return {"plan": plan_text, "subtasks": subtasks,
            "events": [AgentEvent(agent="planner", content=content, data={"subtasks": len(subtasks)})]}
