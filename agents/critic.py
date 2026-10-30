"""Critic: checks whether the evidence answers the question, and sends gaps back for research."""
from __future__ import annotations

from agents.runtime import Runtime
from agents.state import AgentEvent, ResearchState, SubTask
from llm import parse_json

SYSTEM = """You review research evidence before an answer is written.

For EACH sub-question, decide whether some passage in the evidence actually answers it
(names the specific fact asked for, e.g. the person, year, mechanism). A passage that is
merely on the same topic does not count.

The evidence is sufficient only if every sub-question is answered. For each one that is
not, write a NEW sub-question that takes a different route from the searches already tried
(listed under each sub-question): name the likely source, paper title, author or event,
e.g. "Lewis et al. 2020 paper that introduced retrieval-augmented generation". Never just
repeat the sub-question. At most 2 gaps.

Reply with JSON only:
{"coverage": [{"subquestion": "<text>", "answered": true|false}, ...],
 "sufficient": true|false, "critique": "<two sentences at most>", "gaps": ["<sub-question>", ...]}"""


def _evidence(state: ResearchState, chars: int = 350) -> str:
    lines = []
    for i, f in enumerate(state.findings, 1):
        lines.append(f"[{i}] ({f.tool}) {f.title}: {f.content[:chars].replace(chr(10), ' ')}")
    return "\n".join(lines) or "(no evidence found)"


async def critique(state: ResearchState, rt: Runtime) -> dict:
    latest = {t.id: t for t in state.subtasks}.values()
    tasks = "\n".join(f"- {t.question}" + (f"\n    searches tried: {'; '.join(t.queries)}" if t.queries else "")
                      for t in latest)
    prompt = f"Question: {state.question}\n\nSub-questions researched:\n{tasks}\n\nEvidence:\n{_evidence(state)}"
    reply = await rt.llms["critic"].complete(SYSTEM, [{"role": "user", "content": prompt}], role="critic",
                                             json_mode=True, max_tokens=300)
    try:
        parsed = parse_json(reply)
        coverage = [c for c in parsed.get("coverage", []) if isinstance(c, dict)]
        # Trust the per-question check over the overall verdict: any unanswered one means not sufficient.
        sufficient = bool(parsed.get("sufficient", True)) and all(c.get("answered", True) for c in coverage)
        text = str(parsed.get("critique", ""))
        gaps = [g.strip() for g in parsed.get("gaps", []) if isinstance(g, str) and g.strip()][:2]
    except (ValueError, AttributeError):
        sufficient, text, gaps = True, "Could not parse the critique; proceeding.", []

    iteration = state.iterations + 1
    can_loop = iteration <= rt.cfg("agent.max_research_iterations", 2)
    fresh = _drop_repeats(gaps, [t.question for t in state.subtasks], rt)
    new = []
    if not sufficient and fresh and can_loop:
        base = len({t.id for t in state.subtasks})
        new = [SubTask(id=f"t{base + i + 1}", question=g, origin="critic") for i, g in enumerate(fresh)]
    if sufficient:
        verdict = "sufficient"
    elif new:
        verdict = "gaps found"
    elif gaps and not fresh:
        verdict = "gaps found, but only ones already researched"
    else:
        verdict = "gaps found, research budget spent"
    content = f"Pass {iteration}: {verdict}. {text}" + "".join(f"\n- {t.question}" for t in new)
    await rt.emit("critic", content, {"sufficient": sufficient, "gaps": [t.question for t in new],
                                      "iteration": iteration})
    return {"iterations": iteration, "critique": text, "sufficient": sufficient, "subtasks": new,
            "events": [AgentEvent(agent="critic", content=content, data={"sufficient": sufficient,
                                                                         "gaps": len(new)})]}


def _drop_repeats(gaps: list[str], asked: list[str], rt: Runtime, threshold: float = 0.90) -> list[str]:
    """Gaps that are not just rewordings of a sub-question already researched: asking again finds the same sources.

    With bge-small, a paraphrase of an earlier sub-question scores ~0.93 and a gap that takes
    a new route to the same fact (naming the paper or author) ~0.85, hence 0.90."""
    if not gaps or not asked:
        return gaps
    emb = rt.semantic.embedder
    sims = emb.embed(gaps) @ emb.embed(asked).T
    return [g for g, row in zip(gaps, sims) if float(row.max()) < threshold]


def route_after_critic(state: ResearchState) -> str:
    return "researcher" if state.pending() else "synthesizer"
