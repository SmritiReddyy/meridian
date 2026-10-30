"""The LangGraph pipeline.

    planner → researcher → critic ─┬─ gaps, budget left ─→ researcher
                                   └─ otherwise ─────────→ synthesizer → END

With the critic disabled (the evaluation's ablation) it is
planner → researcher → synthesizer.
"""
from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph

from agents.critic import critique, route_after_critic
from agents.planner import plan
from agents.researcher import research
from agents.runtime import Runtime
from agents.state import ResearchState
from agents.synthesizer import synthesize


def build_graph(rt: Runtime):
    g = StateGraph(ResearchState)

    async def planner(s: ResearchState) -> dict:
        return await plan(s, rt)

    async def researcher(s: ResearchState) -> dict:
        return await research(s, rt)

    async def critic(s: ResearchState) -> dict:
        return await critique(s, rt)

    async def synthesizer(s: ResearchState) -> dict:
        return await synthesize(s, rt)

    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("synthesizer", synthesizer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    if rt.critic_enabled:
        g.add_node("critic", critic)
        g.add_edge("researcher", "critic")
        g.add_conditional_edges("critic", route_after_critic, {"researcher": "researcher",
                                                                "synthesizer": "synthesizer"})
    else:
        g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


async def run_research(question: str, rt: Runtime, session_id: str | None = None) -> ResearchState:
    graph = build_graph(rt)
    out = await graph.ainvoke(ResearchState(session_id=session_id or uuid.uuid4().hex, question=question),
                              {"recursion_limit": 25})
    return ResearchState(**out) if isinstance(out, dict) else out
