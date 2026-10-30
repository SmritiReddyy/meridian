"""Shared state for the LangGraph pipeline.

Planner → Researcher → Critic ─(gaps)→ Researcher … → Synthesizer.
Nodes return partial updates; list fields marked `operator.add` accumulate.
"""
from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """One piece of evidence: a passage and where it came from."""
    title: str
    url: str
    content: str
    tool: str                 # wikipedia | arxiv | web_search | pdf | python
    query: str = ""
    subtask_id: str = ""
    score: float = 0.0        # similarity to the sub-task it was gathered for


class SubTask(BaseModel):
    id: str
    question: str
    origin: str = "planner"   # planner | critic
    done: bool = False
    queries: list[str] = Field(default_factory=list)   # tool calls the researcher ran for it


class Citation(BaseModel):
    id: int
    title: str
    url: str
    snippet: str


class AgentEvent(BaseModel):
    agent: str
    content: str
    data: dict = Field(default_factory=dict)


class ResearchState(BaseModel):
    session_id: str
    question: str

    plan: str = ""
    subtasks: Annotated[list[SubTask], operator.add] = Field(default_factory=list)
    findings: Annotated[list[Finding], operator.add] = Field(default_factory=list)

    iterations: int = 0           # critic passes completed
    critique: str = ""
    sufficient: bool = False

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)

    events: Annotated[list[AgentEvent], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    def pending(self) -> list[SubTask]:
        """Sub-tasks the researcher has not handled yet (latest copy of each id wins)."""
        latest: dict[str, SubTask] = {}
        for t in self.subtasks:
            latest[t.id] = t
        return [t for t in latest.values() if not t.done]
