from __future__ import annotations
from typing import Annotated, Any
from pydantic import BaseModel, Field
import operator


class Finding(BaseModel):
    source: str
    content: str
    url: str | None = None
    relevance_score: float = 1.0
    tool_used: str = ""


class SubTask(BaseModel):
    id: str
    question: str
    assigned_to: str = "researcher"
    status: str = "pending"  # pending | in_progress | done
    findings: list[Finding] = Field(default_factory=list)


class Citation(BaseModel):
    id: int
    source: str
    url: str | None = None
    excerpt: str = ""


class AgentMessage(BaseModel):
    agent: str
    content: str
    timestamp: str = ""


class ResearchState(BaseModel):
    # Input
    session_id: str
    original_question: str

    # Planner output
    sub_tasks: list[SubTask] = Field(default_factory=list)
    research_plan: str = ""

    # Researcher output
    all_findings: Annotated[list[Finding], operator.add] = Field(default_factory=list)
    current_task_idx: int = 0

    # Critic output
    critique: str = ""
    needs_more_research: bool = False
    research_iterations: int = 0
    gaps: list[str] = Field(default_factory=list)

    # Synthesizer output
    final_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)

    # Streaming log
    agent_log: Annotated[list[AgentMessage], operator.add] = Field(default_factory=list)

    # Error tracking
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)


# Resolve forward references now that all classes are defined
SubTask.model_rebuild()