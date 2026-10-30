"""What the agent nodes need besides the state: models, tools, memory, settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:   # imported for annotations only: tools and memory import agents.state
    from llm import LLM, Settings
    from memory import ProceduralMemory, SemanticMemory
    from tools import Tool

Emit = Callable[[str, str, dict], Awaitable[None]]   # (agent, content, data)


async def _no_emit(agent: str, content: str, data: dict) -> None:
    return None


@dataclass
class Runtime:
    llms: dict[str, LLM]
    tools: dict[str, Tool]
    semantic: SemanticMemory
    procedural: ProceduralMemory
    settings: Settings
    critic_enabled: bool = True
    emit: Emit = field(default=_no_emit)
    # (tool, input) -> in-flight or finished call, shared by the run's sub-tasks: identical
    # calls run once. Create a new Runtime per research session.
    tool_calls: dict = field(default_factory=dict)

    def cfg(self, dotted: str, default: Any = None) -> Any:
        return self.settings.get(dotted, default)
