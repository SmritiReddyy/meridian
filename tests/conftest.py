import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Finding, Runtime  # noqa: E402
from llm import ScriptedLLM, Settings  # noqa: E402
from memory import HashEmbedder, ProceduralMemory, SemanticMemory, Store  # noqa: E402
from tools import ToolError  # noqa: E402

SETTINGS = Settings({
    "agent": {"max_research_iterations": 2, "max_subtasks": 4},
    "researcher": {"findings_per_subtask": 3, "semantic_cache_threshold": 0.95, "finding_chars": 800},
    "synthesizer": {"max_sources": 6},
    "tools": {"timeout_s": 2},
})


class FakeTool:
    def __init__(self, name: str, corpus: dict[str, list[str]], fail: bool = False) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.corpus = corpus
        self.fail = fail
        self.queries: list[str] = []

    async def run(self, query: str) -> list[Finding]:
        self.queries.append(query)
        if self.fail:
            raise ToolError("boom")
        out = []
        for key, passages in self.corpus.items():
            if key in query.lower():
                out += [Finding(title=f"{key} article", url=f"https://example.org/{key}/{i}", content=p, tool=self.name,
                                query=query) for i, p in enumerate(passages)]
        return out


def js(obj) -> str:
    return json.dumps(obj)


@pytest.fixture
def make_runtime(tmp_path):
    def make(replies: dict, tools: dict | None = None, critic: bool = True) -> Runtime:
        store = Store(":memory:")
        return Runtime(llms={r: ScriptedLLM(replies) for r in ("planner", "researcher", "critic", "synthesizer")},
                       tools=tools or {}, semantic=SemanticMemory(HashEmbedder(), str(tmp_path / "sem")),
                       procedural=ProceduralMemory(store), settings=SETTINGS, critic_enabled=critic)
    return make
