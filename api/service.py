"""Builds the real runtime from settings: models, tools and the three memory tiers."""
from __future__ import annotations

from dataclasses import dataclass

from agents import Runtime
from agents.runtime import Emit
from llm import LLM, Settings, UsageMeter, build_llms, load_settings
from memory import EpisodicMemory, FastEmbedder, ProceduralMemory, SemanticMemory, Store
from tools import Tool, build_tools


@dataclass
class Services:
    settings: Settings
    meter: UsageMeter
    llms: dict[str, LLM]
    tools: dict[str, Tool]
    store: Store
    episodic: EpisodicMemory
    procedural: ProceduralMemory
    semantic: SemanticMemory

    def runtime(self, emit: Emit | None = None, critic: bool = True,
                semantic: SemanticMemory | None = None) -> Runtime:
        rt = Runtime(llms=self.llms, tools=self.tools, semantic=semantic or self.semantic,
                     procedural=self.procedural, settings=self.settings, critic_enabled=critic)
        if emit:
            rt.emit = emit
        return rt


def build_services(settings: Settings | None = None) -> Services:
    s = settings or load_settings()
    meter = UsageMeter()
    store = Store(s.path("memory.db_path"))
    embedder = FastEmbedder(s.get("memory.embedding_model"))
    return Services(
        settings=s, meter=meter, llms=build_llms(s, meter),
        tools=build_tools(s.get("tools.enabled"), s.get("tools.timeout_s", 15), s.get("researcher.finding_chars", 1200),
                          quota_path=s.path("memory.db_path"), web_daily_limit=s.get("tools.web_search_daily_limit", 100)),
        store=store, episodic=EpisodicMemory(store), procedural=ProceduralMemory(store),
        semantic=SemanticMemory(embedder, s.path("memory.semantic_index_dir")),
    )
