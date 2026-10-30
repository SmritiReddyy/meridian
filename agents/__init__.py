from agents.runtime import Runtime
from agents.state import AgentEvent, Citation, Finding, ResearchState, SubTask

__all__ = ["AgentEvent", "Citation", "Finding", "ResearchState", "Runtime", "SubTask", "build_graph",
           "run_research"]


def __getattr__(name: str):
    # The graph imports the tools, and the tools import agents.state, so load it on first use.
    if name in ("build_graph", "run_research"):
        from agents import graph

        return getattr(graph, name)
    raise AttributeError(name)
