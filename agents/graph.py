
```python
from langgraph.graph import StateGraph, END, START
from agents.state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.critic import critic_node
from agents.synthesizer import synthesizer_node


def route_after_critic(state: ResearchState) -> str:
    if state.needs_more_research and state.research_iterations < 3:
        return "researcher"
    return "synthesizer"


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


graph = build_graph()
```