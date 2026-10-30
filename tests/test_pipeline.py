"""The whole graph with a scripted model and fake tools."""
from agents import run_research
from conftest import FakeTool, js

CORPUS = {
    "transformer": ["The Transformer was introduced in 2017 in the paper Attention Is All You Need.",
                    "It replaces recurrence with self-attention, which lets training run in parallel."],
    "bert": ["BERT is an encoder-only Transformer pre-trained with masked language modelling."],
}


async def test_full_run_plans_researches_critiques_and_cites(make_runtime):
    wiki = FakeTool("wikipedia", CORPUS)
    rt = make_runtime({
        "planner": js({"plan": "two parts", "subtasks": ["Who introduced the transformer?", "What is BERT?"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer bert"}]}),
        "critic": js({"sufficient": True, "critique": "Covered.", "gaps": []}),
        "synthesizer": "The Transformer appeared in 2017 [1]. BERT is encoder-only [3]. Bogus [9].",
    }, tools={"wikipedia": wiki})
    state = await run_research("Explain transformers and BERT", rt)

    assert [t.question for t in state.pending()] == []
    assert len({t.id for t in state.subtasks}) == 2
    assert state.findings and all(f.subtask_id in ("t1", "t2") for f in state.findings)
    assert state.iterations == 1 and state.sufficient
    assert [c.id for c in state.citations] == [1, 3]           # [9] is not a source and is dropped
    assert state.confidence["coverage"] == 1.0
    agents = [e.agent for e in state.events]
    assert agents[0] == "planner" and agents[-1] == "synthesizer" and "critic" in agents


async def test_critic_gap_triggers_a_second_research_pass(make_runtime):
    wiki = FakeTool("wikipedia", CORPUS)
    rt = make_runtime({
        "planner": js({"plan": "", "subtasks": ["transformer origin"]}),
        "researcher": [js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
                       js({"calls": [{"tool": "wikipedia", "input": "bert"}]})],
        "critic": [js({"sufficient": False, "critique": "BERT missing.", "gaps": ["What is BERT?"]}),
                   js({"sufficient": True, "critique": "Now covered.", "gaps": []})],
        "synthesizer": "Answer [1].",
    }, tools={"wikipedia": wiki})
    state = await run_research("transformers and BERT", rt)

    assert state.iterations == 2
    assert [t.origin for t in {t.id: t for t in state.subtasks}.values()] == ["planner", "critic"]
    assert wiki.queries == ["transformer", "bert"]
    assert any("BERT" in f.content for f in state.findings)


async def test_research_budget_stops_the_loop(make_runtime):
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "critic": [js({"sufficient": False, "critique": "never happy", "gaps": [g]})
                   for g in ("bert pretraining objective", "gpt scaling laws", "vision models")],
        "synthesizer": "Done [1].",
    }, tools={"wikipedia": FakeTool("wikipedia", CORPUS)})
    state = await run_research("transformers", rt)
    assert state.iterations == 3            # first pass + max_research_iterations (2) loops, then stop
    assert state.answer == "Done [1]."


async def test_a_gap_that_repeats_an_earlier_subquestion_is_not_researched_again(make_runtime):
    wiki = FakeTool("wikipedia", CORPUS)
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer origin"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "critic": js({"sufficient": False, "critique": "want more", "gaps": ["transformer origin?"]}),
        "synthesizer": "Done [1].",
    }, tools={"wikipedia": wiki})
    state = await run_research("transformers", rt)
    assert state.iterations == 1 and wiki.queries == ["transformer"]
    assert "already researched" in [e for e in state.events if e.agent == "critic"][0].content


async def test_wide_bracket_citations_are_normalized(make_runtime):
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "critic": js({"sufficient": True, "gaps": []}),
        "synthesizer": "Introduced in 2017【1】 and parallel【2†source】 [1, 2].",
    }, tools={"wikipedia": FakeTool("wikipedia", CORPUS)})
    state = await run_research("transformers", rt)
    assert state.answer == "Introduced in 2017[1] and parallel[2] [1][2]."
    assert [c.id for c in state.citations] == [1, 2]


async def test_without_critic_goes_straight_to_synthesis(make_runtime):
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "synthesizer": "Answer [1].",
    }, tools={"wikipedia": FakeTool("wikipedia", CORPUS)}, critic=False)
    state = await run_research("transformers", rt)
    assert state.iterations == 0 and "critic" not in [e.agent for e in state.events]
    assert "critic_satisfied" not in state.confidence


async def test_unparseable_plan_falls_back_to_the_question(make_runtime):
    rt = make_runtime({
        "planner": "I think we should look at transformers.",
        "researcher": "not json either",
        "critic": js({"sufficient": True, "gaps": []}),
        "synthesizer": "Answer.",
    }, tools={"wikipedia": FakeTool("wikipedia", CORPUS)})
    state = await run_research("transformer history", rt)
    assert [t.question for t in {t.id: t for t in state.subtasks}.values()] == ["transformer history"]
    assert state.findings   # the researcher fell back to searching the sub-question with the best tool


async def test_a_failing_tool_is_recorded_and_the_others_still_count(make_runtime):
    good, bad = FakeTool("wikipedia", CORPUS), FakeTool("arxiv", CORPUS, fail=True)
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer"]}),
        "researcher": js({"calls": [{"tool": "arxiv", "input": "transformer"},
                                    {"tool": "wikipedia", "input": "transformer"}]}),
        "critic": js({"sufficient": True, "gaps": []}),
        "synthesizer": "Answer [1].",
    }, tools={"wikipedia": good, "arxiv": bad})
    state = await run_research("transformers", rt)
    assert state.findings and state.errors and "arxiv" in state.errors[0]
    stats = rt.procedural.stats()
    assert stats["arxiv"]["success_rate"] == 0.0 and stats["wikipedia"]["success_rate"] == 1.0
    assert stats["wikipedia"]["cited_per_call"] > 0


async def test_one_long_article_cannot_crowd_out_the_other_sources(make_runtime):
    from agents.state import Finding, ResearchState
    from agents.synthesizer import select_sources
    rt = make_runtime({})
    many = [Finding(title="Big", url="https://big", content=f"transformer passage {i}", tool="wikipedia")
            for i in range(8)]
    other = [Finding(title="Small", url="https://small", content="transformer overview", tool="arxiv")]
    picked = select_sources(ResearchState(session_id="s", question="transformer", findings=many + other), rt)
    assert sum(f.url == "https://big" for f in picked) == 2 and any(f.url == "https://small" for f in picked)


async def test_identical_tool_calls_run_once_per_session(make_runtime):
    wiki = FakeTool("wikipedia", CORPUS)
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer origin", "transformer training", "transformer uses"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "Transformer"}]}),
        "critic": js({"sufficient": True, "gaps": []}),
        "synthesizer": "Answer [1].",
    }, tools={"wikipedia": wiki})
    state = await run_research("transformers", rt)
    assert wiki.queries == ["Transformer"]            # three sub-tasks, one request
    assert {f.subtask_id for f in state.findings} == {"t1", "t2", "t3"}


async def test_an_unanswered_subquestion_overrides_a_sufficient_verdict(make_runtime):
    rt = make_runtime({
        "planner": js({"subtasks": ["transformer origin"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "critic": [js({"coverage": [{"subquestion": "transformer origin", "answered": False}],
                       "sufficient": True, "critique": "looks fine", "gaps": ["original 2017 bert paper"]}),
                   js({"coverage": [], "sufficient": True, "critique": "ok", "gaps": []})],
        "synthesizer": "Answer [1].",
    }, tools={"wikipedia": FakeTool("wikipedia", CORPUS)})
    state = await run_research("transformers", rt)
    assert state.iterations == 2       # the unanswered sub-question sent research round two
