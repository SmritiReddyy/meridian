/Users/usmri/Desktop/clm/meridian/tests/test_agents.py

The file contains all four test classes with the requested nine test methods:

**TestPlannerAgent**
- `test_planner_decomposes_question` — patches `_build_llm` and `_call_llm`, returns a JSON plan with 3 sub-tasks, asserts `sub_tasks` list length and `research_plan` string are populated on the returned state dict.
- `test_planner_handles_invalid_json` — patches `_call_llm` to return a malformed string, asserts `ValueError` with the expected message is raised (the planner propagates the parse error; the errors list is populated by the calling graph, not the node itself).

**TestResearcherAgent**
- `test_researcher_calls_tools` — mocks all three tool `run` methods and the Anthropic client; asserts each tool is awaited exactly once and `all_findings` has at least 3 entries.
- `test_researcher_graceful_degradation` — makes `tavily_search.run` raise `RuntimeError`; asserts arxiv findings are still returned.
- `test_researcher_checks_semantic_cache` — patches `check_semantic_cache` to return cached findings; asserts none of the three tool mocks nor the Claude client were called.

**TestCriticAgent**
- `test_critic_approves_good_findings` — mocks Claude to return `needs_more_research=False`; asserts the flag, `research_iterations`, and empty `gaps`.
- `test_critic_requests_more_research` — mocks Claude to return `needs_more_research=True` with two gaps; asserts both gap strings appear in the result.

**TestSynthesizerAgent**
- `test_synthesizer_produces_answer_with_citations` — mocks Claude response and both memory save functions; asserts `final_answer`, two `Citation` objects, and confidence scores.
- `test_synthesizer_saves_to_memory` — same setup, additionally asserts `save_findings` and `save_session` were each awaited once with the correct `session_id`.