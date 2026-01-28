/Users/usmri/Desktop/clm/meridian/tests/test_memory.py

The file contains three test classes as specified:

**TestSemanticMemory** — all four tests mock the embedding function with deterministic numpy unit vectors so FAISS (if present) is exercised via the real `SemanticMemory` interface but without any live model. `test_cache_miss` constructs a Gram-Schmidt orthogonal vector to guarantee cosine similarity is below 0.8. `test_persist_and_reload` uses pytest's `tmp_path` fixture.

**TestEpisodicMemory** — uses `sqlite+aiosqlite:///:memory:` via a `pytest_asyncio` fixture. All three tests (`test_save_and_retrieve_session`, `test_list_sessions`, `test_save_and_get_messages`) call the real `EpisodicMemory` methods against SQLite; the message-order test also verifies ascending ISO-8601 timestamp ordering.

**TestProceduralMemory** — same in-memory SQLite fixture. `test_log_and_retrieve_stats` logs exactly 3 successes + 2 failures and asserts `success_rate == 0.6` with `pytest.approx`. `test_recommend_tools` logs three tools at 100 / 40 / 0 % success and asserts `tool_a` is first and `tool_b` precedes `tool_c`.