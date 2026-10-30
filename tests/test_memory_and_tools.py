from agents import Finding
from llm import parse_json
from llm.client import _retry_after
from memory import EpisodicMemory, HashEmbedder, ProceduralMemory, SemanticMemory, Store
from tools.base import chunk
from tools.code_executor import PythonTool


def f(content: str, url: str = "u") -> Finding:
    return Finding(title="t", url=url, content=content, tool="wikipedia")


def test_semantic_cache_hits_only_above_threshold(tmp_path):
    sem = SemanticMemory(HashEmbedder(), str(tmp_path))
    sem.remember("history of the transformer architecture", [f("2017 paper")])
    hit = sem.lookup("history of the transformer architecture", 0.95)
    assert hit and hit[0][0].content == "2017 paper"
    assert sem.lookup("photosynthesis in plants", 0.5) is None
    # persisted: a new instance sees the same memory
    again = SemanticMemory(HashEmbedder(), str(tmp_path))
    assert again.lookup("history of the transformer architecture", 0.95)


def test_rank_puts_the_relevant_passage_first():
    sem = SemanticMemory(HashEmbedder())
    ranked = sem.rank("who introduced the transformer", [f("bananas are yellow"), f("the transformer was introduced by vaswani")], 2)
    assert "vaswani" in ranked[0].content and ranked[0].score > ranked[1].score


def test_procedural_memory_ranks_reliable_tools_first():
    proc = ProceduralMemory(Store(":memory:"))
    for _ in range(4):
        proc.record("arxiv", ok=False, empty=True, ms=900)
        proc.record("wikipedia", ok=True, empty=False, ms=300)
    proc.record_cited({"wikipedia": 3})
    assert proc.rank(["arxiv", "wikipedia"]) == ["wikipedia", "arxiv"]
    assert "0% returned results" in proc.describe(["arxiv"])


def test_episodic_memory_records_sessions_and_events():
    ep = EpisodicMemory(Store(":memory:"))
    ep.start("s1", "q?")
    ep.event("s1", "agent_update", "planner", "plan", {"n": 1})
    ep.event("s1", "final", "synthesizer", "done", {})
    ep.finish("s1", "done", answer="a")
    assert [e["seq"] for e in ep.events("s1")] == [0, 1]
    assert [e["type"] for e in ep.events("s1", after=0)] == ["final"]
    assert ep.sessions()[0]["status"] == "done"


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_retry_after_reads_groq_reset_headers():
    class R:
        def __init__(self, h): self.headers = h
    assert _retry_after(R({"retry-after": "3"})) == 3.0
    assert abs(_retry_after(R({"x-ratelimit-reset-tokens": "1m2.5s"})) - 63.0) < 1e-6
    assert _retry_after(R({})) is None


def test_chunk_respects_the_limit_and_paragraphs():
    pieces = chunk("a" * 50 + "\n" + "b" * 50 + "\n" + "c" * 120, 100)
    assert all(len(p) <= 100 for p in pieces) and pieces[0].startswith("a")


async def test_python_tool_runs_code_and_enforces_limits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-in-parent")
    tool = PythonTool(timeout_s=3)
    out, _, code = await tool.execute("print(2 ** 10)")
    assert code == 0 and out.strip() == "1024"
    _, err, code = await tool.execute("while True: pass")
    assert code != 0                       # killed by the CPU limit or the timeout
    out, _, code = await tool.execute("import os; print(os.environ.get('GROQ_API_KEY', 'absent'))")
    assert out.strip() == "absent"         # the parent's environment, API keys included, is not passed on


def test_daily_quota_is_a_hard_cap():
    from tools.web_search import DailyQuota
    q = DailyQuota(":memory:", "web_search", 3)
    assert [q.take() for _ in range(5)] == [True, True, True, False, False]
    assert q.used() == 3 and q.remaining() == 0


async def test_web_search_refuses_once_the_quota_is_spent():
    import pytest
    from tools import ToolError
    from tools.web_search import DailyQuota, WebSearchTool
    tool = WebSearchTool(DailyQuota(":memory:", "web_search", 0))
    assert not tool.available
    with pytest.raises(ToolError):
        await tool.run("anything")      # refused before any request is sent


def test_you_search_results_are_parsed_from_an_sse_stream():
    import json as _json
    from tools.web_search import mcp_result, parse_you_results
    payload = {"results": {"web": [
        {"url": "https://en.wikipedia.org/wiki/Raft_(algorithm)", "title": "Raft (algorithm)", "description": "",
         "contents": {"highlights": ["Raft elects a leader.", "Terms number elections."]}},
        {"url": "https://x.example", "title": "X", "description": "a description"}]}}
    body = ('data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
            'data: ' + _json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
                "content": [{"type": "text", "text": _json.dumps(payload)}]}}) + "\n\n")
    rows = parse_you_results(mcp_result(body, "text/event-stream", request_id=2))
    assert rows[0] == ("Raft (algorithm)", "https://en.wikipedia.org/wiki/Raft_(algorithm)",
                       "Raft elects a leader.\nTerms number elections.")
    assert rows[1][2] == "a description"


def test_arxiv_strict_query_requires_words_not_years():
    from tools.arxiv_tool import ArxivTool
    q = ArxivTool.strict_query("retrieval-augmented generation knowledge-intensive NLP tasks Lewis 2020")
    assert q == ("all:retrieval-augmented AND all:generation AND all:knowledge-intensive AND all:NLP AND "
                 "all:tasks AND all:Lewis")
