import asyncio
import json

import httpx

import api.main as main
from api.service import Services
from conftest import SETTINGS, FakeTool, js
from llm import ScriptedLLM, UsageMeter
from memory import EpisodicMemory, HashEmbedder, ProceduralMemory, SemanticMemory, Store


def scripted_services(tmp_path) -> Services:
    llm = ScriptedLLM({
        "planner": js({"plan": "p", "subtasks": ["transformer origin"]}),
        "researcher": js({"calls": [{"tool": "wikipedia", "input": "transformer"}]}),
        "critic": js({"sufficient": True, "critique": "ok", "gaps": []}),
        "synthesizer": "The Transformer dates from 2017 [1].",
    })
    store = Store(":memory:")
    return Services(settings=SETTINGS, meter=UsageMeter(), llms={r: llm for r in
                    ("planner", "researcher", "critic", "synthesizer", "judge")},
                    tools={"wikipedia": FakeTool("wikipedia", {"transformer": ["Introduced in 2017."]})},
                    store=store, episodic=EpisodicMemory(store), procedural=ProceduralMemory(store),
                    semantic=SemanticMemory(HashEmbedder(), str(tmp_path / "sem")))


async def test_research_streams_agent_updates_then_the_final_answer(tmp_path):
    main.services = scripted_services(tmp_path)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/research", json={"question": "Where does the transformer come from?"})
        assert r.status_code == 200
        body = r.json()
        events = []
        async with c.stream("GET", body["stream_url"]) as s:
            async for line in s.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        types = [e["type"] for e in events]
        assert types[-1] == "final" and types.count("agent_update") >= 4
        assert [e["agent"] for e in events[:4]] == ["planner", "researcher", "critic", "synthesizer"]
        final = events[-1]["data"]
        assert final["answer"].endswith("[1].") and final["citations"][0]["url"].startswith("https://example.org")
        assert final["confidence_scores"]["coverage"] == 1.0

        await asyncio.sleep(0.05)
        sessions = (await c.get("/api/sessions")).json()["sessions"]
        assert sessions[0]["session_id"] == body["session_id"] and sessions[0]["status"] == "done"
        replay = (await c.get(f"/api/sessions/{body['session_id']}")).json()
        assert len(replay["events"]) == len(events)
        assert (await c.get("/api/research/nope/stream")).status_code == 404
