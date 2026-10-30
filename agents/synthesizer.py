"""Synthesizer: writes the answer from numbered sources, citing each claim inline."""
from __future__ import annotations

import asyncio
import re
from collections import Counter

from agents.runtime import Runtime
from agents.state import AgentEvent, Citation, Finding, ResearchState

SYSTEM = """You write the final answer to a research question using ONLY the numbered sources given.

- Cite every factual sentence with the number(s) of the source(s) that contain that fact, in
  square brackets like [2] or [1][3]. Check each citation: the cited source must actually state
  the claim. Different claims usually come from different sources; do not cite one source
  for everything.
- Do not state anything the sources do not support. If the sources do not cover part of the
  question, say so plainly instead of filling the gap from memory.
- Be specific and concise: a short overview, then the details. Markdown is fine.
- Never cite a number that is not in the source list."""

CITE = re.compile(r"\[(\d{1,2})\]")
# Some models (gpt-oss) cite as 【1】 or 【1†source】; rewrite those as [1].
WIDE_CITE = re.compile(r"【\s*(\d{1,2})(?:[^】]*)】")


def normalize_citations(text: str) -> str:
    text = WIDE_CITE.sub(r"[\1]", text)
    return re.sub(r"\[(\d{1,2})\s*,\s*(\d{1,2})\]", r"[\1][\2]", text)   # [1, 3] -> [1][3]


def select_sources(state: ResearchState, rt: Runtime, per_url: int = 2) -> list[Finding]:
    """De-duplicated findings, most relevant to the whole question first, at most
    `per_url` passages from any one page so a long article cannot crowd out the rest."""
    seen, unique = set(), []
    for f in state.findings:
        k = (f.url, f.content[:200])
        if k not in seen:
            seen.add(k)
            unique.append(f)
    ranked = rt.semantic.rank(state.question, unique, len(unique))
    out, per = [], {}
    for f in ranked:
        if per.get(f.url, 0) < per_url:
            out.append(f)
            per[f.url] = per.get(f.url, 0) + 1
        if len(out) == rt.cfg("synthesizer.max_sources", 10):
            break
    return out


async def synthesize(state: ResearchState, rt: Runtime) -> dict:
    sources = await asyncio.to_thread(select_sources, state, rt)
    listing = "\n\n".join(f"[{i}] {f.title} ({f.url})\n{f.content}" for i, f in enumerate(sources, 1))
    prompt = f"Question: {state.question}\n\nSources:\n{listing or '(none found)'}"
    answer = await rt.llms["synthesizer"].complete(SYSTEM, [{"role": "user", "content": prompt}],
                                                   role="synthesizer", max_tokens=1200)
    answer = normalize_citations(answer)

    cited = sorted({int(n) for n in CITE.findall(answer) if 1 <= int(n) <= len(sources)})
    invalid = sorted({int(n) for n in CITE.findall(answer) if not 1 <= int(n) <= len(sources)})
    citations = [Citation(id=i, title=sources[i - 1].title, url=sources[i - 1].url,
                          snippet=sources[i - 1].content[:240]) for i in cited]
    rt.procedural.record_cited(Counter(sources[i - 1].tool for i in cited))

    tasks = {t.id: t for t in state.subtasks}
    covered = {f.subtask_id for f in state.findings}
    confidence = {
        "coverage": round(sum(1 for t in tasks if t in covered) / max(len(tasks), 1), 2),
        "sources_cited": round(len(cited) / max(len(sources), 1), 2),
    }
    if rt.critic_enabled:
        confidence["critic_satisfied"] = 1.0 if state.sufficient else 0.0
    data = {"answer": answer, "citations": [c.model_dump() for c in citations], "confidence_scores": confidence,
            "invalid_citations": invalid,
            "contexts": [f.content for f in sources]}   # what the answer was allowed to use (for evaluation)
    await rt.emit("synthesizer", f"Answer written from {len(sources)} sources, {len(cited)} cited.", data)
    return {"answer": answer, "citations": citations, "confidence": confidence,
            "events": [AgentEvent(agent="synthesizer", content=f"{len(cited)} of {len(sources)} sources cited",
                                  data={"invalid_citations": invalid})]}
