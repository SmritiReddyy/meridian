"""Score the recorded answers with RAGAS.

  faithfulness       share of the answer's claims supported by the sources it was given
                     (1 - faithfulness is the hallucination rate)
  answer_relevancy   how directly the answer addresses the question
  context_recall     share of the reference answer's claims found in the retrieved sources

Each metric has its own judge model (configs: evaluation.judges), the same one for
every answer in both configurations, so scores are comparable across them. Using
three judges also spreads the work over three daily token budgets. Faithfulness
is never judged by the model that wrote the answers.

Scores go to results/scores/<config>__<metric>.jsonl, one line per answer. A
score the judge could not produce (rate limit, unparseable reply) is not
written, so the next run retries it; judge calls go through a SQLite cache, so
nothing already answered is paid for twice.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

from llm import Settings
from memory import FastEmbedder

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
SCORES = ROOT / "results" / "scores"
METRICS = ("faithfulness", "answer_relevancy", "context_recall")


class LocalEmbeddings(Embeddings):
    """Meridian's local embedder as a LangChain Embeddings. RAGAS logs `.model` and
    expects a string there, which LangChain's FastEmbed wrapper does not provide."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._embedder = FastEmbedder(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0].tolist()


def judge_llm(settings: Settings, model: str) -> LangchainLLMWrapper:
    provider = settings.get("llm.provider")
    p = settings.get(f"llm.providers.{provider}")
    extra = dict((settings.get("llm.model_params") or {}).get(model) or {})
    extra.pop("max_tokens_floor", None)
    chat = ChatOpenAI(model=model, base_url=p["base_url"], api_key=os.environ[p["api_key_env"]], temperature=0,
                      max_retries=10, timeout=180, max_tokens=4096,   # room for long verdict lists
                      **({"reasoning_effort": extra["reasoning_effort"]} if "reasoning_effort" in extra else {}))
    # Groq rejects n > 1; bypass_n makes RAGAS ask n times instead (answer relevancy uses 3).
    return LangchainLLMWrapper(chat, bypass_n=True)


def metric_object(name: str, llm: LangchainLLMWrapper, emb: LangchainEmbeddingsWrapper):
    if name == "faithfulness":
        return Faithfulness(llm=llm)
    if name == "answer_relevancy":
        return ResponseRelevancy(llm=llm, embeddings=emb)
    return LLMContextRecall(llm=llm)


def _read(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def scores_path(config: str, metric: str) -> Path:
    return SCORES / f"{config}__{metric}.jsonl"


def load_scores(config: str) -> dict[str, dict]:
    """{question id: {metric: score}} merged from the per-metric files."""
    merged: dict[str, dict] = {}
    for m in METRICS:
        for qid, row in _read(scores_path(config, m)).items():
            merged.setdefault(qid, {"id": qid, "config": config})[m] = row[m]
    return merged


def score_config(settings: Settings, config: str, metrics: tuple[str, ...] = METRICS) -> dict[str, dict]:
    runs = list(_read(RUNS / f"{config}.jsonl").values())
    SCORES.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(ROOT / "data" / "judge_cache.sqlite")))
    emb = LangchainEmbeddingsWrapper(LocalEmbeddings(settings.get("memory.embedding_model")))
    judges = settings.get("evaluation.judges")
    for name in metrics:
        out = scores_path(config, name)
        have = _read(out)
        todo = [r for r in runs if r["id"] not in have]
        if not todo:
            continue
        metric = metric_object(name, judge_llm(settings, judges[name]), emb)
        for i in range(0, len(todo), 5):   # small batches: each judge allows ~30 requests a minute
            batch = todo[i:i + 5]
            ds = EvaluationDataset.from_list([{"user_input": r["question"], "response": r["answer"],
                                               "retrieved_contexts": r["contexts"] or [""],
                                               "reference": r["reference"]} for r in batch])
            result = evaluate(ds, metrics=[metric], show_progress=False,
                              run_config=RunConfig(max_workers=2, max_retries=10, max_wait=90, timeout=240))
            written = 0
            with out.open("a") as f:
                for r, row in zip(batch, result.to_pandas().to_dict("records")):
                    v = _clean(row.get(name))
                    if v is not None:   # a missing score is retried next run, not stored as a gap
                        f.write(json.dumps({"id": r["id"], "config": config, name: v, "judge": judges[name]}) + "\n")
                        written += 1
            print(f"[{config}] {name}: {len(_read(out))}/{len(runs)} scored ({written}/{len(batch)} in this batch)",
                  flush=True)
    return load_scores(config)


def _clean(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)
