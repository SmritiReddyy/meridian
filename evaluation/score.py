"""Score the recorded answers with RAGAS.

  faithfulness       share of the answer's claims supported by the sources it was given
                     (1 - faithfulness is the hallucination rate)
  answer_relevancy   how directly the answer addresses the question
  context_recall     share of the reference answer's claims found in the retrieved sources

The judge is a model on the same provider (configs: llm.models.judge). Its
calls go through a LangChain SQLite cache, so re-scoring costs nothing and an
interrupted run picks up where it stopped. Scores per question are written to
results/scores/<config>.jsonl.
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


def judge(settings: Settings) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    provider = settings.get("llm.provider")
    p = settings.get(f"llm.providers.{provider}")
    set_llm_cache(SQLiteCache(database_path=str(ROOT / "data" / "judge_cache.sqlite")))
    chat = ChatOpenAI(model=settings.get("llm.models.judge"), base_url=p["base_url"],
                      api_key=os.environ[p["api_key_env"]], temperature=0, max_retries=10, timeout=180,
                      max_tokens=4096)   # room for the judge's longer verdict lists
    emb = LocalEmbeddings(settings.get("memory.embedding_model"))
    # Groq rejects n > 1; bypass_n makes RAGAS ask n times instead (answer relevancy uses 3).
    return LangchainLLMWrapper(chat, bypass_n=True), LangchainEmbeddingsWrapper(emb)


def score_config(settings: Settings, config: str) -> list[dict]:
    runs = [json.loads(line) for line in (RUNS / f"{config}.jsonl").read_text().splitlines() if line.strip()]
    SCORES.mkdir(parents=True, exist_ok=True)
    out = SCORES / f"{config}.jsonl"
    have = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in runs if r["id"] not in have]
    if todo:
        llm, emb = judge(settings)
        metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb), LLMContextRecall(llm=llm)]
        # A few at a time: the free tier allows ~30 requests a minute.
        for i in range(0, len(todo), 5):
            batch = todo[i:i + 5]
            ds = EvaluationDataset.from_list([{"user_input": r["question"], "response": r["answer"],
                                               "retrieved_contexts": r["contexts"] or [""],
                                               "reference": r["reference"]} for r in batch])
            result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, show_progress=False,
                              run_config=RunConfig(max_workers=2, max_retries=10, max_wait=90, timeout=240))
            rows = result.to_pandas().to_dict("records")
            with out.open("a") as f:
                for r, row in zip(batch, rows):
                    scores = {m: _clean(row.get(m)) for m in METRICS}
                    f.write(json.dumps({"id": r["id"], "config": config, **scores}) + "\n")
            print(f"[{config}] scored {min(i + 5, len(todo))}/{len(todo)}", flush=True)
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def _clean(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)
