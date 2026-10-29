"""Semantic memory: findings from earlier sessions, retrievable by meaning.

Two uses:
  * a cache: before calling any tool, the researcher looks for an earlier
    sub-task whose question is close enough (cosine >= threshold) and reuses
    the findings gathered for it;
  * ranking: passages are scored against the sub-task question so only the
    most relevant ones go forward.

FAISS inner-product index over normalized embeddings, persisted to a directory.
"""
from __future__ import annotations

import json
import os
import threading

import faiss
import numpy as np

from agents.state import Finding
from memory.embeddings import Embedder

RANK_CHARS = 600


class SemanticMemory:
    def __init__(self, embedder: Embedder, index_dir: str | None = None) -> None:
        self.embedder = embedder
        self.index_dir = index_dir
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.entries: list[dict] = []   # parallel to index rows: {"question": str, "findings": [...]}
        self._vectors: dict[str, np.ndarray] = {}   # passage text -> embedding, so nothing is embedded twice
        # One embedding batch at a time: concurrent ONNX runs fight over the same cores and
        # each redo the others' work; serialized, later callers find their passages cached.
        self._embed_lock = threading.Lock()
        if index_dir and os.path.exists(os.path.join(index_dir, "entries.json")):
            self._load()

    # ------------------------------------------------------------ cache

    def lookup(self, question: str, threshold: float) -> tuple[list[Finding], float] | None:
        """Findings stored for the most similar earlier question, if it is similar enough."""
        if self.index.ntotal == 0:
            return None
        q = self.embedder.embed([question])
        scores, ids = self.index.search(q, 1)
        score, idx = float(scores[0][0]), int(ids[0][0])
        if idx < 0 or score < threshold:
            return None
        return [Finding(**f) for f in self.entries[idx]["findings"]], score

    def remember(self, question: str, findings: list[Finding]) -> None:
        if not findings:
            return
        self.index.add(self.embedder.embed([question]))
        self.entries.append({"question": question, "findings": [f.model_dump() for f in findings]})
        self._save()

    # ------------------------------------------------------------ ranking

    def rank(self, question: str, findings: list[Finding], top_k: int) -> list[Finding]:
        """The top_k findings by similarity to the question, with their scores set."""
        if not findings:
            return []
        # Title plus the start of the passage is enough to rank it, at about half the cost.
        texts = [f"{f.title}\n{f.content[:RANK_CHARS]}" for f in findings]
        with self._embed_lock:
            q = self.embedder.embed([question])[0]
            missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
            if missing:
                for t, v in zip(missing, self.embedder.embed(missing)):
                    self._vectors[t] = v
        scores = np.stack([self._vectors[t] for t in texts]) @ q
        order = np.argsort(-scores)[:top_k]
        return [findings[i].model_copy(update={"score": round(float(scores[i]), 4)}) for i in order]

    # ------------------------------------------------------------ persistence

    def _save(self) -> None:
        if not self.index_dir:
            return
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_dir, "questions.faiss"))
        with open(os.path.join(self.index_dir, "entries.json"), "w") as f:
            json.dump(self.entries, f)

    def _load(self) -> None:
        self.index = faiss.read_index(os.path.join(self.index_dir, "questions.faiss"))
        with open(os.path.join(self.index_dir, "entries.json")) as f:
            self.entries = json.load(f)
