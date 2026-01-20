```python
"""
Semantic memory module for storing and retrieving research findings using FAISS.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticMemory:
    """
    Stores past research findings as embeddings in a FAISS index,
    enabling similarity-based retrieval before hitting external APIs.
    """

    def __init__(
        self,
        index_path: str = "data/faiss_index",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.index_path = index_path
        self.metadata_path = f"{index_path}.metadata.pkl"
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._load_or_create_index()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_or_create_index(self) -> None:
        """Load index from disk if it exists, otherwise create a new flat L2 index."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self._metadata = pickle.load(f)
        else:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatL2(embedding_dim)
            self._metadata = []

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text string and return as a float32 row vector."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding.astype(np.float32)

    def _l2_to_similarity(self, l2_distances: np.ndarray) -> np.ndarray:
        """
        Convert L2 distances to similarity scores in [0, 1].

        Uses the formula: similarity = 1 / (1 + distance)
        so that distance=0 gives similarity=1 and larger distances approach 0.
        """
        return 1.0 / (1.0 + l2_distances)

    async def add_finding(self, content: str, metadata: dict[str, Any]) -> None:
        """
        Embed the content string and add it to the FAISS index.

        Saves metadata to a parallel list and persists both the index and metadata.
        """
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed, content)

        async with self._lock:
            self._index.add(embedding)
            record = dict(metadata)
            record["content"] = content
            self._metadata.append(record)
            await loop.run_in_executor(None, self.save)

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Embed the query and search the FAISS index for the top_k nearest neighbors.

        Returns a list of result dicts, each containing the stored metadata,
        the original content, and a 'similarity_score' field.
        """
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed, query)

        async with self._lock:
            n_stored = self._index.ntotal
            if n_stored == 0:
                return []

            k = min(top_k, n_stored)
            distances, indices = await loop.run_in_executor(
                None, lambda: self._index.search(embedding, k)
            )

        similarities = self._l2_to_similarity(distances[0])
        results: list[dict[str, Any]] = []

        for idx, similarity in zip(indices[0], similarities):
            if idx < 0:
                continue
            record = dict(self._metadata[idx])
            record["similarity_score"] = float(similarity)
            results.append(record)

        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_dir = os.path.dirname(self.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

        faiss.write_index(self._index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)

    async def check_cache(
        self, query: str, threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        """
        Search the index with a high similarity threshold.

        Returns cached results only if a very similar query was previously stored.
        An empty list is returned when no results meet the threshold.
        """
        results = await self.search(query, top_k=5)
        return [r for r in results if r.get("similarity_score", 0.0) >= threshold]
```