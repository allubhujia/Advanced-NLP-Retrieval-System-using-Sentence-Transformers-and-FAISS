from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from embedder import Embedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalResult:
    def __init__(
        self,
        text: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        rank: int = 0,
    ):
        self.text = text
        self.score = score
        self.metadata = metadata if metadata is not None else {}
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("'top_k' must be >= 1.")

        self._embedder = embedder
        self._store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

        self._validate_dim_alignment()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        k = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        logger.debug("Retrieving top-%d documents for query: '%s'.", k, query[:80])

        try:
            query_vec = self._embedder.embed_query(query)
            raw_results = self._store.search(query_vec, top_k=k)
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        results = self._build_results(raw_results, threshold)
        logger.debug("Retrieved %d results after filtering.", len(results))
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[List[RetrievalResult]]:
        if not queries:
            raise ValueError("'queries' must not be empty.")

        return [
            self.retrieve(q, top_k=top_k, score_threshold=score_threshold)
            for q in queries
        ]

    def get_context_for_llm(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        return separator.join(r.text for r in results)

    def _build_results(
        self,
        raw: List[tuple],
        threshold: Optional[float],
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        rank = 1

        for metadata, score in raw:
            if threshold is not None and score < threshold:
                continue

            text = metadata.pop("text", "")
            results.append(
                RetrievalResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                    rank=rank,
                )
            )
            rank += 1

        return results

    def _validate_dim_alignment(self) -> None:
        if self._embedder.embedding_dim != self._store.dim:
            raise ValueError(
                f"Dimension mismatch: Embedder produces {self._embedder.embedding_dim}-d "
                f"vectors but VectorStore expects {self._store.dim}-d."
            )

    def __repr__(self) -> str:
        return (
            f"Retriever(model='{self._embedder.model_name}', "
            f"store_size={self._store.size}, top_k={self.top_k})"
        )