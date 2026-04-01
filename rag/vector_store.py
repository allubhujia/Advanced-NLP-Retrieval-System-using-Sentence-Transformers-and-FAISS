from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    _META_FILE = "metadata.pkl"
    _INDEX_FILE = "faiss.index"
    _CONFIG_FILE = "config.json"

    def __init__(self, dim: int, index_type: str = "flat", nlist: int = 100, nprobe: int = 10) -> None:
        if dim <= 0:
            raise ValueError("'dim' must be a positive integer.")
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

        self._index: faiss.Index = self._build_index()
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

    def _build_index(self) -> faiss.Index:
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
        else:
            raise ValueError(f"Unsupported index_type='{self.index_type}'")

        id_index = faiss.IndexIDMap2(index)
        logger.info("FAISS index created: type=%s, dim=%d.", self.index_type, self.dim)
        return id_index

    def add(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> List[int]:
        self._validate_embeddings(embeddings)

        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents count mismatch.")

        if self.index_type == "ivf" and not self._inner_index().is_trained:
            raise RuntimeError("IVF index not trained.")

        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)

        for i, doc in zip(ids.tolist(), documents):
            self._metadata[i] = dict(doc)

        self._next_id += len(embeddings)
        return ids.tolist()

    def train(self, embeddings: np.ndarray) -> None:
        if self.index_type != "ivf":
            return
        self._validate_embeddings(embeddings)
        self._inner_index().train(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self._next_id == 0:
            raise RuntimeError("Vector store is empty.")

        query = self._prepare_query(query_embeddings)
        actual_k = min(top_k, self._next_id)

        scores, ids = self._index.search(query, actual_k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id == -1:
                continue
            meta = self._metadata.get(int(doc_id), {})
            results.append((meta, float(score)))

        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / self._INDEX_FILE))

        with open(path / self._META_FILE, "wb") as f:
            pickle.dump({"metadata": self._metadata, "next_id": self._next_id}, f)

        config = {
            "dim": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }

        with open(path / self._CONFIG_FILE, "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        path = Path(directory)

        with open(path / cls._CONFIG_FILE) as f:
            config = json.load(f)

        instance = cls(
            dim=config["dim"],
            index_type=config["index_type"],
            nlist=config["nlist"],
            nprobe=config["nprobe"],
        )

        instance._index = faiss.read_index(str(path / cls._INDEX_FILE))

        with open(path / cls._META_FILE, "rb") as f:
            state = pickle.load(f)

        instance._metadata = state["metadata"]
        instance._next_id = state["next_id"]

        return instance

    def size(self) -> int:
        return self._next_id

    def _inner_index(self) -> faiss.Index:
        return faiss.downcast_index(self._index.index)

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be numpy array.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError("Invalid embedding shape.")
        if embeddings.dtype != np.float32:
            raise TypeError("embeddings must be float32.")

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        query = np.atleast_2d(query).astype(np.float32)
        if query.shape[1] != self.dim:
            raise ValueError("Query dimension mismatch.")
        return query

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.dim}, type='{self.index_type}', size={self.size()})"