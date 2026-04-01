from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Embedder:

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64, normalize: bool = True, device: str | None = None,) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or self._resolve_device()

        logger.info("Loading embedding model '%s' on device '%s'.",model_name,self.device)
        self._model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d.", self.embedding_dim)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Input 'texts' must not be empty.")
        try:
            embeddings: np.ndarray = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings = self.normalize,
                convert_to_numpy=True,
                show_progress_bar = False,
            )
        except Exception as exc:
            raise ValueError(f"Embedding encoding failed: {exc}") from exc
        
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str)->np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        return self.embed([query])[0]
    
    @staticmethod
    def _resolve_device()->str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __repr__(self)->str:
        return(
            f"Embedder(model='{self.model_name}', dim={self.embedding_dim}, "
            f"Device='{self.device}', batch_size={self.batch_size})"
        )
        