from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingModel:
    """
    Lightweight wrapper around SentenceTransformer for dense vector embeddings.
    Produces 384-dimensional normalized float32 vectors using all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedding model.

        Args:
            model_name: Sentence transformer model to load.
            device: 'cpu' or 'cuda' for GPU acceleration.
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalized 384-dim float32 vector.

        Args:
            text: Input text to embed.

        Returns:
            np.ndarray of shape (384,), dtype float32, L2-normalized.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a batch of texts into normalized 384-dim float32 vectors.

        Args:
            texts: List of input texts to embed.
            batch_size: Number of texts to process per forward pass.

        Returns:
            np.ndarray of shape (N, 384), dtype float32, L2-normalized.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    def embedding_dim(self) -> int:
        """Returns the output embedding dimension (384)."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return (
            f"EmbeddingModel(model='{self.model_name}', "
            f"device='{self.device}', dim={self.embedding_dim})"
        )