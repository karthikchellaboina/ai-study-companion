from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text chunks into embedding vectors.
        """
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a user question into an embedding vector.
        """
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )