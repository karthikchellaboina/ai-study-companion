from pathlib import Path
import json

import numpy as np

from rag.embeddings import EmbeddingModel


INDEX_DIR = Path("data/index")
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()

        self.embeddings = None
        self.chunks = None

    def build_index(self, chunks):
        """
        Create embeddings for all chunks and save them locally.
        """

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        self.embeddings = self.embedding_model.embed_texts(texts)

        self.chunks = chunks

        np.save(
            EMBEDDINGS_FILE,
            self.embeddings
        )

        with open(
            CHUNKS_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

    def load_index(self):
        """
        Load previously created embeddings and chunks.
        """

        if not EMBEDDINGS_FILE.exists():
            return False

        if not CHUNKS_FILE.exists():
            return False

        self.embeddings = np.load(
            EMBEDDINGS_FILE
        )

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            self.chunks = json.load(f)

        return True

    def search(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Find the most relevant chunks for a query.
        """

        if self.embeddings is None:
            raise RuntimeError(
                "Index has not been loaded."
            )

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        # Because embeddings are normalized,
        # dot product = cosine similarity.
        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            chunk = self.chunks[index]

            results.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[index]),
                }
            )

        return results