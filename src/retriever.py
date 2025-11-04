from typing import List, Tuple, Dict
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding_engine import EmbeddingEngine


class Retriever:
    def __init__(
        self,
        index_dir: str = os.path.join('data', 'faiss_index'),
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    ) -> None:
        self.index_dir = index_dir
        self.engine = EmbeddingEngine(model_name)
        self.index, self.id_to_text = self.engine.load_index(index_dir)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not query or not query.strip():
            return []
        query_embedding = self.engine.encode_texts([query], batch_size=1)
        scores, indices = self._search_embeddings(query_embedding, top_k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            text = self.id_to_text.get(int(idx), '')
            results.append((text, float(score)))
        return results

    def _search_embeddings(self, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        # FAISS IndexFlatIP returns inner product; embeddings are normalized, so this is cosine similarity
        scores, indices = self.index.search(query_emb, top_k)
        return scores, indices


if __name__ == '__main__':
    r = Retriever()
    print(r.search("What happened in the news today?", top_k=3))

