import os
import pickle
from typing import List, Tuple, Dict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype='float32')

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D: [num_items, embedding_dim]")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def build_and_save_index(
        self,
        cleaned_csv_path: str,
        text_column: str = 'text',
        output_dir: str = os.path.join('data', 'faiss_index'),
    ) -> Tuple[str, str]:
        if not os.path.isfile(cleaned_csv_path):
            raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_csv_path}")

        df = pd.read_csv(cleaned_csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {cleaned_csv_path}")

        texts: List[str] = df[text_column].astype(str).tolist()

        os.makedirs(output_dir, exist_ok=True)
        print(f"[Embedding] Encoding {len(texts)} texts with {self.model_name}...")
        embeddings = self.encode_texts(texts)
        print("[Embedding] Building FAISS index...")
        index = self.build_faiss_index(embeddings)

        index_path = os.path.join(output_dir, 'index.faiss')
        meta_path = os.path.join(output_dir, 'meta.pkl')

        print(f"[Embedding] Saving index to {index_path} and metadata to {meta_path}...")
        faiss.write_index(index, index_path)
        id_to_text: Dict[int, str] = {i: t for i, t in enumerate(texts)}
        with open(meta_path, 'wb') as f:
            pickle.dump({'id_to_text': id_to_text}, f)

        print("[Embedding] Done.")
        return index_path, meta_path

    def load_index(
        self, index_dir: str = os.path.join('data', 'faiss_index')
    ) -> Tuple[faiss.Index, Dict[int, str]]:
        index_path = os.path.join(index_dir, 'index.faiss')
        meta_path = os.path.join(index_dir, 'meta.pkl')
        if not (os.path.isfile(index_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(
                f"Missing index files in {index_dir}. Expected 'index.faiss' and 'meta.pkl'"
            )
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        id_to_text = meta['id_to_text']
        return index, id_to_text


if __name__ == '__main__':
    # Convenience CLI: builds index from the default cleaned dataset if present
    cleaned_path = os.path.join('data', 'cleaned', 'cleaned_news.csv')
    engine = EmbeddingEngine()
    engine.build_and_save_index(cleaned_path)

