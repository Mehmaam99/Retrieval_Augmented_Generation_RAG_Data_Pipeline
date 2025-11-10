import os
import pickle
from typing import List, Tuple, Dict

import faiss
import numpy as np
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

    def build_and_save_index_for_each_file(
        self,
        transcripts_dir: str,
        output_dir: str | None = None,
    ) -> None:
        """
        Reads each .txt file, creates embeddings, and saves an individual FAISS index
        and metadata file with the same base name.
        """
        if not os.path.isdir(transcripts_dir):
            raise FileNotFoundError(f"Directory not found: {transcripts_dir}")

        # Collect all .txt files
        txt_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {transcripts_dir}")

        # Define output directory
        if output_dir is None:
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(src_dir)
            output_dir = os.path.join(project_root, 'data', 'faiss_index')

        os.makedirs(output_dir, exist_ok=True)

        for txt_file in txt_files:
            file_path = os.path.join(transcripts_dir, txt_file)
            base_name = os.path.splitext(txt_file)[0]

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                print(f"[Warning] Skipping empty file: {txt_file}")
                continue

            print(f"[Embedding] Encoding text from: {txt_file}")
            embeddings = self.encode_texts([content])

            print(f"[Embedding] Building FAISS index for: {txt_file}")
            index = self.build_faiss_index(embeddings)

            index_path = os.path.join(output_dir, f"{base_name}.faiss")
            meta_path = os.path.join(output_dir, f"{base_name}.pkl")

            print(f"[Embedding] Saving index as {base_name}.faiss and metadata as {base_name}.pkl...")
            faiss.write_index(index, index_path)

            id_to_text: Dict[int, str] = {0: content}
            with open(meta_path, 'wb') as f:
                pickle.dump({'id_to_text': id_to_text, 'source_file': txt_file}, f)

        print("[Embedding] ✅ All text files processed and indexed successfully.")

    def load_index(
        self, index_path: str, meta_path: str
    ) -> Tuple[faiss.Index, Dict[int, str]]:
        if not (os.path.isfile(index_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(
                f"Missing index or metadata file: {index_path}, {meta_path}"
            )
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        id_to_text = meta['id_to_text']
        return index, id_to_text


if __name__ == '__main__':
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    transcripts_dir = os.path.join(project_root, 'transcriptions', 'downloads')

    engine = EmbeddingEngine()
    engine.build_and_save_index_for_each_file(transcripts_dir)
