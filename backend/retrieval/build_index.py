# backend/retrieval/build_index.py
from typing import List
import pickle
import os

from backend.retrieval.processing import Embedder
from backend.retrieval.retrieval import VectorStore
from backend.retrieval.schemas import Chunk


DEFAULT_CHUNK_PATH = "storage/chunks.pkl"
DEFAULT_STORAGE_PATH = "storage"


def load_chunks(path: str) -> List[Chunk]:
    """
    Load serialized Chunk objects from disk.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with open(path, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        raise RuntimeError("No chunks found. Cannot build index.")

    if not isinstance(chunks[0], Chunk):
        raise RuntimeError(
            "Invalid chunk format. Expected list of Chunk objects."
        )

    return chunks


def main(
    chunk_path: str = DEFAULT_CHUNK_PATH,
    storage_path: str = DEFAULT_STORAGE_PATH,
) -> None:
    """
    Build FAISS index from serialized chunks.
    """

    print(f"Loading chunks from {chunk_path}...")
    chunks = load_chunks(chunk_path)

    embedder = Embedder()

    print("Encoding documents...")
    texts = [chunk.text for chunk in chunks]

    embeddings = embedder.encode_texts(texts)

    vector_store = VectorStore(storage_path=storage_path)

    print("Adding embeddings to FAISS...")
    vector_store.add_embeddings(embeddings, chunks)

    print("Saving index...")
    vector_store.save()

    print(f"Index rebuilt successfully in /{storage_path}.")


if __name__ == "__main__":
    main()