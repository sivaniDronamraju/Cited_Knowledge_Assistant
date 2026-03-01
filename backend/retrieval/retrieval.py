# backend/retrieval/retrieval.py
from typing import List, Dict, Tuple
import os
import pickle

import faiss
import numpy as np

from backend.retrieval.processing import Embedder
from backend.retrieval.schemas import Chunk


 
# Vector Store
 


class VectorStore:
    """
    FAISS-backed vector storage layer.

    Responsibilities:
    - Store embeddings
    - Persist index and metadata
    - Load index safely
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        storage_path: str = "test_storage",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path

        os.makedirs(self.storage_path, exist_ok=True)

        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: List[Chunk],
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks length mismatch.")

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def save(self) -> None:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Cannot save empty index.")

        faiss.write_index(
            self.index,
            os.path.join(self.storage_path, "faiss.index"),
        )

        with open(
            os.path.join(self.storage_path, "chunks.pkl"),
            "wb",
        ) as f:
            pickle.dump(self.chunks, f)

        np.save(
            os.path.join(self.storage_path, "embeddings.npy"),
            self.embeddings,
        )

    def load(self) -> None:
        index_path = os.path.join(self.storage_path, "faiss.index")
        chunks_path = os.path.join(self.storage_path, "chunks.pkl")
        embeddings_path = os.path.join(self.storage_path, "embeddings.npy")

        required = [index_path, chunks_path, embeddings_path]
        missing = [p for p in required if not os.path.exists(p)]

        if missing:
            raise FileNotFoundError(
                f"Missing index files: {missing}"
            )

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings = np.load(embeddings_path)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise RuntimeError(
                "Embeddings and chunk metadata count mismatch."
            )


 
# MMR Reranking
 


def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    lambda_param: float = 0.5,
    top_k: int = 5,
) -> Tuple[List[int], np.ndarray]:
    """
    Maximal Marginal Relevance (MMR) reranking.
    """

    if candidate_embeddings.size == 0:
        return [], np.array([])

    similarity = np.dot(
        candidate_embeddings,
        query_embedding.T,
    ).squeeze()

    selected: List[int] = []
    candidate_pool = list(range(len(candidate_embeddings)))

    while len(selected) < min(top_k, len(candidate_pool)):
        best_idx = None
        best_score = -np.inf

        for idx in candidate_pool:
            if not selected:
                diversity_penalty = 0.0
            else:
                selected_embeds = candidate_embeddings[selected]
                sim_to_selected = np.dot(
                    selected_embeds,
                    candidate_embeddings[idx],
                )
                diversity_penalty = float(np.max(sim_to_selected))

            score = (
                lambda_param * similarity[idx]
                - (1 - lambda_param) * diversity_penalty
            )

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        candidate_pool.remove(best_idx)

    return selected, similarity


 
# Retriever
 


class Retriever:
    """
    High-level retrieval interface.
    """

    def __init__(
        self,
        storage_path: str = "test_storage",
    ) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore(storage_path=storage_path)
        self.vector_store.load()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        lambda_param: float = 0.5,
    ) -> List[Dict[str, object]]:

        if self.vector_store.index is None:
            raise RuntimeError("Vector index not loaded.")

        query_embedding = self.embedder.encode_query(query)

        scores, indices = self.vector_store.index.search(
            query_embedding.astype(np.float32),
            candidate_k,
        )

        valid_indices = [
            idx
            for idx in indices[0]
            if 0 <= idx < len(self.vector_store.chunks)
        ]

        if not valid_indices:
            return []

        candidate_embeddings = self.vector_store.embeddings[
            valid_indices
        ]
        candidate_chunks = [
            self.vector_store.chunks[i]
            for i in valid_indices
        ]

        selected_indices, similarity = mmr_rerank(
            query_embedding,
            candidate_embeddings,
            lambda_param=lambda_param,
            top_k=top_k,
        )

        results: List[Dict[str, object]] = []

        for idx in selected_indices:
            results.append(
                {
                    "score": float(similarity[idx]),
                    "chunk": candidate_chunks[idx],
                }
            )

        return results

    def confidence_gate(
        self,
        results: List[Dict[str, object]],
        threshold: float = 0.55,
    ) -> bool:
        """
        Return True if retrieval confidence meets threshold.
        """

        if not results:
            return False

        max_score = max(float(r["score"]) for r in results)
        return max_score >= threshold