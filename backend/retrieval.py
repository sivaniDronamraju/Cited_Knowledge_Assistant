# backend/retrieval.py

import os
import pickle
import faiss
import numpy as np

from backend.processing import Embedder


# ----------------------------
# Vector Store
# ----------------------------

class VectorStore:
    def __init__(self, embedding_dim=384, storage_path="storage"):
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        self.index = None
        self.chunks = []
        self.embeddings = None

    def add_embeddings(self, embeddings, chunks):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def save(self):
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Cannot save empty index. Build index first.")

        faiss.write_index(
            self.index,
            os.path.join(self.storage_path, "faiss.index")
        )

        with open(
            os.path.join(self.storage_path, "chunks.pkl"),
            "wb"
        ) as f:
            pickle.dump(self.chunks, f)

        np.save(
            os.path.join(self.storage_path, "embeddings.npy"),
            self.embeddings
        )

    def load(self):
        index_path = os.path.join(self.storage_path, "faiss.index")
        chunks_path = os.path.join(self.storage_path, "chunks.pkl")
        embeddings_path = os.path.join(self.storage_path, "embeddings.npy")

        missing_files = [
            path for path in [index_path, chunks_path, embeddings_path]
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Vector index not found. Missing files: {missing_files}\n"
                f"Please build the index before creating Retriever()."
            )

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings = np.load(embeddings_path)


# ----------------------------
# MMR Reranking
# ----------------------------

def mmr_rerank(query_embedding, candidate_embeddings, lambda_param=0.5, top_k=5):
    similarity = np.dot(candidate_embeddings, query_embedding.T).squeeze()

    selected = []
    candidates = list(range(len(candidate_embeddings)))

    while len(selected) < top_k and candidates:
        mmr_scores = []

        for idx in candidates:
            if not selected:
                diversity_penalty = 0
            else:
                selected_embeddings = candidate_embeddings[selected]
                sim_to_selected = np.dot(
                    selected_embeddings,
                    candidate_embeddings[idx]
                )
                diversity_penalty = np.max(sim_to_selected)

            score = lambda_param * similarity[idx] - (1 - lambda_param) * diversity_penalty
            mmr_scores.append((idx, score))

        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)

    return selected


# ----------------------------
# Retriever
# ----------------------------

class Retriever:
    def __init__(self, storage_path="storage"):
        self.embedder = Embedder()
        self.vector_store = VectorStore(storage_path=storage_path)
        self.vector_store.load()

    def search(self, query, top_k=5, candidate_k=50, lambda_param=0.5):
        query_embedding = self.embedder.encode_query(query)

        # Step 1: FAISS search
        scores, indices = self.vector_store.index.search(query_embedding, candidate_k)

        candidate_embeddings = self.vector_store.embeddings[indices[0]]
        candidate_chunks = [self.vector_store.chunks[i] for i in indices[0]]

        # Step 2: MMR rerank
        selected_indices = mmr_rerank(
            query_embedding,
            candidate_embeddings,
            lambda_param=lambda_param,
            top_k=top_k
        )

        results = []
        for idx in selected_indices:
            results.append({
                "score": float(scores[0][idx]),
                "chunk": candidate_chunks[idx]
            })

        return results