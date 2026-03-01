# backend/retrieval/processing.py

import uuid
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

from backend.retrieval.schemas import Document, Chunk


# chunking

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """
    Perform recursive semantic chunking.

    Strategy:
    1. Keep small documents intact.
    2. Split large documents by paragraph.
    3. Further split by sentences.
    4. Apply sentence-based overlap.
    """

    chunks: List[Chunk] = []

    for doc in documents:
        text = doc.text.strip()

        if not text:
            continue

        # Small document → keep intact
        if len(text) <= chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    text=text,
                    metadata=doc.metadata,
                )
            )
            continue

        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]

        current_sentences: List[str] = []

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)

            for sentence in sentences:
                current_sentences.append(sentence)
                joined = " ".join(current_sentences)

                if len(joined) >= chunk_size:
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            text=joined.strip(),
                            metadata=doc.metadata,
                        )
                    )

                    # Keep overlap
                    current_sentences = current_sentences[
                        -overlap_sentences:
                    ]

        # Remaining tail
        if current_sentences:
            tail_text = " ".join(current_sentences).strip()
            if tail_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        text=tail_text,
                        metadata=doc.metadata,
                    )
                )

    return chunks


# embedding 


class Embedder:
    """
    SentenceTransformer-based embedding generator.

    Responsibilities:
    - Generate embeddings for texts
    - Normalize embeddings for cosine similarity
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of raw text strings into normalized embeddings.
        """

        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        embeddings = embeddings / norms

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalized embedding.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            norm = 1.0

        return embedding / norm