# backend/processing.py

import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

from backend.schemas import Chunk


# -------------------------------------------------------
# Recursive Semantic Chunking
# -------------------------------------------------------

def chunk_documents(documents, chunk_size=800, overlap_sentences=2):
    """
    Recursive semantic chunking:

    1. Split by paragraph
    2. If too large → split by sentences
    3. Apply sentence-based overlap
    """

    chunks = []

    for doc in documents:
        text = doc.text.strip()

        # If structured row or small document → keep intact
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

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk_sentences = []

        for paragraph in paragraphs:

            # Tokenize paragraph into sentences
            sentences = sent_tokenize(paragraph)

            for sentence in sentences:
                current_chunk_sentences.append(sentence)

                # Check length if joined
                joined = " ".join(current_chunk_sentences)

                if len(joined) >= chunk_size:
                    # Create chunk
                    chunk_text = joined.strip()

                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=doc.document_id,
                            text=chunk_text,
                            metadata=doc.metadata,
                        )
                    )

                    # Apply sentence-based overlap
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]

        # Add remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=doc.document_id,
                        text=chunk_text,
                        metadata=doc.metadata,
                    )
                )

    return chunks


# -------------------------------------------------------
# Embedding
# -------------------------------------------------------

class Embedder:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, chunks):
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def encode_query(self, query):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        embedding = embedding / np.linalg.norm(embedding)
        return embedding