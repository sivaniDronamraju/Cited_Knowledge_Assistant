# tests/test_retrieval.py
import os
import shutil

from backend.retrieval.ingestion import load_documents
from backend.retrieval.processing import chunk_documents, Embedder
from backend.retrieval.retrieval import VectorStore, Retriever


DATA_PATH = "data/company_bronze"
TEST_STORAGE = "test_storage"


def test_full_pipeline():
    """
    Integration test:
    ingestion → chunking → embedding → indexing → retrieval
    """

    if os.path.exists(TEST_STORAGE):
        shutil.rmtree(TEST_STORAGE)

    # Step 1: Load documents
    documents = load_documents(DATA_PATH)
    assert len(documents) > 0, "No documents loaded."

    # Step 2: Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=800,
        overlap_sentences=2,
    )
    assert len(chunks) > 0, "No chunks created."

    # Step 3: Generate embeddings
    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode_texts(texts)

    assert embeddings.shape[0] == len(chunks), "Embedding mismatch."

    # Step 4: Build vector store
    vector_store = VectorStore(storage_path=TEST_STORAGE)
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save()

    # Step 5: Retrieval
    retriever = Retriever(storage_path=TEST_STORAGE)

    results = retriever.search(
        query="leave policy",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5,
    )

    assert len(results) == 3, "Retrieval failed."
    assert results[0]["chunk"].text.strip() != "", "Empty chunk returned."


def exploratory_queries():
    """
    Behavioral exploration tests (manual inspection).
    """

    retriever = Retriever(storage_path=TEST_STORAGE)

    print("\n==============================")
    print("Query 1: Redundancy Test")
    print("==============================")

    results = retriever.search(
        query="leave policy reimbursement",
        top_k=5,
        candidate_k=30,
        lambda_param=0.5,
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])

    print("\n==============================")
    print("Query 2: Polish Query Test")
    print("==============================")

    results = retriever.search(
        query="polityka urlopowa",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5,
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])

    print("\n==============================")
    print("Query 3: Structured Data Test")
    print("==============================")

    results = retriever.search(
        query="employee leave balance E0011",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5,
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])


if __name__ == "__main__":
    test_full_pipeline()
    exploratory_queries()