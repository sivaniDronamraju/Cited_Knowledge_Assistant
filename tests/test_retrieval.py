import os
import shutil

from backend.ingestion import load_documents
from backend.processing import chunk_documents, Embedder
from backend.retrieval import VectorStore, Retriever


DATA_PATH = "data/company_bronze"
TEST_STORAGE = "test_storage"


def test_full_pipeline():
    """
    Integration test:
    Validates ingestion → chunking → embedding → indexing → retrieval.
    """

    # ----------------------------
    # Clean previous test storage
    # ----------------------------
    if os.path.exists(TEST_STORAGE):
        shutil.rmtree(TEST_STORAGE)

    # ----------------------------
    # 1. Load documents
    # ----------------------------
    documents = load_documents(DATA_PATH)
    assert len(documents) > 0, "No documents loaded."

    # ----------------------------
    # 2. Chunk documents
    # ----------------------------
    chunks = chunk_documents(
    documents,
    chunk_size=800,
    overlap_sentences=2
)
    assert len(chunks) > 0, "No chunks created."

    # ----------------------------
    # 3. Generate embeddings
    # ----------------------------
    embedder = Embedder()
    embeddings = embedder.encode_chunks(chunks)
    assert embeddings.shape[0] == len(chunks), "Embedding mismatch."

    # ----------------------------
    # 4. Build and save index
    # ----------------------------
    vector_store = VectorStore(storage_path=TEST_STORAGE)
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save()

    # ----------------------------
    # 5. Retrieval test
    # ----------------------------
    retriever = Retriever(storage_path=TEST_STORAGE)

    results = retriever.search(
        query="leave policy",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5
    )

    assert len(results) == 3, "Retrieval failed."
    assert results[0]["chunk"].text.strip() != "", "Empty chunk returned."

    print("\n=== Basic Retrieval Test ===")
    print(results[0]["chunk"].text[:200])


def exploratory_queries():
    """
    Behavioral exploration tests.
    These are not strict pass/fail — they help inspect retrieval quality.
    """

    retriever = Retriever(storage_path=TEST_STORAGE)

    print("\n\n==============================")
    print("Query 1: Redundancy Test")
    print("==============================")

    results = retriever.search(
        query="leave policy reimbursement",
        top_k=5,
        candidate_k=30,
        lambda_param=0.5
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])

    print("\n\n==============================")
    print("Query 2: Polish Query Test")
    print("==============================")

    results = retriever.search(
        query="polityka urlopowa",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])

    print("\n\n==============================")
    print("Query 3: Structured Data Test")
    print("==============================")

    results = retriever.search(
        query="employee leave balance E0011",
        top_k=3,
        candidate_k=20,
        lambda_param=0.5
    )

    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | Source: {r['chunk'].metadata['file_name']}")
        print(r["chunk"].text[:150])


if __name__ == "__main__":
    test_full_pipeline()
    exploratory_queries()