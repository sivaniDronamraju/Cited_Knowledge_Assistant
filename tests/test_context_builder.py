# tests/test_context_builder.py
from backend.generation.context_builder import ContextBuilder
from backend.retrieval.schemas import Chunk


def test_context_builder():
    chunk = Chunk(
        chunk_id="123",
        document_id="doc_1",
        text="Test content.",
        metadata={"file_name": "policy.docx"},
    )

    results = [
        {
            "score": 0.91,
            "chunk": chunk,
        }
    ]

    builder = ContextBuilder()
    context = builder.build(results)

    assert "UUID: 123" in context
    assert "policy.docx" in context
    assert "0.9100" in context
    assert "Test content." in context