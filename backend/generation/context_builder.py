# backend/generation/context_builder.py
from typing import List, Dict, TypedDict
from backend.retrieval.schemas import Chunk


class RetrievalResult(TypedDict):
    score: float
    chunk: Chunk


class ContextBuilder:
    """
    Builds structured LLM context from retrieval results.
    """

    NO_CONTEXT = "NO_CONTEXT_AVAILABLE"

    def __init__(self, separator: str = "-" * 50) -> None:
        self.separator = separator

    def build(self, results: List[RetrievalResult]) -> str:
        """
        Convert retrieval results into formatted LLM context block.
        """

        if not results:
            return self.NO_CONTEXT

        formatted_blocks: List[str] = ["DOCUMENT EXCERPTS:\n"]

        for idx, item in enumerate(results, start=1):
            score = float(item["score"])
            chunk = item["chunk"]

            if not isinstance(chunk, Chunk):
                raise TypeError("Expected Chunk instance in retrieval results.")

            uuid = chunk.chunk_id
            source = chunk.metadata.get("file_name", "UNKNOWN_SOURCE")
            content = chunk.text.strip()

            block = (
                f"[{idx}]\n"
                f"UUID: {uuid}\n"
                f"SOURCE: {source}\n"
                f"SIMILARITY: {score:.4f}\n"
                f"CONTENT:\n"
                f"{content}\n"
                f"\n{self.separator}\n"
            )

            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)