# backend/core/qa_service.py
from typing import List, Dict, TypedDict, Set

from backend.retrieval.retrieval import Retriever
from backend.generation.context_builder import ContextBuilder
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.validators import ResponseValidator
from backend.llm.ollama_streaming import OllamaStreamingLLM
from backend.retrieval.schemas import Chunk


class QAResponse(TypedDict):
    answer: str
    validated: bool
    confidence_score: float
    retrieval_scores: List[float]
    used_chunk_ids: List[str]


class QAService:
    """
    Enterprise-safe QA orchestration layer.
    """

    FALLBACK_MESSAGE = "Insufficient information in provided documents."

    def __init__(
        self,
        storage_path: str = "test_storage",
        threshold: float = 0.65,
    ) -> None:
        self.retriever = Retriever(storage_path=storage_path)
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.llm = OllamaStreamingLLM(model_name="llama3:8b")
        self.validator = ResponseValidator()
        self.threshold = threshold

    def _fallback_response(
        self,
        confidence_score: float,
        retrieval_scores: List[float],
        used_ids: List[str],
    ) -> QAResponse:
        return {
            "answer": self.FALLBACK_MESSAGE,
            "validated": False,
            "confidence_score": confidence_score,
            "retrieval_scores": retrieval_scores,
            "used_chunk_ids": used_ids,
        }

    def ask(self, query: str) -> QAResponse:

        if not query.strip():
            return self._fallback_response(0.0, [], [])

        # Step 1: Retrieve
        results = self.retriever.search(query)

        retrieval_scores: List[float] = [
            float(r["score"]) for r in results
        ]

        max_score = max(retrieval_scores) if retrieval_scores else 0.0

        # Step 2: Confidence Gate
        if not self.retriever.confidence_gate(
            results,
            threshold=self.threshold,
        ):
            return self._fallback_response(
                confidence_score=max_score,
                retrieval_scores=retrieval_scores,
                used_ids=[],
            )

        # Step 3: Build Context + Prompt
        context = self.context_builder.build(results)
        prompt = self.prompt_builder.build(query, context)

        # Step 4: Stream LLM Response
        response_parts: List[str] = []

        for token in self.llm.stream_chat(
            prompt["system"],
            prompt["user"],
        ):
            response_parts.append(token)

        full_response = "".join(response_parts).strip()

        # Step 5: Validation
        allowed_ids: Set[str] = {
            r["chunk"].chunk_id for r in results
        }

        citations_valid = self.validator.validate_citations(
            full_response,
            allowed_ids,
        )

        grounding_valid = self.validator.validate_grounding(
            full_response,
            allowed_ids,
        )

        if not (citations_valid and grounding_valid):
            return self._fallback_response(
                confidence_score=max_score,
                retrieval_scores=retrieval_scores,
                used_ids=list(allowed_ids),
            )

        return {
            "answer": full_response,
            "validated": True,
            "confidence_score": max_score,
            "retrieval_scores": retrieval_scores,
            "used_chunk_ids": list(allowed_ids),
        }