# backend/generation/validators.py

import re
from typing import Set


class ResponseValidator:
    """
    Enterprise response validation layer.

    Responsibilities:
    - Ensure citations exist
    - Ensure citations are valid (no hallucinated UUIDs)
    - Ensure grounding (every sentence contains citation)
    """

    UUID_PATTERN = re.compile(r"\[UUID:\s*([a-f0-9\-]+)\]")

    def _extract_citations(self, text: str) -> list[str]:
        return self.UUID_PATTERN.findall(text)

    def validate_citations(
        self,
        response_text: str,
        allowed_chunk_ids: Set[str],
    ) -> bool:
        """
        Validate:
        - At least one citation exists
        - All citations are allowed
        """

        citations = self._extract_citations(response_text)

        if not citations:
            return False

        for uuid in citations:
            if uuid not in allowed_chunk_ids:
                return False

        return True

    def validate_grounding(
        self,
        response_text: str,
        allowed_chunk_ids: Set[str],
    ) -> bool:
        """
        Validate:
        - Every sentence contains at least one valid citation
        - No hallucinated UUIDs
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            response_text.strip(),
        )

        if not sentences:
            return False

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            citations = self._extract_citations(sentence)

            if not citations:
                return False

            for uuid in citations:
                if uuid not in allowed_chunk_ids:
                    return False

        return True