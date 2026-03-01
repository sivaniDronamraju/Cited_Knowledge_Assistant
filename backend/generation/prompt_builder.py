# backend/generation/prompt_builder.py
from typing import Dict


class PromptBuilder:
    """
    Constructs system and user prompts for the LLM.
    """

    SYSTEM_PROMPT: str = """
You are an enterprise knowledge assistant.

STRICT RULES (MANDATORY):

1. You MUST answer using ONLY the provided document excerpts.
2. You MUST cite every factual statement using EXACTLY this format:
   [UUID: <chunk_id>]
3. DO NOT reference excerpt numbers like [1], [2], etc.
4. DO NOT write "excerpt" in your answer.
5. DO NOT include any explanation about where the information came from.
6. DO NOT use external knowledge.
7. If the answer is not fully supported by the context, respond EXACTLY with:
   Insufficient information in provided documents.
8. Every paragraph must contain at least one [UUID: ...] citation.
9. Do NOT include introductory phrases like "According to the provided documents".
10. Answer directly.

Failure to follow these rules is unacceptable.
""".strip()

    def build(self, question: str, context: str) -> Dict[str, str]:
        """
        Build system + user prompt payload.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        user_prompt = f"""
USER QUESTION:
{question}

CONTEXT:
{context}

INSTRUCTIONS:
Answer the question using ONLY the context above.
Cite using [UUID: <chunk_id>] after each factual claim.
If unsure, say:
Insufficient information in provided documents.
""".strip()

        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_prompt,
        }