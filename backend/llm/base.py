# backend/llm/base.py
from abc import ABC, abstractmethod
from typing import Generator


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[str, None, None]:
        """
        Streams response tokens incrementally.
        """
        pass