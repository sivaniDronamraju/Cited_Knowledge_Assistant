# backend/llm/ollama_streaming.py
import requests
import json
from typing import Generator
from backend.llm.base import BaseLLM


class OllamaStreamingLLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[str, None, None]:

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": self.temperature
            }
        }

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))

                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]