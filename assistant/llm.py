"""Ollama streaming LLM client."""

from typing import AsyncGenerator

import ollama

DEFAULT_MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep your answers concise and conversational — no markdown, no lists."
)


class LLMClient:
    """Streams token-by-token responses from a local Ollama model."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._history: list[dict] = []

    def reset_history(self) -> None:
        self._history = []

    async def stream(self, user_text: str) -> AsyncGenerator[str, None]:
        """Yields LLM tokens one by one, maintaining conversation history."""
        self._history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        client = ollama.AsyncClient()
        full_response = ""

        async for chunk in await client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        ):
            token: str = chunk["message"]["content"]
            full_response += token
            yield token

        self._history.append({"role": "assistant", "content": full_response})
