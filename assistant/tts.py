"""Kokoro TTS wrapper with sentence-level streaming."""

import asyncio
import re
from typing import AsyncGenerator

import numpy as np

# Sentence boundary: end of .!? followed by whitespace or end-of-string
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

DEFAULT_LANG = "a"  # American English
DEFAULT_VOICE = "af_heart"


class TTSEngine:
    """
    Wraps Kokoro KPipeline.

    stream_sentences() consumes an async token stream from the LLM, splits
    it into sentences, and yields synthesized audio chunks as they arrive —
    keeping latency low.
    """

    def __init__(self, lang_code: str = DEFAULT_LANG, voice: str = DEFAULT_VOICE):
        self.lang_code = lang_code
        self.voice = voice
        self._pipeline = None

    def load(self) -> None:
        import os
        from kokoro import KPipeline  # lazy import

        # Force use of local HF cache — avoids network requests that block indefinitely.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print("[TTS] Loading Kokoro ...")
        self._pipeline = KPipeline(
            lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M"
        )
        print("[TTS] Model ready.")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text → float32 numpy array at 24 kHz."""
        chunks = [audio for _, _, audio in self._pipeline(text, voice=self.voice)]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    async def synthesize_async(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

    async def stream_sentences(
        self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Consumes LLM tokens, buffers them into sentences, and yields audio
        for each complete sentence without waiting for the full response.
        """
        buffer = ""
        async for token in token_stream:
            buffer += token
            parts = _SENTENCE_END.split(buffer)
            # parts[-1] is the incomplete sentence still being built
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    yield await self.synthesize_async(sentence)
            buffer = parts[-1]

        # Flush whatever remains
        tail = buffer.strip()
        if tail:
            yield await self.synthesize_async(tail)
