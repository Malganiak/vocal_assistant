"""Main pipeline: Microphone → VAD → ASR → LLM → TTS → Speaker."""

import numpy as np

from .audio import AudioPlayer, MicrophoneStream
from .asr import ASRModel
from .llm import LLMClient
from .tts import TTSEngine
from .vad import VoiceActivityDetector


class VoiceAssistantPipeline:
    """
    Orchestrates the real-time ASR → LLM → TTS pipeline.

    Flow:
      1. Mic chunks stream into the VAD.
      2. When VAD detects end of utterance, audio is sent to Parakeet.
      3. Transcript is streamed to Ollama.
      4. LLM tokens are split into sentences by TTSEngine.
      5. Each sentence is synthesized and played immediately.

    The `is_speaking` flag prevents the assistant from processing its own
    voice output as new input.
    """

    def __init__(
        self,
        asr_model: str = "nvidia/parakeet-tdt-0.6b-v3",
        llm_model: str = "qwen3.5:0.8b",
        tts_lang: str = "a",
        tts_voice: str = "af_heart",
    ):
        self.asr = ASRModel(model_name=asr_model)
        self.llm = LLMClient(model=llm_model)
        self.tts = TTSEngine(lang_code=tts_lang, voice=tts_voice)
        self.player = AudioPlayer()
        self.vad = VoiceActivityDetector()
        self.is_speaking = False  # True while assistant is playing audio

    def load_models(self) -> None:
        self.asr.load()
        self.tts.load()

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        print("[ASR] Transcribing ...", flush=True)
        text = await self.asr.transcribe_async(audio)
        text = text.strip()
        if not text:
            return

        print(f"[YOU] {text}", flush=True)
        print("[ASSISTANT] ", end="", flush=True)

        self.is_speaking = True
        try:
            token_stream = self.llm.stream(text)
            async for audio_chunk in self.tts.stream_sentences(token_stream):
                await self.player.play_async(audio_chunk)
        finally:
            self.is_speaking = False
            print(flush=True)

    async def run(self) -> None:
        print("Voice assistant ready — speak now. Press Ctrl-C to quit.\n", flush=True)

        mic = MicrophoneStream()
        mic.start()

        try:
            while True:
                chunk = await mic.read()

                if self.is_speaking:
                    # Skip VAD while assistant is talking (avoid feedback)
                    continue

                utterance = self.vad.process(chunk)
                if utterance is not None:
                    await self._handle_utterance(utterance)
        except KeyboardInterrupt:
            print("\n[QUIT]")
        finally:
            mic.stop()
