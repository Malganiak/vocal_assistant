"""Microphone capture and audio playback via sounddevice."""

import asyncio
import numpy as np
import sounddevice as sd

MIC_SAMPLE_RATE = 16_000  # Parakeet requires 16 kHz mono
TTS_SAMPLE_RATE = 24_000  # Kokoro outputs 24 kHz
CHUNK_SIZE = 1_024  # samples per sounddevice callback


class MicrophoneStream:
    """Async iterator that yields float32 numpy chunks from the microphone."""

    def __init__(
        self, sample_rate: int = MIC_SAMPLE_RATE, chunk_size: int = CHUNK_SIZE
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time, status):
        # Called from sounddevice thread → must be thread-safe
        chunk = indata.copy().flatten()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

    def start(self):
        self._loop = asyncio.get_event_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> np.ndarray:
        return await self._queue.get()


class AudioPlayer:
    """Plays float32 numpy arrays through the default speaker."""

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def play(self, audio: np.ndarray) -> None:
        sd.play(audio, self.sample_rate)
        sd.wait()

    async def play_async(self, audio: np.ndarray) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.play, audio)
