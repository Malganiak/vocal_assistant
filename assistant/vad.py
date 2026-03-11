"""Energy-based Voice Activity Detector."""

import numpy as np


class VoiceActivityDetector:
    """
    Simple RMS energy VAD with speech/silence state machine.

    Accumulates audio chunks until a silence gap follows a speech segment,
    then returns the full utterance as a single numpy array.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.01,
        min_speech_ms: int = 400,
        silence_ms: int = 800,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self._min_speech_samples = int(min_speech_ms * sample_rate / 1_000)
        self._silence_samples = int(silence_ms * sample_rate / 1_000)

        self._buffer: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_counter = 0
        self._in_speech = False

    def _is_speech(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk**2)))
        return rms > self.energy_threshold

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Feed one audio chunk. Returns a complete utterance array when a
        speech segment ends, otherwise returns None.
        """
        is_speech = self._is_speech(chunk)

        if is_speech:
            self._in_speech = True
            self._silence_counter = 0

        if self._in_speech:
            self._buffer.append(chunk)
            self._speech_samples += len(chunk)

            if not is_speech:
                self._silence_counter += len(chunk)
                if self._silence_counter >= self._silence_samples:
                    if self._speech_samples >= self._min_speech_samples:
                        utterance = np.concatenate(self._buffer)
                    else:
                        utterance = None
                    self._reset()
                    return utterance

        return None

    def _reset(self):
        self._buffer = []
        self._speech_samples = 0
        self._silence_counter = 0
        self._in_speech = False
