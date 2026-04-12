"""Speech-to-text via faster-whisper + sounddevice capture."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("voice.stt")

_model = None


def _load_model(model_size: str = "base"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(model_size, compute_type="float32")
    return _model


class Listener:
    """Captures audio and transcribes speech.

    Designed for deliberate, gesture-triggered activation — not always-on listening.
    Energy-based VAD keeps it simple and reliable without a heavy VAD model.
    """

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self._model = _load_model(model_size)

    def _is_speech(self, audio: np.ndarray) -> bool:
        """Return True if RMS energy of audio chunk is above threshold."""
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.silence_threshold

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper on audio array, return stripped transcript text."""
        segments, _ = self._model.transcribe(audio, language="en")
        text = "".join(s.text for s in segments).strip()
        return text

    def listen_once(self, timeout: float = 30.0) -> str | None:
        """Listen for speech, return transcription when silence detected.

        Waits for speech to start, records until silence_duration of quiet,
        then transcribes. Returns None on timeout with no speech detected,
        or if the captured audio is too short to be meaningful.
        """
        import sounddevice as sd
        import time

        chunk_samples = int(self.sample_rate * 0.1)  # 100ms chunks
        silence_chunks = int(self.silence_duration / 0.1)

        audio_buffer: list[np.ndarray] = []
        silent_count = 0
        heard_speech = False
        start = time.monotonic()

        def callback(indata, frames, time_info, status):
            nonlocal silent_count, heard_speech
            if status:
                logger.debug("sounddevice status: %s", status)
            chunk = indata[:, 0].copy()
            if self._is_speech(chunk):
                heard_speech = True
                silent_count = 0
                audio_buffer.append(chunk)
            elif heard_speech:
                silent_count += 1
                audio_buffer.append(chunk)  # keep recording during silence gap

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=callback,
        ):
            while True:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    logger.debug("listen_once: timeout after %.1fs", elapsed)
                    break
                if heard_speech and silent_count >= silence_chunks:
                    logger.debug(
                        "listen_once: speech ended after %d silent chunks", silent_count
                    )
                    break
                sd.sleep(50)

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer)
        min_samples = int(self.sample_rate * 0.3)  # less than 300ms — too short
        if len(audio) < min_samples:
            logger.debug(
                "listen_once: audio too short (%d samples, min %d)", len(audio), min_samples
            )
            return None

        logger.debug("listen_once: transcribing %.2fs of audio", len(audio) / self.sample_rate)
        return self._transcribe(audio)
