"""Tests for voice/stt.py — mocked so no microphone or Whisper model needed."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import voice.stt as stt_module
from voice.stt import Listener


@pytest.fixture(autouse=True)
def reset_global_model():
    """Reset the global _model between tests to prevent state leakage."""
    original = stt_module._model
    stt_module._model = None
    yield
    stt_module._model = original


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_listener_creates():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener()
        assert l.sample_rate == 16000
        assert l.silence_threshold == 0.01
        assert l.silence_duration == 1.5


def test_listener_custom_params():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(sample_rate=8000, silence_threshold=0.05, silence_duration=2.0)
        assert l.sample_rate == 8000
        assert l.silence_threshold == 0.05
        assert l.silence_duration == 2.0


def test_listener_stores_model():
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        l = Listener(model_size="tiny")
        mock_load.assert_called_once_with("tiny")
        assert l._model is mock_model


# ---------------------------------------------------------------------------
# _is_speech
# ---------------------------------------------------------------------------

def test_energy_above_threshold_is_speech():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        loud = np.ones(1600, dtype=np.float32) * 0.5
        assert l._is_speech(loud)


def test_energy_below_threshold_is_silence():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        quiet = np.ones(1600, dtype=np.float32) * 0.001
        assert not l._is_speech(quiet)


def test_zero_audio_is_silence():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        silent = np.zeros(1600, dtype=np.float32)
        assert not l._is_speech(silent)


def test_is_speech_exactly_at_threshold_is_silence():
    """Boundary: energy equal to threshold is NOT speech (> not >=)."""
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.5)
        # np.ones * 0.5 → RMS = 0.5, which is not > 0.5
        at_threshold = np.ones(1600, dtype=np.float32) * 0.5
        assert not l._is_speech(at_threshold)


def test_is_speech_just_above_threshold():
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.5)
        just_above = np.ones(1600, dtype=np.float32) * 0.501
        assert l._is_speech(just_above)


def test_is_speech_mixed_signal():
    """Array with some loud and some quiet values — RMS should decide."""
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        # Half zeros, half 0.1 → RMS ≈ 0.0707 > 0.01
        mixed = np.array([0.0] * 800 + [0.1] * 800, dtype=np.float32)
        assert l._is_speech(mixed)


def test_is_speech_negative_amplitude():
    """Negative amplitudes should be treated same as positive (squared in RMS)."""
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        negative = np.ones(1600, dtype=np.float32) * -0.5
        assert l._is_speech(negative)


def test_is_speech_single_sample():
    """Edge case: single sample array."""
    with patch("voice.stt._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        l = Listener(silence_threshold=0.01)
        assert l._is_speech(np.array([1.0], dtype=np.float32))
        assert not l._is_speech(np.array([0.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# _transcribe
# ---------------------------------------------------------------------------

def test_transcribe_returns_text():
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_segments = [MagicMock(text=" Hello there ")]
        mock_model.transcribe.return_value = (mock_segments, None)
        mock_load.return_value = mock_model
        l = Listener()
        audio = np.zeros(16000, dtype=np.float32)
        result = l._transcribe(audio)
        assert result == "Hello there"


def test_transcribe_concatenates_segments():
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_segments = [MagicMock(text=" Hello"), MagicMock(text=" world")]
        mock_model.transcribe.return_value = (mock_segments, None)
        mock_load.return_value = mock_model
        l = Listener()
        audio = np.zeros(16000, dtype=np.float32)
        result = l._transcribe(audio)
        assert result == "Hello world"


def test_transcribe_empty_returns_empty():
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_load.return_value = mock_model
        l = Listener()
        audio = np.zeros(16000, dtype=np.float32)
        result = l._transcribe(audio)
        assert result == ""


def test_transcribe_strips_whitespace():
    """Transcription with only whitespace should strip to empty string."""
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_segments = [MagicMock(text="  "), MagicMock(text="   ")]
        mock_model.transcribe.return_value = (mock_segments, None)
        mock_load.return_value = mock_model
        l = Listener()
        audio = np.zeros(16000, dtype=np.float32)
        result = l._transcribe(audio)
        assert result == ""


def test_transcribe_passes_audio_to_model():
    """Ensure the audio array is forwarded to the model's transcribe call."""
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_load.return_value = mock_model
        l = Listener()
        audio = np.ones(16000, dtype=np.float32) * 0.5
        l._transcribe(audio)
        call_args = mock_model.transcribe.call_args
        np.testing.assert_array_equal(call_args[0][0], audio)
        assert call_args[1]["language"] == "en"


def test_transcribe_single_segment_no_spaces():
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_segments = [MagicMock(text="okay")]
        mock_model.transcribe.return_value = (mock_segments, None)
        mock_load.return_value = mock_model
        l = Listener()
        result = l._transcribe(np.zeros(16000, dtype=np.float32))
        assert result == "okay"


# ---------------------------------------------------------------------------
# _load_model (global caching)
# ---------------------------------------------------------------------------

def test_load_model_caches_globally():
    """_load_model should only construct WhisperModel once."""
    with patch("faster_whisper.WhisperModel") as MockWhisper:
        MockWhisper.return_value = MagicMock()
        stt_module._load_model("base")
        stt_module._load_model("base")
        MockWhisper.assert_called_once()


def test_load_model_returns_same_instance():
    with patch("faster_whisper.WhisperModel") as MockWhisper:
        instance = MagicMock()
        MockWhisper.return_value = instance
        m1 = stt_module._load_model("base")
        m2 = stt_module._load_model("base")
        assert m1 is m2


# ---------------------------------------------------------------------------
# listen_once — mocked sounddevice
# ---------------------------------------------------------------------------

def _make_listener_with_mock_model():
    """Helper: return a Listener with a mocked model, no global side effects."""
    with patch("voice.stt._load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([MagicMock(text=" test ")], None)
        mock_load.return_value = mock_model
        return Listener()


def test_listen_once_returns_none_when_no_speech():
    """If no speech detected in the callback, listen_once returns None."""
    l = _make_listener_with_mock_model()

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("sounddevice.InputStream", return_value=FakeStream()), \
         patch("sounddevice.sleep"), \
         patch("time.monotonic", side_effect=[0.0, 31.0, 31.0]):
        result = l.listen_once(timeout=30.0)

    assert result is None


def test_listen_once_returns_none_for_too_short_audio():
    """Audio shorter than 300ms should return None even if speech was heard."""
    l = _make_listener_with_mock_model()

    # Simulate: speech detected, but only 10ms of audio buffered
    short_chunk = np.ones(160, dtype=np.float32) * 0.5  # 10ms at 16kHz

    class FakeStream:
        def __init__(self, **kwargs):
            self._cb = kwargs.get("callback")
        def __enter__(self):
            # Inject one loud chunk to set heard_speech = True
            indata = short_chunk.reshape(-1, 1)
            self._cb(indata, 160, None, None)
            return self
        def __exit__(self, *a): pass

    with patch("sounddevice.InputStream", FakeStream), \
         patch("sounddevice.sleep"), \
         patch("time.monotonic", side_effect=[0.0, 31.0, 31.0]):
        result = l.listen_once(timeout=30.0)

    assert result is None
