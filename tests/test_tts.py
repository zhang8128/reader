"""tests/test_tts.py — Unit tests for voice.tts.Speaker."""

import os
import sys
from unittest.mock import patch, MagicMock, call

import pytest

from voice.tts import Speaker, _detect_platform, _DEFAULT_PIPER_MODEL


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_speaker_creates_with_defaults():
    s = Speaker()
    assert s.rate is not None


def test_speaker_rate_default_value():
    s = Speaker()
    assert s.rate == 175


def test_speaker_length_scale_default():
    s = Speaker()
    assert s.length_scale == 1.0


def test_speaker_voice_default():
    s = Speaker()
    assert s.voice == "Samantha"


def test_speaker_macos_backend():
    with patch("sys.platform", "darwin"):
        s = Speaker(backend="macos")
        assert s._backend == "macos"


def test_speaker_piper_backend():
    s = Speaker(backend="piper")
    assert s._backend == "piper"


def test_speaker_explicit_backend_overrides_platform():
    # Even on macOS, piper can be forced.
    s = Speaker(backend="piper")
    assert s._backend == "piper"


def test_detect_platform_darwin():
    with patch("sys.platform", "darwin"):
        import voice.tts as tts_mod
        with patch.object(tts_mod, "_detect_platform", wraps=tts_mod._detect_platform):
            result = tts_mod._detect_platform()
    # _detect_platform reads sys.platform at call time
    assert result == "macos"


def test_detect_platform_linux():
    with patch("sys.platform", "linux"):
        import voice.tts as tts_mod
        result = tts_mod._detect_platform()
    assert result == "piper"


def test_model_default_path():
    s = Speaker(backend="piper")
    assert "en_US-lessac-medium.onnx" in s.model


def test_model_env_var(monkeypatch):
    monkeypatch.setenv("PIPER_MODEL", "/custom/path/model.onnx")
    s = Speaker(backend="piper")
    assert s.model == "/custom/path/model.onnx"


def test_model_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("PIPER_MODEL", "/env/model.onnx")
    s = Speaker(backend="piper", model="/explicit/model.onnx")
    assert s.model == "/explicit/model.onnx"


# ---------------------------------------------------------------------------
# macOS speak
# ---------------------------------------------------------------------------


def test_macos_speak_calls_say():
    s = Speaker(backend="macos", voice="Samantha", rate=175)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Hello world")
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "say"
        assert "-v" in cmd
        assert "Samantha" in cmd
        assert "-r" in cmd


def test_macos_speak_includes_text():
    s = Speaker(backend="macos", voice="Samantha", rate=175)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Test sentence")
        cmd = mock_popen.call_args[0][0]
        assert "Test sentence" in cmd


def test_macos_speak_rate_in_command():
    s = Speaker(backend="macos", rate=200)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Hello")
        cmd = mock_popen.call_args[0][0]
        rate_idx = cmd.index("-r")
        assert cmd[rate_idx + 1] == "200"


def test_macos_speak_custom_voice():
    s = Speaker(backend="macos", voice="Alex")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Hi")
        cmd = mock_popen.call_args[0][0]
        voice_idx = cmd.index("-v")
        assert cmd[voice_idx + 1] == "Alex"


# ---------------------------------------------------------------------------
# Piper speak
# ---------------------------------------------------------------------------


def test_piper_speak_calls_piper_then_player():
    s = Speaker(backend="piper", model="/tmp/test.onnx")
    with patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen, \
         patch("tempfile.NamedTemporaryFile") as mock_tmp:
        mock_tmp.return_value.__enter__ = lambda s: MagicMock(name="/tmp/out.wav")
        mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
        mock_tmpfile = MagicMock()
        mock_tmpfile.name = "/tmp/out.wav"
        mock_tmp.return_value = mock_tmpfile
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        with patch("shutil.which", return_value="/usr/bin/piper"):
            s.speak("Hello world")
        assert mock_run.called or mock_popen.called


def test_piper_calls_subprocess_run_with_model():
    s = Speaker(backend="piper", model="/models/voice.onnx")
    with patch("shutil.which", return_value="/usr/bin/piper"), \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen, \
         patch("tempfile.NamedTemporaryFile") as mock_tmp, \
         patch("os.unlink"):
        mock_tmpfile = MagicMock()
        mock_tmpfile.name = "/tmp/synth123.wav"
        mock_tmp.return_value = mock_tmpfile
        mock_run.return_value = MagicMock(returncode=0)
        mock_popen.return_value = MagicMock()
        s.speak("Test piper")
        assert mock_run.called
        piper_cmd = mock_run.call_args[0][0]
        assert piper_cmd[0] == "piper"
        assert "--model" in piper_cmd
        assert "/models/voice.onnx" in piper_cmd


def test_piper_missing_binary_raises():
    s = Speaker(backend="piper", model="/tmp/test.onnx")
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="piper binary not found"):
            s.speak("Hello")


def test_piper_synthesis_failure_raises():
    s = Speaker(backend="piper", model="/tmp/test.onnx")
    with patch("shutil.which", return_value="/usr/bin/piper"), \
         patch("tempfile.NamedTemporaryFile") as mock_tmp, \
         patch("subprocess.run") as mock_run, \
         patch("os.unlink"):
        mock_tmpfile = MagicMock()
        mock_tmpfile.name = "/tmp/out.wav"
        mock_tmp.return_value = mock_tmpfile
        mock_run.side_effect = __import__("subprocess").CalledProcessError(
            1, "piper", stderr="model not found"
        )
        with pytest.raises(RuntimeError, match="piper synthesis failed"):
            s.speak("Hello")


def test_piper_length_scale_passed():
    s = Speaker(backend="piper", model="/m/model.onnx", length_scale=0.8)
    with patch("shutil.which", return_value="/usr/bin/piper"), \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen, \
         patch("tempfile.NamedTemporaryFile") as mock_tmp, \
         patch("os.unlink"):
        mock_tmpfile = MagicMock()
        mock_tmpfile.name = "/tmp/s.wav"
        mock_tmp.return_value = mock_tmpfile
        mock_run.return_value = MagicMock(returncode=0)
        mock_popen.return_value = MagicMock()
        s.speak("Speed test")
        piper_cmd = mock_run.call_args[0][0]
        ls_idx = piper_cmd.index("--length-scale")
        assert piper_cmd[ls_idx + 1] == "0.8"


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_terminates_process():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        s.speak("Long text")
        s.stop()
        mock_proc.terminate.assert_called_once()


def test_stop_safe_when_not_speaking():
    """stop() should not raise when nothing is playing."""
    s = Speaker(backend="macos")
    s.stop()  # no exception


def test_stop_safe_when_process_already_finished():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already finished
        mock_popen.return_value = mock_proc
        s.speak("Hi")
        # Should not raise even though process finished.
        s.stop()


def test_stop_clears_internal_state():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        s.speak("Hello")
        assert s._proc is not None
        s.stop()
        assert s._proc is None


# ---------------------------------------------------------------------------
# is_speaking
# ---------------------------------------------------------------------------


def test_is_speaking():
    s = Speaker(backend="macos")
    assert not s.is_speaking
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        s.speak("Hello")
        assert s.is_speaking


def test_is_speaking_false_after_process_exits():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # process finished
        mock_popen.return_value = mock_proc
        s.speak("Hello")
        assert not s.is_speaking


def test_is_speaking_false_before_speak():
    s = Speaker(backend="piper")
    assert not s.is_speaking


# ---------------------------------------------------------------------------
# block=True
# ---------------------------------------------------------------------------


def test_speak_blocks_when_sync():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Hello", block=True)
        mock_proc.wait.assert_called_once()


def test_speak_does_not_block_by_default():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        s.speak("Hello")
        mock_proc.wait.assert_not_called()


# ---------------------------------------------------------------------------
# speak() stops previous speech before starting new
# ---------------------------------------------------------------------------


def test_speak_stops_previous_before_new():
    s = Speaker(backend="macos")
    with patch("subprocess.Popen") as mock_popen:
        first_proc = MagicMock()
        first_proc.poll.return_value = None
        second_proc = MagicMock()
        second_proc.poll.return_value = None
        mock_popen.side_effect = [first_proc, second_proc]

        s.speak("First sentence")
        s.speak("Second sentence")  # should stop first

        first_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------


def test_unknown_backend_raises():
    s = Speaker(backend="espeak")
    with pytest.raises(ValueError, match="Unknown backend"):
        s.speak("Hello")
