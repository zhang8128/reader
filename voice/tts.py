"""
voice/tts.py — Cross-platform text-to-speech.

Backends:
  piper  — Piper TTS (Linux primary target, AMD HX370 / Bazzite)
  macos  — macOS `say` command (development fallback)

Platform is auto-detected unless `backend` is passed explicitly.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

_DEFAULT_PIPER_MODEL = str(
    Path("~/.local/share/piper/en_US-lessac-medium.onnx").expanduser()
)


def _detect_platform() -> str:
    """Return 'piper' for Linux, 'macos' for Darwin."""
    if sys.platform == "darwin":
        return "macos"
    return "piper"


class Speaker:
    """Cross-platform TTS speaker.

    Parameters
    ----------
    backend : str, optional
        'piper' or 'macos'.  Auto-detected from sys.platform if omitted.
    voice : str, optional
        macOS: voice name passed to `say -v` (default 'Samantha').
        Piper: not used (voice is determined by the model file).
    rate : int, optional
        macOS: words-per-minute for `say -r` (default 175).
        Piper: not used directly; use `length_scale` instead.
    model : str, optional
        Piper only — path to the .onnx model file.
        Defaults to PIPER_MODEL env var, then ~/.local/share/piper/en_US-lessac-medium.onnx.
    length_scale : float, optional
        Piper only — affects speech rate via `--length-scale`.
        <1.0 = faster, 1.0 = normal, >1.0 = slower (default 1.0).
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        voice: str = "Samantha",
        rate: int = 175,
        model: Optional[str] = None,
        length_scale: float = 1.0,
    ):
        self._backend: str = backend if backend is not None else _detect_platform()
        self.voice = voice
        self.rate = rate
        self.length_scale = length_scale
        self.model: str = model or os.environ.get("PIPER_MODEL", _DEFAULT_PIPER_MODEL)

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._tmp_wav: Optional[str] = None  # path of current temp file (Piper only)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True if a speech process is currently running."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def speak(self, text: str, block: bool = False) -> None:
        """Speak *text*.

        Parameters
        ----------
        text  : The string to speak.
        block : If True, wait until speech finishes before returning.
                Default is False (non-blocking).
        """
        # Stop any in-progress speech first.
        self.stop()

        if self._backend == "macos":
            self._speak_macos(text, block)
        elif self._backend == "piper":
            self._speak_piper(text, block)
        else:
            raise ValueError(f"Unknown backend: {self._backend!r}")

    def stop(self) -> None:
        """Interrupt current speech.  Safe to call when nothing is playing."""
        with self._lock:
            proc = self._proc
            tmp = self._tmp_wav
            self._proc = None
            self._tmp_wav = None

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # macOS backend
    # ------------------------------------------------------------------

    def _speak_macos(self, text: str, block: bool) -> None:
        cmd = ["say", "-v", self.voice, "-r", str(self.rate), text]
        proc = subprocess.Popen(cmd)
        with self._lock:
            self._proc = proc
        if block:
            proc.wait()

    # ------------------------------------------------------------------
    # Piper backend
    # ------------------------------------------------------------------

    def _speak_piper(self, text: str, block: bool) -> None:
        # Check piper is available; fail gracefully if not.
        if shutil.which("piper") is None:
            raise RuntimeError(
                "piper binary not found on PATH. "
                "Install Piper TTS: https://github.com/rhasspy/piper"
            )

        # Write to a named temp file so we can play it after synthesis.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            piper_cmd = [
                "piper",
                "--model", self.model,
                "--output_file", tmp_path,
                "--length-scale", str(self.length_scale),
            ]
            subprocess.run(
                piper_cmd,
                input=text,
                text=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise RuntimeError(
                f"piper synthesis failed (exit {exc.returncode}): {exc.stderr}"
            ) from exc

        # Choose player: aplay on Linux, afplay on macOS.
        player = "afplay" if sys.platform == "darwin" else "aplay"
        player_cmd = [player, tmp_path]

        proc = subprocess.Popen(player_cmd)
        with self._lock:
            self._proc = proc
            self._tmp_wav = tmp_path

        if block:
            proc.wait()
            # Clean up temp file once done.
            with self._lock:
                self._tmp_wav = None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        else:
            # Clean up in background after playback finishes.
            threading.Thread(
                target=self._cleanup_after,
                args=(proc, tmp_path),
                daemon=True,
            ).start()

    def _cleanup_after(self, proc: subprocess.Popen, tmp_path: str) -> None:
        """Wait for *proc* to exit, then delete *tmp_path*."""
        try:
            proc.wait()
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        # Clear internal state if it still refers to this process.
        with self._lock:
            if self._proc is proc:
                self._proc = None
            if self._tmp_wav == tmp_path:
                self._tmp_wav = None
