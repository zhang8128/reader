"""Voice reading session — orchestrates TTS, STT, and gestures."""
from __future__ import annotations

import logging

from voice.tts import Speaker

logger = logging.getLogger("voice.session")


class VoiceSession:
    """Orchestrates a voice reading session."""

    def __init__(self, store, speaker: Speaker | None = None):
        self.store = store
        self.speaker = speaker or Speaker()
        self._book = None

    @property
    def book(self):
        if self._book is None:
            self._book = self.store.get_active_book()
        return self._book

    def read_current(self) -> dict | None:
        """Read the current passage aloud. Returns passage dict."""
        if not self.book:
            self.speaker.speak("No book is open.", block=True)
            return None

        passage = self.store.get_current_passage(self.book["id"])
        if not passage:
            self.speaker.speak(
                f"You've finished {self.book['title']}.", block=True
            )
            return None

        progress = self.store.get_progress(self.book["id"])
        chapter = passage["chapter_title"]
        pos = passage["position"] + 1
        total = progress["total"]

        # Announce position, then read
        self.speaker.speak(
            f"{chapter}. Passage {pos} of {total}. {passage['text']}",
            block=True,
        )
        return passage

    def reread(self) -> dict | None:
        """Re-read the current passage."""
        return self.read_current()

    def handle_gesture_advance(self, reaction: str = "positive") -> dict:
        """Store a reaction and advance to next passage."""
        book = self.book
        passage = self.store.get_current_passage(book["id"])
        pos = passage["position"] if passage else 0

        self.store.store_reaction(book["id"], pos, reaction)
        has_more = self.store.advance(book["id"])

        return {"advanced": has_more, "reaction_stored": True}

    def handle_voice_reaction(self, transcription: str) -> dict:
        """Store a voice transcription as reaction and advance."""
        return self.handle_gesture_advance(reaction=transcription)
