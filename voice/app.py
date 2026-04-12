"""Voice reader CLI — accessible reading interface.

Usage:
    python -m voice [book.epub]        # read a new book
    python -m voice                    # resume current book
    python -m voice --no-camera        # keyboard/voice only (no gestures)
    python -m voice --no-mic --no-camera  # TTS only, manual advance
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from store import ReadingStore
from voice.tts import Speaker
from voice.gestures import Gesture, GestureWatcher
from voice.session import VoiceSession

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("voice.app")


def main():
    parser = argparse.ArgumentParser(description="Voice reader for shared reading")
    parser.add_argument("book", nargs="?", help="Path to epub (omit to resume)")
    parser.add_argument("--voice", default="Samantha", help="TTS voice name")
    parser.add_argument("--rate", type=int, default=175, help="TTS speech rate (wpm)")
    parser.add_argument("--backend", choices=["piper", "macos"], help="TTS backend override")
    parser.add_argument("--model", help="Piper TTS model path")
    parser.add_argument("--no-camera", action="store_true", help="Disable gesture input")
    parser.add_argument("--no-mic", action="store_true", help="Disable voice input")
    parser.add_argument("--hold-frames", type=int, default=10,
                        help="Gesture stability frames (higher = more tremor-resistant)")
    args = parser.parse_args()

    # Initialize components
    store = ReadingStore()
    speaker = Speaker(
        backend=args.backend,
        voice=args.voice,
        rate=args.rate,
        model=args.model,
    )
    session = VoiceSession(store=store, speaker=speaker)

    # Open book if path given
    if args.book:
        from pathlib import Path
        from epub import parse_epub, extract_title
        from chunker import chunk_chapter

        book_path = Path(args.book)
        if not book_path.exists():
            print(f"Book not found: {book_path}", file=sys.stderr)
            sys.exit(1)

        title = extract_title(book_path)
        chapters = parse_epub(book_path)
        chunked = []
        total = 0
        for ch in chapters:
            chunks = chunk_chapter(ch["text"])
            chunked.append({"chapter_title": ch["title"], "chunks": chunks})
            total += len(chunks)
        store.add_book(title, str(book_path), chunked)
        speaker.speak(f"Opened {title}. {total} passages. Let's begin.", block=True)

    # Start gesture watcher
    watcher = None
    if not args.no_camera:
        try:
            watcher = GestureWatcher(hold_frames=args.hold_frames)
            watcher.start()
            logger.info("Camera ready — watching for gestures")
        except Exception as e:
            logger.warning("Camera unavailable: %s — continuing without gestures", e)
            watcher = None

    # Initialize listener
    listener = None
    if not args.no_mic:
        try:
            from voice.stt import Listener
            listener = Listener()
            logger.info("Microphone ready — listening enabled")
        except Exception as e:
            logger.warning("Microphone unavailable: %s — continuing without voice input", e)

    # Intro and first passage
    speaker.speak(
        "Ready. Thumbs up to advance. Open palm to speak. Here we go.",
        block=True,
    )
    time.sleep(0.5)
    passage = session.read_current()
    if not passage:
        return

    # Main loop
    try:
        while True:
            gesture = watcher.poll() if watcher else None

            if gesture == Gesture.THUMBS_UP:
                speaker.speak("Got it.", block=True)
                result = session.handle_gesture_advance(reaction="[thumbs up]")
                if not result["advanced"]:
                    speaker.speak("That's the end of the book.", block=True)
                    break
                time.sleep(0.3)
                session.read_current()

            elif gesture == Gesture.THUMBS_DOWN:
                speaker.speak("Reading again.", block=True)
                time.sleep(0.3)
                session.reread()

            elif gesture == Gesture.OPEN_PALM:
                if listener:
                    speaker.speak("I'm listening.", block=True)
                    text = listener.listen_once(timeout=30.0)
                    if text:
                        speaker.speak(f"You said: {text}", block=True)
                        time.sleep(0.5)
                        result = session.handle_voice_reaction(text)
                        if not result["advanced"]:
                            speaker.speak("That's the end of the book.", block=True)
                            break
                        time.sleep(0.3)
                        session.read_current()
                    else:
                        speaker.speak("I didn't catch that. Try again or thumbs up to skip.", block=True)
                else:
                    speaker.speak("Voice input not available.", block=True)

            time.sleep(0.03)  # ~30fps polling rate

    except KeyboardInterrupt:
        speaker.speak("Goodbye.", block=True)
    finally:
        if watcher:
            watcher.stop()
        logger.info("Session ended")


if __name__ == "__main__":
    main()
