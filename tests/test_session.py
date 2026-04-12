from unittest.mock import MagicMock
from voice.session import VoiceSession


def test_session_creates():
    store = MagicMock()
    speaker = MagicMock()
    s = VoiceSession(store=store, speaker=speaker)
    assert s.store is store
    assert s.speaker is speaker


def test_session_creates_default_speaker():
    store = MagicMock()
    s = VoiceSession(store=store)
    assert s.speaker is not None


def test_read_current_speaks_passage():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = {
        "text": "Christmas won't be Christmas without any presents.",
        "chapter_title": "Chapter 1",
        "position": 0,
        "word_count": 7,
    }
    store.get_progress.return_value = {"title": "Little Women", "position": 0, "total": 100, "reactions": 0}
    s = VoiceSession(store=store, speaker=speaker)
    passage = s.read_current()
    speaker.speak.assert_called_once()
    spoken_text = speaker.speak.call_args[0][0]
    assert "Christmas" in spoken_text
    assert "Chapter 1" in spoken_text
    assert "1 of 100" in spoken_text
    assert passage is not None
    assert passage["text"] == "Christmas won't be Christmas without any presents."


def test_read_current_no_book_returns_none():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = None
    s = VoiceSession(store=store, speaker=speaker)
    result = s.read_current()
    assert result is None
    speaker.speak.assert_called_once()
    assert "No book" in speaker.speak.call_args[0][0]


def test_read_current_finished_returns_none():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = None  # no more passages
    s = VoiceSession(store=store, speaker=speaker)
    result = s.read_current()
    assert result is None
    assert "finished" in speaker.speak.call_args[0][0].lower()


def test_handle_gesture_advance_stores_and_advances():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = {"position": 5}
    store.advance.return_value = True
    s = VoiceSession(store=store, speaker=speaker)
    result = s.handle_gesture_advance(reaction="liked it")
    store.store_reaction.assert_called_once_with(1, 5, "liked it")
    store.advance.assert_called_once_with(1)
    assert result["advanced"] is True
    assert result["reaction_stored"] is True


def test_handle_gesture_advance_at_end():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = {"position": 99}
    store.advance.return_value = False
    s = VoiceSession(store=store, speaker=speaker)
    result = s.handle_gesture_advance()
    assert result["advanced"] is False


def test_handle_voice_reaction():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = {"position": 3}
    store.advance.return_value = True
    s = VoiceSession(store=store, speaker=speaker)
    result = s.handle_voice_reaction("I loved this passage, it reminded me of home")
    store.store_reaction.assert_called_once_with(1, 3, "I loved this passage, it reminded me of home")
    assert result["advanced"] is True


def test_reread_calls_read_current():
    store = MagicMock()
    speaker = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    store.get_current_passage.return_value = {
        "text": "Some passage.",
        "chapter_title": "Ch1",
        "position": 0,
        "word_count": 2,
    }
    store.get_progress.return_value = {"title": "Little Women", "position": 0, "total": 50, "reactions": 0}
    s = VoiceSession(store=store, speaker=speaker)
    s.reread()
    speaker.speak.assert_called_once()


def test_book_property_caches():
    store = MagicMock()
    store.get_active_book.return_value = {"id": 1, "title": "Little Women"}
    s = VoiceSession(store=store, speaker=MagicMock())
    _ = s.book
    _ = s.book
    store.get_active_book.assert_called_once()  # only called once due to caching
