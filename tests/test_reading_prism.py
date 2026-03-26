"""Tests for the reading prism connector."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from store import ReadingStore


def test_run_prism_stores_questions(tmp_path):
    from reading_prism import run_reading_prism

    store = ReadingStore(tmp_path / "test.db")

    with patch("reading_prism._classify") as mock_classify, \
         patch("reading_prism._ask_lens") as mock_ask:
        mock_classify.return_value = (["attachment", "power"], "information-theory", "formalist")
        mock_ask.side_effect = lambda lens, ctx: f"Question from {lens['name']}"

        lenses = {
            "attachment": {"name": "attachment", "prompt": "test", "temperature": 0.8},
            "power": {"name": "power", "prompt": "test", "temperature": 0.8},
            "information-theory": {"name": "information-theory", "prompt": "test", "temperature": 0.8},
            "formalist": {"name": "formalist", "prompt": "test", "temperature": 0.8},
        }

        questions = run_reading_prism(
            store=store,
            book_id=1,
            chunk_position=42,
            passage="Vin took three baywraps.",
            reaction="Scarcity is in her.",
            lenses=lenses,
        )

        assert len(questions) == 4
        assert questions[0]["lens"] == "attachment"

    stored = store.get_prism_questions(book_id=1, chunk_position=42)
    assert len(stored) == 4


def test_run_prism_returns_empty_on_failure(tmp_path):
    from reading_prism import run_reading_prism

    store = ReadingStore(tmp_path / "test.db")

    with patch("reading_prism._classify") as mock_classify:
        mock_classify.return_value = ([], None, None)

        questions = run_reading_prism(
            store=store,
            book_id=1,
            chunk_position=42,
            passage="test",
            reaction="test",
            lenses={},
        )
        assert questions == []


def test_handle_prism_response(tmp_path):
    from reading_prism import handle_prism_response

    store = ReadingStore(tmp_path / "test.db")
    store.store_prism_question(1, 42, "attachment", "Who does Vin move toward?", "relevant")

    result = handle_prism_response(store, "attachment", "She moves toward information", 1, 42)

    assert result is not None
    assert "prism:attachment" in result["remember"]["source"]
    assert "She moves toward information" in result["remember"]["content"]
    assert result["dream_capture"]["reaction"] == "She moves toward information"

    # Verify marked as engaged
    questions = store.get_prism_questions(1, 42)
    assert questions[0]["engaged"] == 1


def test_handle_prism_response_no_match(tmp_path):
    from reading_prism import handle_prism_response

    store = ReadingStore(tmp_path / "test.db")
    result = handle_prism_response(store, "power", "test", 1, 42)
    assert result is None


def test_recall_query_is_raw_question(tmp_path):
    from reading_prism import run_reading_prism

    store = ReadingStore(tmp_path / "test.db")

    with patch("reading_prism._classify") as mock_classify, \
         patch("reading_prism._ask_lens") as mock_ask:
        mock_classify.return_value = (["attachment"], "power", "formalist")
        mock_ask.side_effect = lambda lens, ctx: f"Question from {lens['name']}"

        lenses = {
            "attachment": {"name": "attachment", "prompt": "test", "domain": "early-childhood-development", "temperature": 0.8},
            "power": {"name": "power", "prompt": "test", "domain": "literary-theory", "temperature": 0.8},
            "formalist": {"name": "formalist", "prompt": "test", "domain": "literary-theory", "temperature": 0.8},
            "authorial": {"name": "authorial", "prompt": "test", "domain": "craft", "temperature": 0.8},
        }

        questions = run_reading_prism(
            store=store, book_id=1, chunk_position=42,
            passage="test", reaction="test", lenses=lenses,
        )

        # Should have classified lenses + authorial
        assert any(q["lens"] == "authorial" for q in questions)
        # Every question should have recall_query = raw question
        for q in questions:
            assert q["recall_query"] == q["question"]
