"""Tests for prism question storage in reader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from store import ReadingStore


def test_store_prism_question(tmp_path):
    store = ReadingStore(tmp_path / "test.db")
    qid = store.store_prism_question(
        book_id=1,
        chunk_position=42,
        lens="attachment",
        question="Who does Vin move toward when afraid?",
        selection_type="relevant",
    )
    assert qid is not None


def test_get_questions_for_position(tmp_path):
    store = ReadingStore(tmp_path / "test.db")
    store.store_prism_question(1, 42, "attachment", "Q1", "relevant")
    store.store_prism_question(1, 42, "power", "Q2", "distant")
    store.store_prism_question(1, 43, "formalist", "Q3", "relevant")

    questions = store.get_prism_questions(book_id=1, chunk_position=42)
    assert len(questions) == 2
    assert questions[0]["lens"] == "attachment"


def test_mark_prism_engaged(tmp_path):
    store = ReadingStore(tmp_path / "test.db")
    qid = store.store_prism_question(1, 42, "attachment", "Q1", "relevant")
    store.mark_prism_engaged(qid, response="She moves toward information")

    questions = store.get_prism_questions(book_id=1, chunk_position=42)
    assert questions[0]["engaged"] == 1
    assert questions[0]["response"] == "She moves toward information"


def test_fired_history(tmp_path):
    store = ReadingStore(tmp_path / "test.db")
    store.store_prism_question(1, 40, "attachment", "Q1", "relevant")
    store.store_prism_question(1, 41, "power", "Q2", "relevant")
    store.store_prism_question(1, 42, "attachment", "Q3", "relevant")

    history = store.prism_fired_history(book_id=1, limit=10)
    assert history == ["attachment", "power", "attachment"]
