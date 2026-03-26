"""Reading state — tracks books, chunks, position, and reactions."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(".reader/state.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReadingStore:
    """SQLite store for reading state."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                opened_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                position INTEGER NOT NULL,
                chapter_title TEXT,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                UNIQUE(book_id, position)
            );

            CREATE TABLE IF NOT EXISTS reading_position (
                book_id INTEGER PRIMARY KEY REFERENCES books(id),
                current_chunk INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                chunk_position INTEGER NOT NULL,
                reaction TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS companion_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                chunk_position INTEGER NOT NULL,
                model TEXT NOT NULL,
                reaction TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instinct_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                chunk_position INTEGER NOT NULL,
                reaction_id INTEGER NOT NULL REFERENCES reactions(id),
                instinct_ids TEXT NOT NULL,
                notes TEXT NOT NULL,
                reviewed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instinct_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                note_ids TEXT NOT NULL,
                verdicts TEXT NOT NULL,
                patterns TEXT,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prism_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL REFERENCES books(id),
                chunk_position INTEGER NOT NULL,
                lens TEXT NOT NULL,
                question TEXT NOT NULL,
                selection_type TEXT NOT NULL,
                engaged BOOLEAN DEFAULT 0,
                response TEXT,
                forwarded_to_engram BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def add_book(self, title: str, path: str, chapters: list[dict]) -> int:
        """Store a parsed and chunked book. chapters: [{chapter_title, chunks: [str]}]"""
        cursor = self.conn.execute(
            "INSERT INTO books (title, path, opened_at) VALUES (?, ?, ?)",
            (title, path, _now()),
        )
        book_id = cursor.lastrowid

        position = 0
        for chapter in chapters:
            chapter_title = chapter["chapter_title"]
            for chunk_text in chapter["chunks"]:
                word_count = len(chunk_text.split())
                self.conn.execute(
                    "INSERT INTO chunks (book_id, position, chapter_title, text, word_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (book_id, position, chapter_title, chunk_text, word_count),
                )
                position += 1

        self.conn.execute(
            "UPDATE books SET total_chunks = ? WHERE id = ?",
            (position, book_id),
        )
        self.conn.execute(
            "INSERT INTO reading_position (book_id, current_chunk, updated_at) VALUES (?, 0, ?)",
            (book_id, _now()),
        )
        self.conn.commit()
        return book_id

    def get_active_book(self) -> dict | None:
        """Get the most recently opened unfinished book."""
        row = self.conn.execute(
            "SELECT * FROM books WHERE finished_at IS NULL ORDER BY opened_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_current_passage(self, book_id: int) -> dict | None:
        """Get the current chunk for reading."""
        pos_row = self.conn.execute(
            "SELECT current_chunk FROM reading_position WHERE book_id = ?",
            (book_id,),
        ).fetchone()
        if not pos_row:
            return None

        chunk = self.conn.execute(
            "SELECT * FROM chunks WHERE book_id = ? AND position = ?",
            (book_id, pos_row["current_chunk"]),
        ).fetchone()
        return dict(chunk) if chunk else None

    def get_progress(self, book_id: int) -> dict:
        """Get reading progress for a book."""
        book = self.conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        pos = self.conn.execute(
            "SELECT current_chunk FROM reading_position WHERE book_id = ?", (book_id,)
        ).fetchone()
        reactions = self.conn.execute(
            "SELECT COUNT(*) as count FROM reactions WHERE book_id = ?", (book_id,)
        ).fetchone()
        return {
            "title": book["title"],
            "position": pos["current_chunk"] if pos else 0,
            "total": book["total_chunks"],
            "reactions": reactions["count"],
        }

    def advance(self, book_id: int) -> bool:
        """Move to next chunk. Returns False if at end of book."""
        book = self.conn.execute("SELECT total_chunks FROM books WHERE id = ?", (book_id,)).fetchone()
        pos = self.conn.execute(
            "SELECT current_chunk FROM reading_position WHERE book_id = ?", (book_id,)
        ).fetchone()

        next_pos = pos["current_chunk"] + 1
        if next_pos >= book["total_chunks"]:
            self.conn.execute(
                "UPDATE books SET finished_at = ? WHERE id = ?",
                (_now(), book_id),
            )
            self.conn.commit()
            return False

        self.conn.execute(
            "UPDATE reading_position SET current_chunk = ?, updated_at = ? WHERE book_id = ?",
            (next_pos, _now(), book_id),
        )
        self.conn.commit()
        return True

    def store_reaction(self, book_id: int, chunk_position: int, reaction: str) -> int:
        """Store a reaction to a passage."""
        cursor = self.conn.execute(
            "INSERT INTO reactions (book_id, chunk_position, reaction, created_at) "
            "VALUES (?, ?, ?, ?)",
            (book_id, chunk_position, reaction, _now()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def store_companion_reaction(
        self, book_id: int, chunk_position: int, model: str, reaction: str
    ) -> int:
        """Store a companion's reaction to a passage."""
        cursor = self.conn.execute(
            "INSERT INTO companion_reactions (book_id, chunk_position, model, reaction, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (book_id, chunk_position, model, reaction, _now()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def store_instinct_note(
        self, book_id: int, chunk_position: int, reaction_id: int,
        instinct_ids: list[str], notes: str,
    ) -> int:
        """Store an instinct note linked to a reaction."""
        import json
        cursor = self.conn.execute(
            "INSERT INTO instinct_notes "
            "(book_id, chunk_position, reaction_id, instinct_ids, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_id, chunk_position, reaction_id, json.dumps(instinct_ids), notes, _now()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def count_unreviewed_notes(self, book_id: int) -> int:
        """Count instinct notes not yet covered by a deep review."""
        row = self.conn.execute(
            "SELECT COUNT(*) as n FROM instinct_notes WHERE book_id = ? AND reviewed = 0",
            (book_id,),
        ).fetchone()
        return row["n"]

    def get_unreviewed_notes(self, book_id: int, limit: int = 10) -> list[dict]:
        """Get unreviewed instinct notes with their passage text and reaction."""
        rows = self.conn.execute(
            "SELECT n.id, n.chunk_position, n.instinct_ids, n.notes, "
            "  c.text AS passage_text, c.chapter_title, r.reaction "
            "FROM instinct_notes n "
            "JOIN chunks c ON c.book_id = n.book_id AND c.position = n.chunk_position "
            "JOIN reactions r ON r.id = n.reaction_id "
            "WHERE n.book_id = ? AND n.reviewed = 0 "
            "ORDER BY n.created_at DESC LIMIT ?",
            (book_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_notes_reviewed(self, note_ids: list[int]) -> None:
        """Mark instinct notes as reviewed."""
        if not note_ids:
            return
        placeholders = ",".join("?" for _ in note_ids)
        self.conn.execute(
            f"UPDATE instinct_notes SET reviewed = 1 WHERE id IN ({placeholders})",
            note_ids,
        )
        self.conn.commit()

    def store_review(
        self, book_id: int, note_ids: list[int], verdicts: list[dict],
        patterns: str | None, model: str,
    ) -> int:
        """Store a deep review result."""
        import json
        cursor = self.conn.execute(
            "INSERT INTO instinct_reviews "
            "(book_id, note_ids, verdicts, patterns, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_id, json.dumps(note_ids), json.dumps(verdicts), patterns, model, _now()),
        )
        self.mark_notes_reviewed(note_ids)
        self.conn.commit()
        return cursor.lastrowid

    def store_prism_question(
        self,
        book_id: int,
        chunk_position: int,
        lens: str,
        question: str,
        selection_type: str,
    ) -> int:
        """Store a prism question for a passage."""
        cur = self.conn.execute(
            "INSERT INTO prism_questions "
            "(book_id, chunk_position, lens, question, selection_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_id, chunk_position, lens, question, selection_type, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_prism_questions(self, book_id: int, chunk_position: int) -> list[dict]:
        """Get prism questions for a specific passage."""
        rows = self.conn.execute(
            "SELECT * FROM prism_questions "
            "WHERE book_id = ? AND chunk_position = ? ORDER BY id",
            (book_id, chunk_position),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_prism_engaged(self, question_id: int, response: str = "") -> None:
        """Mark a prism question as engaged with optional response."""
        self.conn.execute(
            "UPDATE prism_questions SET engaged = 1, response = ? WHERE id = ?",
            (response, question_id),
        )
        self.conn.commit()

    def prism_fired_history(self, book_id: int, limit: int = 50) -> list[str]:
        """Get recent fired lens names in order for rotating selection."""
        rows = self.conn.execute(
            "SELECT lens FROM prism_questions WHERE book_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (book_id, limit),
        ).fetchall()
        return [r["lens"] for r in reversed(rows)]

    def close(self):
        self.conn.close()
