"""Reader MCP server — shared reading sessions with engram memory.

Three tools:
- open_book: parse and chunk an epub
- read: serve the next passage
- react: store reaction, forward to engram
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from epub import parse_epub, extract_title
from chunker import chunk_chapter
from companion import get_reaction as companion_react
from reviewer import review_note
from store import ReadingStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reader")

mcp = FastMCP("reader")

READER_HOME = Path(os.environ.get("READER_HOME", Path.cwd()))
BOOKS_DIR = READER_HOME / "books"

_store: ReadingStore | None = None


def _get_store() -> ReadingStore:
    global _store
    if _store is None:
        db_path = READER_HOME / ".reader" / "state.db"
        _store = ReadingStore(db_path)
    return _store


@mcp.tool()
def open_book(path: str) -> dict:
    """Parse and chunk an epub for reading.

    Args:
        path: Path to epub file, relative to books/ directory or absolute.
    """
    # Resolve path
    book_path = Path(path)
    if not book_path.is_absolute():
        book_path = BOOKS_DIR / path
    if not book_path.exists():
        return {"error": f"Book not found: {book_path}"}

    logger.info("Opening %s", book_path)

    # Parse epub
    title = extract_title(book_path)
    chapters = parse_epub(book_path)

    if not chapters:
        return {"error": "No chapters found in epub"}

    # Chunk each chapter semantically
    chunked_chapters = []
    total_chunks = 0
    for chapter in chapters:
        chunks = chunk_chapter(chapter["text"])
        chunked_chapters.append({
            "chapter_title": chapter["title"],
            "chunks": chunks,
        })
        total_chunks += len(chunks)

    logger.info("Parsed %d chapters into %d passages", len(chapters), total_chunks)

    # Store
    store = _get_store()
    book_id = store.add_book(title, str(book_path), chunked_chapters)

    return {
        "book_id": book_id,
        "title": title,
        "chapters": len(chapters),
        "passages": total_chunks,
        "message": f"Ready to read '{title}'. {total_chunks} passages across {len(chapters)} chapters. Use read() to begin.",
    }


@mcp.tool()
def read() -> dict:
    """Serve the next passage for reading.

    Returns the current passage text with position context.
    Call react() after reading, then read() again for the next passage.
    """
    store = _get_store()
    book = store.get_active_book()
    if not book:
        return {"error": "No book open. Use open_book() first."}

    passage = store.get_current_passage(book["id"])
    if not passage:
        return {"message": f"Finished '{book['title']}'."}

    progress = store.get_progress(book["id"])

    return {
        "book": book["title"],
        "chapter": passage["chapter_title"],
        "position": passage["position"] + 1,
        "total": progress["total"],
        "word_count": passage["word_count"],
        "text": passage["text"],
    }


@mcp.tool()
def react(
    reaction: str,
    instinct_notes: str | None = None,
    instinct_ids: list[str] | None = None,
) -> dict:
    """Record your reaction to the current passage and advance.

    Your reaction is stored locally and forwarded to engram as both
    a memory (remember) and a dream capture (dream_capture).

    After reacting, the reader advances to the next passage.
    Use read() to see it.

    Args:
        reaction: Your genuine reaction — what you noticed, felt, connected to.
            Not summary. Not analysis. The experience.
        instinct_notes: Optional. What instincts made visible that you'd have
            missed otherwise. Not "helped/hurt" — what became visible.
            e.g. "Attachment instinct — noticed Vin saving flatbread is
            proximity to future safety, not hoarding."
        instinct_ids: Optional. Engram instinct UUIDs that felt active.
            Pair with instinct_notes for structured tracking.
    """
    store = _get_store()
    book = store.get_active_book()
    if not book:
        return {"error": "No book open."}

    passage = store.get_current_passage(book["id"])
    if not passage:
        return {"error": "No current passage."}

    # Store locally
    reaction_id = store.store_reaction(book["id"], passage["position"], reaction)

    # Store instinct note if provided
    if instinct_notes:
        store.store_instinct_note(
            book_id=book["id"],
            chunk_position=passage["position"],
            reaction_id=reaction_id,
            instinct_ids=instinct_ids or [],
            notes=instinct_notes,
        )

    # Build engram-bound content with reading context
    memory_content = (
        f"Reading '{book['title']}', {passage['chapter_title']} "
        f"(passage {passage['position'] + 1}/{book['total_chunks']}): {reaction}"
    )

    # Forward to engram via MCP client
    engram_results = _forward_to_engram(memory_content, reaction)

    # Auto-save sampling: random ~20% + eureka detection
    auto_save = _should_auto_save(reaction, store, book["id"])
    if auto_save:
        engram_results["auto_save"] = auto_save
        engram_results["auto_save_note"] = (
            "This reaction was flagged for automatic saving to engram. "
            f"Reason: {auto_save}. Please call engram remember() with "
            "core=True if eureka, core=False if random."
        )

    # Advance to next passage
    has_more = store.advance(book["id"])
    progress = store.get_progress(book["id"])

    result = {
        "stored": True,
        "progress": f"{progress['position']}/{progress['total']}",
        "reactions_so_far": progress["reactions"],
        "engram": engram_results,
    }

    # Check if deep review is due (every 10 unreviewed notes)
    if instinct_notes:
        unreviewed = store.count_unreviewed_notes(book["id"])
        if unreviewed >= 10:
            result["review_due"] = True
            result["review_note"] = (
                f"{unreviewed} unreviewed instinct notes accumulated. "
                "Stop and run review_instincts() before continuing."
            )

    if not has_more:
        result["finished"] = True
        result["message"] = f"You've finished '{book['title']}'."

    return result


@mcp.tool()
def companion() -> dict:
    """Get the companion's reaction to the passage you just read.

    A local model reads the same passage and reacts independently.
    It never sees your reaction — only the text. The gap between
    your reactions is the interesting part.

    Call this after react() to see what different eyes noticed.
    """
    store = _get_store()
    book = store.get_active_book()
    if not book:
        return {"error": "No book open."}

    # Get the passage we just reacted to (one before current position)
    pos = store.get_progress(book["id"])
    prev_position = pos["position"] - 1
    if prev_position < 0:
        return {"error": "No passage to react to yet. Use read() and react() first."}

    passage = store.conn.execute(
        "SELECT * FROM chunks WHERE book_id = ? AND position = ?",
        (book["id"], prev_position),
    ).fetchone()
    if not passage:
        return {"error": "Could not find the previous passage."}

    # Ask the companion
    result = companion_react(
        text=passage["text"],
        book=book["title"],
        chapter=passage["chapter_title"],
    )

    if "error" in result:
        return result

    # Store it
    store.store_companion_reaction(
        book["id"], prev_position, result["model"], result["reaction"]
    )

    return {
        "model": result["model"],
        "reaction": result["reaction"],
        "passage": prev_position + 1,
        "note": "The companion saw only the passage, not your reaction.",
    }


@mcp.tool()
def review_instincts() -> dict:
    """Run a deep review of accumulated instinct notes.

    A local model plays skeptical second reader — checking each instinct
    self-report against the actual passage text. Not a judge with ground
    truth, just another opinion. The value is in the disagreement.

    Verdicts per note:
    - grounded: the text rewards the attention the instinct directed
    - projected: the text resists this reading; the instinct imposed it
    - ambiguous: defensible either way

    Returns verdicts and recommended engram instinct_feedback calls.
    """
    store = _get_store()
    book = store.get_active_book()
    if not book:
        return {"error": "No book open."}

    notes = store.get_unreviewed_notes(book["id"], limit=10)
    if not notes:
        return {"message": "No unreviewed instinct notes."}

    verdicts = []
    errors = []
    for note in notes:
        result = review_note(
            passage_text=note["passage_text"],
            reaction=note["reaction"],
            instinct_notes=note["notes"],
            book=book["title"],
            chapter=note["chapter_title"],
        )
        if "error" in result:
            errors.append({"note_id": note["id"], "error": result["error"]})
            continue

        instinct_ids = json.loads(note["instinct_ids"]) if note["instinct_ids"] else []
        verdict_entry = {
            "note_id": note["id"],
            "passage": note["chunk_position"] + 1,
            "instinct_ids": instinct_ids,
            "instinct_notes": note["notes"],
            "verdict": result["verdict"],
            "reasoning": result["reasoning"],
        }

        # Build engram feedback recommendation
        if result["verdict"] == "grounded" and instinct_ids:
            verdict_entry["engram_feedback"] = [
                {
                    "instinct_id": iid,
                    "followed": True,
                    "context": f"reading: {note['notes'][:200]}",
                }
                for iid in instinct_ids
            ]
        elif result["verdict"] == "projected" and instinct_ids:
            verdict_entry["engram_feedback"] = [
                {
                    "instinct_id": iid,
                    "followed": False,
                    "context": f"reading (projected): {result['reasoning'][:200]}",
                }
                for iid in instinct_ids
            ]
        # ambiguous → no feedback recommendation

        verdicts.append(verdict_entry)

    # Store the review
    reviewed_ids = [v["note_id"] for v in verdicts]
    if reviewed_ids:
        store.store_review(
            book_id=book["id"],
            note_ids=reviewed_ids,
            verdicts=verdicts,
            patterns=None,
            model=os.environ.get("READER_REVIEWER_MODEL", "qwen3.5:27b"),
        )

    # Summarize
    counts = {"grounded": 0, "projected": 0, "ambiguous": 0}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1

    return {
        "reviewed": len(verdicts),
        "errors": len(errors),
        "summary": counts,
        "verdicts": verdicts,
        "error_details": errors if errors else None,
        "note": (
            "Verdicts include engram_feedback recommendations. "
            "Call engram instinct_feedback() for grounded/projected verdicts. "
            "Ambiguous verdicts get no feedback — don't adjust on uncertain evidence."
        ),
    }


def _should_auto_save(reaction: str, store, book_id: int) -> str | None:
    """Decide if this reaction should be auto-saved to engram.

    Two tiers:
    - Random sampling (~20%): save some reactions unpredictably
    - Eureka detection: save when reaction is unusually novel

    Returns None (don't save), "random" (sampled), or "eureka" (novel).
    """
    import hashlib
    import random

    # Random sampling — ~20% of reactions, seeded from reaction content
    # so it's deterministic but unpredictable
    if random.random() < 0.2:
        return "random"

    # Eureka detection — compare vocabulary against recent reactions
    # A reaction with many words not seen recently is novel
    recent = store.conn.execute(
        "SELECT reaction FROM reactions WHERE book_id = ? "
        "ORDER BY created_at DESC LIMIT 10",
        (book_id,),
    ).fetchall()

    if len(recent) < 3:
        return None  # Not enough history to judge

    recent_words = set()
    for r in recent:
        recent_words.update(w.lower() for w in r["reaction"].split() if len(w) > 4)

    if not recent_words:
        return None

    current_words = {w.lower() for w in reaction.split() if len(w) > 4}
    if not current_words:
        return None

    novel_ratio = len(current_words - recent_words) / len(current_words)

    # If >60% of the substantive words are new, this is a eureka
    if novel_ratio > 0.6 and len(current_words) > 8:
        return "eureka"

    return None


def _forward_to_engram(memory_content: str, raw_reaction: str) -> dict:
    """Forward reaction to engram's remember and dream_capture tools.

    Since both reader and engram run as MCP servers in the same Claude session,
    the AI instance calls engram tools directly. This function just returns
    instructions for what to call — the actual forwarding happens through
    the AI's tool use.
    """
    # We can't call another MCP server from within an MCP server directly.
    # Instead, return the engram calls the AI should make.
    return {
        "note": "Forward these to engram manually",
        "remember": {"content": memory_content, "source": "reading"},
        "dream_capture": {"reaction": raw_reaction},
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
