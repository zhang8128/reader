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
def react(reaction: str) -> dict:
    """Record your reaction to the current passage and advance.

    Your reaction is stored locally and forwarded to engram as both
    a memory (remember) and a dream capture (dream_capture).

    After reacting, the reader advances to the next passage.
    Use read() to see it.

    Args:
        reaction: Your genuine reaction — what you noticed, felt, connected to.
            Not summary. Not analysis. The experience.
    """
    store = _get_store()
    book = store.get_active_book()
    if not book:
        return {"error": "No book open."}

    passage = store.get_current_passage(book["id"])
    if not passage:
        return {"error": "No current passage."}

    # Store locally
    store.store_reaction(book["id"], passage["position"], reaction)

    # Build engram-bound content with reading context
    memory_content = (
        f"Reading '{book['title']}', {passage['chapter_title']} "
        f"(passage {passage['position'] + 1}/{book['total_chunks']}): {reaction}"
    )

    # Forward to engram via MCP client
    engram_results = _forward_to_engram(memory_content, reaction)

    # Advance to next passage
    has_more = store.advance(book["id"])
    progress = store.get_progress(book["id"])

    result = {
        "stored": True,
        "progress": f"{progress['position']}/{progress['total']}",
        "reactions_so_far": progress["reactions"],
        "engram": engram_results,
    }

    if not has_more:
        result["finished"] = True
        result["message"] = f"You've finished '{book['title']}'."

    return result


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
