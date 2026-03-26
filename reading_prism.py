"""Reading prism connector — bridges the prism engine to the reader.

Calls prism's classify + ask functions with passage + reaction context.
Stores questions in reader's store. Handles engagement flow to engram.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("reader.prism")

# Add prism scripts to path
_PRISM_SCRIPTS = Path.home() / "Projects" / "prism" / "scripts"
if str(_PRISM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PRISM_SCRIPTS))


def _classify(context, lenses, transition_context="", graph=None, last_frame=None, fired_history=None):
    """Classify via prism — returns (relevant, distant, rotating)."""
    try:
        from prism import classify_reading
        return classify_reading(
            context, lenses,
            transition_context=transition_context,
            graph=graph,
            last_frame=last_frame,
            fired_history=fired_history,
        )
    except Exception:
        logger.warning("Prism classify failed", exc_info=True)
        return [], None, None


def _ask_lens(lens, context):
    """Ask a lens via prism — returns question string or None."""
    try:
        from prism import ask_lens
        return ask_lens(lens, context)
    except Exception:
        logger.warning("Prism ask_lens failed", exc_info=True)
        return None


def _load_reading_lenses():
    """Load reading lens pack."""
    try:
        from prism import load_lenses
        all_lenses = load_lenses()
        reading_domains = {
            "early-childhood-development", "literary-theory",
            "philosophy", "social-sciences", "sciences", "craft",
        }
        return {
            name: lens for name, lens in all_lenses.items()
            if lens.get("domain") in reading_domains
        }
    except Exception:
        logger.warning("Failed to load reading lenses", exc_info=True)
        return {}


def run_reading_prism(
    store,
    book_id: int,
    chunk_position: int,
    passage: str,
    reaction: str,
    book: str = "",
    chapter: str = "",
    lenses: dict | None = None,
    graph=None,
    last_frame: str | None = None,
) -> list[dict]:
    """Run the reading prism on a passage + reaction.

    Classifies, asks up to 4 lenses, stores questions in reader's store.
    Returns list of {id, lens, question, selection_type} dicts.
    """
    if lenses is None:
        lenses = _load_reading_lenses()
    if not lenses:
        return []

    context = {
        "type": "reading",
        "passage": passage,
        "reaction": reaction,
        "book": book,
        "chapter": chapter,
    }

    fired_history = store.prism_fired_history(book_id)
    relevant, distant, rotating = _classify(
        context, lenses,
        graph=graph,
        last_frame=last_frame,
        fired_history=fired_history,
    )

    to_ask = []
    for name in relevant:
        to_ask.append((name, "relevant"))
    if distant:
        to_ask.append((distant, "distant"))
    if rotating:
        to_ask.append((rotating, "rotating"))

    results = []
    for name, selection_type in to_ask:
        if name not in lenses:
            continue
        question = _ask_lens(lenses[name], context)
        if not question:
            continue

        qid = store.store_prism_question(
            book_id=book_id,
            chunk_position=chunk_position,
            lens=name,
            question=question,
            selection_type=selection_type,
        )
        results.append({
            "id": qid,
            "lens": name,
            "question": question,
            "selection_type": selection_type,
            "recall_query": question,
        })

    # Mandatory authorial lens — always fires, not classifier-routed
    authorial = lenses.get("authorial")
    if authorial:
        question = _ask_lens(authorial, context)
        if question:
            qid = store.store_prism_question(
                book_id=book_id,
                chunk_position=chunk_position,
                lens="authorial",
                question=question,
                selection_type="mandatory",
            )
            results.append({
                "id": qid,
                "lens": "authorial",
                "question": question,
                "selection_type": "mandatory",
                "recall_query": question,
            })

    return results


def handle_prism_response(
    store,
    lens: str,
    response: str,
    book_id: int,
    chunk_position: int,
) -> dict | None:
    """Handle a reader's response to a prism question.

    Marks the question as engaged and returns engram-bound content.
    Returns dict with remember/dream_capture instructions, or None.
    """
    questions = store.get_prism_questions(book_id, chunk_position)
    matched = [q for q in questions if q["lens"] == lens and not q["engaged"]]
    if not matched:
        return None

    question = matched[0]
    store.mark_prism_engaged(question["id"], response=response)

    content = (
        f"Prism ({lens}): {question['question']}\n"
        f"Response: {response}"
    )

    return {
        "remember": {
            "content": content,
            "source": f"prism:{lens}",
        },
        "dream_capture": {
            "reaction": response,
        },
    }
