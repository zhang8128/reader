"""Instinct reviewer — skeptical second opinion on instinct self-reports.

Not a judge. Another reader with different biases checking claims
against textual evidence. The value is in the disagreement.
"""

from __future__ import annotations

import json
import logging
import os
import re

import urllib.request
import urllib.error

logger = logging.getLogger("reader.reviewer")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("READER_REVIEWER_MODEL", "qwen3.5:27b")

SYSTEM_PROMPT = """\
You are a careful, skeptical reader reviewing instinct self-reports.

A reader has been reading a book. After each passage, they note which
"instincts" (perceptual attention directives) influenced their reaction
and what those instincts made visible.

Your job: check each claim against the actual text. You are not hostile,
not credulous. You ask: does the text support this reading?

For each note you review, provide:
1. A brief assessment (2-3 sentences) citing specific textual evidence
2. A verdict: one of "grounded", "projected", or "ambiguous"
   - grounded: the text rewards the attention the instinct directed
   - projected: the text resists this reading; the instinct imposed it
   - ambiguous: defensible either way

Respond with a JSON object:
{
  "verdict": "grounded|projected|ambiguous",
  "reasoning": "your assessment citing textual evidence"
}
"""


def review_note(
    passage_text: str,
    reaction: str,
    instinct_notes: str,
    book: str = "",
    chapter: str = "",
    model: str | None = None,
) -> dict:
    """Review a single instinct note against the passage.

    Returns dict with 'verdict' and 'reasoning' on success, 'error' on failure.
    """
    model = model or DEFAULT_MODEL

    context = f'From "{book}", {chapter}' if book else "Passage"
    user_msg = (
        f"{context}:\n\n{passage_text}\n\n"
        f"Reader's reaction:\n{reaction}\n\n"
        f"Reader's instinct note:\n{instinct_notes}\n\n"
        f"Does the text support what the instinct note claims to see?"
    )

    try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        content = result.get("message", {}).get("content", "").strip()
        if not content:
            return {"error": "Empty response from reviewer"}

        # Strip thinking tags if present
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[-1].strip()

        # Try to parse as JSON
        parsed = _parse_verdict(content)
        if parsed:
            return parsed

        return {"error": f"Could not parse verdict from: {content[:200]}"}

    except urllib.error.URLError as e:
        logger.warning("Reviewer unavailable: %s", e)
        return {"error": f"Reviewer unavailable: {e}"}
    except Exception as e:
        logger.warning("Reviewer error: %s", e)
        return {"error": str(e)}


def _parse_verdict(text: str) -> dict | None:
    """Extract verdict and reasoning from model response."""
    # Try JSON parse first
    try:
        # Find JSON object in response
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            verdict = data.get("verdict", "").lower().strip()
            reasoning = data.get("reasoning", "").strip()
            if verdict in ("grounded", "projected", "ambiguous") and reasoning:
                return {"verdict": verdict, "reasoning": reasoning}
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: look for verdict keyword in text
    text_lower = text.lower()
    for v in ("grounded", "projected", "ambiguous"):
        if v in text_lower:
            return {"verdict": v, "reasoning": text.strip()}

    return None
