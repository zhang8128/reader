"""Semantic chunker — split chapter text at natural topic boundaries.

Uses nomic-embed-text-v1.5 embeddings on sliding windows to detect
where the text shifts. Splits happen at cosine similarity drops between
adjacent windows. Chunk sizes vary naturally — that's a feature.

Soft floor: ~50 words. No ceiling.

Embeds in batches to avoid OOM on long chapters.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("reader.chunker")

# Lazy-loaded singleton — FastEmbed is heavy
_embedder = None

BATCH_SIZE = 32  # Embed this many windows at a time


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
    return _embedder


def _embed_batched(texts: list[str]) -> np.ndarray:
    """Embed texts in batches with clustering prefix."""
    model = _get_embedder()
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        prefixed = [f"clustering: {t}" for t in batch]
        vecs = list(model.embed(prefixed))
        all_vecs.extend(vecs)
    arr = np.array(all_vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving natural breaks."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _merge_small(chunks: list[str], min_words: int) -> list[str]:
    """Merge chunks below the soft floor into their neighbors."""
    if not chunks:
        return chunks
    merged = []
    carry = ""
    for chunk in chunks:
        if carry:
            chunk = carry + "\n\n" + chunk
            carry = ""
        if len(chunk.split()) < min_words and chunk != chunks[-1]:
            carry = chunk
        else:
            merged.append(chunk)
    if carry:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + carry
        else:
            merged.append(carry)
    return merged


def chunk_chapter(text: str, *, window_size: int = 3,
                  threshold_percentile: float = 30.0,
                  min_words: int = 50) -> list[str]:
    """Split chapter text into semantic chunks.

    Algorithm:
    1. Split into paragraphs (natural units)
    2. Create sliding windows of `window_size` paragraphs
    3. Embed each window (in batches)
    4. Compute cosine similarity between adjacent windows
    5. Split at similarity drops below the percentile threshold

    Soft minimum: chunks below min_words get merged with the next chunk.
    No maximum: a long continuous scene stays together.
    """
    paragraphs = _split_into_paragraphs(text)

    # Short chapters stay whole
    if len(paragraphs) <= window_size + 1:
        return [text.strip()] if text.strip() else []

    # Build sliding windows
    windows = []
    for i in range(len(paragraphs) - window_size + 1):
        window_text = "\n\n".join(paragraphs[i:i + window_size])
        windows.append(window_text)

    logger.info("Chunking: %d paragraphs, %d windows", len(paragraphs), len(windows))

    # Embed in batches
    embeddings = _embed_batched(windows)

    # Cosine similarity between adjacent windows
    similarities = []
    for i in range(len(embeddings) - 1):
        similarities.append(float(np.dot(embeddings[i], embeddings[i + 1])))

    if not similarities:
        return [text.strip()]

    # Split at similarity drops below threshold
    threshold = np.percentile(similarities, threshold_percentile)
    split_indices = set()
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_after = i + window_size - 1
            if split_after < len(paragraphs):
                split_indices.add(split_after)

    # Build chunks from paragraphs
    chunks = []
    current = []
    for i, para in enumerate(paragraphs):
        current.append(para)
        if i in split_indices:
            chunks.append("\n\n".join(current))
            current = []
    if current:
        chunks.append("\n\n".join(current))

    # Merge small chunks forward
    return _merge_small(chunks, min_words)
