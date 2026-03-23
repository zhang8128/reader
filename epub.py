"""Epub parser — extract ordered chapters as clean text from .epub files.

Handles both multi-file epubs (one HTML per chapter) and single-file epubs
(entire book in one HTML) by detecting chapter boundaries in the text.
"""

from __future__ import annotations

import re
import warnings
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# OPF namespaces
_OPF_NS = "http://www.idpf.org/2007/opf"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

# Patterns for detecting chapter boundaries in single-file epubs
_CHAPTER_PATTERNS = [
    # "PROLOGUE", "EPILOGUE" (standalone or with trailing text on next line)
    r"PROLOGUE",
    r"EPILOGUE",
    # "CHAPTER 1", "Chapter Twenty-Three"
    r"(?:CHAPTER|Chapter)\s+[\dIVXLCDMivxlcdm]+",
    # "PART ONE: THE SURVIVOR OF HATHSIN" etc.
    r"PART\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[\dIVXLCDM]+)\b[^.]*",
]
# Used to find chapter markers anywhere in a paragraph
_CHAPTER_SEARCH_RE = re.compile("|".join(f"(?:{p})" for p in _CHAPTER_PATTERNS))

# Bare numbers as standalone paragraphs (chapter numbers like "1", "26")
_BARE_NUMBER_RE = re.compile(r"^\d{1,3}$")


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    """Find the .opf file path from META-INF/container.xml."""
    container = zf.read("META-INF/container.xml")
    root = ET.fromstring(container)
    rootfile = root.find(f".//{{{_CONTAINER_NS}}}rootfile")
    if rootfile is None:
        raise ValueError("No rootfile found in container.xml")
    return rootfile.attrib["full-path"]


def _parse_spine_order(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    """Return content file paths in spine (reading) order."""
    opf_dir = str(Path(opf_path).parent)
    opf_content = zf.read(opf_path)
    root = ET.fromstring(opf_content)

    manifest = {}
    for item in root.findall(f".//{{{_OPF_NS}}}item"):
        item_id = item.attrib.get("id", "")
        href = item.attrib.get("href", "")
        media = item.attrib.get("media-type", "")
        if "html" in media or "xhtml" in media:
            full_path = str(Path(opf_dir) / href) if opf_dir != "." else href
            manifest[item_id] = full_path

    spine = root.find(f"{{{_OPF_NS}}}spine")
    if spine is None:
        return list(manifest.values())

    ordered = []
    for itemref in spine.findall(f"{{{_OPF_NS}}}itemref"):
        idref = itemref.attrib.get("idref", "")
        if idref in manifest:
            ordered.append(manifest[idref])

    return ordered


def _html_to_text(html_bytes: bytes) -> str:
    """Strip HTML to clean text, preserving paragraph breaks.

    Extracts leaf-level block elements only — avoids double-collecting
    when a blockquote contains <p> tags.
    """
    soup = BeautifulSoup(html_bytes, "lxml")

    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()

    block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
    paragraphs = []

    for tag in soup.find_all(list(block_tags)):
        # Skip if this tag contains other block-level elements (not a leaf)
        if tag.find(list(block_tags)):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        text = soup.get_text(separator="\n", strip=True)
        return text

    return "\n\n".join(paragraphs)


def _split_text_into_chapters(text: str) -> list[dict]:
    """Split a single large text into chapters using text pattern matching.

    For single-file epubs where the entire book is one HTML document.
    Handles chapter markers that appear within paragraphs (common when
    HTML has minimal structure).
    """
    paragraphs = text.split("\n\n")
    chapters = []
    current_title = None
    current_lines = []

    def _save_current():
        if current_lines:
            body = "\n\n".join(current_lines).strip()
            if len(body.split()) >= 50:
                chapters.append({
                    "title": current_title or f"Chapter {len(chapters) + 1}",
                    "text": body,
                })

    def _is_real_chapter_number(idx: int) -> bool:
        """A bare number is a real chapter marker only if followed by prose, not another number."""
        next_idx = idx + 1
        while next_idx < len(paragraphs):
            next_stripped = paragraphs[next_idx].strip()
            if not next_stripped:
                next_idx += 1
                continue
            # If the next non-empty paragraph is also a bare number, this is TOC noise
            if _BARE_NUMBER_RE.match(next_stripped):
                return False
            # If the next paragraph is substantial, this is a real chapter marker
            return len(next_stripped) > 30
        return False

    for i, para in enumerate(paragraphs):
        stripped = para.strip()

        # Bare number on its own line = chapter number (only if followed by prose)
        if _BARE_NUMBER_RE.match(stripped) and _is_real_chapter_number(i):
            _save_current()
            current_title = f"Chapter {stripped}"
            current_lines = []
            continue

        # Check for chapter/part markers within the paragraph
        match = _CHAPTER_SEARCH_RE.search(stripped)
        if match and len(stripped) < 200:
            # The whole paragraph is a heading (short enough)
            _save_current()
            current_title = match.group(0).strip()
            current_lines = []
            continue
        elif match:
            # Chapter marker embedded in a longer paragraph — split here
            before = stripped[:match.start()].strip()
            after = stripped[match.end():].strip()
            marker = match.group(0).strip()

            if before:
                current_lines.append(before)
            _save_current()
            current_title = marker
            current_lines = []
            if after:
                current_lines.append(after)
            continue

        current_lines.append(para)

    # Don't forget the last chapter
    _save_current()

    # Filter: skip front matter before the first real content
    # (Prologue, Chapter 1, or Part One)
    first_real = 0
    for i, ch in enumerate(chapters):
        title_upper = ch.get("title", "").upper()
        if any(marker in title_upper for marker in ["PROLOGUE", "CHAPTER", "PART ONE", "PART I"]):
            first_real = i
            break

    return chapters[first_real:]


def extract_title(epub_path: str | Path) -> str:
    """Extract the book title from epub metadata."""
    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_path = _find_opf_path(zf)
        opf_content = zf.read(opf_path)
        root = ET.fromstring(opf_content)
        title_el = root.find(f".//{{{_DC_NS}}}title")
        if title_el is not None and title_el.text:
            return title_el.text.strip()
    return Path(epub_path).stem


def parse_epub(epub_path: str | Path) -> list[dict]:
    """Parse an epub into ordered chapters.

    Returns a list of dicts: [{index, title, text}]
    Handles both multi-file and single-file epubs.
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"Epub not found: {epub_path}")

    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_path = _find_opf_path(zf)
        spine_paths = _parse_spine_order(zf, opf_path)

        raw_chapters = []
        for content_path in spine_paths:
            try:
                html_bytes = zf.read(content_path)
            except KeyError:
                continue

            text = _html_to_text(html_bytes)
            if len(text.split()) < 20:
                continue

            soup = BeautifulSoup(html_bytes, "lxml")
            heading = soup.find(["h1", "h2", "h3"])
            title = heading.get_text(strip=True) if heading else None

            raw_chapters.append({
                "title": title,
                "text": text,
            })

    # If we got a single massive chapter, split on text patterns
    if len(raw_chapters) == 1 and len(raw_chapters[0]["text"].split()) > 10000:
        chapters = _split_text_into_chapters(raw_chapters[0]["text"])
    elif len(raw_chapters) <= 2 and sum(len(c["text"].split()) for c in raw_chapters) > 50000:
        # Two files but still basically one blob
        full_text = "\n\n".join(c["text"] for c in raw_chapters)
        chapters = _split_text_into_chapters(full_text)
    else:
        chapters = raw_chapters

    # Add indices
    for i, ch in enumerate(chapters):
        ch["index"] = i
        if not ch.get("title"):
            ch["title"] = f"Chapter {i + 1}"

    return chapters
