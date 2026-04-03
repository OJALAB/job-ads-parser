from __future__ import annotations

import re

from esco_skill_batch.text_utils import normalize_text


LEADING_CONTEXT_PATTERNS = [
    re.compile(r"^(?:zna|znac|znajomosc)\s+", re.IGNORECASE),
    re.compile(r"^(?:wymagane|wymagana|wymagany)\s*:\s*", re.IGNORECASE),
    re.compile(r"^(?:must have)\s*:\s*", re.IGNORECASE),
    re.compile(r"^(?:potrzebne sa|potrzebna jest|potrzebny jest)\s+", re.IGNORECASE),
    re.compile(r"^(?:na daily przyda sie)\s+", re.IGNORECASE),
    re.compile(r"^(?:plus)\s+", re.IGNORECASE),
    re.compile(r"^(?:no i)\s+", re.IGNORECASE),
]

SURROUNDING_PUNCTUATION_RE = re.compile(r"^[\s:;,\-+/]+|[\s:;,\-+/]+$")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_extracted_skill_mention(text: str, language: str | None = None) -> str:
    original = text.strip()
    if not original:
        return original

    mention = SURROUNDING_PUNCTUATION_RE.sub("", original)
    mention = MULTISPACE_RE.sub(" ", mention).strip()

    for pattern in LEADING_CONTEXT_PATTERNS:
        mention = pattern.sub("", mention).strip()

    lowered = normalize_text(mention)
    language = (language or "").strip().lower()

    if lowered == "python":
        return "Python"
    if lowered == "sql":
        return "SQL"
    if lowered == "structured query language":
        return "Structured Query Language"

    if language == "pl":
        if lowered in {"python programming", "programming in python"}:
            return "programowanie w Pythonie"
        if lowered == "programowanie w sql":
            return "SQL"
        if lowered in {"komunikacyjne", "komunikacyjnych", "komunikacyjnymi"}:
            return "umiejetnosci komunikacyjne"
        if lowered == "rozwiazywaniem problemow":
            return "rozwiazywanie problemow"

    return mention
