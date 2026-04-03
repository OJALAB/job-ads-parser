from __future__ import annotations

import re
import unicodedata


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: str) -> str:
    text = strip_accents(value or "").lower().strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(strip_accents(value or ""))]


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
