"""Deterministic text normalization and canonical ATS term extraction."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from app.ats.phrases import (
    CANONICAL_PHRASE_MAP,
    CANONICAL_TERM_MAP,
    PHRASE_HEADWORDS,
    PROTECTED_PHRASES,
)
from app.ats.stopwords import COMMON_STOPWORDS, PHRASE_EDGE_STOPWORDS

_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": " ",
        "\u00b7": " ",
        "\u2044": "/",
        "&": " and ",
        ",": " ",
        ";": " ",
        ":": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "!": " ",
        "?": " ",
        "|": " ",
        "\\": " ",
        "_": " ",
        "-": " ",
    }
)

_TOKEN_PATTERN = re.compile(r"\.[a-z0-9]+|[a-z0-9][a-z0-9.+#/]*[a-z0-9+#]|[a-z0-9]")


def normalize_text(text: str | None) -> str:
    """Lowercase and normalize punctuation without inventing semantics."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.translate(_PUNCTUATION_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_text(text: str | None, *, drop_stopwords: bool = False) -> list[str]:
    """Tokenize text while preserving ATS-relevant token forms like node.js and ci/cd."""
    tokens: list[str] = []
    for token in _raw_tokens(text):
        canonical = CANONICAL_TERM_MAP.get(token, token)
        if not canonical:
            continue
        if drop_stopwords and canonical in COMMON_STOPWORDS:
            continue
        tokens.append(canonical)
    return tokens


def normalize_phrase(text: str | None) -> str:
    """Normalize a phrase into a stable whitespace-delimited form."""
    tokens = tokenize_text(text, drop_stopwords=False)
    return " ".join(tokens).strip()


def canonicalize_term(term: str | None) -> str:
    """Map a raw token or phrase into a conservative canonical ATS form."""
    normalized = normalize_phrase(term)
    if not normalized:
        return ""
    if normalized in CANONICAL_PHRASE_MAP:
        return CANONICAL_PHRASE_MAP[normalized]
    if normalized in CANONICAL_TERM_MAP:
        return CANONICAL_TERM_MAP[normalized]
    return normalized


def canonicalize_terms(terms: Iterable[str]) -> list[str]:
    return [canonical for canonical in (canonicalize_term(term) for term in terms) if canonical]


def extract_ngrams(tokens: Sequence[str], min_n: int = 2, max_n: int = 3) -> list[str]:
    if not tokens:
        return []
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")
    ngrams: list[str] = []
    limit = min(max_n, len(tokens))
    for size in range(min_n, limit + 1):
        for index in range(0, len(tokens) - size + 1):
            ngrams.append(" ".join(tokens[index : index + size]))
    return ngrams


def extract_canonical_term_pairs(text: str | None) -> list[tuple[str, str]]:
    """Return canonical/raw term pairs, including conservative ATS phrases."""
    raw_tokens = _raw_tokens(text)
    if not raw_tokens:
        return []

    pairs: list[tuple[str, str]] = []
    for raw_token in raw_tokens:
        canonical = canonicalize_term(raw_token)
        if not canonical or canonical in COMMON_STOPWORDS:
            continue
        pairs.append((canonical, raw_token))

    for raw_phrase in extract_ngrams(raw_tokens, min_n=2, max_n=3):
        canonical_phrase = canonicalize_term(raw_phrase)
        if _should_keep_phrase(raw_phrase, canonical_phrase):
            pairs.append((canonical_phrase, raw_phrase))
    return pairs


def _raw_tokens(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(normalized):
        tokens.extend(_expand_raw_token(token))
    return tokens


def _should_keep_phrase(raw_phrase: str, canonical_phrase: str) -> bool:
    normalized_raw = normalize_phrase(raw_phrase)
    if normalized_raw in CANONICAL_PHRASE_MAP:
        return True
    if not canonical_phrase or " " not in canonical_phrase:
        return any(char in canonical_phrase for char in ".#+/")
    tokens = canonical_phrase.split()
    if tokens[0] in PHRASE_EDGE_STOPWORDS or tokens[-1] in PHRASE_EDGE_STOPWORDS:
        return False
    if canonical_phrase in PROTECTED_PHRASES:
        return True
    return tokens[-1] in PHRASE_HEADWORDS


def _expand_raw_token(token: str) -> list[str]:
    if "/" not in token or token == "ci/cd":
        return [token]
    parts = [part for part in token.split("/") if part]
    if len(parts) < 2:
        return [token]
    if any(not re.fullmatch(r"[a-z0-9.+#]+", part) for part in parts):
        return [token]
    return parts
