"""Conservative stopword sets for ATS signal extraction."""

from __future__ import annotations

COMMON_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

GENERIC_JOB_WORDS = frozenset(
    {
        "ability",
        "abilities",
        "candidate",
        "candidates",
        "collaboration",
        "communicate",
        "communication",
        "cross",
        "experience",
        "experienced",
        "excellent",
        "knowledge",
        "preferred",
        "requirements",
        "responsibilities",
        "requirement",
        "role",
        "skills",
        "strong",
        "team",
        "teams",
        "work",
        "working",
        "year",
        "years",
    }
)

PHRASE_EDGE_STOPWORDS = COMMON_STOPWORDS | frozenset(
    {
        "using",
        "use",
        "build",
        "built",
        "building",
        "maintain",
        "maintaining",
        "lead",
        "leading",
        "support",
        "supporting",
    }
)
