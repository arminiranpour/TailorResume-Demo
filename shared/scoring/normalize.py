import re
import unicodedata

STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "you",
    "we",
    "our",
    "your",
    "their",
    "will",
    "can",
    "able",
    "strong",
    "excellent",
    "good",
    "experience",
    "knowledge",
    "skills",
    "requirements",
    "responsibilities",
}

SYNONYM_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "reactjs": "react",
    "postgres": "postgresql",
}

MULTIWORD_SYNONYMS = {
    ("c", "sharp"): "c#",
    ("dot", "net"): ".net",
    ("asp", "net"): "asp.net",
}

SEPARATOR_PATTERN = re.compile(r"[\/_\-\+\(\)\[\]\{\}:;,\.]", re.UNICODE)
TOKEN_PATTERN = re.compile(r"[a-z0-9#.+]+", re.UNICODE)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    normalized = SEPARATOR_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_tokens(tokens: list[str]) -> list[str]:
    normalized_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        if next_token is not None:
            pair = (token, next_token)
            if pair in MULTIWORD_SYNONYMS:
                mapped = MULTIWORD_SYNONYMS[pair]
                if mapped not in STOPWORDS:
                    normalized_tokens.append(mapped)
                i += 2
                continue
        mapped = SYNONYM_MAP.get(token, token)
        if mapped not in STOPWORDS:
            normalized_tokens.append(mapped)
        i += 1
    return normalized_tokens


def tokenize(text: str) -> list[str]:
    if text is None:
        return []
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    raw_tokens = TOKEN_PATTERN.findall(normalized)
    filtered = [t for t in raw_tokens if any(ch.isalnum() for ch in t)]
    return normalize_tokens(filtered)


def generate_ngrams(tokens: list[str], max_n: int = 3) -> set[str]:
    phrases = set()
    if not tokens or max_n < 2:
        return phrases
    n_max = min(max_n, len(tokens))
    for n in range(2, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            phrases.add(" ".join(tokens[i : i + n]))
    return phrases


def extract_signals(text: str) -> dict:
    normalized = normalize_text(text)
    tokens = tokenize(text)
    phrases = generate_ngrams(tokens, 3)
    return {"normalized": normalized, "tokens": tokens, "phrases": phrases}
