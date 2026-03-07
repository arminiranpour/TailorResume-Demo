from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Set

from shared.scoring.normalize import generate_ngrams, normalize_text, tokenize

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9#.+-]*")
_TRAILING_PERIODS = "."


def build_allowed_vocab(resume_json: Dict[str, Any]) -> Dict[str, Set[str]]:
    terms: Set[str] = set()
    proper_nouns: Set[str] = set()
    for text in _iter_resume_texts(resume_json):
        if not text:
            continue
        _add_terms(text, terms)
        _add_proper_nouns(text, proper_nouns)
    return {"terms": terms, "proper_nouns": proper_nouns}


def _iter_resume_texts(resume_json: Dict[str, Any]) -> Iterable[str]:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    if isinstance(summary.get("text"), str):
        yield summary.get("text")

    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if isinstance(line, dict) and isinstance(line.get("text"), str):
            yield line.get("text")

    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        for key in ("company", "title"):
            if isinstance(exp.get(key), str):
                yield exp.get(key)
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                yield bullet.get("text")

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        for key in ("name", "text"):
            if isinstance(proj.get(key), str):
                yield proj.get(key)
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in bullets:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                yield bullet.get("text")

    education = resume_json.get("education") if isinstance(resume_json.get("education"), list) else []
    for edu in education:
        if not isinstance(edu, dict):
            continue
        for key in ("school", "degree"):
            if isinstance(edu.get(key), str):
                yield edu.get(key)


def _add_terms(text: str, terms: Set[str]) -> None:
    tokens = [_strip_trailing_periods(token) for token in tokenize(text)]
    for token in tokens:
        if token:
            terms.add(token)
    for phrase in generate_ngrams(tokens, 3):
        terms.add(phrase)


def _add_proper_nouns(text: str, proper_nouns: Set[str]) -> None:
    for token in _TOKEN_PATTERN.findall(text):
        if any(ch.isupper() for ch in token):
            normalized = _normalize_proper_token(token)
            normalized = _strip_trailing_periods(normalized)
            if normalized:
                proper_nouns.add(normalized)


def _normalize_proper_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip()


def _strip_trailing_periods(token: str) -> str:
    if not token:
        return token
    return token.rstrip(_TRAILING_PERIODS)


def normalize_terms(terms: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        value = normalize_text(term)
        if value:
            normalized.append(value)
    return normalized
