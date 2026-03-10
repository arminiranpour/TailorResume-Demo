from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

from docx import Document
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

_BULLET_MARKERS = {
    "\u2022",  # bullet
    "\u2023",  # triangular bullet
    "\u25E6",  # white bullet
    "\u2043",  # hyphen bullet
    "\u2219",  # bullet operator
    "\u00B7",  # middle dot
    "\u2027",  # hyphenation point
    "\u25AA",  # black small square
    "\u25AB",  # white small square
    "\u25CF",  # black circle
    "\u25CB",  # white circle
    "\u25A0",  # black square
    "\u25A1",  # white square
    "\u25C6",  # black diamond
    "\u25C7",  # white diamond
    "\u25B6",  # black right-pointing triangle
    "\u25BA",  # black right-pointing pointer
    "\u25B8",  # black right-pointing small triangle
    "\u25B9",  # white right-pointing small triangle
}

_BULLET_PATTERN = re.compile("[" + re.escape("".join(_BULLET_MARKERS)) + "]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_DEFAULT_FUZZY_THRESHOLD = 0.82
_SCORE_EPSILON = 1e-6


class ParagraphMappingError(ValueError):
    """Raised when paragraph mapping is ambiguous or incomplete."""


@dataclass(frozen=True)
class ParagraphInfo:
    paragraph_index: int
    text: str
    normalized: str
    run_count: int
    paragraph: Paragraph


@dataclass(frozen=True)
class Target:
    field_type: str
    field_id: str
    text: str
    normalized: str


@dataclass(frozen=True)
class MatchDiagnostic:
    paragraph_index: int
    original_text: str
    normalized_text: str
    run_count: int
    match_type: str


def normalize_text(text: str) -> str:
    """Normalize text for deterministic matching.

    Rules:
    - lowercase
    - normalize bullet markers
    - strip punctuation/symbols
    - collapse whitespace
    - trim
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = _BULLET_PATTERN.sub(" ", normalized)

    cleaned_chars: List[str] = []
    for char in normalized:
        if char.isalnum() or char.isspace():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    normalized = "".join(cleaned_chars)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def extract_docx_paragraphs(docx_path: str) -> List[ParagraphInfo]:
    """Load a DOCX and extract paragraph metadata for matching."""
    doc = Document(docx_path)
    paragraphs: List[ParagraphInfo] = []
    for index, paragraph in enumerate(doc.paragraphs):
        text = "".join(run.text for run in paragraph.runs)
        normalized = normalize_text(text)
        paragraphs.append(
            ParagraphInfo(
                paragraph_index=index,
                text=text,
                normalized=normalized,
                run_count=len(paragraph.runs),
                paragraph=paragraph,
            )
        )
    return paragraphs


def build_resume_targets(resume_json: Dict[str, Any]) -> List[Target]:
    """Convert ResumeJSON into ordered paragraph mapping targets."""
    targets: List[Target] = []

    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    summary_text = summary.get("text")
    if isinstance(summary_text, str) and summary_text.strip():
        targets.append(
            Target(
                field_type="summary",
                field_id="summary",
                text=summary_text,
                normalized=normalize_text(summary_text),
            )
        )

    skills = resume_json.get("skills")
    if isinstance(skills, dict):
        lines = skills.get("lines")
    elif isinstance(skills, list):
        lines = skills
    else:
        lines = []
    if isinstance(lines, list):
        for idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            line_text = line.get("text")
            line_id = line.get("line_id")
            if isinstance(line_text, str) and isinstance(line_id, str) and line_text.strip():
                targets.append(
                    Target(
                        field_type="skills",
                        field_id=line_id,
                        text=line_text,
                        normalized=normalize_text(line_text),
                    )
                )
            elif isinstance(line_text, str) and line_text.strip() and not isinstance(line_id, str):
                fallback_id = f"skills_{idx}"
                targets.append(
                    Target(
                        field_type="skills",
                        field_id=fallback_id,
                        text=line_text,
                        normalized=normalize_text(line_text),
                    )
                )

    _extend_bullet_targets(targets, resume_json.get("experience"), "experience")
    _extend_bullet_targets(targets, resume_json.get("projects"), "projects")

    return targets


def _extend_bullet_targets(
    targets: List[Target], collection: Any, label: str
) -> None:
    if not isinstance(collection, list):
        return
    for exp_idx, item in enumerate(collection):
        if not isinstance(item, dict):
            continue
        bullets = item.get("bullets") if isinstance(item.get("bullets"), list) else []
        for bullet_idx, bullet in enumerate(bullets):
            if not isinstance(bullet, dict):
                continue
            bullet_text = bullet.get("text")
            bullet_id = bullet.get("bullet_id")
            if isinstance(bullet_text, str) and isinstance(bullet_id, str) and bullet_text.strip():
                targets.append(
                    Target(
                        field_type="bullet",
                        field_id=bullet_id,
                        text=bullet_text,
                        normalized=normalize_text(bullet_text),
                    )
                )
            elif isinstance(bullet_text, str) and bullet_text.strip() and not isinstance(bullet_id, str):
                fallback_id = f"{label}_{exp_idx}_b{bullet_idx}"
                targets.append(
                    Target(
                        field_type="bullet",
                        field_id=fallback_id,
                        text=bullet_text,
                        normalized=normalize_text(bullet_text),
                    )
                )


def similarity_score(a: str, b: str) -> float:
    """Compute a deterministic similarity score based on token overlap and edit distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    union = tokens_a | tokens_b
    overlap = len(tokens_a & tokens_b) / len(union) if union else 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    score = (overlap + ratio) / 2.0
    return score


def match_targets_to_paragraphs(
    targets: List[Target],
    paragraphs: List[ParagraphInfo],
    threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> Tuple[Dict[str, Any], Dict[str, MatchDiagnostic]]:
    """Match ResumeJSON targets to DOCX paragraphs with strict ordering."""
    mapping: Dict[str, Any] = {"summary": {}, "skills": {}, "bullets": {}}
    diagnostics: Dict[str, MatchDiagnostic] = {}
    used_paragraphs: set[int] = set()
    current_index = 0

    for target in targets:
        if not target.normalized:
            raise ParagraphMappingError(f"Empty normalized target for {target.field_id}")

        candidates = [
            paragraph
            for paragraph in paragraphs
            if paragraph.paragraph_index >= current_index
            and paragraph.paragraph_index not in used_paragraphs
        ]
        if not candidates:
            raise ParagraphMappingError(f"No remaining paragraphs for {target.field_id}")

        exact_matches = [
            paragraph for paragraph in candidates if paragraph.normalized == target.normalized
        ]
        match_type = "exact"
        if len(exact_matches) == 1:
            chosen = exact_matches[0]
        elif len(exact_matches) > 1:
            raise ParagraphMappingError(
                f"Ambiguous exact matches for {target.field_id} ({len(exact_matches)} candidates)"
            )
        else:
            match_type = "fuzzy"
            chosen, best_score, tie = _select_best_fuzzy(target, candidates)
            if best_score < threshold:
                raise ParagraphMappingError(
                    f"Fuzzy match below threshold for {target.field_id}: {best_score:.3f}"
                )
            if tie:
                raise ParagraphMappingError(
                    f"Ambiguous fuzzy matches for {target.field_id}: {best_score:.3f}"
                )

        if chosen.paragraph_index < current_index:
            raise ParagraphMappingError(
                f"Order violation for {target.field_id}: {chosen.paragraph_index} < {current_index}"
            )
        if chosen.paragraph_index in used_paragraphs:
            raise ParagraphMappingError(
                f"Paragraph {chosen.paragraph_index} already mapped"
            )

        used_paragraphs.add(chosen.paragraph_index)
        current_index = chosen.paragraph_index + 1

        _store_mapping(mapping, target, chosen.paragraph_index)
        diagnostics[target.field_id] = MatchDiagnostic(
            paragraph_index=chosen.paragraph_index,
            original_text=chosen.text,
            normalized_text=chosen.normalized,
            run_count=chosen.run_count,
            match_type=match_type,
        )

        logger.info(
            "Mapped %s %s -> paragraph %s",
            target.field_type,
            target.field_id,
            chosen.paragraph_index,
        )

    _validate_order(targets, mapping, diagnostics)
    return mapping, diagnostics


def _select_best_fuzzy(
    target: Target, candidates: Iterable[ParagraphInfo]
) -> Tuple[ParagraphInfo, float, bool]:
    best_score = -1.0
    best_matches: List[ParagraphInfo] = []
    for paragraph in candidates:
        score = similarity_score(target.normalized, paragraph.normalized)
        if score > best_score + _SCORE_EPSILON:
            best_score = score
            best_matches = [paragraph]
        elif abs(score - best_score) <= _SCORE_EPSILON:
            best_matches.append(paragraph)
    if not best_matches:
        raise ParagraphMappingError(f"No fuzzy candidates for {target.field_id}")
    best_matches.sort(key=lambda item: item.paragraph_index)
    return best_matches[0], best_score, len(best_matches) > 1


def _store_mapping(mapping: Dict[str, Any], target: Target, paragraph_index: int) -> None:
    if target.field_type == "summary":
        mapping["summary"] = {"paragraph_index": paragraph_index}
    elif target.field_type == "skills":
        mapping["skills"][target.field_id] = paragraph_index
    elif target.field_type == "bullet":
        mapping["bullets"][target.field_id] = paragraph_index
    else:
        raise ParagraphMappingError(f"Unknown field type: {target.field_type}")


def _validate_order(
    targets: List[Target],
    mapping: Dict[str, Any],
    diagnostics: Dict[str, MatchDiagnostic],
) -> None:
    ordered_indices: List[int] = []
    for target in targets:
        diagnostic = diagnostics.get(target.field_id)
        if diagnostic is None:
            raise ParagraphMappingError(f"Missing diagnostic for {target.field_id}")
        ordered_indices.append(diagnostic.paragraph_index)
    if ordered_indices != sorted(ordered_indices):
        raise ParagraphMappingError("Target order not preserved in mapping")
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ParagraphMappingError("Paragraph mapped more than once")


def _compute_template_signature(paragraphs: List[ParagraphInfo]) -> str:
    concatenated = "\n".join(paragraph.normalized for paragraph in paragraphs)
    digest = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()
    return digest


def build_docx_mapping(docx_path: str, resume_json: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level API to build a DOCX paragraph mapping for ResumeJSON fields."""
    paragraphs = extract_docx_paragraphs(docx_path)
    targets = build_resume_targets(resume_json)
    mapping, diagnostics = match_targets_to_paragraphs(targets, paragraphs)

    if len(diagnostics) != len(targets):
        raise ParagraphMappingError(
            f"Only mapped {len(diagnostics)} of {len(targets)} targets"
        )

    mapping["diagnostics"] = {
        field_id: {
            "paragraph_index": diag.paragraph_index,
            "original_text": diag.original_text,
            "normalized_text": diag.normalized_text,
            "run_count": diag.run_count,
            "match_type": diag.match_type,
        }
        for field_id, diag in diagnostics.items()
    }
    mapping["template_signature"] = _compute_template_signature(paragraphs)
    return mapping
