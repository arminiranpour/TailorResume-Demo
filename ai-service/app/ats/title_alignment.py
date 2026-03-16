"""Deterministic ATS title alignment derived from ATS signals and evidence."""

from __future__ import annotations

from dataclasses import dataclass

from app.ats.canonicalize import canonicalize_term, extract_ngrams, tokenize_text
from app.ats.stopwords import PHRASE_EDGE_STOPWORDS
from app.ats.types import (
    JobSignals,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    ResumeSignals,
    SourceEntry,
    TitleAlignment,
)

_TITLE_HEADWORDS = frozenset(
    {
        "administrator",
        "analyst",
        "architect",
        "consultant",
        "coordinator",
        "designer",
        "developer",
        "director",
        "engineer",
        "lead",
        "manager",
        "officer",
        "owner",
        "scientist",
        "specialist",
        "strategist",
        "supervisor",
    }
)
_TITLE_MODIFIER_TOKENS = frozenset({"head", "junior", "lead", "principal", "senior", "staff"})
_EXPERIENCE_EVIDENCE_BONUS = 3
_PROJECT_EVIDENCE_BONUS = 2
_SUMMARY_EVIDENCE_BONUS = 1
_TOKEN_OVERLAP_SCORE = 2
_PHRASE_OVERLAP_SCORE = 4
_MAX_TOKEN_SCORE = 8
_MAX_PHRASE_SCORE = 12
_MAX_ALIGNMENT_SCORE = 20


@dataclass(frozen=True)
class _ResumeTitleCandidate:
    phrase: str
    section: str
    source_id: str
    order: int
    exp_id: str | None
    experience_order: int | None
    tokens: tuple[str, ...]
    phrase_variants: tuple[str, ...]
    overlapping_tokens: tuple[str, ...]
    overlapping_phrases: tuple[str, ...]
    matched_weight: int


@dataclass(frozen=True)
class _SupportingEvidence:
    experience_ids: tuple[str, ...]
    bullet_ids: tuple[str, ...]
    has_experience_support: bool
    has_project_support: bool
    has_summary_support: bool
    has_any_support: bool
    experience_title_overlap_terms: tuple[str, ...]


def build_title_alignment(
    job_signals: JobSignals,
    resume_signals: ResumeSignals,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
) -> TitleAlignment:
    """Build a stable ATS title-alignment summary without inventing unsupported titles."""
    job_title_tokens = _extract_job_title_tokens(job_signals)
    job_title_phrases = _extract_job_title_phrases(job_signals, job_title_tokens)

    resume_candidates = _collect_resume_title_candidates(
        resume_signals=resume_signals,
        job_title_tokens=job_title_tokens,
        job_title_phrases=job_title_phrases,
        job_weights=job_weights,
    )
    resume_title_phrases = _collect_resume_title_phrases(
        resume_signals,
        resume_candidates,
        job_title_tokens,
    )
    resume_title_tokens = _collect_resume_title_tokens(
        resume_signals,
        resume_title_phrases,
        job_title_tokens,
    )

    resume_title_tokens_set = set(resume_title_tokens)
    resume_title_phrases_set = set(resume_title_phrases)
    overlapping_tokens = tuple(token for token in job_title_tokens if token in resume_title_tokens_set)
    overlapping_phrases = tuple(
        phrase for phrase in job_title_phrases if phrase in resume_title_phrases_set
    )

    matched_terms = _matched_title_terms(
        job_weights=job_weights,
        overlapping_tokens=overlapping_tokens,
        overlapping_phrases=overlapping_phrases,
    )
    supporting_evidence = _collect_supporting_evidence(matched_terms, evidence_links)

    strongest_candidate = _strongest_matching_candidate(resume_candidates)
    strongest_matching_resume_title = (
        strongest_candidate.phrase if strongest_candidate is not None else None
    )

    alignment_score = _compute_alignment_score(
        overlapping_tokens=overlapping_tokens,
        overlapping_phrases=overlapping_phrases,
        supporting_evidence=supporting_evidence,
    )
    alignment_strength = _alignment_strength(alignment_score)
    matched_coverage = tuple(
        coverage.coverage_by_term[term]
        for term in matched_terms
        if term in coverage.coverage_by_term
    )
    has_strong_experience_support = any(
        term_coverage.has_experience_support and term_coverage.coverage_strength == "strong"
        for term_coverage in matched_coverage
    )
    is_title_supported = bool(
        (overlapping_tokens or overlapping_phrases) and supporting_evidence.has_any_support
    )
    is_safe_for_summary_alignment = bool(
        is_title_supported
        and (supporting_evidence.has_summary_support or has_strong_experience_support)
    )
    is_safe_for_experience_alignment = _is_safe_for_experience_alignment(
        job_title_tokens=job_title_tokens,
        strongest_candidate=strongest_candidate,
        supporting_evidence=supporting_evidence,
    )
    missing_title_tokens = tuple(
        token for token in job_title_tokens if token not in resume_title_tokens_set
    )

    return TitleAlignment(
        job_title_tokens=job_title_tokens,
        job_title_phrases=job_title_phrases,
        resume_title_tokens=resume_title_tokens,
        resume_title_phrases=resume_title_phrases,
        overlapping_tokens=overlapping_tokens,
        overlapping_phrases=overlapping_phrases,
        strongest_matching_resume_title=strongest_matching_resume_title,
        supporting_experience_ids=supporting_evidence.experience_ids,
        supporting_bullet_ids=supporting_evidence.bullet_ids,
        title_alignment_score=alignment_score,
        alignment_strength=alignment_strength,
        is_title_supported=is_title_supported,
        is_safe_for_summary_alignment=is_safe_for_summary_alignment,
        is_safe_for_experience_alignment=is_safe_for_experience_alignment,
        missing_title_tokens=missing_title_tokens,
    )


def _extract_job_title_tokens(job_signals: JobSignals) -> tuple[str, ...]:
    ordered_tokens: list[str] = []
    for term in job_signals.title_terms:
        for token in term.split():
            _append_unique(ordered_tokens, token)
    return tuple(ordered_tokens)


def _extract_job_title_phrases(
    job_signals: JobSignals,
    job_title_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    phrases: list[str] = []
    headword = _headword(job_title_tokens)

    for term in job_signals.title_terms:
        if " " not in term:
            continue
        if _looks_like_title_phrase(term):
            _append_unique(phrases, term)

    if headword:
        for token in job_title_tokens:
            if token == headword or token in PHRASE_EDGE_STOPWORDS:
                continue
            _append_unique(phrases, canonicalize_term(f"{token} {headword}"))

    return tuple(
        phrase for phrase in phrases if phrase and _looks_like_title_phrase(phrase)
    )


def _collect_resume_title_candidates(
    *,
    resume_signals: ResumeSignals,
    job_title_tokens: tuple[str, ...],
    job_title_phrases: tuple[str, ...],
    job_weights: JobWeights,
) -> tuple[_ResumeTitleCandidate, ...]:
    candidates: list[_ResumeTitleCandidate] = []
    for entry in resume_signals.source_entries:
        if entry.section == "experience_title":
            phrase = canonicalize_term(entry.text)
            if _looks_like_title_candidate(phrase):
                candidates.append(
                    _build_resume_title_candidate(
                        phrase=phrase,
                        section=entry.section,
                        source_id=entry.source_id,
                        order=entry.order,
                        exp_id=entry.exp_id,
                        experience_order=entry.experience_order,
                        job_title_tokens=job_title_tokens,
                        job_title_phrases=job_title_phrases,
                        job_weights=job_weights,
                    )
                )
            continue

        if entry.section != "summary":
            continue

        for phrase in _extract_summary_title_phrases(entry):
            candidates.append(
                _build_resume_title_candidate(
                    phrase=phrase,
                    section=entry.section,
                    source_id=entry.source_id,
                    order=entry.order,
                    exp_id=entry.exp_id,
                    experience_order=entry.experience_order,
                    job_title_tokens=job_title_tokens,
                    job_title_phrases=job_title_phrases,
                    job_weights=job_weights,
                )
            )

    return tuple(candidates)


def _collect_resume_title_phrases(
    resume_signals: ResumeSignals,
    resume_candidates: tuple[_ResumeTitleCandidate, ...],
    job_title_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    phrases: list[str] = []

    for candidate in resume_candidates:
        for phrase in candidate.phrase_variants:
            if _looks_like_title_phrase(phrase):
                _append_unique(phrases, phrase)

    for term in resume_signals.title_like_terms:
        if _looks_like_title_phrase(term) and _keep_resume_signal_phrase(term, job_title_tokens):
            _append_unique(phrases, term)

    for section in ("experience", "summary"):
        for term in resume_signals.section_terms.get(section, ()):
            if _looks_like_title_phrase(term) and _keep_resume_signal_phrase(term, job_title_tokens):
                _append_unique(phrases, term)

    return tuple(phrases)


def _collect_resume_title_tokens(
    resume_signals: ResumeSignals,
    resume_title_phrases: tuple[str, ...],
    job_title_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    tokens: list[str] = []
    for phrase in resume_title_phrases:
        for token in phrase.split():
            _append_unique(tokens, token)

    job_title_tokens_set = set(job_title_tokens)
    for term in resume_signals.title_like_terms:
        if " " in term:
            continue
        if term in _TITLE_HEADWORDS or term in job_title_tokens_set:
            _append_unique(tokens, term)

    for section in ("experience", "summary"):
        for term in resume_signals.section_terms.get(section, ()):
            if " " in term:
                continue
            if term in _TITLE_HEADWORDS:
                _append_unique(tokens, term)

    return tuple(tokens)


def _matched_title_terms(
    *,
    job_weights: JobWeights,
    overlapping_tokens: tuple[str, ...],
    overlapping_phrases: tuple[str, ...],
) -> tuple[str, ...]:
    overlapping = set(overlapping_tokens) | set(overlapping_phrases)
    return tuple(term for term in job_weights.title_priority_terms if term in overlapping)


def _collect_supporting_evidence(
    matched_terms: tuple[str, ...],
    evidence_links: ResumeEvidenceLinks,
) -> _SupportingEvidence:
    experience_ids: list[str] = []
    bullet_ids: list[str] = []
    experience_title_overlap_terms: list[str] = []
    has_experience_support = False
    has_project_support = False
    has_summary_support = False

    for term in matched_terms:
        link = evidence_links.links_by_term.get(term)
        if link is None:
            continue
        has_experience_support = has_experience_support or link.has_experience_backing
        has_project_support = has_project_support or link.has_project_backing
        has_summary_support = has_summary_support or link.has_summary_backing

        for candidate in link.ranked_candidates:
            if candidate.exp_id and candidate.section_bucket == "experience":
                _append_unique(experience_ids, candidate.exp_id)
            if candidate.bullet_id and candidate.section in {"experience_bullet", "project_bullet"}:
                _append_unique(bullet_ids, candidate.bullet_id)
            if candidate.section == "experience_title":
                _append_unique(experience_title_overlap_terms, term)

    has_any_support = has_experience_support or has_project_support or has_summary_support

    return _SupportingEvidence(
        experience_ids=tuple(experience_ids),
        bullet_ids=tuple(bullet_ids),
        has_experience_support=has_experience_support,
        has_project_support=has_project_support,
        has_summary_support=has_summary_support,
        has_any_support=has_any_support,
        experience_title_overlap_terms=tuple(experience_title_overlap_terms),
    )


def _compute_alignment_score(
    *,
    overlapping_tokens: tuple[str, ...],
    overlapping_phrases: tuple[str, ...],
    supporting_evidence: _SupportingEvidence,
) -> int:
    token_score = min(len(overlapping_tokens) * _TOKEN_OVERLAP_SCORE, _MAX_TOKEN_SCORE)
    phrase_score = min(len(overlapping_phrases) * _PHRASE_OVERLAP_SCORE, _MAX_PHRASE_SCORE)
    evidence_score = 0
    if supporting_evidence.has_experience_support:
        evidence_score += _EXPERIENCE_EVIDENCE_BONUS
    if supporting_evidence.has_project_support:
        evidence_score += _PROJECT_EVIDENCE_BONUS
    if supporting_evidence.has_summary_support:
        evidence_score += _SUMMARY_EVIDENCE_BONUS
    return min(token_score + phrase_score + evidence_score, _MAX_ALIGNMENT_SCORE)


def _alignment_strength(score: int) -> str:
    if score >= 8:
        return "strong"
    if score >= 4:
        return "moderate"
    if score >= 2:
        return "weak"
    return "none"


def _is_safe_for_experience_alignment(
    *,
    job_title_tokens: tuple[str, ...],
    strongest_candidate: _ResumeTitleCandidate | None,
    supporting_evidence: _SupportingEvidence,
) -> bool:
    if strongest_candidate is None or strongest_candidate.section != "experience_title":
        return False

    overlapping_tokens = strongest_candidate.overlapping_tokens
    if not overlapping_tokens:
        return False

    if set(overlapping_tokens) != set(job_title_tokens):
        return False

    return bool(supporting_evidence.experience_title_overlap_terms)


def _strongest_matching_candidate(
    candidates: tuple[_ResumeTitleCandidate, ...],
) -> _ResumeTitleCandidate | None:
    ranked = tuple(
        candidate
        for candidate in sorted(candidates, key=_resume_title_candidate_sort_key)
        if candidate.overlapping_tokens or candidate.overlapping_phrases
    )
    if not ranked:
        return None
    return ranked[0]


def _resume_title_candidate_sort_key(candidate: _ResumeTitleCandidate):
    missing_tokens = len(candidate.tokens) - len(candidate.overlapping_tokens)
    return (
        -len(candidate.overlapping_phrases),
        -len(candidate.overlapping_tokens),
        -candidate.matched_weight,
        0 if candidate.section == "experience_title" else 1,
        missing_tokens,
        candidate.experience_order if candidate.experience_order is not None else 999,
        candidate.order,
        candidate.phrase,
    )


def _build_resume_title_candidate(
    *,
    phrase: str,
    section: str,
    source_id: str,
    order: int,
    exp_id: str | None,
    experience_order: int | None,
    job_title_tokens: tuple[str, ...],
    job_title_phrases: tuple[str, ...],
    job_weights: JobWeights,
) -> _ResumeTitleCandidate:
    phrase_variants = _title_phrase_variants(phrase, job_title_tokens)
    token_set = {token for variant in phrase_variants for token in variant.split()}
    ordered_tokens = tuple(token for token in job_title_tokens if token in token_set)
    extra_tokens = [token for token in phrase.split() if token not in ordered_tokens]
    tokens = tuple(list(ordered_tokens) + [token for token in extra_tokens if token not in ordered_tokens])
    phrase_variant_set = set(phrase_variants)
    overlapping_tokens = tuple(token for token in job_title_tokens if token in token_set)
    overlapping_phrases = tuple(
        candidate_phrase for candidate_phrase in job_title_phrases if candidate_phrase in phrase_variant_set
    )
    matched_weight = 0
    for term in overlapping_tokens + overlapping_phrases:
        weight = job_weights.weights_by_term.get(term)
        if weight is not None:
            matched_weight += weight.total_weight

    return _ResumeTitleCandidate(
        phrase=phrase,
        section=section,
        source_id=source_id,
        order=order,
        exp_id=exp_id,
        experience_order=experience_order,
        tokens=tokens,
        phrase_variants=phrase_variants,
        overlapping_tokens=overlapping_tokens,
        overlapping_phrases=overlapping_phrases,
        matched_weight=matched_weight,
    )


def _extract_summary_title_phrases(entry: SourceEntry) -> tuple[str, ...]:
    phrases: list[str] = []
    for raw_phrase in extract_ngrams(tokenize_text(entry.text), min_n=2, max_n=4):
        canonical_phrase = canonicalize_term(raw_phrase)
        if _looks_like_title_phrase(canonical_phrase):
            _append_unique(phrases, canonical_phrase)
    return tuple(phrases)


def _title_phrase_variants(
    phrase: str,
    job_title_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_phrase = canonicalize_term(phrase)
    if not canonical_phrase:
        return ()

    tokens = canonical_phrase.split()
    headword = _headword(tokens)
    job_modifier_tokens = {token for token in job_title_tokens if token != headword}
    variants: list[str] = []
    if _looks_like_title_candidate(canonical_phrase):
        _append_unique(variants, canonical_phrase)

    if len(tokens) >= 2:
        for raw_ngram in extract_ngrams(tokens, min_n=2, max_n=min(3, len(tokens))):
            ngram = canonicalize_term(raw_ngram)
            if _looks_like_title_phrase(ngram) and _keep_resume_subphrase(
                original_phrase=canonical_phrase,
                candidate_phrase=ngram,
                job_modifier_tokens=job_modifier_tokens,
            ):
                _append_unique(variants, ngram)

    if headword:
        for token in tokens:
            if (
                token == headword
                or token in PHRASE_EDGE_STOPWORDS
                or token not in _TITLE_MODIFIER_TOKENS
            ):
                continue
            compressed = canonicalize_term(f"{token} {headword}")
            if _looks_like_title_phrase(compressed):
                _append_unique(variants, compressed)

    return tuple(variants)


def _looks_like_title_candidate(term: str) -> bool:
    if not term:
        return False
    if " " in term:
        return _looks_like_title_phrase(term)
    return term in _TITLE_HEADWORDS


def _looks_like_title_phrase(term: str) -> bool:
    if not term or " " not in term:
        return False
    tokens = term.split()
    if not tokens:
        return False
    if tokens[0] in PHRASE_EDGE_STOPWORDS or tokens[-1] in PHRASE_EDGE_STOPWORDS:
        return False
    return tokens[-1] in _TITLE_HEADWORDS


def _keep_resume_subphrase(
    *,
    original_phrase: str,
    candidate_phrase: str,
    job_modifier_tokens: set[str],
) -> bool:
    if candidate_phrase == original_phrase:
        return True
    candidate_tokens = candidate_phrase.split()
    if not candidate_tokens:
        return False
    if candidate_tokens[0] in _TITLE_MODIFIER_TOKENS:
        return True
    return any(token in job_modifier_tokens for token in candidate_tokens)


def _keep_resume_signal_phrase(term: str, job_title_tokens: tuple[str, ...]) -> bool:
    tokens = term.split()
    if not tokens:
        return False
    headword = _headword(tokens)
    for token in tokens:
        if token in _TITLE_MODIFIER_TOKENS:
            return True
        if token != headword and token in job_title_tokens:
            return True
    return False


def _headword(tokens: tuple[str, ...] | list[str]) -> str | None:
    for token in reversed(tokens):
        if token in _TITLE_HEADWORDS:
            return token
    if not tokens:
        return None
    return tokens[-1]


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)
