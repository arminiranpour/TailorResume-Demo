"""Deterministic ATS keyword weighting derived from JobSignals."""

from __future__ import annotations

from app.ats.phrases import PROTECTED_PHRASES
from app.ats.stopwords import GENERIC_JOB_WORDS
from app.ats.types import JobSignals, JobWeights, TermWeight

_BASE_WEIGHT = 1
_TITLE_BONUS = 5
_REQUIRED_BONUS = 6
_PREFERRED_BONUS = 3
_DOMAIN_BONUS = 2
_TECHNICAL_BONUS = 4
_LOW_SIGNAL_PENALTY = -4
_MAX_REPETITION_BONUS = 4
_HIGH_PRIORITY_THRESHOLD = 10
_MEDIUM_PRIORITY_THRESHOLD = 5

_LOW_SIGNAL_TERMS = GENERIC_JOB_WORDS | {
    "detail",
    "detail oriented",
    "motivated",
    "motivation",
    "oriented",
    "player",
    "responsible",
    "team player",
}

_STRONG_TECHNICAL_PHRASES = PROTECTED_PHRASES | {
    "backend development",
    "cloud infrastructure",
}


def build_job_weights(job_signals: JobSignals) -> JobWeights:
    """Compute stable, auditable ATS term weights from deterministic job signals."""
    title_terms = set(job_signals.title_terms)
    required_terms = set(job_signals.required_terms)
    preferred_terms = set(job_signals.preferred_terms)
    repeated_terms = set(job_signals.repeated_terms)
    domain_terms = set(job_signals.domain_terms)

    weights_by_term: dict[str, TermWeight] = {}
    for term in job_signals.canonical_terms:
        count = int(job_signals.keyword_counts.get(term, 0))
        is_low_signal = _is_low_signal_term(term)

        components: dict[str, int] = {"base": _BASE_WEIGHT}
        reasons: list[str] = ["term_present"]
        source_signals: list[str] = ["canonical_terms"]

        if term in title_terms:
            components["title"] = _TITLE_BONUS
            reasons.append("title_term")
            source_signals.append("title_terms")

        if term in required_terms:
            components["must_have"] = _REQUIRED_BONUS
            reasons.append("must_have")
            source_signals.append("required_terms")

        if term in preferred_terms:
            components["preferred"] = _PREFERRED_BONUS
            reasons.append("preferred")
            source_signals.append("preferred_terms")

        repetition_bonus = _repetition_bonus(count)
        if term in repeated_terms and repetition_bonus > 0:
            components["repetition"] = repetition_bonus
            reasons.append("repeated")
            source_signals.extend(["repeated_terms", "keyword_counts"])

        if term in domain_terms:
            components["domain"] = _DOMAIN_BONUS
            reasons.append("domain_term")
            source_signals.append("domain_terms")

        technical_bonus = _technical_bonus(term)
        if technical_bonus > 0:
            components["technical"] = technical_bonus
            reasons.append("technical_term")
            source_signals.append("technical_phrase_rules")

        if is_low_signal:
            components["low_signal_penalty"] = _LOW_SIGNAL_PENALTY
            reasons.append("low_signal")
            source_signals.append("low_signal_rules")

        total_weight = sum(components.values())

        weights_by_term[term] = TermWeight(
            term=term,
            total_weight=total_weight,
            components=components,
            reasons=tuple(reasons),
            source_sections=tuple(job_signals.term_sources.get(term, ())),
            source_ids=tuple(job_signals.term_source_ids.get(term, ())),
            source_signals=tuple(_unique_preserve_order(source_signals)),
            count=count,
            is_low_signal=is_low_signal,
        )

    ordered_terms = tuple(
        sorted(
            weights_by_term,
            key=lambda term: (-weights_by_term[term].total_weight, term),
        )
    )

    return JobWeights(
        weights_by_term=weights_by_term,
        ordered_terms=ordered_terms,
        high_priority_terms=_filter_terms(
            ordered_terms,
            weights_by_term,
            minimum=_HIGH_PRIORITY_THRESHOLD,
        ),
        medium_priority_terms=_filter_terms(
            ordered_terms,
            weights_by_term,
            minimum=_MEDIUM_PRIORITY_THRESHOLD,
            maximum=_HIGH_PRIORITY_THRESHOLD - 1,
        ),
        low_priority_terms=_filter_terms(
            ordered_terms,
            weights_by_term,
            maximum=_MEDIUM_PRIORITY_THRESHOLD - 1,
        ),
        title_priority_terms=tuple(term for term in ordered_terms if term in title_terms),
        required_priority_terms=tuple(term for term in ordered_terms if term in required_terms),
        preferred_priority_terms=tuple(term for term in ordered_terms if term in preferred_terms),
    )


def _filter_terms(
    ordered_terms: tuple[str, ...],
    weights_by_term: dict[str, TermWeight],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[str, ...]:
    filtered: list[str] = []
    for term in ordered_terms:
        weight = weights_by_term[term].total_weight
        if minimum is not None and weight < minimum:
            continue
        if maximum is not None and weight > maximum:
            continue
        filtered.append(term)
    return tuple(filtered)


def _repetition_bonus(count: int) -> int:
    if count <= 1:
        return 0
    return min(count - 1, _MAX_REPETITION_BONUS)


def _technical_bonus(term: str) -> int:
    if term in _STRONG_TECHNICAL_PHRASES:
        return _TECHNICAL_BONUS
    if any(char in term for char in ".#/+"):
        return _TECHNICAL_BONUS
    return 0


def _is_low_signal_term(term: str) -> bool:
    if term in _LOW_SIGNAL_TERMS:
        return True
    return all(token in _LOW_SIGNAL_TERMS for token in term.split())


def _unique_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
