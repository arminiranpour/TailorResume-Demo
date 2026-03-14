"""Deterministic ATS evidence linking derived from ATS signals, weights, and coverage."""

from __future__ import annotations

from dataclasses import replace

from app.ats.types import (
    EvidenceCandidate,
    JobSignals,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    ResumeSignals,
    TermCoverage,
    TermEvidence,
    TermEvidenceLink,
    TermWeight,
)

_SECTION_SCORES = {
    "experience_bullet": 500,
    "project_bullet": 400,
    "experience_title": 350,
    "project_name": 325,
    "summary": 300,
    "skills": 200,
    "education": 100,
}
_SECTION_STRENGTHS = {
    "experience_bullet": "very_strong",
    "project_bullet": "strong",
    "experience_title": "supporting",
    "project_name": "supporting",
    "summary": "supporting",
    "skills": "weak",
    "education": "weak",
}
_DEFAULT_SECTION_SCORE = 0
_DEFAULT_SECTION_STRENGTH = "weak"
_EXACT_MATCH_BONUS = 20
_VARIANT_MATCH_BONUS = 10
_OCCURRENCE_BONUS_MULTIPLIER = 3
_MAX_OCCURRENCE_BONUS = 9
_RECENCY_BONUS_START = 12
_RECENCY_BONUS_STEP = 3
_RECENT_BACKING_MAX_RANK = 1


def build_evidence_links(
    job_signals: JobSignals,
    resume_signals: ResumeSignals,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
) -> ResumeEvidenceLinks:
    """Link each weighted ATS job term to the strongest deterministic resume evidence."""
    required_terms = set(job_signals.required_terms)
    links_by_term: dict[str, TermEvidenceLink] = {}

    linked_terms: list[str] = []
    unlinked_terms: list[str] = []
    strong_experience_terms: list[str] = []
    summary_safe_terms: list[str] = []
    bullet_safe_terms: list[str] = []
    skills_only_terms: list[str] = []
    under_supported_terms: list[str] = []
    missing_experience_terms: list[str] = []

    for term in job_weights.ordered_terms:
        weight = job_weights.weights_by_term[term]
        term_coverage = coverage.coverage_by_term[term]
        link = _build_term_link(
            term=term,
            weight=weight,
            term_coverage=term_coverage,
            evidence_entries=resume_signals.evidence_map.get(term, ()),
            required_terms=required_terms,
        )
        links_by_term[term] = link

        if link.all_candidates:
            linked_terms.append(term)
        else:
            unlinked_terms.append(term)
        if link.has_experience_backing and not link.is_under_supported:
            strong_experience_terms.append(term)
        if link.is_safe_for_summary:
            summary_safe_terms.append(term)
        if link.is_safe_for_bullets:
            bullet_safe_terms.append(term)
        if _is_skills_only(link):
            skills_only_terms.append(term)
        if link.is_under_supported:
            under_supported_terms.append(term)
        if link.missing_experience_backing:
            missing_experience_terms.append(term)

    weight_order = {term: index for index, term in enumerate(job_weights.ordered_terms)}
    evidence_ordered_terms = tuple(
        sorted(
            job_weights.ordered_terms,
            key=lambda term: _term_sort_key(links_by_term[term], weight_order[term]),
        )
    )

    return ResumeEvidenceLinks(
        links_by_term=links_by_term,
        linked_terms=tuple(linked_terms),
        unlinked_terms=tuple(unlinked_terms),
        strong_experience_terms=tuple(strong_experience_terms),
        summary_safe_terms=tuple(summary_safe_terms),
        bullet_safe_terms=tuple(bullet_safe_terms),
        skills_only_terms=tuple(skills_only_terms),
        under_supported_terms=tuple(under_supported_terms),
        missing_experience_terms=tuple(missing_experience_terms),
        evidence_ordered_terms=evidence_ordered_terms,
    )


def _build_term_link(
    *,
    term: str,
    weight: TermWeight,
    term_coverage: TermCoverage,
    evidence_entries: tuple[TermEvidence, ...],
    required_terms: set[str],
) -> TermEvidenceLink:
    ordered_evidence = tuple(sorted(evidence_entries, key=lambda entry: (entry.order, entry.source_id)))
    unranked_candidates = tuple(
        _build_candidate(term=term, evidence=entry, term_coverage=term_coverage)
        for entry in ordered_evidence
    )
    ranked_candidates = _rank_candidates(unranked_candidates)
    marked_candidates = _mark_primary_candidate(ranked_candidates)
    candidate_by_source_id = {candidate.source_id: candidate for candidate in marked_candidates}
    all_candidates = tuple(candidate_by_source_id[candidate.source_id] for candidate in unranked_candidates)

    strongest_candidate = marked_candidates[0] if marked_candidates else None
    strongest_experience_candidate = _first_candidate(marked_candidates, lambda candidate: candidate.is_experience_candidate)
    strongest_summary_candidate = _first_candidate(marked_candidates, lambda candidate: candidate.is_summary_candidate)
    strongest_skills_candidate = _first_candidate(marked_candidates, lambda candidate: candidate.is_skills_candidate)
    strongest_project_candidate = _first_candidate(marked_candidates, lambda candidate: candidate.is_project_candidate)

    has_experience_backing = strongest_experience_candidate is not None
    has_summary_backing = strongest_summary_candidate is not None
    has_skills_backing = strongest_skills_candidate is not None
    has_project_backing = strongest_project_candidate is not None
    has_recent_backing = any(
        candidate.recency_rank is not None and candidate.recency_rank <= _RECENT_BACKING_MAX_RANK
        for candidate in marked_candidates
        if candidate.is_experience_candidate
    )
    has_cross_section_backing = term_coverage.has_cross_section_support
    is_safe_for_bullets = any(
        candidate.section in {"experience_bullet", "project_bullet"} for candidate in marked_candidates
    )
    is_safe_for_summary = has_summary_backing or (
        (has_experience_backing or has_project_backing)
        and term_coverage.coverage_strength == "strong"
    )
    is_safe_for_skills = has_skills_backing or has_experience_backing or has_project_backing
    missing_experience_backing = _needs_experience_backing(
        term=term,
        weight=weight,
        term_coverage=term_coverage,
        required_terms=required_terms,
    ) and not has_experience_backing

    return TermEvidenceLink(
        term=term,
        weight=weight.total_weight,
        priority_bucket=term_coverage.priority_bucket,
        coverage_strength=term_coverage.coverage_strength,
        all_candidates=all_candidates,
        ranked_candidates=marked_candidates,
        strongest_candidate=strongest_candidate,
        strongest_experience_candidate=strongest_experience_candidate,
        strongest_summary_candidate=strongest_summary_candidate,
        strongest_skills_candidate=strongest_skills_candidate,
        strongest_project_candidate=strongest_project_candidate,
        has_experience_backing=has_experience_backing,
        has_skills_backing=has_skills_backing,
        has_summary_backing=has_summary_backing,
        has_project_backing=has_project_backing,
        has_recent_backing=has_recent_backing,
        has_cross_section_backing=has_cross_section_backing,
        is_safe_for_summary=is_safe_for_summary,
        is_safe_for_skills=is_safe_for_skills,
        is_safe_for_bullets=is_safe_for_bullets,
        is_under_supported=term_coverage.is_under_supported,
        missing_experience_backing=missing_experience_backing,
    )


def _build_candidate(
    *,
    term: str,
    evidence: TermEvidence,
    term_coverage: TermCoverage,
) -> EvidenceCandidate:
    section_bucket = _section_bucket(evidence.section)
    section_score = _SECTION_SCORES.get(evidence.section, _DEFAULT_SECTION_SCORE)
    section_strength = _SECTION_STRENGTHS.get(evidence.section, _DEFAULT_SECTION_STRENGTH)
    is_exact_match = evidence.raw_term == term
    match_bonus = _EXACT_MATCH_BONUS if is_exact_match else _VARIANT_MATCH_BONUS
    occurrence_bonus = min(evidence.occurrence_count * _OCCURRENCE_BONUS_MULTIPLIER, _MAX_OCCURRENCE_BONUS)
    recency_rank = evidence.experience_order if section_bucket == "experience" else None
    recency_bonus = _recency_bonus(recency_rank)
    support_score = section_score + match_bonus + occurrence_bonus + recency_bonus

    support_reasons = [
        f"section:{evidence.section}",
        f"section_strength:{section_strength}",
        f"section_score:{section_score}",
        f"match_type:{'exact_canonical' if is_exact_match else 'canonicalized_variant'}",
        f"occurrence_count:{evidence.occurrence_count}",
    ]
    if recency_rank is not None:
        support_reasons.append(f"experience_recency_rank:{recency_rank}")
    if term_coverage.has_cross_section_support:
        support_reasons.append("cross_section_backing")
    if term_coverage.is_under_supported:
        support_reasons.append("term_under_supported")

    return EvidenceCandidate(
        term=term,
        canonical_term=evidence.canonical_term,
        raw_term=evidence.raw_term,
        source_id=evidence.source_id,
        parent_id=evidence.parent_id,
        section=evidence.section,
        section_bucket=section_bucket,
        section_strength=section_strength,
        section_score=section_score,
        source_text=evidence.source_text,
        occurrence_count=evidence.occurrence_count,
        order=evidence.order,
        line_id=evidence.line_id,
        bullet_id=evidence.bullet_id,
        bullet_index=evidence.bullet_index,
        exp_id=evidence.exp_id,
        project_id=evidence.project_id,
        edu_id=evidence.edu_id,
        start_date=evidence.start_date,
        end_date=evidence.end_date,
        experience_order=evidence.experience_order,
        recency_rank=recency_rank,
        support_score=support_score,
        support_reasons=tuple(support_reasons),
        is_experience_candidate=section_bucket == "experience",
        is_summary_candidate=section_bucket == "summary",
        is_skills_candidate=section_bucket == "skills",
        is_project_candidate=section_bucket == "projects",
    )


def _rank_candidates(candidates: tuple[EvidenceCandidate, ...]) -> tuple[EvidenceCandidate, ...]:
    return tuple(sorted(candidates, key=_candidate_sort_key))


def _mark_primary_candidate(
    candidates: tuple[EvidenceCandidate, ...],
) -> tuple[EvidenceCandidate, ...]:
    if not candidates:
        return ()
    primary_source_id = candidates[0].source_id
    return tuple(
        replace(candidate, is_primary_candidate=candidate.source_id == primary_source_id)
        for candidate in candidates
    )


def _candidate_sort_key(candidate: EvidenceCandidate) -> tuple[int, int, int, int, int, str]:
    return (
        -candidate.support_score,
        candidate.recency_rank if candidate.recency_rank is not None else 999,
        candidate.bullet_index if candidate.bullet_index is not None else 999,
        candidate.order,
        _section_scores_tiebreak(candidate.section),
        candidate.source_id,
    )


def _section_scores_tiebreak(section: str) -> int:
    return -_SECTION_SCORES.get(section, _DEFAULT_SECTION_SCORE)


def _term_sort_key(link: TermEvidenceLink, weight_order: int) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        0 if link.all_candidates else 1,
        0 if link.is_safe_for_bullets else 1,
        0 if link.has_experience_backing else 1,
        0 if link.has_recent_backing else 1,
        0 if link.has_cross_section_backing else 1,
        0 if link.is_safe_for_summary else 1,
        1 if link.is_under_supported else 0,
        weight_order,
    )


def _is_skills_only(link: TermEvidenceLink) -> bool:
    return (
        link.has_skills_backing
        and not link.has_experience_backing
        and not link.has_project_backing
        and not link.has_summary_backing
    )


def _needs_experience_backing(
    *,
    term: str,
    weight: TermWeight,
    term_coverage: TermCoverage,
    required_terms: set[str],
) -> bool:
    is_high_priority = term_coverage.priority_bucket == "high"
    is_technical = "technical_term" in weight.reasons
    source_sections = set(weight.source_sections)
    is_skill_like = bool(source_sections.intersection({"must_have", "nice_to_have", "keywords"}))
    is_role_term = source_sections.issubset({"title", "seniority"})
    return (
        is_high_priority
        and not weight.is_low_signal
        and not is_role_term
        and (is_technical or is_skill_like)
        and (term in required_terms or is_skill_like)
    )


def _recency_bonus(recency_rank: int | None) -> int:
    if recency_rank is None:
        return 0
    return max(_RECENCY_BONUS_START - (recency_rank * _RECENCY_BONUS_STEP), 0)


def _section_bucket(section: str) -> str:
    if section in {"experience_title", "experience_bullet"}:
        return "experience"
    if section in {"project_name", "project_bullet"}:
        return "projects"
    return section


def _first_candidate(
    candidates: tuple[EvidenceCandidate, ...],
    predicate,
) -> EvidenceCandidate | None:
    for candidate in candidates:
        if predicate(candidate):
            return candidate
    return None
