"""Deterministic ATS recency-aware prioritization built on existing ATS signals."""

from __future__ import annotations

from app.ats.types import (
    ATSRecencyPriorities,
    EvidenceCandidate,
    JobSignals,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    ResumeSignals,
    TermEvidenceLink,
    TermRecencyPriority,
    TitleAlignment,
)

_PRIORITY_BUCKET_BASE = {
    "high": 3000,
    "medium": 2000,
    "low": 1000,
}
_COVERAGE_STRENGTH_BONUS = {
    "strong": 80,
    "medium": 40,
    "weak": 15,
    "missing": 0,
}
_RECENT_EXPERIENCE_MAX_RANK = 1
_RECENT_PROJECT_MAX_RANK = 0
_RECENT_EXPERIENCE_BONUS = {
    0: 24,
    1: 16,
}
_RECENT_PROJECT_BONUS = 8
_CROSS_SECTION_BONUS = 6
_TITLE_ALIGNMENT_RECENT_BONUS = 4
_BULLET_SECTIONS = frozenset({"experience_bullet", "project_bullet"})


def build_recency_priorities(
    job_signals: JobSignals,
    resume_signals: ResumeSignals,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    title_alignment: TitleAlignment,
) -> ATSRecencyPriorities:
    """Build a stable ATS recency-prioritization inventory for all weighted job terms."""
    del job_signals  # Recency uses the normalized weighted term inventory downstream.

    experience_rank_by_id = {
        exp_id: index for index, exp_id in enumerate(resume_signals.recent_experience_order)
    }
    project_rank_by_id = _project_rank_by_id(resume_signals)
    title_aligned_terms = set(title_alignment.overlapping_tokens) | set(
        title_alignment.overlapping_phrases
    )
    weight_order = {term: index for index, term in enumerate(job_weights.ordered_terms)}

    priorities_by_term: dict[str, TermRecencyPriority] = {}
    for term in job_weights.ordered_terms:
        priorities_by_term[term] = _build_term_priority(
            term=term,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            title_alignment=title_alignment,
            title_aligned_terms=title_aligned_terms,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        )

    recency_ordered_terms = tuple(
        sorted(
            job_weights.ordered_terms,
            key=lambda term: _term_sort_key(
                priority=priorities_by_term[term],
                weight_order=weight_order[term],
            ),
        )
    )
    prioritized_terms = tuple(
        term
        for term in recency_ordered_terms
        if priorities_by_term[term].strongest_overall_candidate is not None
    )

    return ATSRecencyPriorities(
        priorities_by_term=priorities_by_term,
        prioritized_terms=prioritized_terms,
        recent_high_priority_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].is_recent_high_priority_term
        ),
        recent_bullet_safe_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].is_recent_and_bullet_safe
        ),
        recent_summary_safe_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].is_recent_and_summary_safe
        ),
        stale_high_priority_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].is_stale_high_priority_term
        ),
        recent_experience_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].has_recent_experience_backing
        ),
        stale_only_terms=tuple(
            term
            for term in recency_ordered_terms
            if priorities_by_term[term].has_only_stale_backing
        ),
        recency_ordered_terms=recency_ordered_terms,
    )


def _build_term_priority(
    *,
    term: str,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    title_alignment: TitleAlignment,
    title_aligned_terms: set[str],
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> TermRecencyPriority:
    weight = job_weights.weights_by_term[term]
    term_coverage = coverage.coverage_by_term[term]
    link = evidence_links.links_by_term[term]

    recent_candidates = tuple(
        candidate
        for candidate in link.ranked_candidates
        if _is_recent_candidate(
            candidate,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        )
    )
    recent_experience_candidates = tuple(
        candidate
        for candidate in recent_candidates
        if _is_recent_experience_candidate(candidate, experience_rank_by_id)
    )
    recent_project_candidates = tuple(
        candidate
        for candidate in recent_candidates
        if _is_recent_project_candidate(candidate, project_rank_by_id)
    )
    stale_candidates = tuple(
        candidate
        for candidate in link.ranked_candidates
        if _is_stale_structured_candidate(
            candidate,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        )
    )

    strongest_recent_candidate = _first_ranked_candidate(
        recent_candidates,
        experience_rank_by_id=experience_rank_by_id,
        project_rank_by_id=project_rank_by_id,
    )
    strongest_recent_experience_candidate = _first_ranked_candidate(
        recent_experience_candidates,
        experience_rank_by_id=experience_rank_by_id,
        project_rank_by_id=project_rank_by_id,
    )
    strongest_recent_project_candidate = _first_ranked_candidate(
        recent_project_candidates,
        experience_rank_by_id=experience_rank_by_id,
        project_rank_by_id=project_rank_by_id,
    )

    has_recent_backing = strongest_recent_candidate is not None
    has_recent_experience_backing = strongest_recent_experience_candidate is not None
    has_recent_project_backing = strongest_recent_project_candidate is not None
    has_only_stale_backing = bool(
        link.strongest_candidate is not None
        and not has_recent_backing
        and stale_candidates
    )

    recent_source_ids = tuple(
        candidate.source_id
        for candidate in link.all_candidates
        if _is_recent_candidate(
            candidate,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        )
    )
    stale_source_ids = tuple(
        candidate.source_id
        for candidate in link.all_candidates
        if _is_stale_structured_candidate(
            candidate,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        )
    )

    is_title_aligned_term = term in title_aligned_terms
    is_recent_and_bullet_safe = bool(
        strongest_recent_candidate is not None
        and strongest_recent_candidate.section in _BULLET_SECTIONS
        and link.is_safe_for_bullets
    )
    is_recent_and_summary_safe = bool(
        has_recent_backing
        and (
            link.is_safe_for_summary
            or (
                is_title_aligned_term
                and has_recent_experience_backing
                and title_alignment.is_safe_for_summary_alignment
            )
        )
    )

    recent_bonus = _recent_bonus(
        recent_candidates,
        experience_rank_by_id=experience_rank_by_id,
        project_rank_by_id=project_rank_by_id,
    )
    title_alignment_bonus = _title_alignment_bonus(
        is_title_aligned_term=is_title_aligned_term,
        has_recent_experience_backing=has_recent_experience_backing,
        title_alignment=title_alignment,
    )
    recency_priority_score = _priority_score(
        priority_bucket=term_coverage.priority_bucket,
        weight=weight.total_weight,
        coverage_strength=term_coverage.coverage_strength,
        strongest_candidate=link.strongest_candidate,
        has_cross_section_backing=link.has_cross_section_backing,
        recent_bonus=recent_bonus,
        title_alignment_bonus=title_alignment_bonus,
    )
    recency_boost_applied = recent_bonus > 0

    return TermRecencyPriority(
        term=term,
        weight=weight.total_weight,
        priority_bucket=term_coverage.priority_bucket,
        coverage_strength=term_coverage.coverage_strength,
        strongest_overall_candidate=link.strongest_candidate,
        strongest_recent_candidate=strongest_recent_candidate,
        strongest_recent_experience_candidate=strongest_recent_experience_candidate,
        strongest_recent_project_candidate=strongest_recent_project_candidate,
        has_recent_backing=has_recent_backing,
        has_recent_experience_backing=has_recent_experience_backing,
        has_recent_project_backing=has_recent_project_backing,
        has_only_stale_backing=has_only_stale_backing,
        recent_source_ids=recent_source_ids,
        stale_source_ids=stale_source_ids,
        recency_priority_score=recency_priority_score,
        recency_boost_applied=recency_boost_applied,
        recency_reasons=_build_recency_reasons(
            term=term,
            weight=weight.total_weight,
            term_priority_bucket=term_coverage.priority_bucket,
            coverage_strength=term_coverage.coverage_strength,
            strongest_overall_candidate=link.strongest_candidate,
            strongest_recent_candidate=strongest_recent_candidate,
            strongest_recent_experience_candidate=strongest_recent_experience_candidate,
            strongest_recent_project_candidate=strongest_recent_project_candidate,
            has_recent_backing=has_recent_backing,
            has_only_stale_backing=has_only_stale_backing,
            recent_bonus=recent_bonus,
            title_alignment_bonus=title_alignment_bonus,
            is_recent_and_bullet_safe=is_recent_and_bullet_safe,
            is_recent_and_summary_safe=is_recent_and_summary_safe,
            is_title_aligned_term=is_title_aligned_term,
            title_alignment=title_alignment,
            recent_source_ids=recent_source_ids,
            stale_source_ids=stale_source_ids,
            link=link,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        ),
        is_recent_and_bullet_safe=is_recent_and_bullet_safe,
        is_recent_and_summary_safe=is_recent_and_summary_safe,
        is_recent_high_priority_term=bool(
            term_coverage.priority_bucket == "high" and has_recent_backing
        ),
        is_stale_high_priority_term=bool(
            term_coverage.priority_bucket == "high" and has_only_stale_backing
        ),
    )


def _project_rank_by_id(resume_signals: ResumeSignals) -> dict[str, int]:
    ordered_project_ids: list[str] = []
    for entry in resume_signals.source_entries:
        if entry.project_id is None:
            continue
        if entry.section not in {"project_name", "project_bullet"}:
            continue
        if entry.project_id not in ordered_project_ids:
            ordered_project_ids.append(entry.project_id)
    return {project_id: index for index, project_id in enumerate(ordered_project_ids)}


def _is_recent_candidate(
    candidate: EvidenceCandidate,
    *,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> bool:
    return _is_recent_experience_candidate(
        candidate, experience_rank_by_id
    ) or _is_recent_project_candidate(candidate, project_rank_by_id)


def _is_recent_experience_candidate(
    candidate: EvidenceCandidate,
    experience_rank_by_id: dict[str, int],
) -> bool:
    if not candidate.is_experience_candidate:
        return False
    rank = _experience_rank(candidate, experience_rank_by_id)
    return rank is not None and rank <= _RECENT_EXPERIENCE_MAX_RANK


def _is_recent_project_candidate(
    candidate: EvidenceCandidate,
    project_rank_by_id: dict[str, int],
) -> bool:
    if not candidate.is_project_candidate or candidate.project_id is None:
        return False
    rank = project_rank_by_id.get(candidate.project_id)
    return rank is not None and rank <= _RECENT_PROJECT_MAX_RANK


def _is_stale_structured_candidate(
    candidate: EvidenceCandidate,
    *,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> bool:
    if candidate.is_experience_candidate:
        rank = _experience_rank(candidate, experience_rank_by_id)
        return rank is not None and rank > _RECENT_EXPERIENCE_MAX_RANK
    if candidate.is_project_candidate and candidate.project_id is not None:
        rank = project_rank_by_id.get(candidate.project_id)
        return rank is not None and rank > _RECENT_PROJECT_MAX_RANK
    return False


def _experience_rank(
    candidate: EvidenceCandidate,
    experience_rank_by_id: dict[str, int],
) -> int | None:
    if candidate.recency_rank is not None:
        return candidate.recency_rank
    if candidate.exp_id is None:
        return None
    return experience_rank_by_id.get(candidate.exp_id)


def _first_ranked_candidate(
    candidates: tuple[EvidenceCandidate, ...],
    *,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> EvidenceCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            experience_rank_by_id=experience_rank_by_id,
            project_rank_by_id=project_rank_by_id,
        ),
    )


def _candidate_sort_key(
    candidate: EvidenceCandidate,
    *,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> tuple[int, int, int, int, int, str]:
    return (
        -candidate.support_score,
        _experience_rank(candidate, experience_rank_by_id)
        if candidate.is_experience_candidate
        else 999,
        project_rank_by_id.get(candidate.project_id, 999)
        if candidate.is_project_candidate and candidate.project_id is not None
        else 999,
        candidate.bullet_index if candidate.bullet_index is not None else 999,
        candidate.order,
        candidate.source_id,
    )


def _recent_bonus(
    candidates: tuple[EvidenceCandidate, ...],
    *,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> int:
    bonus = 0
    for candidate in candidates:
        if _is_recent_experience_candidate(candidate, experience_rank_by_id):
            rank = _experience_rank(candidate, experience_rank_by_id)
            if rank is None:
                continue
            bonus = max(bonus, _RECENT_EXPERIENCE_BONUS.get(rank, 0))
            continue
        if _is_recent_project_candidate(candidate, project_rank_by_id):
            bonus = max(bonus, _RECENT_PROJECT_BONUS)
    return bonus


def _priority_score(
    *,
    priority_bucket: str,
    weight: int,
    coverage_strength: str,
    strongest_candidate: EvidenceCandidate | None,
    has_cross_section_backing: bool,
    recent_bonus: int,
    title_alignment_bonus: int,
) -> int:
    strongest_section_score = strongest_candidate.section_score if strongest_candidate else 0
    occurrence_bonus = (
        min(strongest_candidate.occurrence_count, 3) * 2 if strongest_candidate else 0
    )
    return (
        _PRIORITY_BUCKET_BASE[priority_bucket]
        + (weight * 100)
        + _COVERAGE_STRENGTH_BONUS[coverage_strength]
        + strongest_section_score
        + occurrence_bonus
        + (_CROSS_SECTION_BONUS if has_cross_section_backing else 0)
        + recent_bonus
        + title_alignment_bonus
    )


def _title_alignment_bonus(
    *,
    is_title_aligned_term: bool,
    has_recent_experience_backing: bool,
    title_alignment: TitleAlignment,
) -> int:
    if (
        is_title_aligned_term
        and has_recent_experience_backing
        and title_alignment.is_title_supported
    ):
        return _TITLE_ALIGNMENT_RECENT_BONUS
    return 0


def _build_recency_reasons(
    *,
    term: str,
    weight: int,
    term_priority_bucket: str,
    coverage_strength: str,
    strongest_overall_candidate: EvidenceCandidate | None,
    strongest_recent_candidate: EvidenceCandidate | None,
    strongest_recent_experience_candidate: EvidenceCandidate | None,
    strongest_recent_project_candidate: EvidenceCandidate | None,
    has_recent_backing: bool,
    has_only_stale_backing: bool,
    recent_bonus: int,
    title_alignment_bonus: int,
    is_recent_and_bullet_safe: bool,
    is_recent_and_summary_safe: bool,
    is_title_aligned_term: bool,
    title_alignment: TitleAlignment,
    recent_source_ids: tuple[str, ...],
    stale_source_ids: tuple[str, ...],
    link: TermEvidenceLink,
    experience_rank_by_id: dict[str, int],
    project_rank_by_id: dict[str, int],
) -> tuple[str, ...]:
    reasons = [
        f"term:{term}",
        f"priority_bucket:{term_priority_bucket}",
        f"weight:{weight}",
        f"coverage_strength:{coverage_strength}",
    ]

    if strongest_overall_candidate is None:
        reasons.append("no_resume_evidence")
        reasons.append("recency_bonus:0")
        return tuple(reasons)

    reasons.append(
        "strongest_overall:"
        f"{strongest_overall_candidate.section}:{strongest_overall_candidate.source_id}"
    )

    if strongest_recent_candidate is not None:
        reasons.append(
            "strongest_recent:"
            f"{strongest_recent_candidate.section}:{strongest_recent_candidate.source_id}"
        )
    else:
        reasons.append("strongest_recent:none")

    if strongest_recent_experience_candidate is not None:
        rank = _experience_rank(strongest_recent_experience_candidate, experience_rank_by_id)
        reasons.append(
            "recent_experience:"
            f"{strongest_recent_experience_candidate.source_id}:rank_{rank}"
        )
    if strongest_recent_project_candidate is not None:
        project_rank = project_rank_by_id.get(strongest_recent_project_candidate.project_id, 999)
        reasons.append(
            "recent_project:"
            f"{strongest_recent_project_candidate.source_id}:rank_{project_rank}"
        )

    if has_recent_backing:
        reasons.append("has_recent_backing")
    else:
        reasons.append("no_recent_backing")

    if has_only_stale_backing:
        reasons.append("stale_only_backing")
    elif stale_source_ids:
        reasons.append("mixed_recent_and_stale_backing")
    elif link.has_experience_backing or link.has_project_backing:
        reasons.append("structured_support_without_stale_sources")
    elif link.has_summary_backing or link.has_skills_backing:
        reasons.append("summary_or_skills_only_support")

    reasons.append(f"recent_source_count:{len(recent_source_ids)}")
    reasons.append(f"stale_source_count:{len(stale_source_ids)}")
    reasons.append(f"recency_bonus:{recent_bonus}")
    reasons.append(f"title_alignment_bonus:{title_alignment_bonus}")

    if is_recent_and_bullet_safe:
        reasons.append("recent_bullet_safe")
    if is_recent_and_summary_safe:
        reasons.append("recent_summary_safe")
    if (
        is_title_aligned_term
        and title_alignment.is_safe_for_summary_alignment
        and strongest_recent_experience_candidate is not None
    ):
        reasons.append("title_alignment_summary_safe")

    return tuple(reasons)


def _term_sort_key(
    *,
    priority: TermRecencyPriority,
    weight_order: int,
) -> tuple[int, int, int, int, int, int, int, int, str]:
    return (
        0 if priority.strongest_overall_candidate is not None else 1,
        _bucket_order(priority.priority_bucket),
        -priority.recency_priority_score,
        0 if priority.has_recent_experience_backing else 1,
        0 if priority.is_recent_and_bullet_safe else 1,
        0 if priority.has_recent_project_backing else 1,
        0 if priority.has_only_stale_backing else 1,
        weight_order,
        priority.term,
    )


def _bucket_order(priority_bucket: str) -> int:
    if priority_bucket == "high":
        return 0
    if priority_bucket == "medium":
        return 1
    return 2
