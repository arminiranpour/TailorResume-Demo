"""Deterministic ATS term coverage derived from job and resume signals."""

from __future__ import annotations

from app.ats.types import (
    CoverageSectionPresence,
    CoverageSummary,
    JobSignals,
    JobWeights,
    ResumeCoverage,
    ResumeSignals,
    TermCoverage,
    TermEvidence,
)

_SECTION_ORDER = ("summary", "skills", "experience", "projects", "education")
_STRONG_SECTIONS = frozenset({"experience", "projects"})
_STRONG_STRENGTH = "strong"
_MEDIUM_STRENGTH = "medium"
_WEAK_STRENGTH = "weak"
_MISSING_STRENGTH = "missing"


def build_coverage_model(
    job_signals: JobSignals,
    resume_signals: ResumeSignals,
    job_weights: JobWeights,
) -> ResumeCoverage:
    """Build a stable ATS coverage inventory for all weighted job terms."""
    high_priority_terms = set(job_weights.high_priority_terms)
    medium_priority_terms = set(job_weights.medium_priority_terms)
    required_terms = set(job_signals.required_terms)
    preferred_terms = set(job_signals.preferred_terms)
    title_terms = set(job_signals.title_terms)

    coverage_by_term: dict[str, TermCoverage] = {}
    covered_terms: list[str] = []
    missing_terms: list[str] = []
    under_supported_terms: list[str] = []
    strongly_covered_terms: list[str] = []
    medium_covered_terms: list[str] = []
    weakly_covered_terms: list[str] = []
    cross_section_supported_terms: list[str] = []
    high_priority_missing_terms: list[str] = []
    required_missing_terms: list[str] = []
    title_terms_missing: list[str] = []

    for term in job_weights.ordered_terms:
        weight = job_weights.weights_by_term[term]
        evidence = _ordered_evidence(resume_signals.evidence_map.get(term, ()))
        source_ids_by_section = {section: [] for section in _SECTION_ORDER}
        source_ids: list[str] = []
        evidence_count = 0

        for entry in evidence:
            bucket = _section_bucket(entry.section)
            if bucket not in source_ids_by_section:
                continue
            _append_unique(source_ids_by_section[bucket], entry.source_id)
            _append_unique(source_ids, entry.source_id)
            evidence_count += entry.occurrence_count

        section_presence = CoverageSectionPresence(
            summary=bool(source_ids_by_section["summary"]),
            skills=bool(source_ids_by_section["skills"]),
            experience=bool(source_ids_by_section["experience"]),
            projects=bool(source_ids_by_section["projects"]),
            education=bool(source_ids_by_section["education"]),
        )
        source_sections = tuple(
            section for section in _SECTION_ORDER if source_ids_by_section[section]
        )
        priority_bucket = _priority_bucket(term, high_priority_terms, medium_priority_terms)
        is_covered = bool(source_ids)
        is_missing = not is_covered
        has_cross_section_support = len(source_sections) >= 2
        coverage_strength = _coverage_strength(
            section_presence=section_presence,
            priority_bucket=priority_bucket,
            is_required=term in required_terms,
            is_title_term=term in title_terms,
            is_technical=_is_technical_term(weight),
        )
        is_under_supported = _is_under_supported(
            is_covered=is_covered,
            section_presence=section_presence,
            priority_bucket=priority_bucket,
            is_required=term in required_terms,
            is_title_term=term in title_terms,
            is_technical=_is_technical_term(weight),
        )

        term_coverage = TermCoverage(
            term=term,
            weight=weight.total_weight,
            priority_bucket=priority_bucket,
            is_required=term in required_terms,
            is_preferred=term in preferred_terms,
            is_title_term=term in title_terms,
            is_low_signal=weight.is_low_signal,
            is_covered=is_covered,
            is_missing=is_missing,
            is_under_supported=is_under_supported,
            coverage_strength=coverage_strength,
            section_presence=section_presence,
            source_ids=tuple(source_ids),
            source_sections=source_sections,
            source_ids_by_section={
                section: tuple(source_ids_by_section[section]) for section in _SECTION_ORDER
            },
            evidence_count=evidence_count,
            has_summary_support=section_presence.summary,
            has_skills_support=section_presence.skills,
            has_experience_support=section_presence.experience,
            has_project_support=section_presence.projects,
            has_education_support=section_presence.education,
            has_cross_section_support=has_cross_section_support,
        )
        coverage_by_term[term] = term_coverage

        if is_covered:
            covered_terms.append(term)
        else:
            missing_terms.append(term)
            if term in high_priority_terms:
                high_priority_missing_terms.append(term)
            if term in required_terms:
                required_missing_terms.append(term)
            if term in title_terms:
                title_terms_missing.append(term)

        if is_under_supported:
            under_supported_terms.append(term)
        if has_cross_section_support:
            cross_section_supported_terms.append(term)
        if coverage_strength == _STRONG_STRENGTH:
            strongly_covered_terms.append(term)
        elif coverage_strength == _MEDIUM_STRENGTH:
            medium_covered_terms.append(term)
        elif coverage_strength == _WEAK_STRENGTH:
            weakly_covered_terms.append(term)

    high_priority_covered = sum(1 for term in job_weights.high_priority_terms if term in coverage_by_term and coverage_by_term[term].is_covered)
    required_covered = sum(1 for term in job_weights.required_priority_terms if term in coverage_by_term and coverage_by_term[term].is_covered)
    title_covered = sum(1 for term in job_weights.title_priority_terms if term in coverage_by_term and coverage_by_term[term].is_covered)

    summary = CoverageSummary(
        total_terms=len(job_weights.ordered_terms),
        covered_terms=len(covered_terms),
        missing_terms=len(missing_terms),
        under_supported_terms=len(under_supported_terms),
        strongly_covered_terms=len(strongly_covered_terms),
        medium_covered_terms=len(medium_covered_terms),
        weakly_covered_terms=len(weakly_covered_terms),
        cross_section_supported_terms=len(cross_section_supported_terms),
        high_priority_total=len(job_weights.high_priority_terms),
        high_priority_covered=high_priority_covered,
        required_total=len(job_weights.required_priority_terms),
        required_covered=required_covered,
        title_total=len(job_weights.title_priority_terms),
        title_covered=title_covered,
        overall_distinct_coverage=_ratio(len(covered_terms), len(job_weights.ordered_terms)),
        high_priority_coverage=_ratio(high_priority_covered, len(job_weights.high_priority_terms)),
        required_coverage=_ratio(required_covered, len(job_weights.required_priority_terms)),
        title_coverage=_ratio(title_covered, len(job_weights.title_priority_terms)),
    )

    return ResumeCoverage(
        coverage_by_term=coverage_by_term,
        coverage_ordered_terms=job_weights.ordered_terms,
        covered_terms=tuple(covered_terms),
        missing_terms=tuple(missing_terms),
        under_supported_terms=tuple(under_supported_terms),
        strongly_covered_terms=tuple(strongly_covered_terms),
        medium_covered_terms=tuple(medium_covered_terms),
        weakly_covered_terms=tuple(weakly_covered_terms),
        cross_section_supported_terms=tuple(cross_section_supported_terms),
        high_priority_missing_terms=tuple(high_priority_missing_terms),
        required_missing_terms=tuple(required_missing_terms),
        title_terms_missing=tuple(title_terms_missing),
        overall_distinct_coverage=summary.overall_distinct_coverage,
        high_priority_coverage=summary.high_priority_coverage,
        required_coverage=summary.required_coverage,
        title_coverage=summary.title_coverage,
        summary=summary,
    )


def _ordered_evidence(evidence: tuple[TermEvidence, ...]) -> tuple[TermEvidence, ...]:
    return tuple(sorted(evidence, key=lambda entry: (entry.order, entry.source_id)))


def _priority_bucket(
    term: str,
    high_priority_terms: set[str],
    medium_priority_terms: set[str],
) -> str:
    if term in high_priority_terms:
        return "high"
    if term in medium_priority_terms:
        return "medium"
    return "low"


def _coverage_strength(
    *,
    section_presence: CoverageSectionPresence,
    priority_bucket: str,
    is_required: bool,
    is_title_term: bool,
    is_technical: bool,
) -> str:
    if not any(
        (
            section_presence.summary,
            section_presence.skills,
            section_presence.experience,
            section_presence.projects,
            section_presence.education,
        )
    ):
        return _MISSING_STRENGTH
    if section_presence.experience or section_presence.projects:
        return _STRONG_STRENGTH
    if section_presence.summary:
        return _MEDIUM_STRENGTH
    if section_presence.skills:
        if _needs_strong_support(
            priority_bucket=priority_bucket,
            is_required=is_required,
            is_title_term=is_title_term,
            is_technical=is_technical,
        ):
            return _WEAK_STRENGTH
        return _MEDIUM_STRENGTH
    return _WEAK_STRENGTH


def _is_under_supported(
    *,
    is_covered: bool,
    section_presence: CoverageSectionPresence,
    priority_bucket: str,
    is_required: bool,
    is_title_term: bool,
    is_technical: bool,
) -> bool:
    if not is_covered:
        return False
    if section_presence.experience or section_presence.projects:
        return False
    if section_presence.education and not (section_presence.summary or section_presence.skills):
        return True
    return _needs_strong_support(
        priority_bucket=priority_bucket,
        is_required=is_required,
        is_title_term=is_title_term,
        is_technical=is_technical,
    )


def _needs_strong_support(
    *,
    priority_bucket: str,
    is_required: bool,
    is_title_term: bool,
    is_technical: bool,
) -> bool:
    if is_required or is_technical:
        return True
    if priority_bucket == "high" and not is_title_term:
        return True
    return False


def _is_technical_term(weight) -> bool:
    return "technical_term" in weight.reasons


def _section_bucket(section: str) -> str:
    if section in {"experience_title", "experience_bullet"}:
        return "experience"
    if section in {"project_name", "project_bullet"}:
        return "projects"
    return section


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator
