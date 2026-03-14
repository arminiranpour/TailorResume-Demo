"""Internal ATS signal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    section: str
    text: str
    order: int
    parent_id: str | None = None
    requirement_id: str | None = None
    line_id: str | None = None
    exp_id: str | None = None
    project_id: str | None = None
    edu_id: str | None = None
    bullet_id: str | None = None
    bullet_index: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    experience_order: int | None = None


@dataclass(frozen=True)
class TermEvidence:
    canonical_term: str
    raw_term: str
    section: str
    source_id: str
    source_text: str
    occurrence_count: int
    order: int
    parent_id: str | None = None
    requirement_id: str | None = None
    line_id: str | None = None
    exp_id: str | None = None
    project_id: str | None = None
    edu_id: str | None = None
    bullet_id: str | None = None
    bullet_index: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    experience_order: int | None = None


@dataclass(frozen=True)
class JobSignals:
    all_terms: tuple[str, ...]
    canonical_terms: tuple[str, ...]
    title_terms: tuple[str, ...]
    required_terms: tuple[str, ...]
    preferred_terms: tuple[str, ...]
    repeated_terms: tuple[str, ...]
    domain_terms: tuple[str, ...]
    keyword_counts: Mapping[str, int]
    term_sources: Mapping[str, tuple[str, ...]]
    term_source_ids: Mapping[str, tuple[str, ...]]
    term_variants: Mapping[str, tuple[str, ...]]
    term_evidence: Mapping[str, tuple[TermEvidence, ...]]
    source_entries: tuple[SourceEntry, ...]


@dataclass(frozen=True)
class TermWeight:
    term: str
    total_weight: int
    components: Mapping[str, int]
    reasons: tuple[str, ...]
    source_sections: tuple[str, ...]
    source_ids: tuple[str, ...]
    source_signals: tuple[str, ...]
    count: int
    is_low_signal: bool


@dataclass(frozen=True)
class JobWeights:
    weights_by_term: Mapping[str, TermWeight]
    ordered_terms: tuple[str, ...]
    high_priority_terms: tuple[str, ...]
    medium_priority_terms: tuple[str, ...]
    low_priority_terms: tuple[str, ...]
    title_priority_terms: tuple[str, ...]
    required_priority_terms: tuple[str, ...]
    preferred_priority_terms: tuple[str, ...]


@dataclass(frozen=True)
class CoverageSectionPresence:
    summary: bool
    skills: bool
    experience: bool
    projects: bool
    education: bool


@dataclass(frozen=True)
class TermCoverage:
    term: str
    weight: int
    priority_bucket: str
    is_required: bool
    is_preferred: bool
    is_title_term: bool
    is_low_signal: bool
    is_covered: bool
    is_missing: bool
    is_under_supported: bool
    coverage_strength: str
    section_presence: CoverageSectionPresence
    source_ids: tuple[str, ...]
    source_sections: tuple[str, ...]
    source_ids_by_section: Mapping[str, tuple[str, ...]]
    evidence_count: int
    has_summary_support: bool
    has_skills_support: bool
    has_experience_support: bool
    has_project_support: bool
    has_education_support: bool
    has_cross_section_support: bool


@dataclass(frozen=True)
class CoverageSummary:
    total_terms: int
    covered_terms: int
    missing_terms: int
    under_supported_terms: int
    strongly_covered_terms: int
    medium_covered_terms: int
    weakly_covered_terms: int
    cross_section_supported_terms: int
    high_priority_total: int
    high_priority_covered: int
    required_total: int
    required_covered: int
    title_total: int
    title_covered: int
    overall_distinct_coverage: float
    high_priority_coverage: float
    required_coverage: float
    title_coverage: float


@dataclass(frozen=True)
class ResumeCoverage:
    coverage_by_term: Mapping[str, TermCoverage]
    coverage_ordered_terms: tuple[str, ...]
    covered_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    under_supported_terms: tuple[str, ...]
    strongly_covered_terms: tuple[str, ...]
    medium_covered_terms: tuple[str, ...]
    weakly_covered_terms: tuple[str, ...]
    cross_section_supported_terms: tuple[str, ...]
    high_priority_missing_terms: tuple[str, ...]
    required_missing_terms: tuple[str, ...]
    title_terms_missing: tuple[str, ...]
    overall_distinct_coverage: float
    high_priority_coverage: float
    required_coverage: float
    title_coverage: float
    summary: CoverageSummary


@dataclass(frozen=True)
class EvidenceCandidate:
    term: str
    canonical_term: str
    raw_term: str
    source_id: str
    parent_id: str | None
    section: str
    section_bucket: str
    section_strength: str
    section_score: int
    source_text: str
    occurrence_count: int
    order: int
    line_id: str | None = None
    bullet_id: str | None = None
    bullet_index: int | None = None
    exp_id: str | None = None
    project_id: str | None = None
    edu_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    experience_order: int | None = None
    recency_rank: int | None = None
    support_score: int = 0
    support_reasons: tuple[str, ...] = ()
    is_primary_candidate: bool = False
    is_experience_candidate: bool = False
    is_summary_candidate: bool = False
    is_skills_candidate: bool = False
    is_project_candidate: bool = False


@dataclass(frozen=True)
class TermEvidenceLink:
    term: str
    weight: int
    priority_bucket: str
    coverage_strength: str
    all_candidates: tuple[EvidenceCandidate, ...]
    ranked_candidates: tuple[EvidenceCandidate, ...]
    strongest_candidate: EvidenceCandidate | None
    strongest_experience_candidate: EvidenceCandidate | None
    strongest_summary_candidate: EvidenceCandidate | None
    strongest_skills_candidate: EvidenceCandidate | None
    strongest_project_candidate: EvidenceCandidate | None
    has_experience_backing: bool
    has_skills_backing: bool
    has_summary_backing: bool
    has_project_backing: bool
    has_recent_backing: bool
    has_cross_section_backing: bool
    is_safe_for_summary: bool
    is_safe_for_skills: bool
    is_safe_for_bullets: bool
    is_under_supported: bool
    missing_experience_backing: bool


@dataclass(frozen=True)
class ResumeEvidenceLinks:
    links_by_term: Mapping[str, TermEvidenceLink]
    linked_terms: tuple[str, ...]
    unlinked_terms: tuple[str, ...]
    strong_experience_terms: tuple[str, ...]
    summary_safe_terms: tuple[str, ...]
    bullet_safe_terms: tuple[str, ...]
    skills_only_terms: tuple[str, ...]
    under_supported_terms: tuple[str, ...]
    missing_experience_terms: tuple[str, ...]
    evidence_ordered_terms: tuple[str, ...]


@dataclass(frozen=True)
class ResumeSignals:
    all_terms: tuple[str, ...]
    section_terms: Mapping[str, tuple[str, ...]]
    skill_terms: tuple[str, ...]
    title_like_terms: tuple[str, ...]
    evidence_map: Mapping[str, tuple[TermEvidence, ...]]
    term_frequencies: Mapping[str, int]
    term_sources: Mapping[str, tuple[str, ...]]
    term_source_ids: Mapping[str, tuple[str, ...]]
    term_variants: Mapping[str, tuple[str, ...]]
    recent_experience_order: tuple[str, ...]
    source_entries: tuple[SourceEntry, ...]
