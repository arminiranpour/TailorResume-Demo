"""Deterministic ATS frequency balancing and post-rewrite validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.ats.canonicalize import canonicalize_term
from app.ats.resume_signals import extract_resume_signals
from app.ats.types import (
    ATSFrequencyBalance,
    ATSRecencyPriorities,
    FrequencyRangeRule,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    TermFrequencyStatus,
    TitleAlignment,
)

_SECTION_ORDER = ("summary", "skills", "experience", "projects", "education")
_ROLLBACK_ORDER_DEFAULT = ("summary", "projects", "education", "skills", "experience")
_ROLLBACK_ORDER_TITLE = ("skills", "projects", "education", "experience", "summary")
_SCOPE_TO_SECTIONS = {
    "summary": ("summary",),
    "skills": ("skills",),
    "bullets": ("experience", "projects"),
}
_STATUS_SEVERITY = {
    "hard_capped": 0,
    "stuffed": 1,
    "over_target": 2,
    "under_target": 3,
    "within_target": 4,
}


def build_frequency_balance(
    *,
    source_resume_json: Mapping[str, Any] | None,
    tailored_resume_json: Mapping[str, Any],
    tailoring_plan: Mapping[str, Any] | None,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    recency: ATSRecencyPriorities | None = None,
    title_alignment: TitleAlignment | None = None,
) -> ATSFrequencyBalance:
    """Build deterministic per-term ATS frequency status for the tailored resume."""
    if not isinstance(tailored_resume_json, Mapping):
        raise ValueError("tailored_resume_json must be a mapping")
    if source_resume_json is not None and not isinstance(source_resume_json, Mapping):
        raise ValueError("source_resume_json must be a mapping or None")

    source_signals = extract_resume_signals(dict(source_resume_json)) if source_resume_json is not None else None
    tailored_signals = extract_resume_signals(dict(tailored_resume_json))

    plan = tailoring_plan if isinstance(tailoring_plan, Mapping) else {}
    blocked_sections_by_term = _blocked_sections_by_term(plan)
    safe_sections_by_term = _safe_sections_by_term(plan)
    priority_by_term = _plan_priority_by_term(plan)

    ordered_terms = _ordered_monitored_terms(
        job_weights=job_weights,
        plan=plan,
        source_signals=source_signals,
        tailored_signals=tailored_signals,
    )

    frequency_by_term: dict[str, TermFrequencyStatus] = {}
    rules_by_term: dict[str, FrequencyRangeRule] = {}
    overused_terms: list[str] = []
    underused_terms: list[str] = []
    within_range_terms: list[str] = []
    capped_terms: list[str] = []

    for term in ordered_terms:
        section_counts = _section_counts_for_term(tailored_signals, term)
        baseline_section_counts = (
            _section_counts_for_term(source_signals, term) if source_signals is not None else _empty_section_counts()
        )
        rule = _build_frequency_rule(
            term=term,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            recency=recency,
            title_alignment=title_alignment,
            blocked_sections=blocked_sections_by_term.get(term, frozenset()),
            safe_sections=safe_sections_by_term.get(term),
            baseline_section_counts=baseline_section_counts,
            priority_hint=priority_by_term.get(term),
        )
        status = _build_term_status(
            rule=rule,
            weight=job_weights.weights_by_term.get(term).total_weight if term in job_weights.weights_by_term else 0,
            section_counts=section_counts,
        )
        rules_by_term[term] = rule
        frequency_by_term[term] = status
        if status.is_overused:
            overused_terms.append(term)
        elif status.is_underused:
            underused_terms.append(term)
        else:
            within_range_terms.append(term)
        if status.is_capped or status.target_max_total == 0:
            capped_terms.append(term)

    section_distribution_summary = _section_distribution_summary(frequency_by_term)
    balance = ATSFrequencyBalance(
        frequency_by_term=frequency_by_term,
        rules_by_term=rules_by_term,
        overused_terms=tuple(overused_terms),
        underused_terms=tuple(underused_terms),
        within_range_terms=tuple(within_range_terms),
        capped_terms=tuple(capped_terms),
        frequency_ordered_terms=tuple(ordered_terms),
        section_distribution_summary=section_distribution_summary,
        validation_errors=(),
        balancing_actions=(),
    )
    validation_errors = validate_frequency_balance(balance)
    return ATSFrequencyBalance(
        frequency_by_term=balance.frequency_by_term,
        rules_by_term=balance.rules_by_term,
        overused_terms=balance.overused_terms,
        underused_terms=balance.underused_terms,
        within_range_terms=balance.within_range_terms,
        capped_terms=balance.capped_terms,
        frequency_ordered_terms=balance.frequency_ordered_terms,
        section_distribution_summary=balance.section_distribution_summary,
        validation_errors=tuple(validation_errors),
        balancing_actions=balance.balancing_actions,
    )


def validate_frequency_balance(balance: ATSFrequencyBalance) -> list[str]:
    """Return deterministic validation errors for frequency violations."""
    if not isinstance(balance, ATSFrequencyBalance):
        raise ValueError("balance must be an ATSFrequencyBalance")

    errors: list[str] = []
    for term in balance.frequency_ordered_terms:
        status = balance.frequency_by_term[term]
        if not status.is_overused:
            continue
        if status.target_max_total == 0 and status.total_count > 0:
            errors.append(f"term '{term}' exceeds hard cap 0 with count {status.total_count}")
        elif status.total_count > status.target_max_total:
            errors.append(
                f"term '{term}' exceeds total cap {status.target_max_total} with count {status.total_count}"
            )
        for section in _SECTION_ORDER:
            count = int(status.section_counts.get(section, 0))
            cap = int(status.target_section_caps.get(section, 0))
            if count > cap:
                errors.append(f"term '{term}' exceeds {section} cap {cap} with count {count}")
    return errors


def _ordered_monitored_terms(
    *,
    job_weights: JobWeights,
    plan: Mapping[str, Any],
    source_signals,
    tailored_signals,
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    plan_terms: list[str] = []

    def add(term: str) -> None:
        canonical = canonicalize_term(term)
        if not canonical or canonical in seen:
            return
        seen.add(canonical)
        ordered.append(canonical)

    for values in (
        plan.get("supported_priority_terms"),
        plan.get("recent_priority_terms"),
        plan.get("skill_priority_terms"),
        plan.get("summary_alignment_terms"),
        _extract_item_terms(plan.get("under_supported_terms")),
        _extract_item_terms(plan.get("blocked_terms")),
        plan.get("prioritized_keywords"),
    ):
        for term in _extract_terms(values):
            if term not in plan_terms:
                plan_terms.append(term)
    title_status = plan.get("title_alignment_status")
    if isinstance(title_status, Mapping):
        for values in (title_status.get("supported_terms"), title_status.get("missing_tokens")):
            for term in _extract_terms(values):
                if term not in plan_terms:
                    plan_terms.append(term)

    for term in job_weights.high_priority_terms:
        add(term)
    for term in job_weights.required_priority_terms:
        add(term)
    for term in job_weights.title_priority_terms:
        add(term)
    for term in plan_terms:
        add(term)
    for term in job_weights.medium_priority_terms:
        if " " in term or any(char in term for char in ".#/+"):
            add(term)

    return tuple(
        term
        for term in ordered
        if not _is_subsumed_single_token_term(
            term=term,
            monitored_terms=ordered,
            explicit_terms=set(plan_terms),
            job_weights=job_weights,
        )
    )


def _is_subsumed_single_token_term(
    *,
    term: str,
    monitored_terms: Sequence[str],
    explicit_terms: set[str],
    job_weights: JobWeights,
) -> bool:
    if term in explicit_terms or " " in term or any(char in term for char in ".#/+"):
        return False
    weight = job_weights.weights_by_term.get(term)
    term_weight = weight.total_weight if weight is not None else 0
    for other in monitored_terms:
        if other == term or " " not in other:
            continue
        if term not in other.split():
            continue
        other_weight = job_weights.weights_by_term.get(other)
        if other_weight is not None and other_weight.total_weight >= term_weight:
            return True
    return False


def _blocked_sections_by_term(plan: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    blocked: dict[str, set[str]] = {}
    for item in _iter_plan_items(plan.get("blocked_terms")):
        term = item["term"]
        scopes = item.get("blocked_for")
        if not isinstance(scopes, Sequence):
            continue
        target = blocked.setdefault(term, set())
        for scope in scopes:
            if not isinstance(scope, str):
                continue
            target.update(_SCOPE_TO_SECTIONS.get(scope, ()))
    summary_plan = plan.get("summary_rewrite")
    if isinstance(summary_plan, Mapping):
        for term in _extract_terms(summary_plan.get("blocked_terms")):
            blocked.setdefault(term, set()).update(_SCOPE_TO_SECTIONS["summary"])
    return {term: frozenset(sorted(sections)) for term, sections in blocked.items()}


def _safe_sections_by_term(plan: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    safe: dict[str, set[str]] = {}
    for item in _iter_plan_items(plan.get("under_supported_terms")):
        term = item["term"]
        scopes = item.get("safe_for")
        if not isinstance(scopes, Sequence):
            continue
        target = safe.setdefault(term, set())
        for scope in scopes:
            if not isinstance(scope, str):
                continue
            target.update(_SCOPE_TO_SECTIONS.get(scope, ()))
    return {term: frozenset(sorted(sections)) for term, sections in safe.items()}


def _plan_priority_by_term(plan: Mapping[str, Any]) -> dict[str, str]:
    priority_by_term: dict[str, str] = {}
    for items in (plan.get("under_supported_terms"), plan.get("blocked_terms")):
        for item in _iter_plan_items(items):
            bucket = item.get("priority_bucket")
            if isinstance(bucket, str):
                priority_by_term[item["term"]] = bucket
    return priority_by_term


def _build_frequency_rule(
    *,
    term: str,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    recency: ATSRecencyPriorities | None,
    title_alignment: TitleAlignment | None,
    blocked_sections: frozenset[str],
    safe_sections: frozenset[str] | None,
    baseline_section_counts: Mapping[str, int],
    priority_hint: str | None,
) -> FrequencyRangeRule:
    weight_entry = job_weights.weights_by_term.get(term)
    coverage_entry = coverage.coverage_by_term.get(term)
    evidence_entry = evidence_links.links_by_term.get(term)
    baseline_total = sum(int(baseline_section_counts.get(section, 0)) for section in _SECTION_ORDER)

    priority_bucket = _priority_bucket(term, job_weights, coverage_entry, priority_hint)
    is_low_signal = bool(weight_entry and weight_entry.is_low_signal)
    is_title_term = bool(coverage_entry and coverage_entry.is_title_term)
    is_required = bool(coverage_entry and coverage_entry.is_required)
    is_under_supported = bool(coverage_entry and coverage_entry.is_under_supported)
    is_technical = bool(weight_entry and "technical_term" in weight_entry.reasons)
    has_cross_section_support = bool(coverage_entry and coverage_entry.has_cross_section_support)
    has_recent_backing = bool(
        recency and term in recency.priorities_by_term and recency.priorities_by_term[term].has_recent_backing
    )
    title_safe = bool(
        title_alignment
        and title_alignment.is_safe_for_summary_alignment
        and term
        in set(title_alignment.overlapping_tokens + title_alignment.overlapping_phrases)
    )

    max_summary = 0
    max_skills = 0
    max_experience = 0
    max_projects = 0
    max_education = 0
    min_total = 0
    max_total = 0
    reasons: list[str] = []

    if is_title_term:
        min_total = 1 if title_safe or baseline_total > 0 else 0
        max_total = 2
        max_summary = 1 if title_safe else 0
        max_skills = 0
        max_experience = 1
        max_projects = 0
        reasons.append("title_term_distribution")
    elif is_low_signal:
        min_total = 0
        max_total = 1
        max_summary = 0
        max_skills = 1
        max_experience = 1
        max_projects = 0
        reasons.append("low_signal_cap")
    elif is_under_supported:
        min_total = 0
        max_total = 1
        max_summary = 1 if safe_sections and "summary" in safe_sections else 0
        max_skills = 1 if safe_sections and "skills" in safe_sections else 0
        max_experience = 1 if safe_sections and "experience" in safe_sections else 0
        max_projects = 1 if safe_sections and "projects" in safe_sections else 0
        reasons.append("under_supported_surface_limit")
    elif is_required or priority_bucket == "high":
        min_total = 2 if has_cross_section_support or baseline_total >= 2 else 1
        max_total = 3
        max_summary = 1 if evidence_entry and evidence_entry.is_safe_for_summary else 0
        max_skills = 1 if evidence_entry and evidence_entry.is_safe_for_skills else 0
        max_experience = 2 if evidence_entry and evidence_entry.has_experience_backing else 0
        max_projects = 1 if evidence_entry and evidence_entry.has_project_backing else 0
        reasons.append("high_priority_cap")
    elif priority_bucket == "medium" or is_technical:
        min_total = 1 if baseline_total > 0 and not is_under_supported else 0
        max_total = 2
        max_summary = 1 if evidence_entry and evidence_entry.is_safe_for_summary else 0
        max_skills = 1 if evidence_entry and evidence_entry.is_safe_for_skills else 0
        max_experience = 1 if evidence_entry and evidence_entry.has_experience_backing else 0
        max_projects = 1 if evidence_entry and evidence_entry.has_project_backing else 0
        reasons.append("medium_priority_cap")
    else:
        min_total = 0
        max_total = 1
        max_summary = 0
        max_skills = 1 if evidence_entry and evidence_entry.is_safe_for_skills else 0
        max_experience = 1 if evidence_entry and evidence_entry.has_experience_backing else 0
        max_projects = 1 if evidence_entry and evidence_entry.has_project_backing else 0
        reasons.append("default_frequency_cap")

    if has_recent_backing:
        reasons.append("recent_evidence_backing")
    if is_technical:
        reasons.append("technical_term")
    if not evidence_entry or not evidence_entry.all_candidates:
        max_total = 0
        max_summary = 0
        max_skills = 0
        max_experience = 0
        max_projects = 0
        min_total = 0
        reasons.append("unsupported_term_hard_cap")

    max_education = 0

    if blocked_sections:
        reasons.append("section_blocked_cap")
        if "summary" in blocked_sections:
            max_summary = 0
        if "skills" in blocked_sections:
            max_skills = 0
        if "experience" in blocked_sections:
            max_experience = 0
        if "projects" in blocked_sections:
            max_projects = 0

    adjusted_caps = {
        "summary": max(max_summary, int(baseline_section_counts.get("summary", 0))),
        "skills": max(max_skills, int(baseline_section_counts.get("skills", 0))),
        "experience": max(max_experience, int(baseline_section_counts.get("experience", 0))),
        "projects": max(max_projects, int(baseline_section_counts.get("projects", 0))),
        "education": max(max_education, int(baseline_section_counts.get("education", 0))),
    }
    adjusted_max_total = max(max_total, baseline_total)
    adjusted_min_total = min(min_total, adjusted_max_total)
    preferred_sections = _preferred_sections(
        is_title_term=is_title_term,
        title_safe=title_safe,
        section_caps=adjusted_caps,
    )
    deprioritized_sections = _deprioritized_sections(
        is_title_term=is_title_term,
        preferred_sections=preferred_sections,
    )

    return FrequencyRangeRule(
        term=term,
        priority_bucket=priority_bucket,
        min_total=adjusted_min_total,
        max_total=adjusted_max_total,
        max_summary=adjusted_caps["summary"],
        max_skills=adjusted_caps["skills"],
        max_experience=adjusted_caps["experience"],
        max_projects=adjusted_caps["projects"],
        max_education=adjusted_caps["education"],
        baseline_total=baseline_total,
        baseline_section_counts={section: int(baseline_section_counts.get(section, 0)) for section in _SECTION_ORDER},
        preferred_sections=preferred_sections,
        deprioritized_sections=deprioritized_sections,
        reasons=tuple(reasons),
    )


def _build_term_status(
    *,
    rule: FrequencyRangeRule,
    weight: int,
    section_counts: Mapping[str, int],
) -> TermFrequencyStatus:
    target_section_caps = {
        "summary": rule.max_summary,
        "skills": rule.max_skills,
        "experience": rule.max_experience,
        "projects": rule.max_projects,
        "education": rule.max_education,
    }
    total_count = sum(int(section_counts.get(section, 0)) for section in _SECTION_ORDER)
    section_overages = {
        section: max(int(section_counts.get(section, 0)) - int(target_section_caps.get(section, 0)), 0)
        for section in _SECTION_ORDER
    }
    total_overage = max(total_count - rule.max_total, 0)
    max_section_overage = max(section_overages.values()) if section_overages else 0

    balancing_reasons = list(rule.reasons)
    for section in _SECTION_ORDER:
        if section_overages[section] > 0:
            balancing_reasons.append(
                f"{section}_cap_exceeded:{section_counts.get(section, 0)}>{target_section_caps.get(section, 0)}"
            )
    if total_overage > 0:
        balancing_reasons.append(f"total_cap_exceeded:{total_count}>{rule.max_total}")
    if total_count < rule.min_total:
        balancing_reasons.append(f"under_target:{total_count}<{rule.min_total}")

    if rule.max_total == 0 and total_count > 0:
        status = "hard_capped"
    elif total_count > rule.max_total + 1 or max_section_overage > 1:
        status = "stuffed"
    elif total_overage > 0 or max_section_overage > 0:
        status = "over_target"
    elif total_count < rule.min_total:
        status = "under_target"
    else:
        status = "within_target"

    return TermFrequencyStatus(
        term=rule.term,
        weight=weight,
        priority_bucket=rule.priority_bucket,
        total_count=total_count,
        section_counts={section: int(section_counts.get(section, 0)) for section in _SECTION_ORDER},
        target_min_total=rule.min_total,
        target_max_total=rule.max_total,
        target_section_caps=target_section_caps,
        status=status,
        is_overused=status in {"hard_capped", "stuffed", "over_target"},
        is_underused=status == "under_target",
        is_capped=total_count >= rule.max_total if rule.max_total > 0 else total_count > 0,
        overuse_amount=max(total_overage, max_section_overage),
        balancing_reasons=tuple(balancing_reasons),
        suggested_preferred_sections=rule.preferred_sections,
        suggested_deprioritized_sections=rule.deprioritized_sections,
    )


def _priority_bucket(term: str, job_weights: JobWeights, coverage_entry, priority_hint: str | None) -> str:
    if coverage_entry is not None:
        return coverage_entry.priority_bucket
    if priority_hint in {"high", "medium", "low"}:
        return priority_hint
    if term in job_weights.high_priority_terms:
        return "high"
    if term in job_weights.medium_priority_terms:
        return "medium"
    return "low"


def _preferred_sections(
    *,
    is_title_term: bool,
    title_safe: bool,
    section_caps: Mapping[str, int],
) -> tuple[str, ...]:
    ordered: list[str] = []
    if is_title_term:
        if title_safe and int(section_caps.get("summary", 0)) > 0:
            ordered.append("summary")
        if int(section_caps.get("experience", 0)) > 0:
            ordered.append("experience")
    else:
        for section in ("experience", "skills", "summary", "projects", "education"):
            if int(section_caps.get(section, 0)) > 0 and section not in ordered:
                ordered.append(section)
    return tuple(ordered)


def _deprioritized_sections(
    *,
    is_title_term: bool,
    preferred_sections: Sequence[str],
) -> tuple[str, ...]:
    base_order = _ROLLBACK_ORDER_TITLE if is_title_term else _ROLLBACK_ORDER_DEFAULT
    ordered: list[str] = []
    for section in base_order:
        if section not in ordered:
            ordered.append(section)
    for section in preferred_sections:
        if section not in ordered:
            ordered.append(section)
    return tuple(ordered)


def _section_counts_for_term(signals, term: str) -> dict[str, int]:
    counts = _empty_section_counts()
    if signals is None:
        return counts
    for entry in signals.evidence_map.get(term, ()):
        counts[_section_bucket(entry.section)] += int(entry.occurrence_count)
    return counts


def _empty_section_counts() -> dict[str, int]:
    return {section: 0 for section in _SECTION_ORDER}


def _section_bucket(section: str) -> str:
    if section in {"experience_title", "experience_bullet"}:
        return "experience"
    if section in {"project_name", "project_bullet"}:
        return "projects"
    if section in _SECTION_ORDER:
        return section
    return "education"


def _section_distribution_summary(
    frequency_by_term: Mapping[str, TermFrequencyStatus],
) -> dict[str, int]:
    summary = {section: 0 for section in _SECTION_ORDER}
    for status in frequency_by_term.values():
        for section in _SECTION_ORDER:
            summary[section] += int(status.section_counts.get(section, 0))
    return summary


def _extract_terms(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    terms: list[str] = []
    for value in values:
        if isinstance(value, str):
            canonical = canonicalize_term(value)
            if canonical and canonical not in terms:
                terms.append(canonical)
    return terms


def _extract_item_terms(values: Any) -> list[str]:
    return [item["term"] for item in _iter_plan_items(values)]


def _iter_plan_items(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    items: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        term = canonicalize_term(value.get("term"))
        if not term:
            continue
        item = dict(value)
        item["term"] = term
        items.append(item)
    return items
