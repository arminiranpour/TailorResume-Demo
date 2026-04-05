from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from app.ats import (
    ATSRecencyPriorities,
    JobSignals,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    ResumeSignals,
    TitleAlignment,
    build_coverage_model,
    build_evidence_links,
    build_job_weights,
    build_recency_priorities,
    build_title_alignment,
    extract_job_signals,
    extract_resume_signals,
)
from app.ats.types import EvidenceCandidate
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json


@dataclass
class TailoringPlanError(Exception):
    details: List[str]
    raw_preview: str


@dataclass
class TailorNotAllowed(Exception):
    decision: str
    reasons: List[Dict[str, Any]]

    def __str__(self) -> str:
        return f"Tailoring plan not allowed for decision={self.decision}"


@dataclass(frozen=True)
class _ATSPlannerContext:
    job_signals: JobSignals
    resume_signals: ResumeSignals
    job_weights: JobWeights
    coverage: ResumeCoverage
    evidence_links: ResumeEvidenceLinks
    title_alignment: TitleAlignment
    recency: ATSRecencyPriorities
    weight_order: Mapping[str, int]


@dataclass(frozen=True)
class _BulletMeta:
    bullet_id: str
    source_section: str
    order: int


_BULLET_SECTIONS = frozenset({"experience_bullet", "project_bullet"})
_MAX_BULLET_REWRITES = 6
_MAX_BULLET_TARGETS = 3
_MAX_SUMMARY_TARGETS = 5


def generate_tailoring_plan(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    provider: Optional[LLMProvider] = None,
    *,
    job_signals: Optional[JobSignals] = None,
    resume_signals: Optional[ResumeSignals] = None,
    job_weights: Optional[JobWeights] = None,
    coverage: Optional[ResumeCoverage] = None,
    evidence_links: Optional[ResumeEvidenceLinks] = None,
    title_alignment: Optional[TitleAlignment] = None,
    recency_priorities: Optional[ATSRecencyPriorities] = None,
) -> Dict[str, Any]:
    del provider  # Planning is deterministic in the ATS pipeline.

    if not isinstance(resume_json, dict) or not isinstance(job_json, dict) or not isinstance(score_result, dict):
        raise ValueError("resume_json, job_json, and score_result must be objects")

    decision = score_result.get("decision")
    if decision != "PROCEED":
        reasons = score_result.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        raise TailorNotAllowed(decision=str(decision), reasons=reasons)

    missing_requirements = score_result.get("missing_requirements")
    if not isinstance(missing_requirements, list):
        raise TailoringPlanError(
            details=["score_result.missing_requirements must be a list"],
            raw_preview="",
        )

    context = _build_ats_context(
        job_json,
        resume_json,
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
        recency_priorities=recency_priorities,
    )
    plan = _build_tailoring_plan(
        resume_json=resume_json,
        score_result=score_result,
        context=context,
    )

    ok, errors = validate_json("tailoring_plan", plan)
    if not ok:
        raise TailoringPlanError(details=errors, raw_preview="")
    return plan


def _build_ats_context(
    job_json: Dict[str, Any],
    resume_json: Dict[str, Any],
    *,
    job_signals: Optional[JobSignals] = None,
    resume_signals: Optional[ResumeSignals] = None,
    job_weights: Optional[JobWeights] = None,
    coverage: Optional[ResumeCoverage] = None,
    evidence_links: Optional[ResumeEvidenceLinks] = None,
    title_alignment: Optional[TitleAlignment] = None,
    recency_priorities: Optional[ATSRecencyPriorities] = None,
) -> _ATSPlannerContext:
    job_signals = job_signals or extract_job_signals(job_json)
    resume_signals = resume_signals or extract_resume_signals(resume_json)
    job_weights = job_weights or build_job_weights(job_signals)
    coverage = coverage or build_coverage_model(job_signals, resume_signals, job_weights)
    evidence_links = evidence_links or build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    title_alignment = title_alignment or build_title_alignment(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    recency = recency_priorities or build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )
    return _ATSPlannerContext(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
        recency=recency,
        weight_order={term: index for index, term in enumerate(job_weights.ordered_terms)},
    )


def _build_tailoring_plan(
    *,
    resume_json: Dict[str, Any],
    score_result: Dict[str, Any],
    context: _ATSPlannerContext,
) -> Dict[str, Any]:
    bullet_order, bullet_meta = _collect_bullet_inventory(resume_json)
    supported_priority_terms = _build_supported_priority_terms(context)
    under_supported_terms = _build_under_supported_terms(context)
    blocked_terms = _build_blocked_terms(context)
    prioritized_keywords = _build_prioritized_keywords(
        context=context,
        supported_priority_terms=supported_priority_terms,
        under_supported_terms=under_supported_terms,
    )
    recent_priority_terms = _build_recent_priority_terms(context)
    skill_priority_terms = _build_skill_priority_terms(context)
    summary_alignment_terms = _build_summary_alignment_terms(context)
    bullet_actions = _build_bullet_actions(
        bullet_order=bullet_order,
        bullet_meta=bullet_meta,
        context=context,
    )

    plan: Dict[str, Any] = {
        "bullet_actions": bullet_actions,
        "missing_requirements": score_result["missing_requirements"],
        "prioritized_keywords": prioritized_keywords,
        "supported_priority_terms": supported_priority_terms,
        "under_supported_terms": under_supported_terms,
        "blocked_terms": blocked_terms,
        "recent_priority_terms": recent_priority_terms,
        "summary_alignment_terms": summary_alignment_terms,
        "skill_priority_terms": skill_priority_terms,
        "title_alignment_status": _build_title_alignment_status(context, summary_alignment_terms),
    }

    summary_plan = _build_summary_rewrite(
        resume_json=resume_json,
        context=context,
        supported_priority_terms=supported_priority_terms,
        blocked_terms=blocked_terms,
        summary_alignment_terms=summary_alignment_terms,
    )
    if summary_plan is not None:
        plan["summary_rewrite"] = summary_plan

    skills_reorder_plan = _build_skills_reorder_plan(
        resume_json=resume_json,
        skill_priority_terms=skill_priority_terms,
        context=context,
    )
    if skills_reorder_plan:
        plan["skills_reorder_plan"] = skills_reorder_plan

    return plan


def _build_supported_priority_terms(context: _ATSPlannerContext) -> List[str]:
    terms: List[str] = []
    for term in context.job_weights.ordered_terms:
        coverage = context.coverage.coverage_by_term[term]
        if not coverage.is_covered or coverage.is_under_supported:
            continue
        if not _has_any_safe_surface(term, context):
            continue
        _append_unique(terms, term)
    return terms


def _build_under_supported_terms(context: _ATSPlannerContext) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for term in context.job_weights.ordered_terms:
        coverage = context.coverage.coverage_by_term[term]
        if not coverage.is_under_supported:
            continue
        safe_for = _safe_surfaces(term, context)
        if not safe_for:
            continue
        items.append(
            {
                "term": term,
                "priority_bucket": coverage.priority_bucket,
                "safe_for": safe_for,
                "reason": "under_supported_resume_evidence",
            }
        )
    return items


def _build_blocked_terms(context: _ATSPlannerContext) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    title_terms = set(context.job_signals.title_terms)

    for term in context.job_weights.ordered_terms:
        coverage = context.coverage.coverage_by_term[term]
        link = context.evidence_links.links_by_term[term]
        blocked_for: List[str] = []

        if not link.all_candidates:
            blocked_for = ["bullets", "summary", "skills"]
        else:
            if _preferred_bullet_candidate(term, context) is None:
                blocked_for.append("bullets")
            if not _is_summary_safe(term, context):
                blocked_for.append("summary")
            if not _is_skills_safe(term, context):
                blocked_for.append("skills")

        if not blocked_for:
            continue
        if not _should_audit_block(term, coverage, link, blocked_for, title_terms):
            continue

        items.append(
            {
                "term": term,
                "priority_bucket": coverage.priority_bucket,
                "blocked_for": blocked_for,
                "reason": _block_reason(
                    term=term,
                    coverage=coverage,
                    link=link,
                    blocked_for=blocked_for,
                    title_terms=title_terms,
                    title_alignment=context.title_alignment,
                ),
            }
        )
    return items


def _build_prioritized_keywords(
    *,
    context: _ATSPlannerContext,
    supported_priority_terms: List[str],
    under_supported_terms: List[Dict[str, Any]],
) -> List[str]:
    keywords: List[str] = []
    title_terms = set(context.job_signals.title_terms)

    for term in supported_priority_terms:
        if term in title_terms and not (_preferred_bullet_candidate(term, context) or _is_skills_safe(term, context)):
            continue
        if _should_include_prioritized_keyword(term, context):
            _append_unique(keywords, term)

    for item in under_supported_terms:
        term = item["term"]
        safe_for = set(item["safe_for"])
        if safe_for.intersection({"bullets", "skills"}):
            _append_unique(keywords, term)

    return keywords


def _build_recent_priority_terms(context: _ATSPlannerContext) -> List[str]:
    items: List[str] = []
    for term in context.recency.recency_ordered_terms:
        priority = context.recency.priorities_by_term[term]
        if not priority.has_recent_backing:
            continue
        if not _has_any_safe_surface(term, context):
            continue
        _append_unique(items, term)
    return items


def _build_skill_priority_terms(context: _ATSPlannerContext) -> List[str]:
    items: List[str] = []
    for term in context.job_weights.ordered_terms:
        if _is_skills_safe(term, context):
            _append_unique(items, term)
    return items


def _build_summary_alignment_terms(context: _ATSPlannerContext) -> List[str]:
    if not context.title_alignment.is_safe_for_summary_alignment:
        return []

    items: List[str] = []
    for phrase in context.title_alignment.overlapping_phrases:
        _append_unique(items, phrase)
    for token in context.title_alignment.overlapping_tokens:
        _append_unique(items, token)
    return items


def _build_summary_rewrite(
    *,
    resume_json: Dict[str, Any],
    context: _ATSPlannerContext,
    supported_priority_terms: List[str],
    blocked_terms: List[Dict[str, Any]],
    summary_alignment_terms: List[str],
) -> Optional[Dict[str, Any]]:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else None
    if summary is None:
        return None
    summary_text = summary.get("text")
    if not isinstance(summary_text, str) or not summary_text.strip():
        return None

    targets: List[str] = []
    for term in summary_alignment_terms:
        _append_unique(targets, term)
    for term in context.recency.recent_summary_safe_terms:
        if _is_summary_safe(term, context):
            _append_unique(targets, term)
    for term in supported_priority_terms:
        if _is_summary_safe(term, context):
            _append_unique(targets, term)

    return {
        "rewrite_intent": "rewrite",
        "target_keywords": targets[:_MAX_SUMMARY_TARGETS],
        "title_alignment_safe": context.title_alignment.is_safe_for_summary_alignment,
        "title_terms": summary_alignment_terms,
        "blocked_terms": [
            item["term"] for item in blocked_terms if "summary" in item.get("blocked_for", [])
        ],
    }


def _build_title_alignment_status(
    context: _ATSPlannerContext,
    summary_alignment_terms: List[str],
) -> Dict[str, Any]:
    return {
        "is_title_supported": context.title_alignment.is_title_supported,
        "is_safe_for_summary_alignment": context.title_alignment.is_safe_for_summary_alignment,
        "alignment_strength": context.title_alignment.alignment_strength,
        "supported_terms": summary_alignment_terms,
        "missing_tokens": list(context.title_alignment.missing_title_tokens),
        "strongest_matching_resume_title": context.title_alignment.strongest_matching_resume_title,
    }


def _build_skills_reorder_plan(
    *,
    resume_json: Dict[str, Any],
    skill_priority_terms: List[str],
    context: _ATSPlannerContext,
) -> List[str]:
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else None
    if skills is None:
        return []
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    if len(lines) <= 1:
        return []

    ordered_ids = _collect_skills_line_ids(resume_json)
    order_index = {line_id: index for index, line_id in enumerate(ordered_ids)}
    scores = {line_id: 0 for line_id in ordered_ids}

    total_terms = len(skill_priority_terms)
    for rank, term in enumerate(skill_priority_terms):
        link = context.evidence_links.links_by_term[term]
        line_ids = []
        for candidate in link.ranked_candidates:
            if candidate.line_id is None:
                continue
            _append_unique(line_ids, candidate.line_id)
        term_score = (total_terms - rank) * 100
        for line_rank, line_id in enumerate(line_ids):
            if line_id not in scores:
                continue
            scores[line_id] += term_score - line_rank

    reordered = sorted(ordered_ids, key=lambda line_id: (-scores[line_id], order_index[line_id]))
    if reordered == ordered_ids or all(scores[line_id] == 0 for line_id in ordered_ids):
        return []
    return reordered


def _build_bullet_actions(
    *,
    bullet_order: List[str],
    bullet_meta: Mapping[str, _BulletMeta],
    context: _ATSPlannerContext,
) -> List[Dict[str, Any]]:
    bullet_evidence_terms: Dict[str, List[str]] = {bullet_id: [] for bullet_id in bullet_order}
    bullet_target_terms: Dict[str, List[str]] = {bullet_id: [] for bullet_id in bullet_order}
    bullet_is_recent: Dict[str, bool] = {bullet_id: False for bullet_id in bullet_order}
    bullet_is_primary: Dict[str, bool] = {bullet_id: False for bullet_id in bullet_order}

    for term in context.job_weights.ordered_terms:
        link = context.evidence_links.links_by_term[term]
        for candidate in link.ranked_candidates:
            if candidate.bullet_id is None or candidate.section not in _BULLET_SECTIONS:
                continue
            if candidate.bullet_id in bullet_evidence_terms:
                _append_unique(bullet_evidence_terms[candidate.bullet_id], term)

        preferred = _preferred_bullet_candidate(term, context)
        if preferred is None or preferred.bullet_id is None:
            continue
        if preferred.bullet_id not in bullet_target_terms:
            continue
        _append_unique(bullet_target_terms[preferred.bullet_id], term)
        bullet_is_primary[preferred.bullet_id] = bool(
            preferred.is_primary_candidate
            or (
                link.strongest_candidate is not None
                and link.strongest_candidate.source_id == preferred.source_id
            )
        )
        if _is_recent_candidate_for_term(preferred, context.recency.priorities_by_term[term]):
            bullet_is_recent[preferred.bullet_id] = True

    selected_bullets = _select_rewrite_bullets(
        bullet_order=bullet_order,
        bullet_target_terms=bullet_target_terms,
        bullet_is_recent=bullet_is_recent,
        bullet_is_primary=bullet_is_primary,
        context=context,
    )

    actions: List[Dict[str, Any]] = []
    for bullet_id in bullet_order:
        targets = bullet_target_terms.get(bullet_id, [])
        action = {
            "bullet_id": bullet_id,
            "rewrite_intent": "rewrite" if bullet_id in selected_bullets and targets else "keep",
            "target_keywords": targets[:_MAX_BULLET_TARGETS] if bullet_id in selected_bullets else [],
            "evidence_terms": bullet_evidence_terms.get(bullet_id, []),
            "source_section": bullet_meta[bullet_id].source_section,
            "is_recent": bullet_is_recent.get(bullet_id, False),
            "is_primary_evidence": bullet_is_primary.get(bullet_id, False),
            "is_safe_for_ats": bool(bullet_evidence_terms.get(bullet_id)),
        }
        actions.append(action)
    return actions


def _select_rewrite_bullets(
    *,
    bullet_order: List[str],
    bullet_target_terms: Mapping[str, List[str]],
    bullet_is_recent: Mapping[str, bool],
    bullet_is_primary: Mapping[str, bool],
    context: _ATSPlannerContext,
) -> Set[str]:
    order_index = {bullet_id: index for index, bullet_id in enumerate(bullet_order)}
    scored: List[Tuple[int, int, str]] = []
    for bullet_id, terms in bullet_target_terms.items():
        if not terms:
            continue
        score = 0
        for term in terms:
            weight = context.job_weights.weights_by_term[term].total_weight
            score += (weight * 100) - context.weight_order[term]
        if bullet_is_recent.get(bullet_id):
            score += 50
        if bullet_is_primary.get(bullet_id):
            score += 25
        scored.append((-score, order_index[bullet_id], bullet_id))

    scored.sort()
    return {bullet_id for _, _, bullet_id in scored[:_MAX_BULLET_REWRITES]}


def _preferred_bullet_candidate(
    term: str,
    context: _ATSPlannerContext,
) -> Optional[EvidenceCandidate]:
    priority = context.recency.priorities_by_term[term]
    link = context.evidence_links.links_by_term[term]

    for candidate in (
        priority.strongest_recent_experience_candidate,
        priority.strongest_recent_project_candidate,
    ):
        if candidate is not None and candidate.section in _BULLET_SECTIONS:
            return candidate

    for candidate in link.ranked_candidates:
        if candidate.section in _BULLET_SECTIONS:
            return candidate
    return None


def _has_any_safe_surface(term: str, context: _ATSPlannerContext) -> bool:
    return bool(_safe_surfaces(term, context))


def _safe_surfaces(term: str, context: _ATSPlannerContext) -> List[str]:
    safe_for: List[str] = []
    if _preferred_bullet_candidate(term, context) is not None:
        safe_for.append("bullets")
    if _is_summary_safe(term, context):
        safe_for.append("summary")
    if _is_skills_safe(term, context):
        safe_for.append("skills")
    return safe_for


def _is_summary_safe(term: str, context: _ATSPlannerContext) -> bool:
    link = context.evidence_links.links_by_term[term]
    coverage = context.coverage.coverage_by_term[term]
    if term in context.job_signals.title_terms:
        return (
            context.title_alignment.is_safe_for_summary_alignment
            and term in _title_alignment_supported_terms(context.title_alignment)
        )
    return bool(
        link.is_safe_for_summary
        and not coverage.is_under_supported
        and not link.missing_experience_backing
    )


def _is_skills_safe(term: str, context: _ATSPlannerContext) -> bool:
    link = context.evidence_links.links_by_term[term]
    return bool(link.all_candidates and link.is_safe_for_skills)


def _title_alignment_supported_terms(title_alignment: TitleAlignment) -> Set[str]:
    supported = set(title_alignment.overlapping_tokens)
    supported.update(title_alignment.overlapping_phrases)
    return supported


def _should_include_prioritized_keyword(term: str, context: _ATSPlannerContext) -> bool:
    if _preferred_bullet_candidate(term, context) is not None:
        return True
    if _is_skills_safe(term, context):
        return True
    if term not in context.job_signals.title_terms and _is_summary_safe(term, context):
        return True
    return False


def _should_audit_block(
    term: str,
    coverage,
    link,
    blocked_for: List[str],
    title_terms: Set[str],
) -> bool:
    return bool(
        coverage.is_missing
        or coverage.is_under_supported
        or link.missing_experience_backing
        or term in title_terms
        or coverage.priority_bucket != "low"
        or len(blocked_for) == 3
    )


def _block_reason(
    *,
    term: str,
    coverage,
    link,
    blocked_for: List[str],
    title_terms: Set[str],
    title_alignment: TitleAlignment,
) -> str:
    if not link.all_candidates:
        return "no_resume_evidence"
    if term in title_terms and "summary" in blocked_for and not title_alignment.is_safe_for_summary_alignment:
        return "unsupported_title_alignment"
    if link.missing_experience_backing:
        return "missing_experience_backing"
    if coverage.is_under_supported:
        return "under_supported_resume_evidence"
    return "surface_not_safe"


def _is_recent_candidate_for_term(candidate: EvidenceCandidate, priority) -> bool:
    recent_candidates = (
        priority.strongest_recent_candidate,
        priority.strongest_recent_experience_candidate,
        priority.strongest_recent_project_candidate,
    )
    return any(
        recent is not None and recent.source_id == candidate.source_id for recent in recent_candidates
    )


def _collect_bullet_inventory(resume_json: Dict[str, Any]) -> Tuple[List[str], Dict[str, _BulletMeta]]:
    order: List[str] = []
    inventory: Dict[str, _BulletMeta] = {}
    cursor = 0

    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            bullet_id = bullet.get("bullet_id") if isinstance(bullet, dict) else None
            if not isinstance(bullet_id, str) or bullet_id.strip() == "":
                continue
            order.append(bullet_id)
            inventory[bullet_id] = _BulletMeta(
                bullet_id=bullet_id,
                source_section="experience_bullet",
                order=cursor,
            )
            cursor += 1

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for project in projects:
        if not isinstance(project, dict):
            continue
        bullets = project.get("bullets") if isinstance(project.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            bullet_id = bullet.get("bullet_id") if isinstance(bullet, dict) else None
            if not isinstance(bullet_id, str) or bullet_id.strip() == "":
                continue
            order.append(bullet_id)
            inventory[bullet_id] = _BulletMeta(
                bullet_id=bullet_id,
                source_section="project_bullet",
                order=cursor,
            )
            cursor += 1
    return order, inventory


def _sorted_bullets(bullets: List[Any]) -> List[Dict[str, Any]]:
    indexed: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, bullet in enumerate(bullets):
        if not isinstance(bullet, dict):
            continue
        bullet_index = bullet.get("bullet_index")
        effective_index = bullet_index if isinstance(bullet_index, int) else idx
        indexed.append((effective_index, idx, bullet))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def _collect_skills_line_ids(resume_json: Dict[str, Any]) -> List[str]:
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    ids: List[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if isinstance(line_id, str) and line_id.strip():
            ids.append(line_id)
    return ids


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


# Backward-compatible public name for the deterministic tailoring plan builder.
build_tailoring_plan = generate_tailoring_plan

__all__ = [
    "TailoringPlanError",
    "TailorNotAllowed",
    "build_tailoring_plan",
    "generate_tailoring_plan",
]
