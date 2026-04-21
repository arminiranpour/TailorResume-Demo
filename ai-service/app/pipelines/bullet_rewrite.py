from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from app.ats import (
    ATSFrequencyBalance,
    ATSRecencyPriorities,
    FrequencyBalancingAction,
    JobWeights,
    ResumeCoverage,
    ResumeEvidenceLinks,
    ResumeSignals,
    build_coverage_model,
    build_evidence_links,
    build_frequency_balance,
    build_job_weights,
    build_recency_priorities,
    build_title_alignment,
    extract_job_signals,
    extract_resume_signals,
    validate_frequency_balance,
)
from app.ats.canonicalize import (
    canonicalize_term,
    extract_canonical_term_pairs,
    normalize_text as normalize_ats_text,
)
from app.config import get_config
from app.pipelines.allowed_vocab import build_allowed_vocab, normalize_terms
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages
from app.scoring_normalize import generate_ngrams, normalize_text, tokenize


@dataclass
class BulletRewriteError(Exception):
    details: List[str]
    raw_preview: str = ""


@dataclass
class BulletRewriteNotAllowed(Exception):
    decision: str
    reasons: List[Dict[str, Any]]

    def __str__(self) -> str:
        return f"Bullet rewrite not allowed for decision={self.decision}"


@dataclass(frozen=True)
class TailoringATSContext:
    supported_priority_terms: Tuple[str, ...]
    under_supported_terms: Tuple[str, ...]
    blocked_bullet_terms: Tuple[str, ...]
    recent_priority_terms: Tuple[str, ...]
    summary_alignment_terms: Tuple[str, ...]
    skill_priority_terms: Tuple[str, ...]
    title_supported_terms: Tuple[str, ...]
    title_missing_terms: Tuple[str, ...]
    title_alignment_safe: bool
    plan_terms: FrozenSet[str]
    avoid_terms: FrozenSet[str]


@dataclass(frozen=True)
class BulletATSRewritePolicy:
    bullet_id: str
    source_section: Optional[str]
    requested_target_keywords: Tuple[str, ...]
    evidence_terms: Tuple[str, ...]
    safe_target_terms: Tuple[str, ...]
    surface_terms: Tuple[str, ...]
    blocked_terms: Tuple[str, ...]
    avoid_terms: Tuple[str, ...]
    required_terms: Tuple[str, ...]
    allowed_new_terms: FrozenSet[str]
    blocked_term_set: FrozenSet[str]
    salvageable_blocked_terms: FrozenSet[str]
    plan_term_set: FrozenSet[str]
    is_recent: bool
    is_primary_evidence: bool
    is_safe_for_ats: bool
    emphasis_strength: str


@dataclass(frozen=True)
class SummaryATSRewritePolicy:
    requested_target_keywords: Tuple[str, ...]
    preferred_surface_terms: Tuple[str, ...]
    safe_title_terms: Tuple[str, ...]
    unsafe_title_terms: Tuple[str, ...]
    supported_priority_terms: Tuple[str, ...]
    recent_priority_terms: Tuple[str, ...]
    required_terms: Tuple[str, ...]
    blocked_terms: Tuple[str, ...]
    avoid_terms: Tuple[str, ...]
    allowed_new_terms: FrozenSet[str]
    blocked_term_set: FrozenSet[str]
    significant_terms: FrozenSet[str]
    title_missing_terms: Tuple[str, ...]
    title_alignment_safe: bool


@dataclass(frozen=True)
class SkillTermPolicy:
    term: str
    display: str
    weight: int
    support_score: int
    priority_rank: int
    supported_rank: int
    recent_rank: int
    has_recent_backing: bool
    has_cross_section_backing: bool
    has_experience_backing: bool
    has_project_backing: bool
    is_supported_priority: bool
    is_skill_priority: bool
    is_recent_priority: bool
    is_under_supported: bool
    is_blocked: bool
    is_allowed_surface: bool
    is_skills_only: bool
    missing_experience_backing: bool


@dataclass(frozen=True)
class SkillsATSOptimizationPolicy:
    resume_signals: ResumeSignals
    job_weights: JobWeights
    coverage: ResumeCoverage
    evidence_links: ResumeEvidenceLinks
    recency: ATSRecencyPriorities
    supported_priority_terms: Tuple[str, ...]
    skill_priority_terms: Tuple[str, ...]
    recent_priority_terms: Tuple[str, ...]
    under_supported_terms: FrozenSet[str]
    blocked_terms: Tuple[str, ...]
    allowed_surface_terms: FrozenSet[str]
    preferred_line_ids: Tuple[str, ...]
    line_budget_by_id: Dict[str, int]
    term_policies: Dict[str, SkillTermPolicy]


_REWRITE_KEEP = {"keep", "skip", "none"}
_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9#.+-]*")
_TRAILING_PERIODS = "."
_MIN_BUDGET = 80

_SOFT_SKILL_TERMS = {
    "communication",
    "communication skills",
    "collaboration",
    "collaborative",
    "teamwork",
    "team player",
    "leadership",
    "problem solving",
    "problem-solving",
    "customer service",
    "time management",
    "adaptability",
    "organization",
    "organizational",
    "interpersonal",
    "interpersonal skills",
    "cross functional",
    "cross-functional",
    "stakeholder management",
    "mentorship",
    "coaching",
    "initiative",
    "ownership",
    "strategic thinking",
    "critical thinking",
    "attention to detail",
    "detail oriented",
    "presentation",
    "writing",
    "verbal communication",
    "conflict resolution",
    "negotiation",
    "prioritization",
}

_TECHNICAL_INDICATORS = {
    "api",
    "apis",
    "backend",
    "frontend",
    "full stack",
    "full-stack",
    "database",
    "databases",
    "data",
    "sql",
    "etl",
    "pipeline",
    "pipelines",
    "cloud",
    "infrastructure",
    "devops",
    "ci cd",
    "ci/cd",
    "microservices",
    "distributed systems",
    "systems",
    "architecture",
    "security",
    "network",
    "performance",
    "scalability",
    "automation",
    "testing",
}

_SKILL_DISPLAY_OVERRIDES = {
    ".net": ".NET",
    "api": "API",
    "aws": "AWS",
    "c#": "C#",
    "ci/cd": "CI/CD",
    "fastapi": "FastAPI",
    "graphql": "GraphQL",
    "javascript": "JavaScript",
    "node.js": "Node.js",
    "postgresql": "PostgreSQL",
    "react": "React",
    "rest api": "REST API",
    "sql": "SQL",
    "typescript": "TypeScript",
}


def rewrite_resume_text(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    character_budgets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tailored, _ = rewrite_resume_text_with_audit(
        resume_json,
        job_json,
        score_result,
        tailoring_plan,
        provider,
        character_budgets=character_budgets,
    )
    return tailored


def apply_bullet_rewrites(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    character_budgets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return rewrite_resume_text(
        resume_json,
        job_json,
        score_result,
        tailoring_plan,
        provider,
        character_budgets=character_budgets,
    )


def rewrite_resume_text_with_audit(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    character_budgets: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(resume_json, dict) or not isinstance(job_json, dict) or not isinstance(score_result, dict):
        raise ValueError("resume_json, job_json, and score_result must be objects")
    if not isinstance(tailoring_plan, dict):
        raise ValueError("tailoring_plan must be an object")

    decision = score_result.get("decision")
    if decision != "PROCEED":
        reasons = score_result.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        raise BulletRewriteNotAllowed(decision=str(decision), reasons=reasons)

    _validate_schema_or_raise("resume", resume_json)
    _validate_schema_or_raise("tailoring_plan", tailoring_plan)

    allowed_vocab = build_allowed_vocab(resume_json)
    budgets = _derive_budgets(resume_json, character_budgets)

    tailored = _clone_resume(resume_json)
    audit_log = {
        "rewritten_bullets": [],
        "kept_bullets": [],
        "rejected_for_new_terms": [],
        "compressed": [],
        "bullet_details": [],
        "summary_detail": None,
        "skills_details": [],
        "frequency_actions": [],
        "frequency_balance": None,
    }

    _rewrite_summary(
        tailored,
        resume_json,
        job_json,
        score_result,
        tailoring_plan,
        provider,
        allowed_vocab,
        budgets,
        audit_log,
    )
    _reorder_skills(tailored, tailoring_plan)
    _tailor_skill_lines(
        tailored,
        resume_json,
        job_json,
        score_result,
        tailoring_plan,
        allowed_vocab,
        budgets,
        audit_log,
    )
    _rewrite_bullets(
        tailored,
        job_json,
        tailoring_plan,
        provider,
        allowed_vocab,
        budgets,
        audit_log,
    )
    _enforce_frequency_balance(
        tailored,
        resume_json,
        job_json,
        tailoring_plan,
        audit_log,
    )

    ok, errors = validate_json("resume", tailored)
    if not ok:
        raise BulletRewriteError(details=errors, raw_preview="")

    invariant_errors = _check_invariants(resume_json, tailored)
    if invariant_errors:
        raise BulletRewriteError(details=invariant_errors, raw_preview="")
    return tailored, audit_log


def _enforce_frequency_balance(
    tailored: Dict[str, Any],
    source_resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    job_signals = extract_job_signals(job_json)
    source_resume_signals = extract_resume_signals(source_resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, source_resume_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, source_resume_signals, job_weights, coverage)
    title_alignment = build_title_alignment(
        job_signals=job_signals,
        resume_signals=source_resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    recency = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=source_resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )

    surfaces = _build_frequency_surface_states(tailored, audit_log)
    balance = _build_frequency_balance_snapshot(
        source_resume_json=source_resume_json,
        tailored=tailored,
        tailoring_plan=tailoring_plan,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        recency=recency,
        title_alignment=title_alignment,
    )
    actions: List[FrequencyBalancingAction] = []
    max_rollbacks = len(surfaces)

    while balance.validation_errors and len(actions) < max_rollbacks:
        term = _select_frequency_violation_term(balance)
        if term is None:
            break
        surface = _select_frequency_surface_for_term(balance, term, surfaces)
        if surface is None:
            break
        action = _rollback_frequency_surface(surface, term, balance, audit_log)
        if action is None:
            break
        actions.append(action)
        balance = _build_frequency_balance_snapshot(
            source_resume_json=source_resume_json,
            tailored=tailored,
            tailoring_plan=tailoring_plan,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            recency=recency,
            title_alignment=title_alignment,
        )

    final_balance = replace(balance, balancing_actions=tuple(actions))
    audit_log["frequency_actions"] = [asdict(action) for action in actions]
    audit_log["frequency_balance"] = asdict(final_balance)


def _build_frequency_balance_snapshot(
    *,
    source_resume_json: Dict[str, Any],
    tailored: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    recency: ATSRecencyPriorities,
    title_alignment,
) -> ATSFrequencyBalance:
    balance = build_frequency_balance(
        source_resume_json=source_resume_json,
        tailored_resume_json=tailored,
        tailoring_plan=tailoring_plan,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        recency=recency,
        title_alignment=title_alignment,
    )
    validation_errors = tuple(validate_frequency_balance(balance))
    return replace(balance, validation_errors=validation_errors)


def _build_frequency_surface_states(
    tailored: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> List[Dict[str, Any]]:
    summary_node = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else None
    skill_nodes: Dict[str, Dict[str, Any]] = {}
    skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else None
    if isinstance(skills, dict):
        for line in skills.get("lines", []):
            if isinstance(line, dict) and isinstance(line.get("line_id"), str):
                skill_nodes[line["line_id"]] = line

    bullet_nodes: Dict[str, Tuple[Dict[str, Any], str]] = {}
    for exp in tailored.get("experience", []) if isinstance(tailored.get("experience"), list) else []:
        if not isinstance(exp, dict):
            continue
        for bullet in exp.get("bullets", []) if isinstance(exp.get("bullets"), list) else []:
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str):
                bullet_nodes[bullet["bullet_id"]] = (bullet, "experience")
    for project in tailored.get("projects", []) if isinstance(tailored.get("projects"), list) else []:
        if not isinstance(project, dict):
            continue
        for bullet in project.get("bullets", []) if isinstance(project.get("bullets"), list) else []:
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str):
                bullet_nodes[bullet["bullet_id"]] = (bullet, "projects")

    surfaces: List[Dict[str, Any]] = []
    order = 0
    summary_detail = audit_log.get("summary_detail")
    if (
        isinstance(summary_detail, dict)
        and summary_detail.get("changed") is True
        and isinstance(summary_node, dict)
        and isinstance(summary_node.get("text"), str)
        and isinstance(summary_detail.get("original_text"), str)
    ):
        surfaces.append(
            {
                "surface_key": "summary:summary",
                "surface_type": "summary",
                "surface_id": "summary",
                "section": "summary",
                "node": summary_node,
                "current_text": summary_node["text"],
                "original_text": summary_detail["original_text"],
                "order": order,
            }
        )
        order += 1

    details = audit_log.get("skills_details")
    if isinstance(details, list):
        for detail in details:
            line_id = detail.get("line_id") if isinstance(detail, dict) else None
            node = skill_nodes.get(line_id) if isinstance(line_id, str) else None
            if (
                isinstance(detail, dict)
                and detail.get("changed") is True
                and isinstance(line_id, str)
                and isinstance(node, dict)
                and isinstance(node.get("text"), str)
                and isinstance(detail.get("original_text"), str)
            ):
                surfaces.append(
                    {
                        "surface_key": f"skills:{line_id}",
                        "surface_type": "skills",
                        "surface_id": line_id,
                        "section": "skills",
                        "node": node,
                        "current_text": node["text"],
                        "original_text": detail["original_text"],
                        "order": order,
                    }
                )
                order += 1

    bullet_details = audit_log.get("bullet_details")
    if isinstance(bullet_details, list):
        for detail in bullet_details:
            bullet_id = detail.get("bullet_id") if isinstance(detail, dict) else None
            node_and_section = bullet_nodes.get(bullet_id) if isinstance(bullet_id, str) else None
            if (
                isinstance(detail, dict)
                and detail.get("changed") is True
                and isinstance(bullet_id, str)
                and node_and_section is not None
                and isinstance(detail.get("original_text"), str)
            ):
                node, section = node_and_section
                if not isinstance(node.get("text"), str):
                    continue
                surfaces.append(
                    {
                        "surface_key": f"bullet:{bullet_id}",
                        "surface_type": "bullet",
                        "surface_id": bullet_id,
                        "section": section,
                        "node": node,
                        "current_text": node["text"],
                        "original_text": detail["original_text"],
                        "order": order,
                    }
                )
                order += 1

    return surfaces


def _select_frequency_violation_term(balance: ATSFrequencyBalance) -> Optional[str]:
    statuses = balance.frequency_by_term
    ordered_index = {term: index for index, term in enumerate(balance.frequency_ordered_terms)}
    candidates = [status for status in statuses.values() if status.is_overused]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda status: (
            _frequency_status_rank(status.status),
            -status.overuse_amount,
            ordered_index.get(status.term, 999),
            status.term,
        ),
    )
    return selected.term


def _frequency_status_rank(status: str) -> int:
    order = {
        "hard_capped": 0,
        "stuffed": 1,
        "over_target": 2,
        "under_target": 3,
        "within_target": 4,
    }
    return order.get(status, 99)


def _select_frequency_surface_for_term(
    balance: ATSFrequencyBalance,
    term: str,
    surfaces: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    status = balance.frequency_by_term.get(term)
    if status is None:
        return None

    section_rank = {
        section: index for index, section in enumerate(status.suggested_deprioritized_sections)
    }
    sections_over_cap = [
        section
        for section in status.suggested_deprioritized_sections
        if int(status.section_counts.get(section, 0)) > int(status.target_section_caps.get(section, 0))
    ]
    if not sections_over_cap:
        sections_over_cap = [
            section
            for section in status.suggested_deprioritized_sections
            if int(status.section_counts.get(section, 0)) > 0
        ]

    candidates = [
        surface
        for surface in surfaces
        if surface.get("changed") is not False
        and surface.get("section") in sections_over_cap
        and _contains_canonical_term(str(surface.get("current_text", "")), term)
    ]
    if not candidates:
        candidates = [
            surface
            for surface in surfaces
            if surface.get("changed") is not False
            and _contains_canonical_term(str(surface.get("current_text", "")), term)
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda surface: (
            section_rank.get(str(surface.get("section")), 999),
            int(surface.get("order", 999)),
            str(surface.get("surface_id", "")),
        ),
    )


def _rollback_frequency_surface(
    surface: Dict[str, Any],
    term: str,
    balance: ATSFrequencyBalance,
    audit_log: Dict[str, Any],
) -> Optional[FrequencyBalancingAction]:
    node = surface.get("node")
    original_text = surface.get("original_text")
    previous_text = surface.get("current_text")
    if not isinstance(node, dict) or not isinstance(original_text, str) or not isinstance(previous_text, str):
        return None

    node["text"] = original_text
    surface["current_text"] = original_text
    surface["changed"] = False

    reason = _frequency_reason_for_term(term, balance, str(surface.get("section", "")))
    action = FrequencyBalancingAction(
        term=term,
        action="rollback_surface",
        section=str(surface.get("section", "")),
        surface_id=str(surface.get("surface_id", "")),
        reason=reason,
        previous_text=previous_text,
        final_text=original_text,
    )
    _annotate_frequency_rollback(audit_log, surface, balance, term)
    return action


def _frequency_reason_for_term(term: str, balance: ATSFrequencyBalance, section: str) -> str:
    status = balance.frequency_by_term.get(term)
    if status is None:
        return "frequency_violation"
    section_count = int(status.section_counts.get(section, 0))
    section_cap = int(status.target_section_caps.get(section, 0))
    if section_count > section_cap:
        return f"{section}_cap_exceeded:{section_count}>{section_cap}"
    if status.total_count > status.target_max_total:
        return f"total_cap_exceeded:{status.total_count}>{status.target_max_total}"
    return status.status


def _annotate_frequency_rollback(
    audit_log: Dict[str, Any],
    surface: Dict[str, Any],
    balance: ATSFrequencyBalance,
    term: str,
) -> None:
    validation_errors = list(balance.validation_errors)
    surface_type = surface.get("surface_type")
    surface_id = surface.get("surface_id")

    if surface_type == "summary":
        detail = audit_log.get("summary_detail")
        if isinstance(detail, dict):
            detail["final_text"] = surface.get("original_text")
            detail["changed"] = False
            detail["skip_reason"] = "frequency_balance_rollback"
            detail["reject_reason"] = "frequency_balance"
            detail["frequency_terms"] = [term]
            detail["validation_errors"] = validation_errors
        return

    if surface_type == "skills":
        details = audit_log.get("skills_details")
        if isinstance(details, list):
            for detail in reversed(details):
                if isinstance(detail, dict) and detail.get("line_id") == surface_id:
                    detail["final_text"] = surface.get("original_text")
                    detail["changed"] = False
                    detail["skip_reason"] = "frequency_balance_rollback"
                    detail["reject_reason"] = "frequency_balance"
                    detail["frequency_terms"] = [term]
                    detail["validation_errors"] = validation_errors
                    break
        return

    if surface_type == "bullet":
        details = audit_log.get("bullet_details")
        if isinstance(details, list):
            for detail in reversed(details):
                if isinstance(detail, dict) and detail.get("bullet_id") == surface_id:
                    detail["final_text"] = surface.get("original_text")
                    detail["changed"] = False
                    detail["skip_reason"] = "frequency_balance_rollback"
                    detail["reject_reason"] = "frequency_balance"
                    detail["frequency_terms"] = [term]
                    detail["validation_errors"] = validation_errors
                    break
        rewritten = audit_log.get("rewritten_bullets")
        if isinstance(rewritten, list) and surface_id in rewritten:
            rewritten[:] = [bullet_id for bullet_id in rewritten if bullet_id != surface_id]
        kept = audit_log.get("kept_bullets")
        if isinstance(kept, list) and isinstance(surface_id, str):
            _append_unique(kept, surface_id)


def _validate_schema_or_raise(schema_name: str, obj: Dict[str, Any]) -> None:
    ok, errors = validate_json(schema_name, obj)
    if not ok:
        raise BulletRewriteError(details=errors, raw_preview="")


def _clone_resume(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(resume_json))


def _derive_budgets(resume_json: Dict[str, Any], character_budgets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary_budget = None
    bullet_budgets: Dict[str, int] = {}
    skills_line_budgets: Dict[str, int] = {}
    if isinstance(character_budgets, dict):
        summary_value = character_budgets.get("summary")
        if isinstance(summary_value, int) and summary_value > 0:
            summary_budget = summary_value
        bullets_value = character_budgets.get("bullets")
        if isinstance(bullets_value, dict):
            for bullet_id, value in bullets_value.items():
                if isinstance(bullet_id, str) and isinstance(value, int) and value > 0:
                    bullet_budgets[bullet_id] = value
        skills_value = character_budgets.get("skills_lines")
        if isinstance(skills_value, dict):
            for line_id, value in skills_value.items():
                if isinstance(line_id, str) and isinstance(value, int) and value > 0:
                    skills_line_budgets[line_id] = value
        skills_value = character_budgets.get("skills_line_max_chars")
        if isinstance(skills_value, dict):
            for line_id, value in skills_value.items():
                if isinstance(line_id, str) and isinstance(value, int) and value > 0:
                    skills_line_budgets[line_id] = value

    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    summary_text = summary.get("text") if isinstance(summary.get("text"), str) else ""
    if summary_budget is None and summary_text:
        summary_budget = max(len(summary_text), _MIN_BUDGET)

    for bullet_id, text in _iter_bullet_texts(resume_json):
        if bullet_id in bullet_budgets:
            continue
        budget = max(len(text), _MIN_BUDGET)
        bullet_budgets[bullet_id] = budget

    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if not isinstance(line_id, str) or not isinstance(text, str):
            continue
        skills_line_budgets.setdefault(line_id, len(text))

    return {
        "summary": summary_budget,
        "bullets": bullet_budgets,
        "skills_line_max_chars": skills_line_budgets,
    }


def _iter_bullet_texts(resume_json: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                yield bullet_id, text

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                yield bullet_id, text


def _rewrite_summary(
    tailored: Dict[str, Any],
    source_resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    summary_detail = {
        "original_text": "",
        "rewrite_intent": None,
        "target_keywords": [],
        "candidate_text": None,
        "final_text": None,
        "skip_reason": None,
        "reject_reason": None,
        "disallowed_terms": [],
        "preferred_surface_terms": [],
        "required_terms": [],
        "supported_priority_terms": [],
        "recent_priority_terms": [],
        "allowed_title_terms": [],
        "blocked_terms": [],
        "blocked_title_terms": [],
        "changed": False,
        "fallback_used": False,
        "fallback_reason": None,
    }
    summary_plan = tailoring_plan.get("summary_rewrite")
    if not isinstance(summary_plan, dict):
        summary_detail["skip_reason"] = "no_summary_plan"
        audit_log["summary_detail"] = summary_detail
        return
    rewrite_intent = summary_plan.get("rewrite_intent")
    if isinstance(rewrite_intent, str):
        summary_detail["rewrite_intent"] = rewrite_intent
    target_keywords = summary_plan.get("target_keywords") if isinstance(summary_plan.get("target_keywords"), list) else []

    summary = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else None
    if summary is None:
        summary_detail["skip_reason"] = "missing_summary"
        audit_log["summary_detail"] = summary_detail
        return
    original_text = summary.get("text")
    if not isinstance(original_text, str) or original_text.strip() == "":
        summary_detail["skip_reason"] = "missing_summary_text"
        audit_log["summary_detail"] = summary_detail
        return
    summary_detail["original_text"] = original_text
    summary_policy = _build_summary_ats_policy(
        source_resume_json,
        job_json,
        score_result,
        tailoring_plan,
        original_text,
    )
    if summary_policy is not None:
        summary_detail["target_keywords"] = list(summary_policy.requested_target_keywords)
        summary_detail["preferred_surface_terms"] = list(summary_policy.preferred_surface_terms)
        summary_detail["required_terms"] = list(summary_policy.required_terms)
        summary_detail["supported_priority_terms"] = list(summary_policy.supported_priority_terms)
        summary_detail["recent_priority_terms"] = list(summary_policy.recent_priority_terms)
        summary_detail["allowed_title_terms"] = list(summary_policy.safe_title_terms)
        summary_detail["blocked_terms"] = list(summary_policy.blocked_terms)
        summary_detail["blocked_title_terms"] = list(summary_policy.unsafe_title_terms)
    else:
        summary_detail["target_keywords"] = list(target_keywords)

    budget = budgets.get("summary")
    rewritten_text, reject_reason, rejected_terms = _call_summary_rewrite(
        original_text,
        job_json,
        summary_policy,
        target_keywords,
        allowed_vocab,
        provider,
        budget,
        force_change=True,
    )
    summary_detail["candidate_text"] = rewritten_text
    if reject_reason:
        summary_detail["reject_reason"] = reject_reason
    if rejected_terms:
        summary_detail["disallowed_terms"] = list(rejected_terms)
    if _needs_summary_fallback(original_text, rewritten_text, reject_reason):
        summary_detail["fallback_used"] = True
        summary_detail["fallback_reason"] = _fallback_reason(original_text, rewritten_text, reject_reason)
        fallback_text = _fallback_summary_rewrite_with_context(
            original_text=original_text,
            candidate_text=rewritten_text,
            reject_reason=reject_reason,
            rejected_terms=rejected_terms or [],
            summary_policy=summary_policy,
            allowed_vocab=allowed_vocab,
            target_keywords=target_keywords,
        )
        if fallback_text is None or _is_effectively_same(original_text, fallback_text):
            summary_detail["skip_reason"] = "fallback_failed"
            audit_log["summary_detail"] = summary_detail
            return
        rewritten_text = fallback_text
        summary_detail["candidate_text"] = rewritten_text

    if rewritten_text is None:
        summary_detail["skip_reason"] = "llm_no_rewrite_or_rejected"
        audit_log["summary_detail"] = summary_detail
        return
    reject_reason_after, rejected_terms_after = _validate_summary_candidate(
        original_text,
        rewritten_text,
        summary_policy,
        allowed_vocab,
        target_keywords,
    )
    if reject_reason_after:
        summary_detail["reject_reason"] = reject_reason_after
        summary_detail["disallowed_terms"] = list(rejected_terms_after)
        audit_log["summary_detail"] = summary_detail
        return
    if budget is not None and len(rewritten_text) > budget:
        preserve_terms = summary_policy.required_terms if summary_policy is not None else ()
        blocked_terms = ()
        emphasis_terms = ()
        avoid_terms = ()
        if summary_policy is not None:
            blocked_terms = summary_policy.blocked_terms + summary_policy.unsafe_title_terms
            emphasis_terms = summary_policy.preferred_surface_terms
            avoid_terms = summary_policy.avoid_terms
        compressed = _compress_text(
            original_text,
            rewritten_text,
            budget,
            provider,
            preserve_terms=preserve_terms,
            blocked_terms=blocked_terms,
            emphasis_terms=emphasis_terms,
            avoid_terms=avoid_terms,
        )
        if compressed is not None:
            rewritten_text = compressed
            reject_reason_after, rejected_terms_after = _validate_summary_candidate(
                original_text,
                rewritten_text,
                summary_policy,
                allowed_vocab,
                target_keywords,
            )
            if reject_reason_after:
                summary_detail["reject_reason"] = f"{reject_reason_after}_after_compress"
                summary_detail["disallowed_terms"] = list(rejected_terms_after)
                audit_log["summary_detail"] = summary_detail
                return
            _append_unique(audit_log["compressed"], "summary")
        if budget is not None and len(rewritten_text) > budget:
            rewritten_text = _truncate_to_budget(rewritten_text, budget)
            reject_reason_after, rejected_terms_after = _validate_summary_candidate(
                original_text,
                rewritten_text,
                summary_policy,
                allowed_vocab,
                target_keywords,
            )
            if reject_reason_after:
                summary_detail["reject_reason"] = f"{reject_reason_after}_after_budget"
                summary_detail["disallowed_terms"] = list(rejected_terms_after)
                audit_log["summary_detail"] = summary_detail
                return
    if _is_effectively_same(original_text, rewritten_text):
        summary_detail["skip_reason"] = "final_identical"
        audit_log["summary_detail"] = summary_detail
        return
    summary["text"] = rewritten_text
    summary_detail["final_text"] = rewritten_text
    summary_detail["changed"] = rewritten_text != original_text
    audit_log["summary_detail"] = summary_detail


def _call_summary_rewrite(
    original_text: str,
    job_json: Dict[str, Any],
    summary_policy: Optional[SummaryATSRewritePolicy],
    target_keywords: Sequence[str],
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
    force_change: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    requested_target_keywords = (
        list(summary_policy.requested_target_keywords)
        if summary_policy is not None
        else list(_normalize_ordered_terms(target_keywords))
    )
    preferred_surface_terms = list(summary_policy.preferred_surface_terms) if summary_policy is not None else []
    allowed_target, missing_target = _split_target_keywords(requested_target_keywords, allowed_terms)
    allowed_surface, missing_surface = _split_target_keywords(preferred_surface_terms, allowed_terms)
    payload = {
        "original_text": original_text,
        "job_title": job_json.get("title"),
        "target_keywords": requested_target_keywords,
        "allowed_target_keywords": allowed_target,
        "missing_target_keywords": missing_target,
        "preferred_surface_terms": preferred_surface_terms,
        "required_supported_terms": list(summary_policy.required_terms) if summary_policy is not None else [],
        "supported_priority_terms": list(summary_policy.supported_priority_terms) if summary_policy is not None else [],
        "recent_preferred_terms": list(summary_policy.recent_priority_terms) if summary_policy is not None else [],
        "allowed_title_terms": list(summary_policy.safe_title_terms) if summary_policy is not None else [],
        "blocked_title_terms": list(summary_policy.unsafe_title_terms) if summary_policy is not None else [],
        "blocked_terms": list(summary_policy.blocked_terms) if summary_policy is not None else [],
        "avoid_terms": list(summary_policy.avoid_terms) if summary_policy is not None else [],
        "title_alignment_safe": bool(summary_policy.title_alignment_safe) if summary_policy is not None else False,
        "allowed_surface_terms": allowed_surface,
        "missing_surface_terms": missing_surface,
        "allowed_terms": _select_allowed_terms(
            allowed_terms,
            original_text,
            requested_target_keywords + preferred_surface_terms,
        ),
        "allowed_proper_nouns": _select_allowed_proper_nouns(allowed_proper, original_text),
        "force_change": force_change,
    }
    if budget is not None:
        payload["max_chars"] = budget

    system_prompt = load_system_prompt("summary_rewrite")
    config = get_config()
    last_rewritten_text: Optional[str] = None
    for _ in range(2):
        messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="summary_rewrite")
        raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
        obj = _parse_llm_json(raw, "summary_rewrite", provider)
        if obj is None:
            return None, "invalid_response", None
        rewritten_text = obj.get("rewritten_text")
        if not isinstance(rewritten_text, str):
            return None, "invalid_text", None
        last_rewritten_text = rewritten_text

        reject_reason, rejected_terms = _validate_summary_candidate(
            original_text,
            rewritten_text,
            summary_policy,
            allowed_vocab,
            requested_target_keywords,
        )
        if reject_reason == "unsafe_title_alignment":
            payload["blocked_title_terms_found"] = rejected_terms
            payload["retry_instruction"] = (
                "Remove unsafe title-alignment phrasing. Only use allowed_title_terms when title_alignment_safe is true."
            )
            continue
        if reject_reason == "blocked_terms":
            payload["blocked_terms_found"] = rejected_terms
            payload["retry_instruction"] = "Remove the blocked summary terms. Preserve only supported ATS terms."
            continue
        if reject_reason == "disallowed_terms":
            payload["disallowed_terms"] = rejected_terms
            payload["retry_instruction"] = "Remove the disallowed terms. Do not replace them with new terms."
            continue
        if reject_reason == "unsupported_ats_terms":
            payload["unsupported_ats_terms"] = rejected_terms
            payload["retry_instruction"] = (
                "Remove ATS-significant terms that are not supported for the summary. Keep only preferred_surface_terms and safe title terms."
            )
            continue
        if reject_reason == "missing_required_summary_terms":
            payload["missing_required_terms"] = rejected_terms
            payload["retry_instruction"] = "Preserve at least one required supported ATS signal in the rewritten summary."
            continue

        return rewritten_text, None, None

    if isinstance(payload.get("blocked_title_terms_found"), list) and payload["blocked_title_terms_found"]:
        return last_rewritten_text, "unsafe_title_alignment", list(payload["blocked_title_terms_found"])
    if isinstance(payload.get("blocked_terms_found"), list) and payload["blocked_terms_found"]:
        return last_rewritten_text, "blocked_terms", list(payload["blocked_terms_found"])
    if isinstance(payload.get("disallowed_terms"), list) and payload["disallowed_terms"]:
        return last_rewritten_text, "disallowed_terms", list(payload["disallowed_terms"])
    if isinstance(payload.get("unsupported_ats_terms"), list) and payload["unsupported_ats_terms"]:
        return last_rewritten_text, "unsupported_ats_terms", list(payload["unsupported_ats_terms"])
    if isinstance(payload.get("missing_required_terms"), list) and payload["missing_required_terms"]:
        return last_rewritten_text, "missing_required_summary_terms", list(payload["missing_required_terms"])
    return None, None, None


def _build_summary_ats_policy(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    original_text: str,
) -> Optional[SummaryATSRewritePolicy]:
    summary_plan = tailoring_plan.get("summary_rewrite")
    if not isinstance(summary_plan, dict):
        return None
    if score_result.get("decision") != "PROCEED":
        return None

    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    title_alignment = build_title_alignment(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    recency = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )

    title_status = tailoring_plan.get("title_alignment_status")
    title_alignment_safe = title_alignment.is_safe_for_summary_alignment
    if isinstance(title_status, dict) and isinstance(title_status.get("is_safe_for_summary_alignment"), bool):
        title_alignment_safe = bool(title_status.get("is_safe_for_summary_alignment"))
    if isinstance(summary_plan.get("title_alignment_safe"), bool):
        title_alignment_safe = bool(summary_plan.get("title_alignment_safe"))

    requested_target_keywords = tuple(_normalize_ordered_terms(summary_plan.get("target_keywords")))
    under_supported_terms = tuple(
        _normalize_ordered_terms(_extract_term_items(tailoring_plan.get("under_supported_terms")))
    )
    under_supported_set = frozenset(under_supported_terms)

    blocked_ordered: List[str] = []
    for term in _normalize_ordered_terms(summary_plan.get("blocked_terms")):
        _append_unique(blocked_ordered, term)
    for term in _normalize_ordered_terms(_extract_blocked_terms(tailoring_plan.get("blocked_terms"), "summary")):
        _append_unique(blocked_ordered, term)
    blocked_terms = tuple(blocked_ordered)
    blocked_term_set = frozenset(blocked_terms)

    raw_title_terms: List[str] = []
    for source_terms in (
        summary_plan.get("title_terms"),
        tailoring_plan.get("summary_alignment_terms"),
        title_status.get("supported_terms") if isinstance(title_status, dict) else None,
    ):
        for term in _ordered_prompt_terms(source_terms):
            _append_unique(raw_title_terms, term)
    normalized_title_terms = tuple(_normalize_ordered_terms(raw_title_terms))
    safe_title_terms = normalized_title_terms if title_alignment_safe else ()
    unsafe_title_terms = normalized_title_terms if not title_alignment_safe else ()

    raw_missing_title_terms = ()
    if isinstance(title_status, dict):
        raw_missing_title_terms = title_status.get("missing_tokens")
    title_missing_terms = tuple(
        _normalize_ordered_terms(raw_missing_title_terms if raw_missing_title_terms else title_alignment.missing_title_tokens)
    )

    supported_priority_terms = tuple(
        term
        for term in _normalize_ordered_terms(tailoring_plan.get("supported_priority_terms"))
        if _is_summary_surface_term(term, original_text, coverage, evidence_links, safe_title_terms, blocked_term_set, under_supported_set)
    ) or tuple(
        term
        for term in job_weights.ordered_terms
        if _is_summary_surface_term(term, original_text, coverage, evidence_links, safe_title_terms, blocked_term_set, under_supported_set)
    )
    recent_priority_terms = tuple(
        term
        for term in _normalize_ordered_terms(tailoring_plan.get("recent_priority_terms"))
        if _is_recent_summary_surface_term(term, original_text, coverage, evidence_links, recency, safe_title_terms, blocked_term_set, under_supported_set)
    ) or tuple(
        term
        for term in recency.recent_summary_safe_terms
        if _is_recent_summary_surface_term(term, original_text, coverage, evidence_links, recency, safe_title_terms, blocked_term_set, under_supported_set)
    )

    preferred_surface_terms: List[str] = []
    for term in safe_title_terms:
        _append_unique(preferred_surface_terms, term)
    for source_terms in (requested_target_keywords, recent_priority_terms, supported_priority_terms):
        for term in source_terms:
            if _is_summary_surface_term(
                term,
                original_text,
                coverage,
                evidence_links,
                safe_title_terms,
                blocked_term_set,
                under_supported_set,
            ):
                _append_unique(preferred_surface_terms, term)

    required_terms: List[str] = []
    for term in preferred_surface_terms:
        if _contains_canonical_term(original_text, term):
            _append_unique(required_terms, term)
    if not required_terms:
        for term in preferred_surface_terms[:3]:
            _append_unique(required_terms, term)

    significant_terms = frozenset(
        tuple(job_weights.ordered_terms)
        + requested_target_keywords
        + supported_priority_terms
        + recent_priority_terms
        + blocked_terms
        + under_supported_terms
        + normalized_title_terms
        + title_missing_terms
    )
    allowed_new_terms = frozenset(preferred_surface_terms)
    avoid_terms_list: List[str] = []
    for term in under_supported_terms + title_missing_terms + unsafe_title_terms:
        if term not in allowed_new_terms:
            _append_unique(avoid_terms_list, term)
    avoid_terms = tuple(avoid_terms_list)

    return SummaryATSRewritePolicy(
        requested_target_keywords=requested_target_keywords,
        preferred_surface_terms=tuple(preferred_surface_terms),
        safe_title_terms=tuple(safe_title_terms),
        unsafe_title_terms=tuple(unsafe_title_terms),
        supported_priority_terms=supported_priority_terms,
        recent_priority_terms=recent_priority_terms,
        required_terms=tuple(required_terms),
        blocked_terms=blocked_terms,
        avoid_terms=avoid_terms,
        allowed_new_terms=allowed_new_terms,
        blocked_term_set=blocked_term_set,
        significant_terms=significant_terms,
        title_missing_terms=title_missing_terms,
        title_alignment_safe=title_alignment_safe,
    )


def _is_summary_surface_term(
    term: str,
    original_text: str,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    safe_title_terms: Sequence[str],
    blocked_term_set: FrozenSet[str],
    under_supported_terms: FrozenSet[str],
) -> bool:
    canonical = canonicalize_term(term)
    if not canonical or canonical in blocked_term_set or canonical in under_supported_terms:
        return False
    if canonical in safe_title_terms:
        return True
    if _contains_canonical_term(original_text, canonical):
        return True
    coverage_entry = coverage.coverage_by_term.get(canonical)
    evidence_entry = evidence_links.links_by_term.get(canonical)
    return bool(
        evidence_entry
        and evidence_entry.is_safe_for_summary
        and evidence_entry.all_candidates
        and not (coverage_entry and coverage_entry.is_under_supported)
    )


def _is_recent_summary_surface_term(
    term: str,
    original_text: str,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    recency: ATSRecencyPriorities,
    safe_title_terms: Sequence[str],
    blocked_term_set: FrozenSet[str],
    under_supported_terms: FrozenSet[str],
) -> bool:
    canonical = canonicalize_term(term)
    if not _is_summary_surface_term(
        canonical,
        original_text,
        coverage,
        evidence_links,
        safe_title_terms,
        blocked_term_set,
        under_supported_terms,
    ):
        return False
    recency_entry = recency.priorities_by_term.get(canonical)
    return bool(recency_entry and recency_entry.is_recent_and_summary_safe)


def _validate_summary_candidate(
    original_text: str,
    rewritten_text: str,
    summary_policy: Optional[SummaryATSRewritePolicy],
    allowed_vocab: Dict[str, Any],
    target_keywords: Sequence[str],
) -> Tuple[Optional[str], List[str]]:
    if summary_policy is not None:
        unsafe_title_terms = _find_unsafe_summary_title_terms(original_text, rewritten_text, summary_policy)
        if unsafe_title_terms:
            return "unsafe_title_alignment", unsafe_title_terms

        blocked_terms = _find_blocked_terms_in_text(rewritten_text, summary_policy.blocked_terms)
        if blocked_terms:
            return "blocked_terms", blocked_terms

    requested_target_keywords = (
        summary_policy.requested_target_keywords if summary_policy is not None else tuple(target_keywords)
    )
    disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, requested_target_keywords)
    if disallowed:
        return "disallowed_terms", disallowed

    if summary_policy is not None:
        unsupported_terms = _find_unsupported_summary_terms(original_text, rewritten_text, summary_policy)
        if unsupported_terms:
            return "unsupported_ats_terms", unsupported_terms

        missing_required_terms = _find_missing_required_terms(rewritten_text, summary_policy.required_terms)
        if missing_required_terms:
            return "missing_required_summary_terms", missing_required_terms

    return None, []


def _find_unsafe_summary_title_terms(
    original_text: str,
    rewritten_text: str,
    summary_policy: SummaryATSRewritePolicy,
) -> List[str]:
    if summary_policy.title_alignment_safe or not summary_policy.unsafe_title_terms:
        return []
    hits: List[str] = []
    for term in summary_policy.unsafe_title_terms:
        if _contains_canonical_term(rewritten_text, term) and not _contains_canonical_term(original_text, term):
            _append_unique(hits, term)
    return hits


def _find_unsupported_summary_terms(
    original_text: str,
    rewritten_text: str,
    summary_policy: SummaryATSRewritePolicy,
) -> List[str]:
    original_terms = _extract_canonical_terms(original_text)
    rewritten_terms = _extract_canonical_terms(rewritten_text)
    introduced_terms = sorted(rewritten_terms - original_terms)
    unsupported_terms = [
        term
        for term in introduced_terms
        if term in summary_policy.significant_terms
        and term not in summary_policy.allowed_new_terms
        and term not in summary_policy.blocked_term_set
        and not _is_covered_by_allowed_summary_phrase(term, rewritten_text, summary_policy.allowed_new_terms)
    ]

    original_guard_terms = set(_extract_tool_like_terms(original_text))
    rewritten_guard_terms = set(_extract_tool_like_terms(rewritten_text))
    introduced_guard_terms: List[str] = []
    for term in sorted(rewritten_guard_terms - original_guard_terms):
        canonical = canonicalize_term(term) or term
        if canonical in summary_policy.allowed_new_terms:
            continue
        if _is_covered_by_allowed_summary_phrase(canonical, rewritten_text, summary_policy.allowed_new_terms):
            continue
        _append_unique(introduced_guard_terms, canonical)

    unsupported: List[str] = []
    for term in unsupported_terms + introduced_guard_terms:
        _append_unique(unsupported, term)
    return unsupported


def _is_covered_by_allowed_summary_phrase(
    term: str,
    rewritten_text: str,
    allowed_new_terms: FrozenSet[str],
) -> bool:
    for allowed_term in allowed_new_terms:
        if " " not in allowed_term:
            continue
        allowed_tokens = allowed_term.split()
        if term in allowed_tokens and _contains_canonical_term(rewritten_text, allowed_term):
            return True
    return False


def _reorder_skills(tailored: Dict[str, Any], tailoring_plan: Dict[str, Any]) -> None:
    # Skills line order must remain unchanged for DOCX mapping integrity.
    # Tailoring should only adjust text within each line (handled elsewhere).
    _ = tailoring_plan
    return


def _tailor_skill_lines(
    tailored: Dict[str, Any],
    source_resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    del allowed_vocab
    del score_result
    skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else None
    if skills is None:
        return
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    if not lines:
        return
    original_text_by_id = {
        line.get("line_id"): line.get("text")
        for line in lines
        if isinstance(line, dict) and isinstance(line.get("line_id"), str) and isinstance(line.get("text"), str)
    }
    policy = _build_skills_ats_policy(source_resume_json, job_json, tailoring_plan, budgets)

    line_states: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        original_text = line.get("text")
        if not isinstance(line_id, str) or not isinstance(original_text, str):
            continue
        detail = {
            "line_id": line_id,
            "original_text": original_text,
            "target_keywords": [],
            "final_text": original_text,
            "changed": False,
            "skip_reason": None,
            "reject_reason": None,
        }
        segments, separator = _split_skills_line(original_text)
        optimized_infos = _optimize_skill_line_segments(segments, policy)
        reordered = [info["text"] for info in optimized_infos]
        if reordered == segments:
            detail["skip_reason"] = "no_reorder_needed"
        else:
            detail["skip_reason"] = "ats_priority_reorder"
            detail["target_keywords"] = [info["primary_term"] for info in optimized_infos if info.get("is_target_term")]
        new_text = separator.join(reordered)
        line["text"] = new_text
        detail["final_text"] = new_text
        detail["changed"] = new_text != original_text
        audit_log["skills_details"].append(detail)
        line_states.append(
            {
                "line": line,
                "line_id": line_id,
                "original_text": original_text,
                "segments": reordered,
                "infos": optimized_infos,
                "separator": separator,
                "budget": policy.line_budget_by_id.get(line_id, len(original_text)),
            }
        )

    _surface_missing_priority_skill_terms(line_states, policy, audit_log)

    validation_errors = _validate_skills_optimization(
        original_text_by_id=original_text_by_id,
        lines=lines,
        policy=policy,
    )
    if validation_errors:
        _restore_skill_lines_after_validation_failure(line_states, original_text_by_id, validation_errors, audit_log)


def _build_skills_ats_policy(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    budgets: Dict[str, Any],
) -> SkillsATSOptimizationPolicy:
    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    title_alignment = build_title_alignment(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    recency = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )

    supported_priority_terms = tuple(
        term
        for term in _normalize_ordered_terms(tailoring_plan.get("supported_priority_terms"))
        if _is_supported_skill_term(term, coverage, evidence_links)
    ) or tuple(
        term
        for term in job_weights.ordered_terms
        if _is_supported_skill_term(term, coverage, evidence_links)
    )
    skill_priority_terms = tuple(
        term
        for term in _normalize_ordered_terms(tailoring_plan.get("skill_priority_terms"))
        if _is_skill_surface_term(term, evidence_links)
    ) or tuple(term for term in job_weights.ordered_terms if _is_skill_surface_term(term, evidence_links))
    recent_priority_terms = tuple(
        term
        for term in _normalize_ordered_terms(tailoring_plan.get("recent_priority_terms"))
        if _is_skill_surface_term(term, evidence_links) and recency.priorities_by_term[term].has_recent_backing
    ) or tuple(
        term
        for term in recency.recency_ordered_terms
        if _is_skill_surface_term(term, evidence_links) and recency.priorities_by_term[term].has_recent_backing
    )
    under_supported_terms = frozenset(
        _normalize_ordered_terms(_extract_term_items(tailoring_plan.get("under_supported_terms")))
        or [
            term
            for term in job_weights.ordered_terms
            if coverage.coverage_by_term[term].is_under_supported
        ]
    )
    blocked_terms = tuple(
        _normalize_ordered_terms(_extract_blocked_terms(tailoring_plan.get("blocked_terms"), "skills"))
    )
    allowed_surface_terms = frozenset(
        term
        for term in skill_priority_terms
        if term not in blocked_terms and term not in under_supported_terms
    )
    preferred_line_ids = _preferred_skill_line_ids(resume_json, tailoring_plan)
    line_budget_by_id = _skill_line_budgets(resume_json, budgets)

    term_candidates = set(resume_signals.skill_terms)
    term_candidates.update(job_weights.ordered_terms)
    term_candidates.update(supported_priority_terms)
    term_candidates.update(skill_priority_terms)
    term_candidates.update(recent_priority_terms)
    term_candidates.update(blocked_terms)

    resume_texts = list(_iter_resume_texts(resume_json))
    priority_rank = {term: index for index, term in enumerate(skill_priority_terms)}
    supported_rank = {term: index for index, term in enumerate(supported_priority_terms)}
    recent_rank = {term: index for index, term in enumerate(recent_priority_terms)}

    term_policies: Dict[str, SkillTermPolicy] = {}
    for term in sorted(term_candidates):
        term_policies[term] = _build_skill_term_policy(
            term=term,
            resume_texts=resume_texts,
            resume_signals=resume_signals,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            recency=recency,
            priority_rank=priority_rank.get(term, -1),
            supported_rank=supported_rank.get(term, -1),
            recent_rank=recent_rank.get(term, -1),
            supported_priority_terms=set(supported_priority_terms),
            skill_priority_terms=set(skill_priority_terms),
            recent_priority_terms=set(recent_priority_terms),
            under_supported_terms=under_supported_terms,
            blocked_terms=set(blocked_terms),
            allowed_surface_terms=allowed_surface_terms,
        )

    return SkillsATSOptimizationPolicy(
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        recency=recency,
        supported_priority_terms=supported_priority_terms,
        skill_priority_terms=skill_priority_terms,
        recent_priority_terms=recent_priority_terms,
        under_supported_terms=under_supported_terms,
        blocked_terms=blocked_terms,
        allowed_surface_terms=allowed_surface_terms,
        preferred_line_ids=preferred_line_ids,
        line_budget_by_id=line_budget_by_id,
        term_policies=term_policies,
    )


def _build_skill_term_policy(
    *,
    term: str,
    resume_texts: Sequence[str],
    resume_signals: ResumeSignals,
    job_weights: JobWeights,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
    recency: ATSRecencyPriorities,
    priority_rank: int,
    supported_rank: int,
    recent_rank: int,
    supported_priority_terms: Set[str],
    skill_priority_terms: Set[str],
    recent_priority_terms: Set[str],
    under_supported_terms: FrozenSet[str],
    blocked_terms: Set[str],
    allowed_surface_terms: FrozenSet[str],
) -> SkillTermPolicy:
    weight_entry = job_weights.weights_by_term.get(term)
    coverage_entry = coverage.coverage_by_term.get(term)
    evidence_entry = evidence_links.links_by_term.get(term)
    recency_entry = recency.priorities_by_term.get(term)

    display = _preferred_skill_display(term, resume_texts, resume_signals, evidence_entry)
    support_score = 0
    has_recent_backing = False
    has_cross_section_backing = False
    has_experience_backing = False
    has_project_backing = False
    is_skills_only = False
    missing_experience_backing = False
    if evidence_entry is not None:
        support_score = (
            evidence_entry.strongest_candidate.support_score
            if evidence_entry.strongest_candidate is not None
            else 0
        )
        has_experience_backing = evidence_entry.has_experience_backing
        has_project_backing = evidence_entry.has_project_backing
        is_skills_only = term in evidence_links.skills_only_terms
        missing_experience_backing = evidence_entry.missing_experience_backing
    if coverage_entry is not None:
        has_cross_section_backing = coverage_entry.has_cross_section_support
    if recency_entry is not None:
        has_recent_backing = recency_entry.has_recent_backing

    return SkillTermPolicy(
        term=term,
        display=display,
        weight=weight_entry.total_weight if weight_entry is not None else 0,
        support_score=support_score,
        priority_rank=priority_rank,
        supported_rank=supported_rank,
        recent_rank=recent_rank,
        has_recent_backing=has_recent_backing,
        has_cross_section_backing=has_cross_section_backing,
        has_experience_backing=has_experience_backing,
        has_project_backing=has_project_backing,
        is_supported_priority=term in supported_priority_terms,
        is_skill_priority=term in skill_priority_terms,
        is_recent_priority=term in recent_priority_terms,
        is_under_supported=term in under_supported_terms or bool(coverage_entry and coverage_entry.is_under_supported),
        is_blocked=term in blocked_terms,
        is_allowed_surface=term in allowed_surface_terms,
        is_skills_only=is_skills_only,
        missing_experience_backing=missing_experience_backing,
    )


def _is_supported_skill_term(term: str, coverage: ResumeCoverage, evidence_links: ResumeEvidenceLinks) -> bool:
    coverage_entry = coverage.coverage_by_term.get(term)
    evidence_entry = evidence_links.links_by_term.get(term)
    if coverage_entry is None or evidence_entry is None:
        return False
    return bool(
        evidence_entry.is_safe_for_skills
        and not coverage_entry.is_under_supported
        and evidence_entry.all_candidates
    )


def _is_skill_surface_term(term: str, evidence_links: ResumeEvidenceLinks) -> bool:
    evidence_entry = evidence_links.links_by_term.get(term)
    return bool(evidence_entry and evidence_entry.is_safe_for_skills and evidence_entry.all_candidates)


def _preferred_skill_line_ids(
    resume_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
) -> Tuple[str, ...]:
    original_ids = _collect_skill_line_ids(resume_json)
    preferred: List[str] = []
    plan_ids = tailoring_plan.get("skills_reorder_plan")
    if isinstance(plan_ids, list):
        for line_id in plan_ids:
            if isinstance(line_id, str) and line_id in original_ids and line_id not in preferred:
                preferred.append(line_id)
    for line_id in original_ids:
        if line_id not in preferred:
            preferred.append(line_id)
    return tuple(preferred)


def _collect_skill_line_ids(resume_json: Dict[str, Any]) -> List[str]:
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


def _skill_line_budgets(
    resume_json: Dict[str, Any],
    budgets: Dict[str, Any],
) -> Dict[str, int]:
    budget_by_id = budgets.get("skills_line_max_chars")
    effective: Dict[str, int] = {}
    if isinstance(budget_by_id, dict):
        for line_id, value in budget_by_id.items():
            if isinstance(line_id, str) and isinstance(value, int) and value > 0:
                effective[line_id] = value
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if isinstance(line_id, str) and isinstance(text, str):
            effective.setdefault(line_id, len(text))
    return effective


def _preferred_skill_display(
    term: str,
    resume_texts: Sequence[str],
    resume_signals: ResumeSignals,
    evidence_entry: Any,
) -> str:
    exact_display = _find_display_in_texts(term, resume_texts)
    if exact_display:
        return exact_display

    if term in _SKILL_DISPLAY_OVERRIDES:
        return _SKILL_DISPLAY_OVERRIDES[term]

    if evidence_entry is not None and evidence_entry.strongest_candidate is not None:
        raw_term = evidence_entry.strongest_candidate.raw_term
        if isinstance(raw_term, str):
            raw_display = _find_display_in_texts(raw_term, resume_texts)
            if raw_display and canonicalize_term(raw_display) == term and raw_display != raw_term:
                return raw_display

    for variant in resume_signals.term_variants.get(term, ()):
        variant_display = _find_display_in_texts(variant, resume_texts)
        if variant_display and canonicalize_term(variant_display) == term and variant_display == term:
            return variant_display

    return _canonical_skill_display(term)


def _canonical_skill_display(term: str) -> str:
    if term in _SKILL_DISPLAY_OVERRIDES:
        return _SKILL_DISPLAY_OVERRIDES[term]
    if any(ch in term for ch in ".#/"):
        return term
    words = []
    for part in term.split():
        if len(part) <= 3 and part.isalpha():
            words.append(part.upper())
        else:
            words.append(part.capitalize())
    return " ".join(words).strip()


def _optimize_skill_line_segments(
    segments: Sequence[str],
    policy: SkillsATSOptimizationPolicy,
) -> List[Dict[str, Any]]:
    infos = _build_skill_segment_infos(segments, policy)
    return sorted(
        infos,
        key=lambda info: (-info["score"], info["original_index"], normalize_text(info["text"])),
    )


def _build_skill_segment_infos(
    segments: Sequence[str],
    policy: SkillsATSOptimizationPolicy,
) -> List[Dict[str, Any]]:
    infos = [_analyze_skill_segment(segment, policy, index) for index, segment in enumerate(segments)]
    counts: Dict[str, int] = {}
    for info in infos:
        primary_term = info.get("primary_term")
        if isinstance(primary_term, str) and primary_term:
            counts[primary_term] = counts.get(primary_term, 0) + 1
    for info in infos:
        primary_term = info.get("primary_term")
        info["score"] = _score_skill_segment(
            info,
            policy,
            duplicate_count=counts.get(primary_term, 0) if isinstance(primary_term, str) else 0,
        )
    return infos


def _analyze_skill_segment(
    segment: str,
    policy: SkillsATSOptimizationPolicy,
    original_index: int,
) -> Dict[str, Any]:
    stripped = segment.strip()
    candidate_terms: List[str] = []
    canonical_segment = canonicalize_term(stripped)
    if canonical_segment:
        candidate_terms.append(canonical_segment)
    for canonical, _ in extract_canonical_term_pairs(stripped):
        if canonical and canonical not in candidate_terms:
            candidate_terms.append(canonical)

    primary_term = ""
    for term in candidate_terms:
        if term in policy.term_policies:
            primary_term = term
            break
    if not primary_term and candidate_terms:
        primary_term = candidate_terms[0]

    normalized = normalize_text(stripped)
    is_soft = normalized in _SOFT_SKILL_TERMS or primary_term in _SOFT_SKILL_TERMS
    is_hard = bool(
        _segment_has_tool_like_token(stripped)
        or _segment_has_technical_indicator(stripped)
        or (primary_term and (_term_is_tool_like(primary_term) or _term_has_technical_indicator(primary_term)))
    )
    term_policy = policy.term_policies.get(primary_term) if primary_term else None
    is_blocked = bool(term_policy and term_policy.is_blocked)
    is_unsupported_ats = any(
        term in policy.job_weights.weights_by_term
        and term not in policy.allowed_surface_terms
        and term not in policy.under_supported_terms
        and term not in policy.blocked_terms
        for term in candidate_terms
    )
    display_text = _preferred_skill_segment_display(
        stripped,
        primary_term,
        candidate_terms,
        policy,
        is_unsupported_ats,
    )
    is_target_term = bool(
        term_policy
        and (
            term_policy.is_supported_priority
            or term_policy.is_skill_priority
            or term_policy.is_recent_priority
        )
    )
    return {
        "source_text": stripped,
        "text": display_text,
        "normalized_text": normalize_text(display_text),
        "primary_term": primary_term,
        "canonical_terms": tuple(candidate_terms),
        "is_soft": is_soft,
        "is_hard": is_hard,
        "is_blocked": is_blocked,
        "is_under_supported": bool(term_policy and term_policy.is_under_supported),
        "is_unsupported_ats": is_unsupported_ats,
        "is_target_term": is_target_term,
        "is_canonicalized_variant": normalize_text(display_text) != normalize_text(stripped),
        "original_index": original_index,
    }


def _preferred_skill_segment_display(
    original_segment: str,
    primary_term: str,
    candidate_terms: Sequence[str],
    policy: SkillsATSOptimizationPolicy,
    is_unsupported_ats: bool,
) -> str:
    if not primary_term or is_unsupported_ats:
        return original_segment
    term_policy = policy.term_policies.get(primary_term)
    if term_policy is None or term_policy.is_blocked:
        return original_segment
    if canonicalize_term(original_segment) != normalize_text(original_segment):
        return term_policy.display
    if term_policy.is_supported_priority or term_policy.is_skill_priority:
        return term_policy.display
    if len(candidate_terms) > 1 and candidate_terms[0] != primary_term:
        return term_policy.display
    return original_segment


def _score_skill_segment(
    info: Dict[str, Any],
    policy: SkillsATSOptimizationPolicy,
    duplicate_count: int,
) -> int:
    if info.get("is_blocked"):
        return -100000
    if info.get("is_unsupported_ats"):
        return -50000

    primary_term = info.get("primary_term")
    term_policy = policy.term_policies.get(primary_term) if isinstance(primary_term, str) and primary_term else None
    if term_policy is not None:
        if term_policy.is_supported_priority:
            score = 3000
        elif term_policy.is_allowed_surface:
            score = 2600
        elif term_policy.is_under_supported:
            score = 1400
        elif info.get("is_hard"):
            score = 1700
        elif info.get("is_soft"):
            score = 800
        else:
            score = 900

        score += term_policy.weight * 10
        score += term_policy.support_score
        if term_policy.has_recent_backing or term_policy.is_recent_priority:
            score += 300
        if term_policy.has_cross_section_backing:
            score += 150
        if term_policy.has_experience_backing:
            score += 100
        if term_policy.has_project_backing:
            score += 75
        if term_policy.is_skills_only:
            score -= 200
        if term_policy.missing_experience_backing:
            score -= 125
        if term_policy.priority_rank >= 0:
            score += max(0, 180 - (term_policy.priority_rank * 4))
        if term_policy.supported_rank >= 0:
            score += max(0, 220 - (term_policy.supported_rank * 6))
        if term_policy.recent_rank >= 0:
            score += max(0, 160 - (term_policy.recent_rank * 8))
    else:
        if info.get("is_hard"):
            score = 1700
        elif info.get("is_soft"):
            score = 800
        else:
            score = 900

    if duplicate_count > 1:
        score -= 250
    return score


def _surface_missing_priority_skill_terms(
    line_states: List[Dict[str, Any]],
    policy: SkillsATSOptimizationPolicy,
    audit_log: Dict[str, Any],
) -> None:
    if not line_states:
        return

    state_by_id = {
        state["line_id"]: state
        for state in line_states
        if isinstance(state.get("line_id"), str)
    }
    present_terms = {
        info["primary_term"]
        for state in line_states
        for info in state.get("infos", [])
        if isinstance(info, dict) and isinstance(info.get("primary_term"), str) and info.get("primary_term")
    }

    for term in _surface_priority_skill_order(policy):
        if term in present_terms:
            continue
        term_policy = policy.term_policies.get(term)
        if term_policy is None or not term_policy.is_allowed_surface or term_policy.is_under_supported:
            continue
        for line_id in policy.preferred_line_ids:
            state = state_by_id.get(line_id)
            if state is None:
                continue
            if _surface_priority_skill_in_line(state, term_policy, policy):
                _update_skills_detail(audit_log, state["line_id"], state["line"]["text"], "priority_skill_surface")
                present_terms.add(term)
                break


def _surface_priority_skill_order(policy: SkillsATSOptimizationPolicy) -> Tuple[str, ...]:
    ordered: List[str] = []
    for collection in (
        policy.supported_priority_terms,
        policy.recent_priority_terms,
        policy.skill_priority_terms,
    ):
        for term in collection:
            term_policy = policy.term_policies.get(term)
            if term_policy is None:
                continue
            coverage_entry = policy.coverage.coverage_by_term.get(term)
            if coverage_entry is not None and coverage_entry.is_title_term:
                continue
            if term_policy.is_blocked or term_policy.is_under_supported or not term_policy.is_allowed_surface:
                continue
            _append_unique(ordered, term)
    return tuple(ordered)


def _surface_priority_skill_in_line(
    state: Dict[str, Any],
    term_policy: SkillTermPolicy,
    policy: SkillsATSOptimizationPolicy,
) -> bool:
    infos = state.get("infos")
    separator = state.get("separator")
    budget = state.get("budget")
    if not isinstance(infos, list) or not isinstance(separator, str) or not isinstance(budget, int):
        return False
    if state.get("surface_locked"):
        return False
    if any(info.get("primary_term") == term_policy.term for info in infos if isinstance(info, dict)):
        return False

    current_segments = [info["text"] for info in infos if isinstance(info, dict)]
    if len(current_segments) < 3:
        candidate_infos = _optimize_skill_line_segments(current_segments + [term_policy.display], policy)
        candidate_segments = [info["text"] for info in candidate_infos]
        candidate_text = separator.join(candidate_segments)
        if len(candidate_text) <= budget:
            _apply_skill_line_candidate(state, candidate_infos, candidate_text)
            state["surface_locked"] = True
            return True

    candidate_entry = _build_skill_segment_infos([term_policy.display], policy)[0]
    for index in _replacement_indexes_for_skill_line(infos):
        existing_info = infos[index]
        if not isinstance(existing_info, dict):
            continue
        if _should_preserve_skill_segment(existing_info, policy):
            continue
        if existing_info.get("score", 0) >= candidate_entry.get("score", 0):
            continue
        replacement_segments = list(current_segments)
        replacement_segments[index] = term_policy.display
        replacement_infos = _optimize_skill_line_segments(replacement_segments, policy)
        replacement_text = separator.join(info["text"] for info in replacement_infos)
        if len(replacement_text) > budget:
            continue
        if term_policy.term not in {
            info.get("primary_term") for info in replacement_infos if isinstance(info, dict)
        }:
            continue
        _apply_skill_line_candidate(state, replacement_infos, replacement_text)
        state["surface_locked"] = True
        return True
    return False


def _apply_skill_line_candidate(
    state: Dict[str, Any],
    candidate_infos: List[Dict[str, Any]],
    candidate_text: str,
) -> None:
    state["infos"] = candidate_infos
    state["segments"] = [info["text"] for info in candidate_infos]
    state["line"]["text"] = candidate_text


def _replacement_indexes_for_skill_line(infos: Sequence[Dict[str, Any]]) -> List[int]:
    indexed: List[Tuple[Tuple[int, int, int], int]] = []
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            continue
        if info.get("is_blocked") or info.get("is_unsupported_ats"):
            priority = 0
        elif info.get("is_canonicalized_variant") and info.get("is_target_term"):
            priority = 5
        elif info.get("is_under_supported"):
            priority = 1
        elif info.get("is_soft"):
            priority = 2
        elif not info.get("is_hard"):
            priority = 3
        else:
            priority = 4
        indexed.append(((priority, int(info.get("score", 0)), -index), index))
    indexed.sort()
    return [index for _, index in indexed]


def _should_preserve_skill_segment(
    info: Dict[str, Any],
    policy: SkillsATSOptimizationPolicy,
) -> bool:
    primary_term = info.get("primary_term")
    if not isinstance(primary_term, str) or not primary_term:
        return False
    term_policy = policy.term_policies.get(primary_term)
    if term_policy is None:
        return False
    return bool(
        term_policy.is_supported_priority
        or (
            term_policy.is_allowed_surface
            and (term_policy.is_recent_priority or term_policy.has_recent_backing)
        )
    )


def _validate_skills_optimization(
    *,
    original_text_by_id: Dict[str, str],
    lines: List[Any],
    policy: SkillsATSOptimizationPolicy,
) -> List[str]:
    errors: List[str] = []

    final_ids = [
        line.get("line_id")
        for line in lines
        if isinstance(line, dict) and isinstance(line.get("line_id"), str)
    ]
    original_ids = list(original_text_by_id.keys())
    if len(lines) != len(original_ids):
        errors.append("skills.lines count changed")
        return errors
    if final_ids != original_ids:
        errors.append("skills.lines order or ids changed")

    original_counts = _collect_skill_term_counts(original_text_by_id.values())
    final_text_by_id: Dict[str, str] = {}
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if not isinstance(line_id, str) or not isinstance(text, str):
            continue
        final_text_by_id[line_id] = text
        budget = policy.line_budget_by_id.get(line_id)
        if isinstance(budget, int) and len(text) > budget:
            errors.append(f"skills line {line_id} exceeds budget")

    final_counts = _collect_skill_term_counts(final_text_by_id.values())
    introduced_terms = sorted(
        term for term, count in final_counts.items() if count > original_counts.get(term, 0)
    )
    blocked_introduced = sorted(term for term in introduced_terms if term in policy.blocked_terms)
    if blocked_introduced:
        errors.append(f"blocked skill terms introduced: {', '.join(blocked_introduced)}")

    unsupported_introduced = sorted(
        term
        for term in introduced_terms
        if term in policy.job_weights.weights_by_term
        and term not in policy.allowed_surface_terms
        and term not in policy.under_supported_terms
    )
    if unsupported_introduced:
        errors.append(f"unsupported ATS skill terms introduced: {', '.join(unsupported_introduced)}")

    for term, count in final_counts.items():
        if count <= 1:
            continue
        if count > original_counts.get(term, 0):
            errors.append(f"duplicate canonical skill variants introduced: {term}")

    for line_id, final_text in final_text_by_id.items():
        original_text = original_text_by_id.get(line_id, "")
        if final_text == original_text:
            continue
        segments, _ = _split_skills_line(final_text)
        optimized_segments = [info["text"] for info in _optimize_skill_line_segments(segments, policy)]
        if optimized_segments != segments:
            errors.append(f"skills line {line_id} violates ATS ordering")

    return errors


def _collect_skill_term_counts(texts: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        if not isinstance(text, str):
            continue
        segments, _ = _split_skills_line(text)
        for segment in segments:
            canonical = canonicalize_term(segment)
            if canonical:
                counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def _restore_skill_lines_after_validation_failure(
    line_states: Sequence[Dict[str, Any]],
    original_text_by_id: Dict[str, str],
    validation_errors: Sequence[str],
    audit_log: Dict[str, Any],
) -> None:
    for state in line_states:
        line = state.get("line")
        line_id = state.get("line_id")
        if not isinstance(line, dict) or not isinstance(line_id, str):
            continue
        original_text = original_text_by_id.get(line_id)
        if not isinstance(original_text, str):
            continue
        line["text"] = original_text
        _reject_skills_detail(audit_log, line_id, original_text, validation_errors)


def _reject_skills_detail(
    audit_log: Dict[str, Any],
    line_id: str,
    final_text: str,
    validation_errors: Sequence[str],
) -> None:
    details = audit_log.get("skills_details")
    if not isinstance(details, list):
        return
    for detail in reversed(details):
        if isinstance(detail, dict) and detail.get("line_id") == line_id:
            detail["final_text"] = final_text
            detail["changed"] = False
            detail["skip_reason"] = "skills_validation_rejected"
            detail["reject_reason"] = "skills_validation"
            detail["validation_errors"] = list(validation_errors)
            return


def _rewrite_bullets(
    tailored: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    action_map = _build_action_map(tailoring_plan)
    tailoring_ats_context = _build_tailoring_ats_context(tailoring_plan)
    bullet_budgets = budgets.get("bullets", {})

    experience = tailored.get("experience") if isinstance(tailored.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            _rewrite_bullet(
                bullet,
                action_map,
                tailoring_ats_context,
                job_json,
                provider,
                allowed_vocab,
                bullet_budgets,
                audit_log,
            )

    projects = tailored.get("projects") if isinstance(tailored.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            _rewrite_bullet(
                bullet,
                action_map,
                tailoring_ats_context,
                job_json,
                provider,
                allowed_vocab,
                bullet_budgets,
                audit_log,
            )


def _rewrite_bullet(
    bullet: Dict[str, Any],
    action_map: Dict[str, Dict[str, Any]],
    tailoring_ats_context: TailoringATSContext,
    job_json: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    bullet_budgets: Dict[str, int],
    audit_log: Dict[str, Any],
) -> None:
    bullet_id = bullet.get("bullet_id")
    original_text = bullet.get("text")
    if not isinstance(bullet_id, str) or not isinstance(original_text, str):
        return
    detail = {
        "bullet_id": bullet_id,
        "original_text": original_text,
        "rewrite_intent": None,
        "target_keywords": [],
        "evidence_terms": [],
        "source_section": None,
        "ats_emphasis": None,
        "candidate_text": None,
        "final_text": None,
        "skip_reason": None,
        "reject_reason": None,
        "changed": False,
    }

    action = action_map.get(bullet_id)
    if not isinstance(action, dict):
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "missing_action"
        _append_bullet_detail(audit_log, detail)
        return

    rewrite_intent = action.get("rewrite_intent")
    if isinstance(rewrite_intent, str):
        detail["rewrite_intent"] = rewrite_intent
    if isinstance(rewrite_intent, str) and rewrite_intent.strip().lower() in _REWRITE_KEEP:
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "rewrite_intent_keep"
        _append_bullet_detail(audit_log, detail)
        return

    target_keywords = action.get("target_keywords") if isinstance(action.get("target_keywords"), list) else []
    detail["target_keywords"] = list(target_keywords)
    detail["evidence_terms"] = list(action.get("evidence_terms")) if isinstance(action.get("evidence_terms"), list) else []
    detail["source_section"] = action.get("source_section") if isinstance(action.get("source_section"), str) else None
    if not target_keywords:
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "empty_target_keywords"
        _append_bullet_detail(audit_log, detail)
        return

    ats_policy = _build_bullet_ats_policy(bullet_id, original_text, action, tailoring_ats_context)
    detail["ats_emphasis"] = ats_policy.emphasis_strength
    budget = bullet_budgets.get(bullet_id)

    rewritten_text, reject_reason, rejected_terms = _call_bullet_rewrite(
        bullet_id,
        original_text,
        job_json,
        ats_policy,
        allowed_vocab,
        provider,
        budget,
    )
    if reject_reason:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = reject_reason
        if rejected_terms:
            detail["disallowed_terms"] = rejected_terms
        _append_bullet_detail(audit_log, detail)
        return
    if rewritten_text is None:
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "llm_no_rewrite_or_rejected"
        _append_bullet_detail(audit_log, detail)
        return

    did_shorten = False
    detail["candidate_text"] = rewritten_text
    if isinstance(budget, int) and len(rewritten_text) > budget:
        compressed = _compress_text(
            original_text,
            rewritten_text,
            budget,
            provider,
            preserve_terms=ats_policy.required_terms,
            blocked_terms=ats_policy.blocked_terms,
            emphasis_terms=ats_policy.surface_terms,
        )
        if compressed is not None:
            if len(compressed) < len(rewritten_text):
                did_shorten = True
            rewritten_text = compressed

            blocked_after = _find_blocked_terms_in_text(rewritten_text, ats_policy.blocked_terms)
            if blocked_after:
                audit_log["rejected_for_new_terms"].append(bullet_id)
                detail["reject_reason"] = "blocked_terms_after_compress"
                detail["disallowed_terms"] = blocked_after
                _append_bullet_detail(audit_log, detail)
                return
            disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, ats_policy.requested_target_keywords)
            if disallowed_after:
                audit_log["rejected_for_new_terms"].append(bullet_id)
                detail["reject_reason"] = "disallowed_terms_after_compress"
                detail["disallowed_terms"] = disallowed_after
                _append_bullet_detail(audit_log, detail)
                return
            unsupported_after = _find_unsupported_ats_terms(original_text, rewritten_text, ats_policy)
            if unsupported_after:
                audit_log["rejected_for_new_terms"].append(bullet_id)
                detail["reject_reason"] = "unsupported_ats_terms_after_compress"
                detail["disallowed_terms"] = unsupported_after
                _append_bullet_detail(audit_log, detail)
                return
            missing_required_after = _find_missing_required_terms(rewritten_text, ats_policy.required_terms)
            if missing_required_after:
                audit_log["rejected_for_new_terms"].append(bullet_id)
                detail["reject_reason"] = "missing_required_evidence_terms_after_compress"
                detail["disallowed_terms"] = missing_required_after
                _append_bullet_detail(audit_log, detail)
                return

    if isinstance(budget, int) and len(rewritten_text) > budget:
        truncated = _truncate_to_budget(rewritten_text, budget)
        if len(truncated) < len(rewritten_text):
            did_shorten = True
        rewritten_text = truncated

    blocked_after = _find_blocked_terms_in_text(rewritten_text, ats_policy.blocked_terms)
    if blocked_after:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = "blocked_terms_after_budget"
        detail["disallowed_terms"] = blocked_after
        _append_bullet_detail(audit_log, detail)
        return
    disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, ats_policy.requested_target_keywords)
    if disallowed_after:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = "disallowed_terms_after_budget"
        detail["disallowed_terms"] = disallowed_after
        _append_bullet_detail(audit_log, detail)
        return
    unsupported_after = _find_unsupported_ats_terms(original_text, rewritten_text, ats_policy)
    if unsupported_after:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = "unsupported_ats_terms_after_budget"
        detail["disallowed_terms"] = unsupported_after
        _append_bullet_detail(audit_log, detail)
        return
    missing_required_after = _find_missing_required_terms(rewritten_text, ats_policy.required_terms)
    if missing_required_after:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = "missing_required_evidence_terms_after_budget"
        detail["disallowed_terms"] = missing_required_after
        _append_bullet_detail(audit_log, detail)
        return

    if did_shorten:
        _append_unique(audit_log["compressed"], bullet_id)
    bullet["text"] = rewritten_text
    audit_log["rewritten_bullets"].append(bullet_id)
    detail["final_text"] = rewritten_text
    detail["changed"] = rewritten_text != original_text
    _append_bullet_detail(audit_log, detail)

def _call_bullet_rewrite(
    bullet_id: str,
    original_text: str,
    job_json: Dict[str, Any],
    ats_policy: BulletATSRewritePolicy,
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    requested_target_keywords = list(ats_policy.requested_target_keywords)
    allowed_target, missing_target = _split_target_keywords(requested_target_keywords, allowed_terms)
    surface_terms = list(ats_policy.surface_terms)
    allowed_surface, missing_surface = _split_target_keywords(surface_terms, allowed_terms)

    payload = {
        "bullet_id": bullet_id,
        "original_text": original_text,
        "job_title": job_json.get("title"),
        "source_section": ats_policy.source_section,
        "target_keywords": requested_target_keywords,
        "evidence_terms": list(ats_policy.evidence_terms),
        "preferred_surface_terms": surface_terms,
        "required_terms": list(ats_policy.required_terms),
        "blocked_terms": list(ats_policy.blocked_terms),
        "avoid_terms": list(ats_policy.avoid_terms),
        "ats_emphasis": ats_policy.emphasis_strength,
        "is_recent": ats_policy.is_recent,
        "is_primary_evidence": ats_policy.is_primary_evidence,
        "is_safe_for_ats": ats_policy.is_safe_for_ats,
        "allowed_target_keywords": allowed_target,
        "missing_target_keywords": missing_target,
        "allowed_surface_terms": allowed_surface,
        "missing_surface_terms": missing_surface,
        "allowed_terms": _select_allowed_terms(
            allowed_terms,
            original_text,
            requested_target_keywords + list(ats_policy.evidence_terms) + surface_terms,
        ),
        "allowed_proper_nouns": _select_allowed_proper_nouns(allowed_proper, original_text),
    }
    if budget is not None:
        payload["max_chars"] = budget

    system_prompt = load_system_prompt("bullet_rewrite")
    config = get_config()
    for _ in range(2):
        messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="bullet_rewrite")
        raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
        obj = _parse_llm_json(raw, "bullet_rewrite", provider)
        if obj is None:
            return None, None, None

        if obj.get("bullet_id") != bullet_id:
            return None, None, None
        rewritten_text = obj.get("rewritten_text")
        if not isinstance(rewritten_text, str):
            return None, None, None

        blocked_terms = _find_blocked_terms_in_text(rewritten_text, ats_policy.blocked_terms)
        if blocked_terms:
            repaired_text = _salvage_bullet_candidate(
                original_text=original_text,
                rewritten_text=rewritten_text,
                blocked_terms=blocked_terms,
                ats_policy=ats_policy,
                allowed_vocab=allowed_vocab,
                requested_target_keywords=requested_target_keywords,
            )
            if repaired_text is not None:
                return repaired_text, None, None
            payload["blocked_terms_found"] = blocked_terms
            payload["retry_instruction"] = "Remove the blocked terms. Preserve only supported evidence-backed ATS terms."
            continue

        disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, requested_target_keywords)
        if disallowed:
            payload["disallowed_terms"] = disallowed
            payload["retry_instruction"] = "Remove the disallowed terms. Do not replace them with new terms."
            continue

        unsupported_terms = _find_unsupported_ats_terms(original_text, rewritten_text, ats_policy)
        if unsupported_terms:
            payload["unsupported_ats_terms"] = unsupported_terms
            payload["retry_instruction"] = (
                "Remove ATS terms that are not supported for this bullet. Keep only evidence-backed or allowed surface terms."
            )
            continue

        missing_required_terms = _find_missing_required_terms(rewritten_text, ats_policy.required_terms)
        if missing_required_terms:
            payload["missing_required_terms"] = missing_required_terms
            payload["retry_instruction"] = "Preserve the required evidence-backed ATS signal in the rewritten bullet."
            continue

        return rewritten_text, None, None

    if isinstance(payload.get("blocked_terms_found"), list) and payload["blocked_terms_found"]:
        return None, "blocked_terms", list(payload["blocked_terms_found"])
    if isinstance(payload.get("disallowed_terms"), list) and payload["disallowed_terms"]:
        return None, "disallowed_terms", list(payload["disallowed_terms"])
    if isinstance(payload.get("unsupported_ats_terms"), list) and payload["unsupported_ats_terms"]:
        return None, "unsupported_ats_terms", list(payload["unsupported_ats_terms"])
    if isinstance(payload.get("missing_required_terms"), list) and payload["missing_required_terms"]:
        return None, "missing_required_evidence_terms", list(payload["missing_required_terms"])
    return None, None, None


def _compress_text(
    original_text: str,
    candidate_text: str,
    max_chars: int,
    provider: LLMProvider,
    preserve_terms: Sequence[str] = (),
    blocked_terms: Sequence[str] = (),
    emphasis_terms: Sequence[str] = (),
    avoid_terms: Sequence[str] = (),
) -> Optional[str]:
    payload = {
        "original_text": original_text,
        "candidate_text": candidate_text,
        "max_chars": max_chars,
        "preserve_terms": list(preserve_terms),
        "blocked_terms": list(blocked_terms),
        "emphasis_terms": list(emphasis_terms),
        "avoid_terms": list(avoid_terms),
    }
    system_prompt = load_system_prompt("compress_text")
    messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="compress_text")
    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    obj = _parse_llm_json(raw, "compress_text", provider)
    if obj is None:
        return None
    compressed_text = obj.get("compressed_text")
    if not isinstance(compressed_text, str):
        return None
    return compressed_text


def _parse_llm_json(raw: str, schema_name: str, provider: LLMProvider) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors = [f"JSON parse error: {exc.msg} at pos {exc.pos}"]
        try:
            obj = repair_json_to_schema(raw, schema_name, errors, provider)
        except Exception:
            return None
    ok, errors = validate_json(schema_name, obj)
    if not ok:
        return None
    return obj


def _select_allowed_terms(allowed_terms: Iterable[str], original_text: str, target_keywords: Sequence[str]) -> List[str]:
    tokens = tokenize(original_text)
    phrases = generate_ngrams(tokens, 3)
    bullet_terms = set(tokens) | phrases
    keyword_terms = set(normalize_terms(target_keywords))
    candidates = [term for term in allowed_terms if term in bullet_terms or term in keyword_terms]
    candidates = sorted(set(candidates))
    return candidates[:50]


def _select_allowed_proper_nouns(allowed_proper: Iterable[str], original_text: str) -> List[str]:
    normalized = normalize_text(original_text)
    candidates = [term for term in allowed_proper if term and term in normalized]
    candidates = sorted(set(candidates))
    return candidates[:30]


def _split_target_keywords(target_keywords: Sequence[str], allowed_terms: Iterable[str]) -> Tuple[List[str], List[str]]:
    allowed_set = set(allowed_terms)
    allowed: List[str] = []
    missing: List[str] = []
    for keyword in target_keywords:
        if not isinstance(keyword, str):
            continue
        normalized = normalize_text(keyword)
        if not normalized:
            continue
        if normalized in allowed_set:
            allowed.append(keyword)
        else:
            missing.append(keyword)
    return allowed, missing


def _build_tailoring_ats_context(tailoring_plan: Dict[str, Any]) -> TailoringATSContext:
    supported_priority_terms = tuple(_normalize_ordered_terms(tailoring_plan.get("supported_priority_terms")))
    under_supported_terms = tuple(
        _normalize_ordered_terms(_extract_term_items(tailoring_plan.get("under_supported_terms")))
    )
    blocked_bullet_terms = tuple(
        _normalize_ordered_terms(_extract_blocked_terms(tailoring_plan.get("blocked_terms"), "bullets"))
    )
    recent_priority_terms = tuple(_normalize_ordered_terms(tailoring_plan.get("recent_priority_terms")))
    summary_alignment_terms = tuple(_normalize_ordered_terms(tailoring_plan.get("summary_alignment_terms")))
    skill_priority_terms = tuple(_normalize_ordered_terms(tailoring_plan.get("skill_priority_terms")))

    title_status = tailoring_plan.get("title_alignment_status")
    if isinstance(title_status, dict):
        title_supported_terms = tuple(_normalize_ordered_terms(title_status.get("supported_terms")))
        title_missing_terms = tuple(_normalize_ordered_terms(title_status.get("missing_tokens")))
        title_alignment_safe = bool(title_status.get("is_safe_for_summary_alignment"))
    else:
        title_supported_terms = ()
        title_missing_terms = ()
        title_alignment_safe = False

    plan_terms = frozenset(
        supported_priority_terms
        + under_supported_terms
        + blocked_bullet_terms
        + recent_priority_terms
        + summary_alignment_terms
        + skill_priority_terms
        + title_supported_terms
        + title_missing_terms
    )
    avoid_terms = frozenset(blocked_bullet_terms + under_supported_terms + title_missing_terms)
    return TailoringATSContext(
        supported_priority_terms=supported_priority_terms,
        under_supported_terms=under_supported_terms,
        blocked_bullet_terms=blocked_bullet_terms,
        recent_priority_terms=recent_priority_terms,
        summary_alignment_terms=summary_alignment_terms,
        skill_priority_terms=skill_priority_terms,
        title_supported_terms=title_supported_terms,
        title_missing_terms=title_missing_terms,
        title_alignment_safe=title_alignment_safe,
        plan_terms=plan_terms,
        avoid_terms=avoid_terms,
    )


def _build_bullet_ats_policy(
    bullet_id: str,
    original_text: str,
    action: Dict[str, Any],
    tailoring_ats_context: TailoringATSContext,
) -> BulletATSRewritePolicy:
    requested_target_keywords = tuple(_ordered_prompt_terms(action.get("target_keywords")))
    evidence_terms = tuple(_ordered_prompt_terms(action.get("evidence_terms")))
    evidence_term_set = frozenset(_normalize_ordered_terms(evidence_terms))
    original_term_set = _extract_canonical_terms(original_text)
    blocked_term_set = frozenset(tailoring_ats_context.blocked_bullet_terms)
    salvageable_blocked_terms = frozenset(
        term
        for term in blocked_term_set
        if term in tailoring_ats_context.title_supported_terms
        or term in tailoring_ats_context.summary_alignment_terms
        or term in tailoring_ats_context.title_missing_terms
    )

    safe_target_terms: List[str] = []
    safe_target_set: Set[str] = set()
    for keyword in requested_target_keywords:
        canonical = canonicalize_term(keyword)
        if not canonical or canonical in blocked_term_set:
            continue
        if (
            canonical in evidence_term_set
            or canonical in original_term_set
            or canonical in tailoring_ats_context.supported_priority_terms
        ):
            if canonical not in safe_target_set:
                safe_target_terms.append(canonical)
                safe_target_set.add(canonical)

    evidence_surface_terms: List[str] = []
    for term in evidence_terms:
        canonical = canonicalize_term(term)
        if canonical and canonical not in blocked_term_set:
            _append_unique(evidence_surface_terms, canonical)

    ordered_targets = list(safe_target_terms)
    recent_priority = set(tailoring_ats_context.recent_priority_terms)
    if recent_priority:
        ordered_targets.sort(key=lambda term: (term not in recent_priority, term not in evidence_term_set, term))
    surface_terms: List[str] = list(evidence_surface_terms)
    for term in ordered_targets:
        _append_unique(surface_terms, term)

    is_recent = bool(action.get("is_recent"))
    is_primary_evidence = bool(action.get("is_primary_evidence"))
    is_safe_for_ats = bool(action.get("is_safe_for_ats"))

    required_terms: List[str] = []
    for term in surface_terms:
        if term in original_term_set:
            _append_unique(required_terms, term)
    if not required_terms and is_recent and is_primary_evidence and is_safe_for_ats and surface_terms:
        required_terms.append(surface_terms[0])

    emphasis_strength = "light"
    if surface_terms and is_recent and is_primary_evidence:
        emphasis_strength = "strong"
    elif surface_terms and (is_recent or is_primary_evidence or is_safe_for_ats):
        emphasis_strength = "medium"

    return BulletATSRewritePolicy(
        bullet_id=bullet_id,
        source_section=action.get("source_section") if isinstance(action.get("source_section"), str) else None,
        requested_target_keywords=requested_target_keywords,
        evidence_terms=tuple(evidence_surface_terms),
        safe_target_terms=tuple(safe_target_terms),
        surface_terms=tuple(surface_terms),
        blocked_terms=tailoring_ats_context.blocked_bullet_terms,
        avoid_terms=tuple(term for term in tailoring_ats_context.avoid_terms if term not in surface_terms),
        required_terms=tuple(required_terms),
        allowed_new_terms=frozenset(surface_terms),
        blocked_term_set=blocked_term_set,
        salvageable_blocked_terms=salvageable_blocked_terms,
        plan_term_set=tailoring_ats_context.plan_terms,
        is_recent=is_recent,
        is_primary_evidence=is_primary_evidence,
        is_safe_for_ats=is_safe_for_ats,
        emphasis_strength=emphasis_strength,
    )


def _extract_term_items(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    extracted: List[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("term")
        else:
            value = item
        if isinstance(value, str):
            extracted.append(value)
    return extracted


def _extract_blocked_terms(items: Any, blocked_for: str) -> List[str]:
    if not isinstance(items, list):
        return []
    blocked: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        scopes = item.get("blocked_for")
        if not isinstance(term, str) or not isinstance(scopes, list):
            continue
        if blocked_for in scopes:
            blocked.append(term)
    return blocked


def _ordered_prompt_terms(terms: Any) -> List[str]:
    if not isinstance(terms, list):
        return []
    ordered: List[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        value = term.strip()
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _normalize_ordered_terms(terms: Any) -> List[str]:
    ordered: List[str] = []
    if not isinstance(terms, (list, tuple)):
        return ordered
    for term in terms:
        if not isinstance(term, str):
            continue
        canonical = canonicalize_term(term)
        if canonical and canonical not in ordered:
            ordered.append(canonical)
    return ordered


def _extract_canonical_terms(text: str) -> FrozenSet[str]:
    terms = {canonical for canonical, _ in extract_canonical_term_pairs(text) if canonical}
    return frozenset(terms)


def _contains_canonical_term(text: str, canonical_term: str) -> bool:
    if not canonical_term:
        return False
    if canonical_term in _extract_canonical_terms(text):
        return True
    normalized_text = normalize_ats_text(text)
    return _contains_phrase(normalized_text, canonical_term)


def _find_blocked_terms_in_text(text: str, blocked_terms: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for term in blocked_terms:
        canonical = canonicalize_term(term)
        if canonical and _contains_canonical_term(text, canonical):
            _append_unique(hits, canonical)
    return hits


def _find_unsupported_ats_terms(
    original_text: str,
    rewritten_text: str,
    policy: BulletATSRewritePolicy,
) -> List[str]:
    original_terms = _extract_canonical_terms(original_text)
    rewritten_terms = _extract_canonical_terms(rewritten_text)
    introduced_plan_terms = [
        term
        for term in sorted(rewritten_terms - original_terms)
        if term in policy.plan_term_set and term not in policy.allowed_new_terms and term not in policy.blocked_term_set
    ]

    original_guard_terms = set(_extract_tool_like_terms(original_text))
    rewritten_guard_terms = set(_extract_tool_like_terms(rewritten_text))
    introduced_guard_terms: List[str] = []
    for term in sorted(rewritten_guard_terms - original_guard_terms):
        canonical = canonicalize_term(term) or term
        if canonical in policy.allowed_new_terms:
            continue
        _append_unique(introduced_guard_terms, canonical)

    unsupported: List[str] = []
    for term in introduced_plan_terms + introduced_guard_terms:
        _append_unique(unsupported, term)
    return unsupported


def _find_missing_required_terms(text: str, required_terms: Sequence[str]) -> List[str]:
    if not required_terms:
        return []
    present = [term for term in required_terms if _contains_canonical_term(text, term)]
    if present:
        return []
    return list(required_terms)


def _salvage_bullet_candidate(
    *,
    original_text: str,
    rewritten_text: str,
    blocked_terms: Sequence[str],
    ats_policy: BulletATSRewritePolicy,
    allowed_vocab: Dict[str, Any],
    requested_target_keywords: Sequence[str],
) -> Optional[str]:
    sanitized = _remove_blocked_terms_from_text(rewritten_text, blocked_terms)
    if not sanitized or _is_effectively_same(rewritten_text, sanitized):
        return None
    if _is_effectively_same(original_text, sanitized):
        return None
    normalized_blocked_terms = _normalize_ordered_terms(blocked_terms)
    if not ats_policy.salvageable_blocked_terms or any(
        not _is_salvageable_blocked_term(term, ats_policy.salvageable_blocked_terms)
        for term in normalized_blocked_terms
    ):
        return None
    if _find_blocked_terms_in_text(sanitized, ats_policy.blocked_terms):
        return None
    if _find_disallowed_terms(sanitized, allowed_vocab, requested_target_keywords):
        return None
    if _find_unsupported_ats_terms(original_text, sanitized, ats_policy):
        return None
    if _find_missing_required_terms(sanitized, ats_policy.required_terms):
        return None
    return sanitized


def _remove_blocked_terms_from_text(text: str, blocked_terms: Sequence[str]) -> str:
    sanitized = text
    removable_terms = _reduce_blocked_terms_for_removal(blocked_terms)
    for term in sorted(removable_terms, key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"(?i)\b{re.escape(term).replace(r'\\ ', r'\\s+')}\b")
        sanitized = pattern.sub("", sanitized)
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    sanitized = re.sub(r"([(\[{])\s+", r"\1", sanitized)
    sanitized = re.sub(r"\s+([)\]}])", r"\1", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\b(with|for|in|on|to|and|or)\s*([,.;:])", r"\2", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"[ ,;:]+$", "", sanitized)
    return sanitized.strip()


def _is_salvageable_blocked_term(term: str, salvageable_terms: FrozenSet[str]) -> bool:
    if term in salvageable_terms:
        return True
    term_tokens = set(term.split())
    for salvageable in salvageable_terms:
        if not salvageable:
            continue
        if " " in salvageable and salvageable in term:
            return True
        if salvageable in term_tokens:
            return True
    return False


def _reduce_blocked_terms_for_removal(blocked_terms: Sequence[str]) -> List[str]:
    reduced: List[str] = []
    for term in sorted(_normalize_ordered_terms(blocked_terms), key=len):
        if any(_blocked_term_contains_term(term, existing) for existing in reduced):
            continue
        reduced.append(term)
    return reduced


def _blocked_term_contains_term(container: str, term: str) -> bool:
    if not container or not term:
        return False
    if container == term:
        return True
    if " " in term and term in container:
        return True
    return term in set(container.split())


def _find_disallowed_terms(
    rewritten_text: str,
    allowed_vocab: Dict[str, Any],
    target_keywords: Sequence[str],
) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())

    disallowed = set()
    normalized_text = normalize_text(rewritten_text)

    for keyword in target_keywords:
        if not isinstance(keyword, str):
            continue
        kw_norm = normalize_text(keyword)
        if not kw_norm or kw_norm in allowed_terms or kw_norm in allowed_proper:
            continue
        if _contains_phrase(normalized_text, kw_norm):
            disallowed.add(kw_norm)

    for term in _extract_tool_like_terms(rewritten_text):
        if term not in allowed_terms and term not in allowed_proper:
            disallowed.add(term)

    for term in _extract_proper_noun_candidates(rewritten_text):
        if term not in allowed_proper and term not in allowed_terms:
            disallowed.add(term)

    return sorted(disallowed)


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    if not phrase:
        return False
    haystack = f" {normalized_text} "
    needle = f" {phrase} "
    if " " in phrase:
        return needle in haystack
    return phrase in normalized_text.split()


def _extract_tool_like_terms(text: str) -> List[str]:
    terms: List[str] = []
    for raw in _RAW_TOKEN_PATTERN.findall(text):
        cleaned = _strip_trailing_periods(raw)
        if not cleaned:
            continue
        if _is_tool_like_raw(cleaned):
            normalized = _normalize_tool_token(cleaned)
            if normalized:
                terms.append(normalized)
    return terms


def _is_tool_like_raw(raw: str) -> bool:
    if any(ch.isdigit() for ch in raw):
        return True
    if any(ch in "#+." for ch in raw):
        return True
    if raw.isupper() and len(raw) >= 2:
        return True
    if any(ch.isupper() for ch in raw[1:]):
        return True
    return False


def _normalize_tool_token(raw: str) -> str:
    tokens = tokenize(raw)
    if tokens:
        return tokens[0]
    return normalize_text(raw)


def _extract_proper_noun_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    tokens = _RAW_TOKEN_PATTERN.findall(text)
    for idx, raw in enumerate(tokens):
        if idx == 0 and raw[:1].isupper() and raw[1:].islower():
            continue
        if any(ch.isupper() for ch in raw):
            candidates.append(_normalize_proper_token(_strip_trailing_periods(raw)))
    return candidates


def _normalize_proper_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip()


def _strip_trailing_periods(token: str) -> str:
    if not token:
        return token
    return token.rstrip(_TRAILING_PERIODS)


def _truncate_to_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space].rstrip()
    return truncated


def _is_effectively_same(original_text: str, candidate_text: Optional[str]) -> bool:
    if not candidate_text:
        return True
    return normalize_text(original_text) == normalize_text(candidate_text)


def _needs_summary_fallback(
    original_text: str,
    candidate_text: Optional[str],
    reject_reason: Optional[str],
) -> bool:
    if reject_reason:
        return True
    if not candidate_text or not candidate_text.strip():
        return True
    if _is_effectively_same(original_text, candidate_text):
        return True
    return False


def _fallback_reason(
    original_text: str,
    candidate_text: Optional[str],
    reject_reason: Optional[str],
) -> str:
    if reject_reason:
        return reject_reason
    if not candidate_text or not candidate_text.strip():
        return "empty_or_invalid"
    if _is_effectively_same(original_text, candidate_text):
        return "identical_to_original"
    return "unknown"


def _fallback_summary_rewrite_with_context(
    *,
    original_text: str,
    candidate_text: Optional[str],
    reject_reason: Optional[str],
    rejected_terms: Sequence[str],
    summary_policy: Optional[SummaryATSRewritePolicy],
    allowed_vocab: Dict[str, Any],
    target_keywords: Sequence[str],
) -> Optional[str]:
    if candidate_text and reject_reason in {"blocked_terms", "unsafe_title_alignment"}:
        sanitized = _remove_blocked_terms_from_text(candidate_text, rejected_terms)
        sanitized = _normalize_summary_candidate_text(sanitized)
        if sanitized and summary_policy is not None:
            composed = _compose_summary_from_policy(
                original_text=original_text,
                candidate_text=sanitized,
                summary_policy=summary_policy,
            )
            if composed is not None:
                reject_reason_after, _ = _validate_summary_candidate(
                    original_text,
                    composed,
                    summary_policy,
                    allowed_vocab,
                    target_keywords,
                )
                if reject_reason_after is None and not _is_effectively_same(original_text, composed):
                    return composed
        if sanitized and not _is_effectively_same(original_text, sanitized):
            reject_reason_after, _ = _validate_summary_candidate(
                original_text,
                sanitized,
                summary_policy,
                allowed_vocab,
                target_keywords,
            )
            if reject_reason_after is None:
                return sanitized
    return _fallback_summary_rewrite(original_text)


def _fallback_summary_rewrite(original_text: str) -> Optional[str]:
    stripped = original_text.strip()
    if not stripped:
        return None
    clauses, joiner, suffix = _split_summary_clauses(stripped)
    if len(clauses) >= 2:
        reordered = [clauses[-1]] + clauses[:-1]
        candidate = joiner.join(reordered).strip()
        if suffix:
            candidate = candidate.rstrip() + suffix
        if not _is_effectively_same(original_text, candidate):
            return candidate
    swapped = _swap_conjunction_phrases(stripped, " and ")
    if swapped and not _is_effectively_same(original_text, swapped):
        return swapped
    pivoted = _rephrase_by_pivot(
        stripped,
        [
            "focused on",
            "specializing in",
            "experienced in",
            "expertise in",
            "skilled in",
            "proficient in",
        ],
    )
    if pivoted and not _is_effectively_same(original_text, pivoted):
        return pivoted
    rotated = _rotate_summary_tokens(stripped)
    if rotated and not _is_effectively_same(original_text, rotated):
        return rotated
    return None


def _split_summary_clauses(text: str) -> Tuple[List[str], str, str]:
    core, suffix = _strip_summary_suffix(text)
    separators = ["; ", " | ", " / ", " - ", " — ", ": ", ", "]
    for sep in separators:
        if sep in core:
            parts = [part.strip() for part in core.split(sep) if part.strip()]
            if len(parts) >= 2:
                return parts, sep, suffix
    for sep in [";", ",", ":"]:
        if sep in core:
            parts = [part.strip() for part in core.split(sep) if part.strip()]
            if len(parts) >= 2:
                return parts, f"{sep} ", suffix
    return [core], " ", suffix


def _strip_summary_suffix(text: str) -> Tuple[str, str]:
    stripped = text.strip()
    if stripped and stripped[-1] in ".!?":
        return stripped[:-1].rstrip(), stripped[-1]
    return stripped, ""


def _swap_conjunction_phrases(text: str, conjunction: str) -> Optional[str]:
    lower = text.lower()
    conj_lower = conjunction.lower()
    idx = lower.find(conj_lower)
    if idx == -1:
        return None
    before = text[:idx].strip()
    after = text[idx + len(conjunction) :].strip()
    after_core, suffix = _strip_summary_suffix(after)
    if not before or not after_core:
        return None
    candidate = f"{after_core}{conjunction}{before}".strip()
    if suffix:
        candidate = candidate.rstrip() + suffix
    return candidate


def _rephrase_by_pivot(text: str, pivots: Sequence[str]) -> Optional[str]:
    lower = text.lower()
    for pivot in pivots:
        marker = f" {pivot} "
        idx = lower.find(marker)
        if idx == -1:
            continue
        before = text[:idx].strip()
        after = text[idx + len(marker) :].strip()
        after_core, suffix = _strip_summary_suffix(after)
        if not before or not after_core:
            continue
        candidate = f"{pivot.capitalize()} {after_core}, {before}".strip()
        if suffix:
            candidate = candidate.rstrip() + suffix
        return candidate
    return None


def _rotate_summary_tokens(text: str) -> Optional[str]:
    core, suffix = _strip_summary_suffix(text)
    tokens = [tok for tok in core.split() if tok]
    if len(tokens) < 2:
        return None
    rotated = [tokens[-1]] + tokens[:-1]
    if tokens[0][:1].isupper():
        rotated[0] = rotated[0][:1].upper() + rotated[0][1:]
    candidate = " ".join(rotated).strip()
    if suffix:
        candidate = candidate.rstrip() + suffix
    return candidate


def _normalize_summary_candidate_text(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[-1] not in ".!?":
        stripped = f"{stripped}."
    if stripped[0].isalpha():
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def _compose_summary_from_policy(
    *,
    original_text: str,
    candidate_text: str,
    summary_policy: SummaryATSRewritePolicy,
) -> Optional[str]:
    if not summary_policy.title_alignment_safe or not summary_policy.safe_title_terms:
        return None
    title_phrase = next((term for term in summary_policy.safe_title_terms if " " in term), None)
    if title_phrase is None:
        title_phrase = summary_policy.safe_title_terms[0]

    focus_terms = [
        term
        for term in summary_policy.requested_target_keywords
        if term not in summary_policy.safe_title_terms
    ]
    if not focus_terms:
        return None

    normalized_candidate = normalize_text(candidate_text)
    ordered_focus = sorted(
        focus_terms,
        key=lambda term: (
            normalized_candidate.find(normalize_text(term))
            if normalize_text(term) in normalized_candidate
            else 10_000 + focus_terms.index(term)
        ),
    )
    ordered_focus = ordered_focus[:2]
    display_focus = [_canonical_skill_display(term) for term in ordered_focus]
    title_display = _canonical_skill_display(title_phrase)
    verb = "building" if "build" in normalize_text(candidate_text) or "build" in normalize_text(original_text) else "delivering"
    tail = "services" if "service" in normalize_text(candidate_text) or "service" in normalize_text(original_text) else ""

    if len(display_focus) == 1:
        summary = f"{title_display} {verb} {display_focus[0]}"
    else:
        summary = f"{title_display} {verb} {display_focus[0]} and {display_focus[1]}"
    if tail:
        summary = f"{summary} {tail}"
    return _normalize_summary_candidate_text(summary)


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _append_bullet_detail(audit_log: Dict[str, Any], detail: Dict[str, Any]) -> None:
    details = audit_log.get("bullet_details")
    if isinstance(details, list):
        details.append(detail)


def _collect_candidate_keywords(
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    allowed_vocab: Dict[str, Any],
) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    candidates: List[str] = []

    prioritized = tailoring_plan.get("prioritized_keywords") if isinstance(tailoring_plan.get("prioritized_keywords"), list) else []
    for keyword in prioritized:
        normalized = normalize_text(keyword)
        if normalized and (normalized in allowed_terms or normalized in allowed_proper):
            _append_unique(candidates, normalized)

    for text in _iter_job_texts(job_json):
        for term in _extract_job_terms(text):
            if term in allowed_terms or term in allowed_proper:
                _append_unique(candidates, term)
        if len(candidates) >= 50:
            break
    return candidates


def _iter_job_texts(job_json: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    title = job_json.get("title")
    if isinstance(title, str):
        texts.append(title)
    keywords = job_json.get("keywords")
    if isinstance(keywords, list):
        texts.extend([kw for kw in keywords if isinstance(kw, str)])
    responsibilities = job_json.get("responsibilities")
    if isinstance(responsibilities, list):
        texts.extend([item for item in responsibilities if isinstance(item, str)])
    for field in ("must_have", "nice_to_have"):
        items = job_json.get(field)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
    return texts


def _extract_job_terms(text: str) -> List[str]:
    tokens = tokenize(text)
    ordered: List[str] = []
    for token in tokens:
        _append_unique(ordered, token)
    for phrase in _ordered_ngrams(tokens, 3):
        _append_unique(ordered, phrase)
    return ordered


def _ordered_ngrams(tokens: List[str], max_n: int) -> List[str]:
    ordered: List[str] = []
    if not tokens or max_n < 2:
        return ordered
    n_max = min(max_n, len(tokens))
    for n in range(2, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            ordered.append(" ".join(tokens[i : i + n]))
    return ordered


def _split_skills_line(text: str) -> Tuple[List[str], str]:
    if " | " in text:
        parts = [part.strip() for part in text.split(" | ")]
        return parts, " | "
    if " / " in text:
        parts = [part.strip() for part in text.split(" / ")]
        return parts, " / "
    if ";" in text:
        parts = [part.strip() for part in text.split(";") if part.strip() != ""]
        return parts, "; "
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip() != ""]
        return parts, ", "
    return [text.strip()], ", "


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


def _extract_tool_like_terms_with_display(text: str) -> List[Tuple[str, str]]:
    terms: List[Tuple[str, str]] = []
    for raw in _RAW_TOKEN_PATTERN.findall(text):
        cleaned = _strip_trailing_periods(raw)
        if not cleaned:
            continue
        if _is_tool_like_raw(cleaned):
            normalized = _normalize_tool_token(cleaned)
            if normalized:
                terms.append((normalized, cleaned))
    return terms


def _find_display_in_texts(term: str, resume_texts: Sequence[str]) -> Optional[str]:
    if not term:
        return None
    has_special = any(ch in "#+." for ch in term)
    if has_special:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
    else:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for text in resume_texts:
        if not isinstance(text, str) or not text:
            continue
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _segment_has_tool_like_token(segment: str) -> bool:
    for raw in _RAW_TOKEN_PATTERN.findall(segment):
        cleaned = _strip_trailing_periods(raw)
        if cleaned and _is_tool_like_raw(cleaned):
            return True
    return False


def _term_is_tool_like(term: str) -> bool:
    if not term:
        return False
    return any(ch.isdigit() for ch in term) or any(ch in "#+." for ch in term)


def _segment_has_technical_indicator(segment: str) -> bool:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    return any(indicator in terms for indicator in _TECHNICAL_INDICATORS)


def _term_has_technical_indicator(term: str) -> bool:
    tokens = tokenize(term)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    return any(indicator in terms for indicator in _TECHNICAL_INDICATORS)


def _build_resume_hard_skill_inventory(
    resume_json: Dict[str, Any],
    candidate_keywords: Sequence[str],
    resume_texts: Sequence[str],
    soft_skill_terms: Set[str],
    resume_normalized: str,
) -> Tuple[Set[str], Dict[str, str]]:
    hard_terms: Set[str] = set()
    display_map: Dict[str, str] = {}

    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if not isinstance(text, str):
            continue
        segments, _ = _split_skills_line(text)
        for segment in segments:
            normalized = normalize_text(segment)
            if not normalized or normalized in soft_skill_terms:
                continue
            if _segment_has_tool_like_token(segment) or _segment_has_technical_indicator(segment):
                for term in _segment_term_set(segment) | {normalized}:
                    if term in soft_skill_terms:
                        continue
                    if term not in hard_terms:
                        hard_terms.add(term)
                        display_map[term] = segment

    for text in resume_texts:
        if not isinstance(text, str):
            continue
        for normalized, display in _extract_tool_like_terms_with_display(text):
            if normalized not in hard_terms:
                hard_terms.add(normalized)
                display_map[normalized] = display

    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized in soft_skill_terms:
            continue
        if normalized in hard_terms:
            continue
        if normalized and resume_normalized and normalized not in resume_normalized:
            continue
        display = _find_display_in_texts(normalized, resume_texts) or keyword
        hard_terms.add(normalized)
        display_map.setdefault(normalized, display)

    return hard_terms, display_map


def _collect_existing_skill_terms(lines: List[Any]) -> Set[str]:
    existing: Set[str] = set()
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if not isinstance(text, str):
            continue
        segments, _ = _split_skills_line(text)
        for segment in segments:
            normalized = normalize_text(segment)
            if normalized:
                existing.add(normalized)
            existing.update(_segment_term_set(segment))
    return existing


def _collect_matched_hard_terms(candidate_keywords: Sequence[str], hard_skill_terms: Set[str]) -> List[str]:
    matched: List[str] = []
    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized:
            continue
        if normalized in hard_skill_terms and normalized not in matched:
            matched.append(normalized)
    return matched


def _collect_matched_soft_terms(
    candidate_keywords: Sequence[str],
    soft_skill_terms: Set[str],
    resume_normalized: str,
) -> List[str]:
    matched: List[str] = []
    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized not in soft_skill_terms:
            continue
        if resume_normalized and normalized not in resume_normalized:
            continue
        if normalized not in matched:
            matched.append(normalized)
    return matched


def _classify_skill_segment(segment: str, hard_skill_terms: Set[str], soft_skill_terms: Set[str]) -> str:
    normalized = normalize_text(segment)
    if normalized in soft_skill_terms:
        return "soft"
    if normalized in hard_skill_terms:
        return "hard"
    if _segment_has_tool_like_token(segment) or _segment_has_technical_indicator(segment):
        return "hard"
    return "unknown"


def _segment_term_set(segment: str) -> Set[str]:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    return set(tokens) | phrases


def _reorder_skill_segments(
    segments: List[str],
    classifications: List[str],
    matched_hard_terms: Sequence[str],
    matched_soft_terms: Sequence[str],
) -> List[str]:
    matched_hard_set = set(matched_hard_terms)
    matched_soft_set = set(matched_soft_terms)
    buckets: Dict[str, List[Tuple[int, str]]] = {
        "hard_matched": [],
        "hard": [],
        "soft_matched": [],
        "soft": [],
        "unknown": [],
    }
    for idx, (segment, classification) in enumerate(zip(segments, classifications)):
        terms = _segment_term_set(segment)
        if classification == "hard":
            if matched_hard_set and terms.intersection(matched_hard_set):
                buckets["hard_matched"].append((idx, segment))
            else:
                buckets["hard"].append((idx, segment))
        elif classification == "soft":
            if matched_soft_set and terms.intersection(matched_soft_set):
                buckets["soft_matched"].append((idx, segment))
            else:
                buckets["soft"].append((idx, segment))
        else:
            buckets["unknown"].append((idx, segment))

    ordered: List[str] = []
    for key in ("hard_matched", "hard", "soft_matched", "soft", "unknown"):
        ordered.extend([segment for _, segment in sorted(buckets[key], key=lambda item: item[0])])
    return ordered


def _surface_missing_hard_skills(
    line_states: List[Dict[str, Any]],
    missing_hard_terms: Sequence[str],
    hard_skill_display: Dict[str, str],
    soft_skill_terms: Set[str],
    audit_log: Dict[str, Any],
) -> None:
    if not line_states:
        return

    for term in missing_hard_terms:
        display = hard_skill_display.get(term)
        if not display:
            continue
        inserted = False
        for state in line_states:
            if inserted:
                break
            segments = state["segments"]
            separator = state["separator"]
            budget = state["budget"]
            current_text = separator.join(segments)
            current_len = len(current_text)
            normalized_segments = {normalize_text(seg) for seg in segments if normalize_text(seg)}
            if term in normalized_segments:
                continue

            classifications = state["classifications"]
            insert_index = _preferred_hard_insert_index(classifications)
            candidate_segments = list(segments)
            candidate_segments.insert(insert_index, display)
            candidate_len = len(separator.join(candidate_segments))
            if candidate_len <= budget:
                segments.insert(insert_index, display)
                classifications.insert(insert_index, "hard")
                state["segments"] = segments
                state["classifications"] = classifications
                state["line"]["text"] = separator.join(segments)
                _update_skills_detail(audit_log, state["line_id"], state["line"]["text"], "hard_skill_surface")
                inserted = True
                continue

            replacement_index = _find_replacement_index(
                segments,
                classifications,
                display,
                budget,
                separator,
                soft_skill_terms,
            )
            if replacement_index is not None:
                segments[replacement_index] = display
                classifications[replacement_index] = "hard"
                state["segments"] = segments
                state["classifications"] = classifications
                state["line"]["text"] = separator.join(segments)
                _update_skills_detail(audit_log, state["line_id"], state["line"]["text"], "hard_skill_surface")
                inserted = True


def _preferred_hard_insert_index(classifications: Sequence[str]) -> int:
    last_hard_index = -1
    for idx, classification in enumerate(classifications):
        if classification == "hard":
            last_hard_index = idx
    return last_hard_index + 1 if last_hard_index >= 0 else 0


def _find_replacement_index(
    segments: Sequence[str],
    classifications: Sequence[str],
    replacement: str,
    budget: int,
    separator: str,
    soft_skill_terms: Set[str],
) -> Optional[int]:
    candidates: List[int] = []
    for idx, classification in enumerate(classifications):
        if classification == "hard":
            continue
        candidates.append(idx)

    def candidate_rank(index: int) -> Tuple[int, int]:
        normalized = normalize_text(segments[index])
        is_soft = normalized in soft_skill_terms
        classification = classifications[index]
        if classification == "soft" and is_soft:
            return (0, index)
        if classification == "unknown":
            return (1, index)
        return (2, index)

    for idx in sorted(candidates, key=candidate_rank):
        candidate_segments = list(segments)
        candidate_segments[idx] = replacement
        if len(separator.join(candidate_segments)) <= budget:
            return idx
    return None


def _update_skills_detail(
    audit_log: Dict[str, Any],
    line_id: str,
    final_text: str,
    reason: str,
) -> None:
    details = audit_log.get("skills_details")
    if not isinstance(details, list):
        return
    for detail in reversed(details):
        if isinstance(detail, dict) and detail.get("line_id") == line_id:
            detail["final_text"] = final_text
            detail["changed"] = True
            detail["skip_reason"] = reason
            return


def _score_skill_segments(segments: List[str], candidates: List[str]) -> List[Tuple[str, int, int]]:
    scored: List[Tuple[str, int, int]] = []
    for idx, segment in enumerate(segments):
        score = _count_segment_overlap(segment, candidates)
        scored.append((segment, score, idx))
    return scored


def _reorder_segments(scored: List[Tuple[str, int, int]]) -> List[str]:
    ordered = sorted(scored, key=lambda item: (-item[1], item[2]))
    return [item[0] for item in ordered]


def _count_segment_overlap(segment: str, candidates: List[str]) -> int:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    score = 0
    for keyword in candidates:
        if keyword in terms:
            score += 1
    return score


def _extract_matching_keywords(segments: List[str], candidates: List[str]) -> List[str]:
    matched: List[str] = []
    for segment in segments:
        tokens = tokenize(segment)
        phrases = generate_ngrams(tokens, 3)
        terms = set(tokens) | phrases
        for keyword in candidates:
            if keyword in terms:
                _append_unique(matched, keyword)
    return matched


def _build_action_map(tailoring_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    actions = tailoring_plan.get("bullet_actions") if isinstance(tailoring_plan.get("bullet_actions"), list) else []
    action_map: Dict[str, Dict[str, Any]] = {}
    for item in actions:
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if not isinstance(bullet_id, str) or bullet_id.strip() == "":
            continue
        if bullet_id not in action_map:
            action_map[bullet_id] = item
    return action_map


def _sorted_bullets(bullets: List[Any]) -> List[Dict[str, Any]]:
    indexed: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, bullet in enumerate(bullets):
        if not isinstance(bullet, dict):
            continue
        bi = bullet.get("bullet_index")
        bullet_index = bi if isinstance(bi, int) else idx
        indexed.append((bullet_index, idx, bullet))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def _check_invariants(original: Dict[str, Any], tailored: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    _compare_summary(original, tailored, errors)
    _compare_skills(original, tailored, errors)
    _compare_experience(original, tailored, errors)
    _compare_projects(original, tailored, errors)

    return errors


def _compare_summary(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_summary = original.get("summary") if isinstance(original.get("summary"), dict) else {}
    tailored_summary = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else {}
    if original_summary.get("id") != tailored_summary.get("id"):
        errors.append("summary.id changed")


def _compare_skills(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_skills = original.get("skills") if isinstance(original.get("skills"), dict) else {}
    tailored_skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else {}
    original_lines = original_skills.get("lines") if isinstance(original_skills.get("lines"), list) else []
    tailored_lines = tailored_skills.get("lines") if isinstance(tailored_skills.get("lines"), list) else []
    if len(original_lines) != len(tailored_lines):
        errors.append("skills.lines count changed")
        return
    original_ids = [line.get("line_id") for line in original_lines if isinstance(line, dict)]
    tailored_ids = [line.get("line_id") for line in tailored_lines if isinstance(line, dict)]
    if original_ids != tailored_ids:
        errors.append("skills.lines order or ids changed")


def _compare_experience(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_exps = original.get("experience") if isinstance(original.get("experience"), list) else []
    tailored_exps = tailored.get("experience") if isinstance(tailored.get("experience"), list) else []
    if len(original_exps) != len(tailored_exps):
        errors.append("experience count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_exps, tailored_exps)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("exp_id") != new.get("exp_id"):
            errors.append(f"experience[{idx}].exp_id changed")
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"experience[{idx}].bullets")


def _compare_projects(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_projects = original.get("projects") if isinstance(original.get("projects"), list) else []
    tailored_projects = tailored.get("projects") if isinstance(tailored.get("projects"), list) else []
    if len(original_projects) != len(tailored_projects):
        errors.append("projects count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_projects, tailored_projects)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("project_id") != new.get("project_id"):
            errors.append(f"projects[{idx}].project_id changed")
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"projects[{idx}].bullets")


def _compare_bullets(original_bullets: Any, tailored_bullets: Any, errors: List[str], prefix: str) -> None:
    original_list = original_bullets if isinstance(original_bullets, list) else []
    tailored_list = tailored_bullets if isinstance(tailored_bullets, list) else []
    if len(original_list) != len(tailored_list):
        errors.append(f"{prefix} count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_list, tailored_list)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("bullet_id") != new.get("bullet_id"):
            errors.append(f"{prefix}[{idx}].bullet_id changed")
        if orig.get("bullet_index") != new.get("bullet_index"):
            errors.append(f"{prefix}[{idx}].bullet_index changed")
