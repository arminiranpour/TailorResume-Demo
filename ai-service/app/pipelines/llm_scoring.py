from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from app.config import get_config
from app.pipelines.scoring import (
    finalize_ats_scoring_result,
    run_ats_scoring_analysis,
    serialize_evidence_candidate,
    summarize_requirement_match,
)
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.schema_loader import load_schema
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages


class LLMScoreError(Exception):
    pass


_LLM_WEIGHTED_TERMS_LIMIT = 12


def build_scoring_packet(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    job_weights = analysis["job_weights"]
    coverage = analysis["coverage"]
    evidence_links = analysis["evidence_links"]
    title_alignment = analysis["title_alignment"]
    recency = analysis["recency"]
    resume_signals = analysis["resume_signals"]
    hard_gate = analysis["hard_gate"]

    requirement_packets = {
        "must_have": [
            _serialize_requirement(
                "must_have",
                requirement,
                match,
                evidence_links,
                analysis["hard_gate_ids"],
            )
            for requirement, match in zip(job_json.get("must_have", []), analysis["must_matches"])
        ],
        "nice_to_have": [
            _serialize_requirement(
                "nice_to_have",
                requirement,
                match,
                evidence_links,
                analysis["hard_gate_ids"],
            )
            for requirement, match in zip(job_json.get("nice_to_have", []), analysis["nice_matches"])
        ],
    }

    weighted_terms = []
    for term in job_weights.ordered_terms:
        term_weight = job_weights.weights_by_term[term]
        term_coverage = coverage.coverage_by_term[term]
        link = evidence_links.links_by_term[term]
        recency_priority = recency.priorities_by_term[term]
        weighted_terms.append(
            {
                "term": term,
                "weight": term_weight.total_weight,
                "priority_bucket": term_coverage.priority_bucket,
                "is_required": term_coverage.is_required,
                "is_preferred": term_coverage.is_preferred,
                "is_title_term": term_coverage.is_title_term,
                "is_low_signal": term_coverage.is_low_signal,
                "covered": term_coverage.is_covered,
                "under_supported": term_coverage.is_under_supported,
                "coverage_strength": term_coverage.coverage_strength,
                "has_recent_backing": recency_priority.has_recent_backing,
                "has_experience_backing": link.has_experience_backing,
                "has_project_backing": link.has_project_backing,
                "has_summary_backing": link.has_summary_backing,
                "missing_experience_backing": link.missing_experience_backing,
                "components": dict(term_weight.components),
                "reasons": list(term_weight.reasons),
                "source_sections": list(term_weight.source_sections),
                "strongest_candidate": serialize_evidence_candidate(link.strongest_candidate),
            }
        )

    evidence_sources = {
        entry.source_id: _serialize_source_entry(entry)
        for entry in resume_signals.source_entries
    }

    return {
        "job": {
            "title": job_json.get("title"),
            "company": job_json.get("company"),
            "location": job_json.get("location"),
            "remote": job_json.get("remote"),
            "seniority": analysis["job_seniority"],
            "must_have": requirement_packets["must_have"],
            "nice_to_have": requirement_packets["nice_to_have"],
        },
        "deterministic_summary": {
            "score_total": analysis["score_total"],
            "decision": analysis["decision"],
            "score_breakdown": dict(analysis["score_breakdown"]),
            "must_have_coverage_percent": analysis["must_have_coverage_percent"],
            "must_have_strict_match_percent": analysis["must_have_strict_match_percent"],
            "nice_to_have_strict_match_percent": analysis["nice_to_have_strict_match_percent"],
            "recent_must_percent": analysis["recent_must_percent"],
            "seniority_ok": analysis["seniority_ok"],
            "resume_level": analysis["resume_level"],
            "hard_gate_failed": hard_gate["hard_gate_failed"],
            "hard_gate_missing": list(hard_gate["hard_gate_missing"]),
            "reasons": list(analysis["reasons"]),
        },
        "coverage_summary": {
            "covered_terms": list(coverage.covered_terms),
            "missing_terms": list(coverage.missing_terms),
            "under_supported_terms": list(coverage.under_supported_terms),
            "high_priority_missing_terms": list(coverage.high_priority_missing_terms),
            "required_missing_terms": list(coverage.required_missing_terms),
            "title_terms_missing": list(coverage.title_terms_missing),
            "overall_distinct_coverage": coverage.overall_distinct_coverage,
            "high_priority_coverage": coverage.high_priority_coverage,
            "required_coverage": coverage.required_coverage,
            "title_coverage": coverage.title_coverage,
        },
        "weighted_terms": weighted_terms,
        "evidence_summary": {
            "skills_only_terms": list(evidence_links.skills_only_terms),
            "under_supported_terms": list(evidence_links.under_supported_terms),
            "missing_experience_terms": list(evidence_links.missing_experience_terms),
            "bullet_safe_terms": list(evidence_links.bullet_safe_terms),
            "summary_safe_terms": list(evidence_links.summary_safe_terms),
        },
        "title_alignment": {
            "job_title_tokens": list(title_alignment.job_title_tokens),
            "job_title_phrases": list(title_alignment.job_title_phrases),
            "resume_title_tokens": list(title_alignment.resume_title_tokens),
            "resume_title_phrases": list(title_alignment.resume_title_phrases),
            "overlapping_tokens": list(title_alignment.overlapping_tokens),
            "overlapping_phrases": list(title_alignment.overlapping_phrases),
            "strongest_matching_resume_title": title_alignment.strongest_matching_resume_title,
            "title_alignment_score": title_alignment.title_alignment_score,
            "alignment_strength": title_alignment.alignment_strength,
            "is_title_supported": title_alignment.is_title_supported,
            "is_safe_for_summary_alignment": title_alignment.is_safe_for_summary_alignment,
            "is_safe_for_experience_alignment": title_alignment.is_safe_for_experience_alignment,
            "missing_title_tokens": list(title_alignment.missing_title_tokens),
        },
        "recency_summary": {
            "recent_high_priority_terms": list(recency.recent_high_priority_terms),
            "recent_bullet_safe_terms": list(recency.recent_bullet_safe_terms),
            "recent_summary_safe_terms": list(recency.recent_summary_safe_terms),
            "stale_high_priority_terms": list(recency.stale_high_priority_terms),
            "stale_only_terms": list(recency.stale_only_terms),
        },
        "evidence_sources": evidence_sources,
        "resume_meta": {
            "experience_count": len(resume_json.get("experience", [])),
            "project_count": len(resume_json.get("projects", [])),
            "education_count": len(resume_json.get("education", [])),
        },
    }


def build_llm_prompt_packet(scoring_packet: Mapping[str, Any]) -> Dict[str, Any]:
    must_requirements = list(scoring_packet["job"]["must_have"])
    nice_requirements = list(scoring_packet["job"]["nice_to_have"])
    weighted_terms = sorted(
        list(scoring_packet["weighted_terms"]),
        key=lambda item: float(item.get("weight", 0.0)),
        reverse=True,
    )[:_LLM_WEIGHTED_TERMS_LIMIT]

    compact_weighted_terms = [
        {
            "term": item["term"],
            "weight": item["weight"],
            "priority_bucket": item["priority_bucket"],
            "is_required": item["is_required"],
            "is_preferred": item["is_preferred"],
            "is_title_term": item["is_title_term"],
            "covered": item["covered"],
            "under_supported": item["under_supported"],
            "coverage_strength": item["coverage_strength"],
            "has_recent_backing": item["has_recent_backing"],
            "missing_experience_backing": item["missing_experience_backing"],
            "strongest_evidence": _compact_candidate_evidence(item.get("strongest_candidate")),
        }
        for item in weighted_terms
    ]

    return {
        "job": {
            "title": scoring_packet["job"].get("title"),
            "seniority": scoring_packet["job"].get("seniority"),
            "must_have": [_compact_requirement_summary(item) for item in must_requirements],
            "nice_to_have": [_compact_requirement_summary(item) for item in nice_requirements],
        },
        "deterministic_baseline": {
            "score_total": scoring_packet["deterministic_summary"]["score_total"],
            "decision": scoring_packet["deterministic_summary"]["decision"],
            "component_scores": dict(scoring_packet["deterministic_summary"]["score_breakdown"]),
            "must_have_coverage_percent": scoring_packet["deterministic_summary"]["must_have_coverage_percent"],
            "must_have_strict_match_percent": scoring_packet["deterministic_summary"]["must_have_strict_match_percent"],
            "nice_to_have_strict_match_percent": scoring_packet["deterministic_summary"]["nice_to_have_strict_match_percent"],
        },
        "baseline_missing_requirements": [
            {
                "requirement_id": item["requirement_id"],
                "text": item["text"],
                "category": item["category"],
                "hard_gate": item["hard_gate"],
                "status": item["status"],
            }
            for item in must_requirements + nice_requirements
            if item["status"] != "matched"
        ],
        "weighted_term_snapshot": compact_weighted_terms,
        "hard_gate_snapshot": {
            "hard_gate_failed": scoring_packet["deterministic_summary"]["hard_gate_failed"],
            "missing_requirement_ids": [
                item["requirement_id"]
                for item in scoring_packet["deterministic_summary"]["hard_gate_missing"]
            ],
        },
        "seniority_snapshot": {
            "resume_level": scoring_packet["deterministic_summary"]["resume_level"],
            "seniority_ok": scoring_packet["deterministic_summary"]["seniority_ok"],
        },
        "title_alignment_snapshot": {
            "strongest_matching_resume_title": scoring_packet["title_alignment"]["strongest_matching_resume_title"],
            "title_alignment_score": scoring_packet["title_alignment"]["title_alignment_score"],
            "alignment_strength": scoring_packet["title_alignment"]["alignment_strength"],
            "is_title_supported": scoring_packet["title_alignment"]["is_title_supported"],
            "missing_title_tokens": list(scoring_packet["title_alignment"]["missing_title_tokens"]),
        },
    }


def extract_llm_json_object(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        raise LLMScoreError("LLM score JSON parse error: provider returned non-string output")

    stripped = raw.strip()
    if stripped == "":
        raise LLMScoreError("LLM score JSON parse error: empty response")

    parsed = _try_parse_json_object(stripped)
    if parsed is not None:
        return parsed

    without_fences = _strip_markdown_fences(stripped)
    if without_fences != stripped:
        parsed = _try_parse_json_object(without_fences)
        if parsed is not None:
            return parsed
        stripped = without_fences

    candidate = _extract_first_balanced_json_object(stripped)
    if candidate is None:
        raise LLMScoreError("LLM score JSON parse error: no JSON object found")

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMScoreError(f"LLM score JSON parse error: {exc.msg} at pos {exc.pos}") from exc
    if not isinstance(obj, dict):
        raise LLMScoreError("LLM score JSON parse error: top-level JSON value was not an object")
    return obj


def build_llm_adjudication_messages(scoring_packet: Mapping[str, Any]) -> List[Dict[str, str]]:
    prompt_packet = build_llm_prompt_packet(scoring_packet)
    untrusted_payload = _render_llm_prompt_packet(prompt_packet)
    return build_llm_messages(
        _llm_adjudicator_system_prompt(),
        untrusted_payload,
        task_label="llm_score_adjudicator",
    )


def adjudicate_score_with_llm(
    scoring_packet: Mapping[str, Any],
    provider: LLMProvider,
) -> Dict[str, Any]:
    config = get_config()
    schema = load_schema("llm_score_result")
    messages = build_llm_adjudication_messages(scoring_packet)
    raw = provider.generate(
        messages,
        json_schema=schema,
        temperature=0,
        seed=0,
        timeout=config.llm_timeout_seconds,
    )
    return extract_llm_json_object(raw)


def validate_llm_score_result(
    llm_result: Mapping[str, Any],
    scoring_packet: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_result = _normalize_llm_score_result(llm_result, scoring_packet)
    ok, errors = validate_json("llm_score_result", normalized_result)
    if not ok:
        raise LLMScoreError("Schema validation failed: " + " | ".join(errors))

    if normalized_result.get("decision") == "PROCEED" and float(normalized_result.get("score_total", 0)) < 70:
        raise LLMScoreError("LLM returned PROCEED with score_total below 70")

    job_requirements = _job_requirement_map(scoring_packet)
    valid_requirement_ids = set(job_requirements)
    matched_by_id = _requirements_by_id(
        scoring_packet["job"]["must_have"],
        scoring_packet["job"]["nice_to_have"],
        statuses={"matched"},
        require_supported=True,
    )
    transferable_by_id = _requirements_by_id(
        scoring_packet["job"]["must_have"],
        scoring_packet["job"]["nice_to_have"],
        statuses={"partial", "matched"},
    )
    valid_evidence_ids = set(scoring_packet["evidence_sources"])

    matched_ids: set[str] = set()
    for item in normalized_result["matched_requirements"]:
        requirement_id = item["requirement_id"]
        if requirement_id not in valid_requirement_ids:
            raise LLMScoreError(f"Unknown matched requirement_id: {requirement_id}")
        if requirement_id in matched_ids:
            raise LLMScoreError(f"Duplicate matched requirement_id: {requirement_id}")
        if requirement_id not in matched_by_id:
            raise LLMScoreError(
                f"Matched requirement is not eligible for strong match: {requirement_id}"
            )
        allowed_evidence_ids = set(matched_by_id[requirement_id]["allowed_evidence_source_ids"])
        _validate_evidence_ids(
            requirement_id=requirement_id,
            evidence_ids=item["evidence_source_ids"],
            valid_evidence_ids=valid_evidence_ids,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        matched_ids.add(requirement_id)

    transferable_ids: set[str] = set()
    for item in normalized_result["transferable_matches"]:
        requirement_id = item["requirement_id"]
        if requirement_id not in valid_requirement_ids:
            raise LLMScoreError(f"Unknown transferable requirement_id: {requirement_id}")
        if requirement_id in matched_ids:
            raise LLMScoreError(
                f"Transferable requirement duplicates a matched requirement: {requirement_id}"
            )
        if requirement_id in transferable_ids:
            raise LLMScoreError(f"Duplicate transferable requirement_id: {requirement_id}")
        if requirement_id not in transferable_by_id:
            raise LLMScoreError(
                f"Transferable requirement lacks deterministic partial evidence: {requirement_id}"
            )
        eligible = transferable_by_id[requirement_id]
        allowed_evidence_ids = set(eligible["allowed_evidence_source_ids"])
        _validate_evidence_ids(
            requirement_id=requirement_id,
            evidence_ids=item["evidence_source_ids"],
            valid_evidence_ids=valid_evidence_ids,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        transferable_ids.add(requirement_id)

    missing_ids: set[str] = set()
    for item in normalized_result["missing_requirements"]:
        requirement_id = item["requirement_id"]
        if requirement_id not in valid_requirement_ids:
            raise LLMScoreError(f"Unknown missing requirement_id: {requirement_id}")
        if requirement_id in matched_ids:
            raise LLMScoreError(
                f"Missing requirement duplicates a matched requirement: {requirement_id}"
            )
        if requirement_id in missing_ids:
            raise LLMScoreError(f"Duplicate missing requirement_id: {requirement_id}")
        if requirement_id in transferable_by_id or _requirement_has_covered_evidence(job_requirements[requirement_id]):
            raise LLMScoreError(
                f"Covered requirement must not be listed as missing; use transferable_matches or matched_requirements: {requirement_id}"
            )
        missing_ids.add(requirement_id)

    return {
        "score_total": round(float(normalized_result["score_total"]), 2),
        "decision": str(normalized_result["decision"]),
        "confidence": str(normalized_result["confidence"]),
        "fit_summary": str(normalized_result["fit_summary"]),
        "matched_requirements": list(normalized_result["matched_requirements"]),
        "missing_requirements": list(normalized_result["missing_requirements"]),
        "transferable_matches": list(normalized_result["transferable_matches"]),
        "risk_flags": [str(item) for item in normalized_result["risk_flags"]],
        "reasons": [
            {
                "code": str(item["code"]),
                "message": str(item["message"]),
                "severity": str(item["severity"]),
            }
            for item in normalized_result["reasons"]
        ],
        "score_breakdown": {
            key: round(float(value), 2)
            for key, value in dict(normalized_result["score_breakdown"]).items()
        },
    }


def _normalize_llm_score_result(
    llm_result: Mapping[str, Any],
    scoring_packet: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = dict(llm_result)
    normalized["reasons"] = _normalize_reason_severities(llm_result.get("reasons"))
    normalized = _repair_covered_missing_requirements(normalized, scoring_packet)
    return normalized


def _normalize_reason_severities(reasons: Any) -> Any:
    if not isinstance(reasons, list):
        return reasons

    severity_map = {
        "low": "info",
        "medium": "warning",
        "high": "blocker",
        "critical": "blocker",
    }
    normalized_reasons = []
    for item in reasons:
        if not isinstance(item, Mapping):
            normalized_reasons.append(item)
            continue
        normalized_item = dict(item)
        severity = normalized_item.get("severity")
        if isinstance(severity, str):
            normalized_item["severity"] = severity_map.get(severity.strip().lower(), severity)
        normalized_reasons.append(normalized_item)
    return normalized_reasons


def _repair_covered_missing_requirements(
    llm_result: Mapping[str, Any],
    scoring_packet: Mapping[str, Any],
) -> Dict[str, Any]:
    missing_requirements = llm_result.get("missing_requirements")
    transferable_matches = llm_result.get("transferable_matches")
    matched_requirements = llm_result.get("matched_requirements")

    if not isinstance(missing_requirements, list):
        return dict(llm_result)
    if not isinstance(transferable_matches, list):
        return dict(llm_result)
    if not isinstance(matched_requirements, list):
        return dict(llm_result)

    requirement_map = _job_requirement_map(scoring_packet)
    matched_ids = {
        item.get("requirement_id")
        for item in matched_requirements
        if isinstance(item, Mapping) and isinstance(item.get("requirement_id"), str)
    }
    transferable_ids = {
        item.get("requirement_id")
        for item in transferable_matches
        if isinstance(item, Mapping) and isinstance(item.get("requirement_id"), str)
    }

    repaired_missing: List[Any] = []
    repaired_transferable = list(transferable_matches)
    for item in missing_requirements:
        if not isinstance(item, Mapping):
            repaired_missing.append(item)
            continue
        requirement_id = item.get("requirement_id")
        if not isinstance(requirement_id, str):
            repaired_missing.append(item)
            continue
        requirement = requirement_map.get(requirement_id)
        if requirement is None or not _should_repair_missing_requirement(requirement):
            repaired_missing.append(item)
            continue

        evidence_ids = [
            str(source_id)
            for source_id in requirement.get("allowed_evidence_source_ids", [])
            if isinstance(source_id, str) and source_id
        ]
        if not evidence_ids:
            repaired_missing.append(item)
            continue

        if requirement_id in matched_ids or requirement_id in transferable_ids:
            continue

        repaired_transferable.append(
            {
                "requirement_id": requirement_id,
                "evidence_source_ids": evidence_ids,
                "rationale": "Requirement has partial deterministic coverage and is treated as transferable rather than missing.",
            }
        )
        transferable_ids.add(requirement_id)

    normalized = dict(llm_result)
    normalized["missing_requirements"] = repaired_missing
    normalized["transferable_matches"] = repaired_transferable
    return normalized


def enforce_final_score_safety(
    llm_result: Mapping[str, Any],
    scoring_packet: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {
        "score_total": round(float(llm_result["score_total"]), 2),
        "decision": str(llm_result["decision"]),
        "confidence": str(llm_result["confidence"]),
        "fit_summary": str(llm_result["fit_summary"]),
        "matched_requirements": list(llm_result["matched_requirements"]),
        "missing_requirements": list(llm_result["missing_requirements"]),
        "transferable_matches": list(llm_result["transferable_matches"]),
        "risk_flags": list(llm_result["risk_flags"]),
        "reasons": list(llm_result["reasons"]),
        "score_breakdown": dict(llm_result["score_breakdown"]),
        "safety_overrides": [],
    }

    deterministic = scoring_packet["deterministic_summary"]
    if deterministic["hard_gate_failed"] and result["decision"] == "PROCEED":
        result["decision"] = "SKIP"
        result["score_total"] = min(result["score_total"], 69.0)
        result["risk_flags"].append("hard_gate_missing")
        result["safety_overrides"].append("hard_gate_missing")
        result["reasons"].append(
            {
                "code": "HARD_GATE_SAFETY_OVERRIDE",
                "message": "Final decision forced to SKIP because a hard gate is missing.",
                "severity": "blocker",
            }
        )

    if deterministic["seniority_ok"] is False and result["decision"] == "PROCEED":
        result["decision"] = "SKIP"
        result["score_total"] = min(result["score_total"], 69.0)
        result["risk_flags"].append("seniority_mismatch")
        result["safety_overrides"].append("seniority_mismatch")
        result["reasons"].append(
            {
                "code": "SENIORITY_SAFETY_OVERRIDE",
                "message": "Final decision forced to SKIP because seniority alignment failed.",
                "severity": "blocker",
            }
        )

    return result


def run_llm_adjudicated_scoring(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    provider: LLMProvider | None,
) -> Dict[str, Any]:
    analysis = run_ats_scoring_analysis(resume_json, job_json)
    scoring_packet = build_scoring_packet(resume_json, job_json, analysis)

    if provider is None:
        return _fallback_result(analysis, "LLM provider unavailable")

    try:
        llm_raw = adjudicate_score_with_llm(scoring_packet, provider)
        llm_validated = validate_llm_score_result(llm_raw, scoring_packet)
        llm_safe = enforce_final_score_safety(llm_validated, scoring_packet)
        return _build_final_llm_result(analysis, scoring_packet, llm_safe)
    except Exception as exc:
        return _fallback_result(analysis, str(exc))


def _build_final_llm_result(
    analysis: Mapping[str, Any],
    scoring_packet: Mapping[str, Any],
    llm_result: Mapping[str, Any],
) -> Dict[str, Any]:
    requirement_map = _job_requirement_map(scoring_packet)
    matched_ids = {item["requirement_id"] for item in llm_result["matched_requirements"]}
    transferable_ids = {item["requirement_id"] for item in llm_result["transferable_matches"]}
    missing_reason_by_id = {
        item["requirement_id"]: item for item in llm_result["missing_requirements"]
    }

    matched_requirements = [
        _enrich_matched_requirement(
            requirement_map[item["requirement_id"]],
            item,
            scoring_packet["evidence_sources"],
        )
        for item in llm_result["matched_requirements"]
    ]

    missing_requirements = []
    for requirement_id, requirement in requirement_map.items():
        if requirement_id in matched_ids:
            continue
        if requirement_id in transferable_ids:
            continue
        if _requirement_has_covered_evidence(requirement):
            continue
        rationale = missing_reason_by_id.get(requirement_id, {})
        missing_requirements.append(
            {
                "requirement_id": requirement_id,
                "text": requirement["text"],
                "hard_gate": requirement["hard_gate"],
                "reason": rationale.get("rationale"),
            }
        )

    reasons = [
        {
            "code": item["code"],
            "message": item["message"],
            "details": {"severity": item["severity"]},
            "severity": item["severity"],
        }
        for item in llm_result["reasons"]
    ]

    extra_fields = {
        "confidence": llm_result["confidence"],
        "fit_summary": llm_result["fit_summary"],
        "transferable_matches": [
            _enrich_transferable_requirement(requirement_map[item["requirement_id"]], item)
            for item in llm_result["transferable_matches"]
        ],
        "risk_flags": list(llm_result["risk_flags"]),
        "llm_scoring_packet": scoring_packet,
        "llm_safety_overrides": list(llm_result["safety_overrides"]),
    }

    if llm_result["decision"] == "PROCEED" and not any(
        reason["code"] == "OK" for reason in reasons
    ):
        reasons.append(
            {
                "code": "OK",
                "message": "LLM adjudicator found the resume to be a proceed candidate.",
                "details": {"severity": "info"},
                "severity": "info",
            }
        )

    return finalize_ats_scoring_result(
        analysis,
        scoring_mode="llm_adjudicated",
        decision=str(llm_result["decision"]),
        score_total=float(llm_result["score_total"]),
        score_breakdown=dict(llm_result["score_breakdown"]),
        reasons=reasons,
        matched_requirements=matched_requirements,
        missing_requirements=missing_requirements,
        extra_fields=extra_fields,
    )


def _fallback_result(analysis: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    fallback_reason = {
        "code": "LLM_ADJUDICATION_FALLBACK",
        "message": "Returned deterministic ATS score because LLM adjudication failed.",
        "details": {"reason": reason},
        "severity": "warning",
    }
    reasons = list(analysis["reasons"]) + [fallback_reason]
    return finalize_ats_scoring_result(
        analysis,
        scoring_mode="ats_deterministic_fallback",
        reasons=reasons,
        extra_fields={
            "confidence": "low",
            "fit_summary": "Returned deterministic ATS score because the LLM adjudication step failed.",
            "transferable_matches": [],
            "risk_flags": ["llm_adjudication_fallback"],
            "llm_fallback_reason": reason,
        },
    )


def _serialize_requirement(
    category: str,
    requirement: Mapping[str, Any],
    match: Mapping[str, Any],
    evidence_links: Any,
    hard_gate_ids: Iterable[str],
) -> Dict[str, Any]:
    requirement_id = str(requirement["requirement_id"])
    hard_gate_id_set = set(hard_gate_ids)
    allowed_source_ids: set[str] = set()
    evidence_candidates = []
    seen_candidate_ids: set[str] = set()
    for term in set(match.get("supported_terms", [])) | set(match.get("covered_terms", [])):
        link = evidence_links.links_by_term.get(term)
        if link is None:
            continue
        for candidate in link.all_candidates:
            allowed_source_ids.add(candidate.source_id)
        for candidate in link.ranked_candidates[:3]:
            if candidate.source_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate.source_id)
            evidence_candidates.append(serialize_evidence_candidate(candidate))
    payload = summarize_requirement_match(match)
    payload["category"] = category
    payload["hard_gate"] = requirement_id in hard_gate_id_set or requirement.get("hard_gate") is True
    payload["allowed_evidence_source_ids"] = sorted(allowed_source_ids)
    payload["evidence_candidates"] = [item for item in evidence_candidates if item is not None]
    return payload


def _compact_requirement_summary(requirement: Mapping[str, Any]) -> Dict[str, Any]:
    support_meta = _requirement_support_meta(requirement)
    return {
        "requirement_id": requirement["requirement_id"],
        "text": requirement["text"],
        "category": requirement["category"],
        "hard_gate": requirement["hard_gate"],
        "status": requirement["status"],
        "supported_ratio": requirement["supported_ratio"],
        "credit_ratio": requirement["credit_ratio"],
        "strong_match_allowed": support_meta["strong_match_allowed"],
        "transferable_match_allowed": support_meta["transferable_match_allowed"],
        "evidence_strength": support_meta["evidence_strength"],
        "support_reason": support_meta["support_reason"],
        "matched_terms": list(requirement.get("matched_terms", []))[:5],
        "supported_terms": list(requirement.get("supported_terms", []))[:5],
        "has_recent_backing": requirement["has_recent_backing"],
        "best_evidence": _compact_requirement_evidence(requirement.get("evidence")),
    }


def _compact_requirement_evidence(evidence: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "source_id": evidence.get("source_id"),
        "source_type": evidence.get("source_type"),
        "snippet": evidence.get("snippet"),
    }


def _compact_candidate_evidence(candidate: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "source_id": candidate.get("source_id"),
        "section": candidate.get("section"),
        "text": candidate.get("source_text"),
    }


def _llm_adjudicator_system_prompt() -> str:
    base_prompt = load_system_prompt("llm_score_adjudicator").rstrip()
    output_contract = """

Return exactly one JSON object with these required fields and no others:
- score_total: number from 0 to 100
- decision: "PROCEED" or "SKIP"
- confidence: "high", "medium", or "low"
- fit_summary: short string
- matched_requirements: array of objects with requirement_id, evidence_source_ids, rationale
- missing_requirements: array of objects with requirement_id, rationale
- transferable_matches: array of objects with requirement_id, evidence_source_ids, rationale
- risk_flags: array of strings
- reasons: array of objects with code, message, severity
- score_breakdown: object with exactly must_have, nice_to_have, transferable_fit, title_seniority, evidence_quality, risk_penalty

    Important:
    - evidence_source_ids must be an array of strings, even when there is only one source id.
    - reasons must be objects, not strings.
    - reason severity must be exactly one of: "info", "warning", "blocker".
    - Do not echo the input packet fields back in the output.
    - matched_requirements may ONLY include items where strong_match_allowed=true.
    - transferable_matches may include items where transferable_match_allowed=true.
    - If unsure whether a requirement is strong/direct-match eligible, put it in transferable_matches, not matched_requirements.
    - missing_requirements may ONLY include requirements with no deterministic coverage and no transferable evidence.
    - If a requirement has covered terms, partial support, summary-only support, skills-only support, or other transferable evidence, do not place it in missing_requirements.
    - Partial requirements must go to transferable_matches, not missing_requirements.
    """.strip()
    return f"{base_prompt}\n\n{output_contract}"


def _render_llm_prompt_packet(prompt_packet: Mapping[str, Any]) -> str:
    lines = [
        "JOB",
        f"title: {prompt_packet['job'].get('title')}",
        f"seniority: {prompt_packet['job'].get('seniority')}",
        "",
        "DETERMINISTIC BASELINE",
        f"score_total: {prompt_packet['deterministic_baseline']['score_total']}",
        f"decision: {prompt_packet['deterministic_baseline']['decision']}",
        "component_scores: "
        + ", ".join(
            f"{key}={value}"
            for key, value in prompt_packet["deterministic_baseline"]["component_scores"].items()
        ),
        f"must_have_coverage_percent: {prompt_packet['deterministic_baseline']['must_have_coverage_percent']}",
        f"must_have_strict_match_percent: {prompt_packet['deterministic_baseline']['must_have_strict_match_percent']}",
        f"nice_to_have_strict_match_percent: {prompt_packet['deterministic_baseline']['nice_to_have_strict_match_percent']}",
        "",
        "MUST-HAVE REQUIREMENTS",
    ]
    lines.extend(_render_requirement_lines(prompt_packet["job"]["must_have"]))
    lines.append("")
    lines.append("NICE-TO-HAVE REQUIREMENTS")
    lines.extend(_render_requirement_lines(prompt_packet["job"]["nice_to_have"]))
    lines.append("")
    lines.append("BASELINE MISSING REQUIREMENTS")
    lines.extend(_render_missing_requirement_lines(prompt_packet["baseline_missing_requirements"]))
    lines.append("")
    lines.append("WEIGHTED TERM SNAPSHOT")
    lines.extend(_render_weighted_term_lines(prompt_packet["weighted_term_snapshot"]))
    lines.append("")
    lines.append("HARD GATE SNAPSHOT")
    lines.append(
        f"hard_gate_failed: {prompt_packet['hard_gate_snapshot']['hard_gate_failed']}"
    )
    lines.append(
        "missing_requirement_ids: "
        + ", ".join(prompt_packet["hard_gate_snapshot"]["missing_requirement_ids"])
    )
    lines.append("")
    lines.append("SENIORITY SNAPSHOT")
    lines.append(f"resume_level: {prompt_packet['seniority_snapshot']['resume_level']}")
    lines.append(f"seniority_ok: {prompt_packet['seniority_snapshot']['seniority_ok']}")
    lines.append("")
    lines.append("TITLE ALIGNMENT SNAPSHOT")
    lines.append(
        f"strongest_matching_resume_title: {prompt_packet['title_alignment_snapshot']['strongest_matching_resume_title']}"
    )
    lines.append(
        f"title_alignment_score: {prompt_packet['title_alignment_snapshot']['title_alignment_score']}"
    )
    lines.append(
        f"alignment_strength: {prompt_packet['title_alignment_snapshot']['alignment_strength']}"
    )
    lines.append(
        f"is_title_supported: {prompt_packet['title_alignment_snapshot']['is_title_supported']}"
    )
    lines.append(
        "missing_title_tokens: "
        + ", ".join(prompt_packet["title_alignment_snapshot"]["missing_title_tokens"])
    )
    return "\n".join(lines)


def _render_requirement_lines(requirements: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    for item in requirements:
        lines.append(
            f"- {item['requirement_id']}: status={item['status']}; hard_gate={item['hard_gate']}; supported_ratio={item['supported_ratio']}; credit_ratio={item['credit_ratio']}; strong_match_allowed={item['strong_match_allowed']}; transferable_match_allowed={item['transferable_match_allowed']}; evidence_strength={item['evidence_strength']}; text={item['text']}"
        )
        lines.append(f"  support_reason: {item['support_reason']}")
        if item["matched_terms"]:
            lines.append("  matched_terms: " + ", ".join(item["matched_terms"]))
        if item["supported_terms"]:
            lines.append("  supported_terms: " + ", ".join(item["supported_terms"]))
        best_evidence = item.get("best_evidence")
        if best_evidence is not None:
            lines.append(f"  best_evidence_source_id: {best_evidence.get('source_id')}")
            lines.append(f"  best_evidence_source_type: {best_evidence.get('source_type')}")
            lines.append(f"  best_evidence_snippet: {best_evidence.get('snippet')}")
    if not lines:
        lines.append("- none")
    return lines


def _requirement_support_meta(requirement: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(requirement.get("status", "missing"))
    supported_ratio = float(requirement.get("supported_ratio", 0.0))
    credit_ratio = float(requirement.get("credit_ratio", 0.0))
    evidence = requirement.get("evidence") or {}
    source_type = str(evidence.get("source_type") or "none")
    has_recent_backing = bool(requirement.get("has_recent_backing") is True)

    strong_match_allowed = status == "matched" and supported_ratio > 0
    transferable_match_allowed = status in {"matched", "partial"} and credit_ratio > 0

    if strong_match_allowed and source_type == "experience_bullet":
        evidence_strength = "strong_direct"
        support_reason = "Deterministic strong match with direct experience-bullet evidence."
    elif strong_match_allowed and source_type == "project_bullet":
        evidence_strength = "moderate_project"
        support_reason = "Deterministic matched requirement backed primarily by project evidence."
    elif status == "partial" and source_type == "experience_bullet":
        evidence_strength = "partial_transferable"
        support_reason = "Only partial deterministic support is available, so treat it as transferable."
    elif source_type == "summary":
        evidence_strength = "weak_summary_only"
        support_reason = "Summary-only evidence is not eligible for strong direct match."
    elif source_type == "skills":
        evidence_strength = "weak_skills_only"
        support_reason = "Skills-only evidence is not eligible for strong direct match."
    elif transferable_match_allowed:
        evidence_strength = "transferable_only"
        support_reason = "Some deterministic support exists, but not enough for strong direct match."
    else:
        evidence_strength = "missing_or_weak"
        support_reason = "No deterministic strong-match evidence is available."

    if has_recent_backing and transferable_match_allowed and not strong_match_allowed:
        support_reason = f"{support_reason} Recent backing exists, but it is still not strong-match eligible."

    return {
        "strong_match_allowed": strong_match_allowed,
        "transferable_match_allowed": transferable_match_allowed,
        "evidence_strength": evidence_strength,
        "support_reason": support_reason,
    }


def _render_missing_requirement_lines(requirements: Sequence[Mapping[str, Any]]) -> List[str]:
    if not requirements:
        return ["- none"]
    return [
        f"- {item['requirement_id']}: category={item['category']}; hard_gate={item['hard_gate']}; status={item['status']}; text={item['text']}"
        for item in requirements
    ]


def _render_weighted_term_lines(items: Sequence[Mapping[str, Any]]) -> List[str]:
    if not items:
        return ["- none"]
    lines: List[str] = []
    for item in items:
        lines.append(
            f"- {item['term']}: weight={item['weight']}; priority_bucket={item['priority_bucket']}; required={item['is_required']}; preferred={item['is_preferred']}; title_term={item['is_title_term']}; covered={item['covered']}; under_supported={item['under_supported']}; recent={item['has_recent_backing']}; missing_experience_backing={item['missing_experience_backing']}"
        )
        strongest = item.get("strongest_evidence")
        if strongest is not None:
            lines.append(f"  strongest_evidence_source_id: {strongest.get('source_id')}")
            lines.append(f"  strongest_evidence_section: {strongest.get('section')}")
            lines.append(f"  strongest_evidence_text: {strongest.get('text')}")
    return lines


def _serialize_source_entry(entry: Any) -> Dict[str, Any]:
    payload = {
        "section": entry.section,
        "text": entry.text,
        "order": entry.order,
    }
    for field in (
        "parent_id",
        "line_id",
        "exp_id",
        "project_id",
        "edu_id",
        "bullet_id",
        "bullet_index",
        "start_date",
        "end_date",
        "experience_order",
    ):
        value = getattr(entry, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _job_requirement_map(scoring_packet: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for category in ("must_have", "nice_to_have"):
        for requirement in scoring_packet["job"][category]:
            payload = dict(requirement)
            payload["category"] = category
            items[requirement["requirement_id"]] = payload
    return items


def _try_parse_json_object(raw: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        raise LLMScoreError("LLM score JSON parse error: top-level JSON value was not an object")
    return obj


def _strip_markdown_fences(raw: str) -> str:
    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return raw
    if not lines[0].lstrip().startswith("```"):
        return raw
    if lines[-1].strip() != "```":
        return raw
    return "\n".join(lines[1:-1]).strip()


def _extract_first_balanced_json_object(raw: str) -> str | None:
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(raw):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return raw[start:index + 1]

    return None


def _requirements_by_id(
    must_requirements: Sequence[Mapping[str, Any]],
    nice_requirements: Sequence[Mapping[str, Any]],
    *,
    statuses: set[str],
    require_supported: bool = False,
) -> Dict[str, Mapping[str, Any]]:
    items: Dict[str, Mapping[str, Any]] = {}
    for requirement in list(must_requirements) + list(nice_requirements):
        if requirement["status"] in statuses:
            if require_supported and float(requirement.get("supported_ratio", 0.0)) <= 0:
                continue
            items[requirement["requirement_id"]] = requirement
    return items


def _requirement_has_covered_evidence(requirement: Mapping[str, Any]) -> bool:
    if requirement.get("status") in {"matched", "partial"}:
        return True
    if float(requirement.get("supported_ratio", 0.0)) > 0:
        return True
    if float(requirement.get("covered_ratio", 0.0)) > 0:
        return True
    if float(requirement.get("credit_ratio", 0.0)) > 0:
        return True
    if requirement.get("supported_terms"):
        return True
    if requirement.get("covered_terms"):
        return True
    if requirement.get("evidence") is not None:
        return True
    return False


def _should_repair_missing_requirement(requirement: Mapping[str, Any]) -> bool:
    support_meta = _requirement_support_meta(requirement)
    if requirement.get("status") in {"matched", "partial"}:
        return True
    if requirement.get("covered_terms"):
        return True
    if float(requirement.get("credit_ratio", 0.0)) > 0:
        return True
    if support_meta["transferable_match_allowed"] is True:
        return True
    return False


def _validate_evidence_ids(
    *,
    requirement_id: str,
    evidence_ids: Sequence[str],
    valid_evidence_ids: set[str],
    allowed_evidence_ids: set[str],
) -> None:
    if not evidence_ids:
        raise LLMScoreError(f"No evidence_source_ids provided for requirement {requirement_id}")
    for source_id in evidence_ids:
        if source_id not in valid_evidence_ids:
            raise LLMScoreError(
                f"Evidence source_id does not exist for requirement {requirement_id}: {source_id}"
            )
        if source_id not in allowed_evidence_ids:
            raise LLMScoreError(
                f"Evidence source_id is not allowed for requirement {requirement_id}: {source_id}"
            )


def _enrich_matched_requirement(
    requirement: Mapping[str, Any],
    llm_item: Mapping[str, Any],
    evidence_sources: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence = [
        {
            "source_id": source_id,
            **dict(evidence_sources.get(source_id, {})),
        }
        for source_id in llm_item["evidence_source_ids"]
    ]
    return {
        "requirement_id": requirement["requirement_id"],
        "text": requirement["text"],
        "evidence": evidence,
        "rationale": llm_item.get("rationale"),
    }


def _enrich_transferable_requirement(
    requirement: Mapping[str, Any],
    llm_item: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "requirement_id": requirement["requirement_id"],
        "text": requirement["text"],
        "evidence_source_ids": list(llm_item["evidence_source_ids"]),
        "rationale": llm_item.get("rationale"),
    }
