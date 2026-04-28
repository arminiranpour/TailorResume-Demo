from typing import Any, Dict, List, Mapping

from app.ats import (
    build_coverage_model,
    build_evidence_links,
    build_job_weights,
    build_recency_priorities,
    build_title_alignment,
    extract_job_signals,
    extract_resume_signals,
)
from app.ats.canonicalize import extract_canonical_term_pairs
from app.ats.types import EvidenceCandidate, ResumeCoverage, ResumeEvidenceLinks
from app.scoring import (
    compute_seniority_alignment,
    evaluate_hard_gates,
    detect_resume_seniority,
    normalize_job_seniority,
    score_job,
)

_PARTIAL_COVERED_BLEND = 0.5
_MAX_UNMATCHED_PARTIAL_CREDIT = 0.75
_MAX_COVERED_ONLY_PARTIAL_CREDIT = 0.25


def _validate_inputs(resume_json: Dict[str, Any], job_json: Dict[str, Any]) -> None:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    if not isinstance(job_json, dict):
        raise ValueError("job_json must be a dict")
    if "summary" not in resume_json or resume_json["summary"] is None:
        raise ValueError("resume_json missing required field: summary")
    if "skills" not in resume_json or resume_json["skills"] is None:
        raise ValueError("resume_json missing required field: skills")
    if "experience" not in resume_json or resume_json["experience"] is None:
        raise ValueError("resume_json missing required field: experience")
    if "must_have" not in job_json or job_json["must_have"] is None:
        raise ValueError("job_json missing required field: must_have")
    if "nice_to_have" not in job_json or job_json["nice_to_have"] is None:
        raise ValueError("job_json missing required field: nice_to_have")


def run_scoring(resume_json: Dict[str, Any], job_json: Dict[str, Any]) -> Dict[str, Any]:
    _validate_inputs(resume_json, job_json)
    return score_job(resume_json, job_json)


def score_fit(resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    return run_scoring(resume, job)


def run_ats_scoring(resume_json: Dict[str, Any], job_json: Dict[str, Any]) -> Dict[str, Any]:
    _validate_inputs(resume_json, job_json)

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

    must_matches = _match_requirements(job_json.get("must_have", []), coverage, evidence_links)
    nice_matches = _match_requirements(job_json.get("nice_to_have", []), coverage, evidence_links)

    must_total = len(must_matches)
    nice_total = len(nice_matches)
    matched_must = sum(1 for match in must_matches if match.get("matched") is True)
    matched_nice = sum(1 for match in nice_matches if match.get("matched") is True)
    recent_must_credit = sum(
        float(match.get("credit_ratio", 0.0))
        for match in must_matches
        if match.get("has_recent_backing") is True
    )

    strict_must_match_percent = _ratio_percent(matched_must, must_total)
    strict_nice_match_percent = _ratio_percent(matched_nice, nice_total)
    must_coverage_percent = _credit_percent(must_matches)
    nice_coverage_percent = _credit_percent(nice_matches)
    recent_must_percent = _credit_ratio_percent(recent_must_credit, must_total)

    resume_level = detect_resume_seniority(resume_json)
    job_seniority = normalize_job_seniority(job_json)
    seniority_alignment = compute_seniority_alignment(job_seniority, resume_level)

    must_score = round((must_coverage_percent / 100) * 60, 2)
    nice_score = round((nice_coverage_percent / 100) * 10, 2)
    seniority_score = round(float(max(0, min(10, seniority_alignment["alignment_points"]))), 2)
    title_score = round((title_alignment.title_alignment_score / 20) * 10, 2)
    recency_score = round((recent_must_percent / 100) * 10, 2)
    score_total = round(must_score + nice_score + seniority_score + title_score + recency_score, 2)
    if score_total > 100:
        score_total = 100.0

    hard_gate = evaluate_hard_gates(
        job_json,
        {
            "matches": {
                "must_have": must_matches,
                "nice_to_have": nice_matches,
            }
        },
    )
    hard_gate_ids = {item["requirement_id"] for item in hard_gate["hard_gate_missing"]}

    matched_requirements = [
        {
            "requirement_id": match["requirement_id"],
            "text": match["text"],
            "evidence": match.get("evidence"),
        }
        for match in must_matches + nice_matches
        if match.get("matched") is True
    ]
    missing_requirements = [
        {
            "requirement_id": match["requirement_id"],
            "text": match["text"],
            "hard_gate": match["requirement_id"] in hard_gate_ids,
        }
        for match in must_matches + nice_matches
        if match.get("matched") is not True
    ]

    reasons: List[Dict[str, Any]] = []
    seniority_ok = seniority_alignment["seniority_ok"]
    if seniority_ok is False:
        reasons.append(
            {
                "code": "SENIORITY_GATE",
                "message": "Seniority gate failed",
                "details": {"seniority_ok": False},
            }
        )
    if hard_gate["hard_gate_failed"]:
        reasons.append(
            {
                "code": "HARD_GATE_MISSING",
                "message": "Hard gate requirement missing",
                "details": {"missing": hard_gate["hard_gate_missing"]},
            }
        )
    if score_total < 70:
        reasons.append(
            {
                "code": "SCORE_TOO_LOW",
                "message": "ATS score below threshold",
                "details": {"score_total": score_total, "threshold": 70},
            }
        )
    if must_coverage_percent < 70:
        reasons.append(
            {
                "code": "MUST_HAVE_COVERAGE_TOO_LOW",
                "message": "ATS must-have coverage below threshold",
                "details": {
                    "must_have_coverage_percent": must_coverage_percent,
                    "must_have_strict_match_percent": strict_must_match_percent,
                    "threshold": 70,
                },
            }
        )

    decision = (
        "PROCEED"
        if seniority_ok is not False
        and not hard_gate["hard_gate_failed"]
        and score_total >= 70
        and strict_must_match_percent >= 70
        else "SKIP"
    )
    if decision == "PROCEED":
        reasons.append({"code": "OK", "message": "Meets ATS thresholds", "details": None})

    return {
        "decision": decision,
        "score_total": score_total,
        "score_breakdown": {
            "must": must_score,
            "nice": nice_score,
            "seniority": seniority_score,
            "title_alignment": title_score,
            "recency": recency_score,
        },
        "must_have_coverage_percent": must_coverage_percent,
        "must_have_strict_match_percent": strict_must_match_percent,
        "nice_to_have_strict_match_percent": strict_nice_match_percent,
        "seniority_ok": seniority_ok,
        "reasons": reasons,
        "matched_requirements": matched_requirements,
        "missing_requirements": missing_requirements,
        "scoring_mode": "ats_upgraded",
    }


def _match_requirements(
    requirements: List[Mapping[str, Any]],
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = requirement.get("requirement_id")
        text = requirement.get("text")
        if not isinstance(requirement_id, str):
            raise ValueError("requirement missing required field: requirement_id")
        if not isinstance(text, str):
            raise ValueError("requirement missing required field: text")

        requirement_terms = _extract_requirement_terms(text)
        supported_terms = [term for term in requirement_terms if _is_supported_term(term, coverage, evidence_links)]
        covered_terms = [term for term in requirement_terms if _is_covered_term(term, coverage, evidence_links)]
        ratio_base = supported_terms if supported_terms else covered_terms
        overlap_score = len(ratio_base) / max(1, len(requirement_terms)) if requirement_terms else 0.0
        supported_ratio = len(supported_terms) / max(1, len(requirement_terms)) if requirement_terms else 0.0
        covered_ratio = len(covered_terms) / max(1, len(requirement_terms)) if requirement_terms else 0.0
        matched = bool(ratio_base) and overlap_score >= _match_threshold(len(requirement_terms))
        candidate = _strongest_candidate(ratio_base, evidence_links)
        credit_ratio = _requirement_credit_ratio(
            matched=matched,
            supported_ratio=supported_ratio,
            covered_ratio=covered_ratio,
        )
        matches.append(
            {
                "requirement_id": requirement_id,
                "text": text,
                "matched": matched,
                "match_method": "ats_evidence" if matched else "none",
                "overlap_score": round(overlap_score, 2),
                "supported_ratio": round(supported_ratio, 4),
                "covered_ratio": round(covered_ratio, 4),
                "credit_ratio": round(credit_ratio, 4),
                "supported_terms": supported_terms,
                "covered_terms": covered_terms,
                "matched_terms": ratio_base if matched else [],
                "evidence": _build_evidence(candidate),
                "has_recent_backing": any(
                    evidence_links.links_by_term[term].has_recent_backing
                    for term in ratio_base
                    if term in evidence_links.links_by_term
                ),
            }
        )
    return matches


def _extract_requirement_terms(text: str) -> List[str]:
    ordered_terms: List[str] = []
    seen: set[str] = set()
    for canonical, _ in extract_canonical_term_pairs(text):
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered_terms.append(canonical)
    return ordered_terms


def _is_covered_term(
    term: str,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
) -> bool:
    term_coverage = coverage.coverage_by_term.get(term)
    link = evidence_links.links_by_term.get(term)
    return bool(term_coverage and link and term_coverage.is_covered and link.all_candidates)


def _is_supported_term(
    term: str,
    coverage: ResumeCoverage,
    evidence_links: ResumeEvidenceLinks,
) -> bool:
    term_coverage = coverage.coverage_by_term.get(term)
    link = evidence_links.links_by_term.get(term)
    if term_coverage is None or link is None:
        return False
    if not term_coverage.is_covered or not link.all_candidates:
        return False
    if link.is_under_supported or link.missing_experience_backing:
        return False
    if link.has_experience_backing or link.has_project_backing:
        return True
    return bool(
        term_coverage.has_cross_section_support
        and link.is_safe_for_summary
        and link.is_safe_for_skills
    )


def _strongest_candidate(
    terms: List[str],
    evidence_links: ResumeEvidenceLinks,
) -> EvidenceCandidate | None:
    candidates: List[EvidenceCandidate] = []
    for term in terms:
        link = evidence_links.links_by_term.get(term)
        if link is None or link.strongest_candidate is None:
            continue
        candidates.append(link.strongest_candidate)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.support_score, candidate.order, candidate.source_id),
    )[0]


def _build_evidence(candidate: EvidenceCandidate | None) -> Dict[str, Any] | None:
    if candidate is None:
        return None
    evidence: Dict[str, Any] = {
        "source_type": candidate.section,
        "source_id": candidate.source_id,
        "snippet": candidate.source_text,
    }
    if candidate.exp_id is not None:
        evidence["exp_id"] = candidate.exp_id
    if candidate.bullet_index is not None:
        evidence["bullet_index"] = candidate.bullet_index
    return evidence


def _match_threshold(term_count: int) -> float:
    if term_count <= 0:
        return 1.1
    if term_count == 1:
        return 1.0
    if term_count <= 3:
        return 0.5
    return 0.4


def _ratio_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 100.0
    return round((numerator / denominator) * 100, 2)


def _credit_ratio_percent(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 100.0
    return round((numerator / denominator) * 100, 2)


def _credit_percent(matches: List[Mapping[str, Any]]) -> float:
    if not matches:
        return 100.0
    total_credit = sum(float(match.get("credit_ratio", 0.0)) for match in matches)
    return _credit_ratio_percent(total_credit, len(matches))


def _requirement_credit_ratio(
    *,
    matched: bool,
    supported_ratio: float,
    covered_ratio: float,
) -> float:
    if matched:
        return 1.0
    if supported_ratio > 0:
        blended_ratio = supported_ratio + ((covered_ratio - supported_ratio) * _PARTIAL_COVERED_BLEND)
        return min(_MAX_UNMATCHED_PARTIAL_CREDIT, blended_ratio)
    if covered_ratio > 0:
        return min(_MAX_COVERED_ONLY_PARTIAL_CREDIT, covered_ratio * _PARTIAL_COVERED_BLEND)
    return 0.0
