from typing import Any, Dict, List, Optional

from .normalize import normalize_text


def _detect_hard_gate_pattern(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    if "security clearance" in normalized:
        return "security clearance"
    if "clearance" in normalized and ("required" in normalized or "must" in normalized):
        return "clearance"
    if "license required" in normalized:
        return "license required"
    if "must be licensed" in normalized:
        return "must be licensed"
    if "certification required" in normalized:
        return "certification required"
    if "required certification" in normalized:
        return "required certification"
    if "must have certification" in normalized:
        return "must have certification"
    if "work authorization" in normalized:
        return "work authorization"
    if "eligible to work" in normalized:
        return "eligible to work"
    return None


def evaluate_hard_gates(job_json: Dict[str, Any], step2_result: Dict[str, Any]) -> Dict[str, Any]:
    matches = step2_result.get("matches")
    must_matches = matches.get("must_have") if isinstance(matches, dict) else None
    must_match_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(must_matches, list):
        for match in must_matches:
            if isinstance(match, dict) and "requirement_id" in match:
                must_match_map[match["requirement_id"]] = match
    missing_ids = set()
    if not must_match_map:
        missing_list = step2_result.get("missing_requirements")
        if isinstance(missing_list, list):
            for item in missing_list:
                if isinstance(item, dict) and "requirement_id" in item:
                    missing_ids.add(item["requirement_id"])
    hard_gate_missing: List[Dict[str, Any]] = []
    for requirement in job_json.get("must_have", []):
        if not isinstance(requirement, dict):
            continue
        req_id = requirement.get("requirement_id")
        if must_match_map:
            match = must_match_map.get(req_id)
            if match and match.get("matched") is True:
                continue
        else:
            if req_id not in missing_ids:
                continue
        pattern = _detect_hard_gate_pattern(requirement.get("text", ""))
        if pattern:
            hard_gate_missing.append(
                {
                    "requirement_id": req_id,
                    "text": requirement.get("text"),
                    "pattern": pattern,
                }
            )
    return {
        "hard_gate_failed": bool(hard_gate_missing),
        "hard_gate_missing": hard_gate_missing,
    }


def _reason(code: str, message: str, details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def decide(step2_result: Dict[str, Any], job_json: Dict[str, Any]) -> Dict[str, Any]:
    hard_gate = evaluate_hard_gates(job_json, step2_result)
    hard_gate_ids = {item["requirement_id"] for item in hard_gate["hard_gate_missing"]}

    matches = step2_result.get("matches")
    must_matches = matches.get("must_have") if isinstance(matches, dict) else []
    nice_matches = matches.get("nice_to_have") if isinstance(matches, dict) else []
    must_match_map: Dict[str, Dict[str, Any]] = {}
    nice_match_map: Dict[str, Dict[str, Any]] = {}
    for match in must_matches or []:
        if isinstance(match, dict) and "requirement_id" in match:
            must_match_map[match["requirement_id"]] = match
    for match in nice_matches or []:
        if isinstance(match, dict) and "requirement_id" in match:
            nice_match_map[match["requirement_id"]] = match

    matched_requirements: List[Dict[str, Any]] = []
    missing_requirements: List[Dict[str, Any]] = []

    for requirement in job_json.get("must_have", []):
        if not isinstance(requirement, dict):
            continue
        req_id = requirement.get("requirement_id")
        match = must_match_map.get(req_id)
        text = requirement.get("text")
        if text is None and match:
            text = match.get("text")
        if match and match.get("matched") is True:
            matched_requirements.append(
                {
                    "requirement_id": req_id,
                    "text": text,
                    "evidence": match.get("evidence"),
                }
            )
        else:
            missing_requirements.append(
                {
                    "requirement_id": req_id,
                    "text": text,
                    "hard_gate": req_id in hard_gate_ids,
                }
            )

    for requirement in job_json.get("nice_to_have", []):
        if not isinstance(requirement, dict):
            continue
        req_id = requirement.get("requirement_id")
        match = nice_match_map.get(req_id)
        text = requirement.get("text")
        if text is None and match:
            text = match.get("text")
        if match and match.get("matched") is True:
            matched_requirements.append(
                {
                    "requirement_id": req_id,
                    "text": text,
                    "evidence": match.get("evidence"),
                }
            )
        else:
            missing_requirements.append(
                {
                    "requirement_id": req_id,
                    "text": text,
                    "hard_gate": False,
                }
            )

    seniority_ok = step2_result.get("seniority", {}).get("seniority_ok")
    score_total = step2_result.get("score_total")
    if not isinstance(score_total, (int, float)):
        score_total = 0
    must_coverage = step2_result.get("must_have", {}).get("coverage_percent")
    if not isinstance(must_coverage, (int, float)):
        must_coverage = 0

    reasons_by_code: Dict[str, Dict[str, Any]] = {}
    if seniority_ok is False:
        reasons_by_code["SENIORITY_GATE"] = _reason(
            "SENIORITY_GATE", "Seniority gate failed", {"seniority_ok": False}
        )
    if hard_gate["hard_gate_failed"]:
        reasons_by_code["HARD_GATE_MISSING"] = _reason(
            "HARD_GATE_MISSING",
            "Hard gate requirement missing",
            {"missing": hard_gate["hard_gate_missing"]},
        )
    if score_total < 55:
        reasons_by_code["SCORE_TOO_LOW"] = _reason(
            "SCORE_TOO_LOW", "Score below threshold", {"score_total": score_total, "threshold": 55}
        )
    elif score_total < 70:
        reasons_by_code["SCORE_TOO_LOW"] = _reason(
            "SCORE_TOO_LOW", "Score below threshold", {"score_total": score_total, "threshold": 70}
        )
    if must_coverage < 70:
        reasons_by_code["MUST_HAVE_COVERAGE_TOO_LOW"] = _reason(
            "MUST_HAVE_COVERAGE_TOO_LOW",
            "Must-have coverage below threshold",
            {"must_have_coverage_percent": must_coverage, "threshold": 70},
        )

    proceed = (
        seniority_ok is not False
        and not hard_gate["hard_gate_failed"]
        and score_total >= 70
        and must_coverage >= 70
    )
    if proceed:
        reasons_by_code["OK"] = _reason("OK", "Meets thresholds", None)
        decision = "PROCEED"
    else:
        decision = "SKIP"

    reason_order = [
        "SENIORITY_GATE",
        "HARD_GATE_MISSING",
        "SCORE_TOO_LOW",
        "MUST_HAVE_COVERAGE_TOO_LOW",
        "OK",
    ]
    reasons: List[Dict[str, Any]] = []
    for code in reason_order:
        if code in reasons_by_code:
            reasons.append(reasons_by_code[code])

    return {
        "decision": decision,
        "reasons": reasons,
        "missing_requirements": missing_requirements,
        "matched_requirements": matched_requirements,
    }
