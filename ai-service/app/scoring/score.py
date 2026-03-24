from typing import Any, Dict, List

from .indexer import build_resume_index
from .matcher import build_job_match
from .seniority import compute_seniority_alignment, detect_resume_seniority, normalize_job_seniority


def _count_matched(matches: List[Dict[str, Any]]) -> int:
    return sum(1 for match in matches if match.get("matched") is True)


def _round_score(value: float) -> float:
    return round(value, 2)


def _clamp_alignment(points: int) -> int:
    if points < 0:
        return 0
    if points > 10:
        return 10
    return points


def _collect_missing(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    missing = []
    for match in matches:
        if match.get("matched") is True:
            continue
        missing.append({"requirement_id": match["requirement_id"], "text": match["text"]})
    return missing


def _collect_matched(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched = []
    for match in matches:
        if match.get("matched") is not True:
            continue
        matched.append(
            {
                "requirement_id": match["requirement_id"],
                "text": match["text"],
                "evidence": match.get("evidence"),
            }
        )
    return matched


def score_job(resume_json: Dict[str, Any], job_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    if not isinstance(job_json, dict):
        raise ValueError("job_json must be a dict")

    resume_index = build_resume_index(resume_json)
    job_match = build_job_match(job_json, resume_index)

    total_must = len(job_json["must_have"])
    total_nice = len(job_json["nice_to_have"])
    matched_must = _count_matched(job_match["must_have"])
    matched_nice = _count_matched(job_match["nice_to_have"])

    if total_must == 0:
        must_coverage_percent = 100
        must_score = 70
    else:
        must_coverage_percent = _round_score((matched_must / total_must) * 100)
        must_score = _round_score((matched_must / total_must) * 70)

    if total_nice == 0:
        nice_score = 0
    else:
        nice_score = _round_score((matched_nice / total_nice) * 20)

    resume_level = detect_resume_seniority(resume_json)
    job_seniority = normalize_job_seniority(job_json)
    seniority_alignment = compute_seniority_alignment(job_seniority, resume_level)
    alignment_points = _clamp_alignment(int(seniority_alignment["alignment_points"]))
    alignment_score = _round_score(float(alignment_points))

    score_total = _round_score(must_score + nice_score + alignment_score)
    if score_total > 100:
        score_total = 100

    missing_requirements = _collect_missing(job_match["must_have"] + job_match["nice_to_have"])
    matched_requirements = _collect_matched(job_match["must_have"] + job_match["nice_to_have"])

    return {
        "score_total": score_total,
        "score_breakdown": {
            "must": must_score,
            "nice": nice_score,
            "alignment": alignment_score,
        },
        "must_have": {
            "total": total_must,
            "matched": matched_must,
            "coverage_percent": must_coverage_percent,
        },
        "nice_to_have": {"total": total_nice, "matched": matched_nice},
        "seniority": {
            "job_seniority": seniority_alignment["job_seniority"],
            "resume_level": seniority_alignment["resume_level"],
            "seniority_ok": seniority_alignment["seniority_ok"],
        },
        "alignment_details": {
            "seniority_points": alignment_points,
            "location_points": 0,
            "location_status": "not_implemented",
        },
        "matches": {
            "must_have": job_match["must_have"],
            "nice_to_have": job_match["nice_to_have"],
        },
        "missing_requirements": missing_requirements,
        "matched_requirements": matched_requirements,
    }
