import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.scoring import decide


def build_job(must_texts, nice_texts):
    return {
        "must_have": [
            {"requirement_id": f"must_{i}", "text": text}
            for i, text in enumerate(must_texts, start=1)
        ],
        "nice_to_have": [
            {"requirement_id": f"nice_{i}", "text": text}
            for i, text in enumerate(nice_texts, start=1)
        ],
        "responsibilities": [],
        "keywords": [],
    }


def build_matches(requirements, matched_ids):
    matches = []
    for req in requirements:
        matched = req["requirement_id"] in matched_ids
        matches.append(
            {
                "requirement_id": req["requirement_id"],
                "text": req["text"],
                "matched": matched,
                "evidence": {
                    "source_type": "summary",
                    "source_id": "summary",
                    "snippet": "Evidence",
                }
                if matched
                else None,
            }
        )
    return matches


def build_step2(score_total, coverage, seniority_ok, job_json, matched_must, matched_nice):
    return {
        "score_total": score_total,
        "score_breakdown": {"must": 0, "nice": 0, "alignment": 0},
        "must_have": {"coverage_percent": coverage},
        "seniority": {"seniority_ok": seniority_ok},
        "matches": {
            "must_have": build_matches(job_json["must_have"], matched_must),
            "nice_to_have": build_matches(job_json["nice_to_have"], matched_nice),
        },
    }


def test_proceed_when_thresholds_met():
    job = build_job(["Python"], [])
    step2 = build_step2(80, 80, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "PROCEED"
    assert [r["code"] for r in result["reasons"]] == ["OK"]


def test_skip_when_seniority_fails():
    job = build_job(["Python"], [])
    step2 = build_step2(90, 90, False, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert result["reasons"][0]["code"] == "SENIORITY_GATE"


def test_skip_when_score_below_55():
    job = build_job(["Python"], [])
    step2 = build_step2(54.99, 80, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert result["reasons"][0]["code"] == "SCORE_TOO_LOW"


def test_skip_when_coverage_below_70():
    job = build_job(["Python"], [])
    step2 = build_step2(90, 69.99, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert result["reasons"][0]["code"] == "MUST_HAVE_COVERAGE_TOO_LOW"


def test_skip_when_hard_gate_missing():
    job = build_job(["Security clearance required"], [])
    step2 = build_step2(90, 0, True, job, set(), set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert result["reasons"][0]["code"] == "HARD_GATE_MISSING"


def test_skip_when_explicit_hard_gate_flag_missing():
    job = build_job(["Bachelor degree or equivalent"], [])
    job["must_have"][0]["hard_gate"] = True
    step2 = build_step2(90, 90, True, job, set(), set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert result["reasons"][0]["code"] == "HARD_GATE_MISSING"
    assert result["missing_requirements"][0]["hard_gate"] is True


def test_determinism_repeated_decide_matches():
    job = build_job(["Python"], [])
    step2 = build_step2(80, 80, True, job, {"must_1"}, set())
    result_a = decide(step2, job)
    result_b = decide(step2, job)
    assert result_a == result_b
