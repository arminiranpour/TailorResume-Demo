import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.scoring import decide


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


def assert_first_reason(result, expected_code):
    assert result["reasons"]
    assert result["reasons"][0]["code"] == expected_code


def run():
    job = build_job(["Python"], [])
    step2 = build_step2(80, 80, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "PROCEED"
    assert [r["code"] for r in result["reasons"]] == ["OK"]

    job = build_job(["Python"], [])
    step2 = build_step2(90, 90, False, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert_first_reason(result, "SENIORITY_GATE")

    job = build_job(["Python"], [])
    step2 = build_step2(54.99, 80, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert_first_reason(result, "SCORE_TOO_LOW")

    job = build_job(["Python"], [])
    step2 = build_step2(90, 69.99, True, job, {"must_1"}, set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert_first_reason(result, "MUST_HAVE_COVERAGE_TOO_LOW")

    job = build_job(["Security clearance required"], [])
    step2 = build_step2(90, 0, True, job, set(), set())
    result = decide(step2, job)
    assert result["decision"] == "SKIP"
    assert_first_reason(result, "HARD_GATE_MISSING")
    missing = {item["requirement_id"]: item for item in result["missing_requirements"]}
    assert missing["must_1"]["hard_gate"] is True


if __name__ == "__main__":
    run()
