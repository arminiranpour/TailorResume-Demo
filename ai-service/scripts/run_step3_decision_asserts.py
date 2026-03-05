import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.scoring import decide

job = {
    "must_have": [
        {"requirement_id": "must_1", "text": "Python"},
        {"requirement_id": "must_2", "text": "Security clearance required"},
    ],
    "nice_to_have": [],
    "responsibilities": [],
    "keywords": [],
}

step2 = {
    "score_total": 80,
    "score_breakdown": {"must": 0, "nice": 0, "alignment": 0},
    "must_have": {"coverage_percent": 50},
    "seniority": {"seniority_ok": True},
    "matches": {
        "must_have": [
            {
                "requirement_id": "must_1",
                "text": "Python",
                "matched": True,
                "evidence": {"source_type": "summary", "source_id": "summary", "snippet": "Python"},
            },
            {
                "requirement_id": "must_2",
                "text": "Security clearance required",
                "matched": False,
                "evidence": None,
            },
        ],
        "nice_to_have": [],
    },
}

result = decide(step2, job)
assert result["decision"] == "SKIP"
assert result["reasons"][0]["code"] == "HARD_GATE_MISSING"
