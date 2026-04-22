from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.scoring import run_scoring  # noqa: E402
from app.scoring import decide  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ats_scenarios"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest() -> list[dict[str, Any]]:
    return _load_json(FIXTURES_DIR / "scenario_manifest.json")


def _score_resume_against_job(resume_json: dict[str, Any], job_json: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score_result = run_scoring(resume_json, job_json)
    decision_result = decide(score_result, job_json)
    return score_result, decision_result


def _check_expectation(scenario_id: str, score_result: dict[str, Any], decision_result: dict[str, Any]) -> tuple[bool, str]:
    score_total = float(score_result["score_total"])
    coverage = float(score_result["must_have"]["coverage_percent"])
    decision = str(decision_result["decision"])

    if scenario_id == "scenario_very_high_fit":
        if decision == "PROCEED" and score_total >= 85 and coverage >= 100:
            return True, "expected PROCEED with high score"
        return False, f"expected PROCEED with score >= 85 and coverage 100, got decision={decision} score={score_total} coverage={coverage}"

    if scenario_id == "scenario_high_imperfect_fit":
        if decision == "PROCEED" and coverage >= 80:
            return True, "expected PROCEED"
        return False, f"expected PROCEED with strong coverage, got decision={decision} coverage={coverage}"

    if scenario_id == "scenario_borderline_fit":
        if decision == "SKIP" and 55 <= score_total < 70:
            return True, "expected near-threshold borderline SKIP"
        if decision == "PROCEED" and 65 <= score_total < 75:
            return True, "expected near-threshold borderline PROCEED"
        return False, f"expected borderline outcome near threshold, got decision={decision} score={score_total}"

    if scenario_id == "scenario_low_fit":
        if decision == "SKIP" and score_total < 55:
            return True, "expected SKIP"
        return False, f"expected SKIP with low score, got decision={decision} score={score_total}"

    if scenario_id == "scenario_false_positive_risk":
        if decision == "SKIP":
            return True, "expected conservative SKIP"
        return False, f"expected SKIP, got decision={decision}"

    return False, f"unexpected scenario id: {scenario_id}"


def main() -> int:
    failures = 0
    job_json = _load_json(FIXTURES_DIR / "shared_job.json")

    for scenario in _load_manifest():
        resume_json = _load_json(FIXTURES_DIR / scenario["resume_fixture"])
        score_result, decision_result = _score_resume_against_job(resume_json, job_json)
        ok, message = _check_expectation(scenario["scenario_id"], score_result, decision_result)

        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {scenario['scenario_name']} "
            f"decision={decision_result['decision']} "
            f"score={score_result['score_total']} "
            f"coverage={score_result['must_have']['coverage_percent']} "
            f"- {message}"
        )
        if not ok:
            failures += 1

    if failures:
        print(f"\nScenario expectation checks failed: {failures}")
        return 1

    print("\nAll ATS scenario expectations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
