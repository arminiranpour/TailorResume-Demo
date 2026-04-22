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


def _score_resume_against_job(resume_json: dict[str, Any], job_json: dict[str, Any]) -> dict[str, Any]:
    score_result = run_scoring(resume_json, job_json)
    decision_result = decide(score_result, job_json)
    return {
        "score": score_result,
        "decision": decision_result,
    }


def _format_requirements(items: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in items:
        text = item.get("text")
        if isinstance(text, str) and text:
            values.append(text)
    return values


def main() -> int:
    job_json = _load_json(FIXTURES_DIR / "shared_job.json")
    manifest = _load_manifest()

    for scenario in manifest:
        resume_path = FIXTURES_DIR / scenario["resume_fixture"]
        resume_json = _load_json(resume_path)
        result = _score_resume_against_job(resume_json, job_json)
        score_result = result["score"]
        decision_result = result["decision"]

        matched = _format_requirements(decision_result["matched_requirements"])
        missing = _format_requirements(decision_result["missing_requirements"])

        print(f"{scenario['scenario_name']} ({scenario['scenario_id']})")
        print(f"  decision: {decision_result['decision']}")
        print(f"  score_total: {score_result['score_total']}")
        print(f"  must_have_coverage_percent: {score_result['must_have']['coverage_percent']}")
        print(f"  matched_requirements: {matched if matched else ['<none>']}")
        print(f"  missing_requirements: {missing if missing else ['<none>']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
