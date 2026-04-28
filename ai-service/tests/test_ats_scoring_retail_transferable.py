import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.scoring import run_ats_scoring  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "ats_diagnostics")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _unrelated_resume() -> dict:
    return {
        "summary": {
            "id": "summary",
            "text": "Warehouse associate focused on shipping, receiving, forklift operation, and cycle counts.",
        },
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Forklift, pallet jack, shipping, receiving"},
                {"line_id": "skills_2", "text": "Cycle counts, warehouse safety, packing"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Distribution Hub",
                "title": "Warehouse Associate",
                "start_date": "2022-01",
                "end_date": "2025-04",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Loaded outbound shipments and completed daily cycle counts.",
                    },
                    {
                        "bullet_id": "exp_1_b2",
                        "bullet_index": 1,
                        "text": "Operated forklifts and pallet jacks while following warehouse safety rules.",
                    },
                ],
            }
        ],
        "education": [],
        "projects": [],
    }


def test_retail_transferable_resume_gets_partial_credit_but_stays_skip():
    resume = _load_fixture("retail_transferable_but_missing_hard_gates_resume.json")
    job = _load_fixture("retail_transferable_but_missing_hard_gates_job.json")

    result = run_ats_scoring(resume, job)
    matched_ids = {item["requirement_id"] for item in result["matched_requirements"]}
    missing_by_id = {item["requirement_id"]: item for item in result["missing_requirements"]}
    hard_gate_ids = {
        item["requirement_id"]
        for reason in result["reasons"]
        if reason["code"] == "HARD_GATE_MISSING"
        for item in reason["details"]["missing"]
    }

    assert result["decision"] == "SKIP"
    assert 30 <= result["score_total"] < 60
    assert result["must_have_coverage_percent"] > 25
    assert result["must_have_strict_match_percent"] < result["must_have_coverage_percent"]
    assert "must_4" in matched_ids
    assert "must_1" not in matched_ids
    assert "must_2" not in matched_ids
    assert "must_7" not in matched_ids
    assert missing_by_id["must_1"]["hard_gate"] is True
    assert missing_by_id["must_2"]["hard_gate"] is True
    assert missing_by_id["must_7"]["hard_gate"] is True
    assert {"must_1", "must_2", "must_7"}.issubset(hard_gate_ids)
    assert "nice_1" in missing_by_id
    assert "nice_2" in missing_by_id
    assert "nice_3" in missing_by_id
    assert "nice_4" in missing_by_id


def test_retail_transferable_resume_still_keeps_missing_domain_requirements_missing():
    resume = _load_fixture("retail_transferable_but_missing_hard_gates_resume.json")
    job = _load_fixture("retail_transferable_but_missing_hard_gates_job.json")

    result = run_ats_scoring(resume, job)
    matched_ids = {item["requirement_id"] for item in result["matched_requirements"]}
    missing_ids = {item["requirement_id"] for item in result["missing_requirements"]}

    assert "must_1" in missing_ids
    assert "must_2" in missing_ids
    assert "must_3" in missing_ids
    assert "must_7" in missing_ids
    assert "nice_1" in missing_ids
    assert "nice_2" in missing_ids
    assert "must_1" not in matched_ids
    assert "must_2" not in matched_ids
    assert "nice_1" not in matched_ids
    assert "nice_2" not in matched_ids


def test_retail_transferable_resume_scores_above_unrelated_resume_without_proceeding():
    retail_resume = _load_fixture("retail_transferable_but_missing_hard_gates_resume.json")
    unrelated_resume = _unrelated_resume()
    job = _load_fixture("retail_transferable_but_missing_hard_gates_job.json")

    retail_result = run_ats_scoring(retail_resume, job)
    unrelated_result = run_ats_scoring(unrelated_resume, job)

    assert retail_result["decision"] == "SKIP"
    assert unrelated_result["decision"] == "SKIP"
    assert retail_result["must_have_coverage_percent"] > unrelated_result["must_have_coverage_percent"]
    assert retail_result["score_total"] > unrelated_result["score_total"]
