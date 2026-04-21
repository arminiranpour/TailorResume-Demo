import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.tailoring_plan import generate_tailoring_plan
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        raise RuntimeError("provider should not be called")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_score() -> dict:
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [
            {
                "requirement_id": "req_3",
                "text": "Own CI/CD automation",
                "hard_gate": True,
            }
        ],
        "must_have_coverage_percent": 67,
    }


def _generate_fixture_plan() -> dict:
    return generate_tailoring_plan(
        _load_fixture("ats_resume.json"),
        _load_fixture("ats_job.json"),
        _fixture_score(),
        DummyProvider(),
    )


def test_planner_uses_ats_weighted_priorities():
    result = _generate_fixture_plan()

    assert "rest api" in result["prioritized_keywords"]
    assert "react" in result["prioritized_keywords"]
    assert result["prioritized_keywords"].index("rest api") < result["prioritized_keywords"].index("react")


def test_unsupported_terms_are_blocked():
    result = _generate_fixture_plan()
    blocked = {item["term"]: item for item in result["blocked_terms"]}

    assert blocked["ci/cd pipelines"]["reason"] == "no_resume_evidence"
    assert blocked["ci/cd pipelines"]["blocked_for"] == ["bullets", "summary", "skills"]


def test_high_priority_evidence_backed_terms_are_prioritized():
    result = _generate_fixture_plan()

    assert "node.js" in result["supported_priority_terms"]
    assert "rest api" in result["supported_priority_terms"]
    assert result["supported_priority_terms"].index("node.js") < result["supported_priority_terms"].index("react")


def test_recent_bullet_safe_evidence_is_preferred_over_stale_evidence():
    resume_json = {
        "summary": {"id": "summary", "text": "Platform engineer."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_old",
                "title": "Engineer",
                "start_date": "2019-01",
                "end_date": "2020-12",
                "bullets": [
                    {"bullet_id": "exp_old_b1", "bullet_index": 0, "text": "Built Python ETL jobs."}
                ],
            },
            {
                "exp_id": "exp_recent",
                "title": "Senior Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {"bullet_id": "exp_recent_b1", "bullet_index": 0, "text": "Built Python APIs."}
                ],
            },
        ],
        "projects": [],
    }
    job_json = {
        "title": "Platform Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build Python services"],
        "keywords": ["Python"],
    }
    score_result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }

    result = generate_tailoring_plan(resume_json, job_json, score_result, DummyProvider())
    actions = {item["bullet_id"]: item for item in result["bullet_actions"]}

    assert actions["exp_recent_b1"]["rewrite_intent"] == "rewrite"
    assert "python" in actions["exp_recent_b1"]["target_keywords"]
    assert actions["exp_recent_b1"]["is_recent"] is True
    assert actions["exp_old_b1"]["rewrite_intent"] == "keep"


def test_summary_title_alignment_is_only_planned_when_safe():
    safe_resume = {
        "summary": {"id": "summary", "text": "Platform engineer delivering backend systems."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python"}]},
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Senior Platform Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {"bullet_id": "exp_1_b1", "bullet_index": 0, "text": "Built Python APIs."}
                ],
            }
        ],
        "projects": [],
    }
    unsafe_resume = {
        "summary": {"id": "summary", "text": "Software engineer delivering backend systems."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python"}]},
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Software Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {"bullet_id": "exp_1_b1", "bullet_index": 0, "text": "Built Python APIs."}
                ],
            }
        ],
        "projects": [],
    }
    safe_job_json = {
        "title": "Senior Platform Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build platform APIs"],
        "keywords": ["Platform", "Python"],
    }
    unsafe_job_json = {
        "title": "Staff Platform Architect",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build platform APIs"],
        "keywords": ["Platform", "Python"],
    }
    score_result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }

    safe_plan = generate_tailoring_plan(safe_resume, safe_job_json, score_result, DummyProvider())
    unsafe_plan = generate_tailoring_plan(unsafe_resume, unsafe_job_json, score_result, DummyProvider())

    assert safe_plan["summary_rewrite"]["title_alignment_safe"] is True
    assert safe_plan["summary_alignment_terms"]
    assert unsafe_plan["summary_rewrite"]["title_alignment_safe"] is False
    assert unsafe_plan["summary_alignment_terms"] == []


def test_skills_reordering_favors_supported_high_priority_skills():
    resume_json = {
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Communication, Teamwork"},
                {"line_id": "skills_2", "text": "Python, AWS, PostgreSQL"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Backend Engineer",
                "start_date": "2021-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs on AWS with PostgreSQL.",
                    }
                ],
            }
        ],
        "projects": [],
    }
    job_json = {
        "title": "Backend Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "AWS"}],
        "responsibilities": ["Build APIs on AWS"],
        "keywords": ["Python", "AWS", "PostgreSQL"],
    }
    score_result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }

    result = generate_tailoring_plan(resume_json, job_json, score_result, DummyProvider())

    assert result["skills_reorder_plan"] == ["skills_2", "skills_1"]
    assert result["skill_priority_terms"][:2] == ["python", "aws"]


def test_missing_unsupported_requirements_remain_missing():
    result = _generate_fixture_plan()

    assert result["missing_requirements"] == _fixture_score()["missing_requirements"]
    assert "ci/cd pipelines" not in result["prioritized_keywords"]


def test_skills_only_terms_are_under_supported_not_fully_supported():
    resume_json = _load_fixture("ats_diagnostics/plan_resume.json")
    job_json = _load_fixture("ats_diagnostics/plan_job.json")
    score_result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }

    result = generate_tailoring_plan(resume_json, job_json, score_result, DummyProvider())
    blocked = {item["term"]: item for item in result["blocked_terms"]}
    under_supported = {item["term"]: item for item in result["under_supported_terms"]}

    assert "fastapi" in result["supported_priority_terms"]
    assert "python" in result["supported_priority_terms"]
    assert "aws" not in result["supported_priority_terms"]
    assert under_supported["aws"]["safe_for"] == ["skills"]
    assert under_supported["aws"]["reason"] == "under_supported_resume_evidence"
    assert blocked["kubernetes"]["blocked_for"] == ["bullets", "summary", "skills"]


def test_plan_ordering_is_deterministic():
    first = _generate_fixture_plan()
    second = _generate_fixture_plan()

    assert first["prioritized_keywords"] == second["prioritized_keywords"]
    assert [item["bullet_id"] for item in first["bullet_actions"]] == [
        item["bullet_id"] for item in second["bullet_actions"]
    ]
    assert first["bullet_actions"] == second["bullet_actions"]


def test_same_input_produces_exact_same_plan_output():
    first = _generate_fixture_plan()
    second = _generate_fixture_plan()

    assert first == second


def test_existing_plan_schema_compatibility_is_preserved():
    result = _generate_fixture_plan()
    ok, errors = validate_json("tailoring_plan", result)

    assert ok is True
    assert errors == []
    assert set(result).issuperset({"bullet_actions", "missing_requirements", "prioritized_keywords"})
