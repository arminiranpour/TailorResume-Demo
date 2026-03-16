import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.tailoring_plan import generate_tailoring_plan
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        raise RuntimeError("provider should not be called")


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer focused on APIs."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Communication, Teamwork"},
                {"line_id": "skills_2", "text": "Python, FastAPI, AWS"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built FastAPI APIs on AWS.",
                        "char_count": 27,
                    },
                    {
                        "bullet_id": "exp_1_b2",
                        "bullet_index": 1,
                        "text": "Mentored engineers across the team.",
                        "char_count": 33,
                    },
                ],
            }
        ],
        "projects": [],
        "education": [],
    }


def sample_job():
    return {
        "title": "Senior Backend Engineer",
        "must_have": [
            {"requirement_id": "req_python", "text": "Python"},
            {"requirement_id": "req_fastapi", "text": "FastAPI"},
        ],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "AWS"}],
        "responsibilities": ["Build backend APIs"],
        "keywords": ["Backend", "Python", "FastAPI", "AWS"],
    }


def sample_score():
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def test_plan_validates_against_schema():
    result = generate_tailoring_plan(sample_resume(), sample_job(), sample_score(), DummyProvider())

    ok, errors = validate_json("tailoring_plan", result)

    assert ok is True
    assert errors == []
    assert "bullet_actions" in result
    assert "prioritized_keywords" in result
    assert "supported_priority_terms" in result
    assert "blocked_terms" in result


def test_plan_generates_summary_and_skills_guidance_deterministically():
    result = generate_tailoring_plan(sample_resume(), sample_job(), sample_score(), DummyProvider())

    assert result["summary_rewrite"]["rewrite_intent"] == "rewrite"
    assert "backend engineer" in result["summary_alignment_terms"]
    assert result["summary_rewrite"]["title_alignment_safe"] is True
    assert result["skills_reorder_plan"] == ["skills_2", "skills_1"]
