import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.tailoring_plan import generate_tailoring_plan
from app.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        raise RuntimeError("provider should not be called")


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Engineer."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, AWS"}]},
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Engineer",
                "start_date": "2021-01",
                "end_date": "2022-12",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs.",
                        "char_count": 18,
                    }
                ],
            }
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Cloud Tool",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Deployed AWS tooling.",
                        "char_count": 22,
                    }
                ],
            }
        ],
        "education": [],
    }


def sample_job():
    return {
        "title": "Backend Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "AWS"}],
        "responsibilities": ["Build backend APIs"],
        "keywords": ["Python", "AWS"],
    }


def sample_score():
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def test_plan_preserves_resume_bullet_ids_and_order():
    result = generate_tailoring_plan(sample_resume(), sample_job(), sample_score(), DummyProvider())

    assert [item["bullet_id"] for item in result["bullet_actions"]] == ["exp_1_b1", "proj_1_b1"]
    assert result["bullet_actions"][0]["source_section"] == "experience_bullet"
    assert result["bullet_actions"][1]["source_section"] == "project_bullet"


def test_plan_marks_only_existing_bullets_for_rewrite():
    result = generate_tailoring_plan(sample_resume(), sample_job(), sample_score(), DummyProvider())

    rewritten_ids = {
        item["bullet_id"] for item in result["bullet_actions"] if item["rewrite_intent"] == "rewrite"
    }

    assert rewritten_ids.issubset({"exp_1_b1", "proj_1_b1"})
