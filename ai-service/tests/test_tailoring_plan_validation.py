import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.tailoring_plan import generate_tailoring_plan
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json


class FixedProvider(LLMProvider):
    def __init__(self, response):
        self.response = response

    def generate(self, messages, *, timeout_seconds=None):
        return self.response


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python"}]},
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Dev",
                "start_date": "2020",
                "end_date": "2021",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built APIs.",
                        "char_count": 11,
                    }
                ],
            }
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Tool",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Built tool.",
                        "char_count": 11,
                    }
                ],
            }
        ],
        "education": [
            {
                "edu_id": "edu_1",
                "school": "Uni",
                "degree": "BS",
                "start_date": "2016",
                "end_date": "2020",
            }
        ],
    }


def sample_job():
    return {
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build APIs"],
        "keywords": ["Backend"],
    }


def test_plan_validates_against_schema():
    score_result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "proj_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["built"],
    }
    provider = FixedProvider(json.dumps(plan))
    result = generate_tailoring_plan(sample_resume(), sample_job(), score_result, provider)
    ok, errors = validate_json("tailoring_plan", result)
    assert ok is True
    assert errors == []
