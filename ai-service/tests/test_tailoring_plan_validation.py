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

    def generate(self, messages, *, timeout=None, **kwargs):
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


def test_plan_autoselects_actionable_bullet():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, FastAPI"}]},
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
                        "text": "Built APIs using FastAPI.",
                        "char_count": 29,
                    }
                ],
            }
        ],
        "projects": [],
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
    job = {
        "must_have": [{"requirement_id": "req_fastapi", "text": "FastAPI"}],
        "nice_to_have": [],
        "responsibilities": ["Build APIs"],
        "keywords": ["Backend"],
    }
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
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }
    provider = FixedProvider(json.dumps(plan))
    result = generate_tailoring_plan(resume, job, score_result, provider)
    actionable = [
        item
        for item in result["bullet_actions"]
        if item.get("rewrite_intent") != "keep" and item.get("target_keywords")
    ]
    assert len(actionable) >= 1


def test_plan_autoselects_multiple_bullets_across_experiences():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, FastAPI, AWS"}]},
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
                        "text": "Built FastAPI services.",
                        "char_count": 24,
                    }
                ],
            },
            {
                "exp_id": "exp_2",
                "company": "Beta",
                "title": "Engineer",
                "start_date": "2021",
                "end_date": "2022",
                "bullets": [
                    {
                        "bullet_id": "exp_2_b1",
                        "bullet_index": 0,
                        "text": "Deployed to AWS.",
                        "char_count": 17,
                    }
                ],
            },
        ],
        "projects": [],
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
    job = {
        "must_have": [{"requirement_id": "req_fastapi", "text": "FastAPI"}],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "AWS"}],
        "responsibilities": ["Build APIs", "Deploy services"],
        "keywords": ["Backend"],
    }
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
            {"bullet_id": "exp_2_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }
    provider = FixedProvider(json.dumps(plan))
    result = generate_tailoring_plan(resume, job, score_result, provider)
    actionable = [
        item
        for item in result["bullet_actions"]
        if item.get("rewrite_intent") != "keep" and item.get("target_keywords")
    ]
    actionable_ids = {item.get("bullet_id") for item in actionable}
    assert len(actionable) >= 2
    assert {"exp_1_b1", "exp_2_b1"}.issubset(actionable_ids)
