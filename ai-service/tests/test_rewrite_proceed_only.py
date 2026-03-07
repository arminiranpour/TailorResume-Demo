import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import RewriteBulletsRequest, rewrite_bullets
from app.providers.base import LLMProvider
from app.schemas.schema_loader import load_schema


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout_seconds=None):
        raise RuntimeError("provider should not be called")


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


def sample_job():
    return {
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build APIs"],
        "keywords": ["Backend"],
    }


def sample_plan():
    return {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []}
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }


def test_refuses_when_decision_skip():
    score_result = {
        "decision": "SKIP",
        "reasons": [{"code": "SCORE_TOO_LOW", "message": "Score low", "details": None}],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 0,
    }
    payload = RewriteBulletsRequest(
        resume_json=sample_resume(),
        job_json=sample_job(),
        score_result=score_result,
        tailoring_plan=sample_plan(),
        character_budgets=None,
    )
    response = rewrite_bullets(payload, provider=DummyProvider(), schema_loader=load_schema)
    assert response.status_code == 409
