import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit
from app.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        return json.dumps(
            {
                "bullet_id": "exp_1_b1",
                "rewritten_text": "Developed FastAPI APIs with PostgreSQL for backend services.",
                "keywords_used": ["FastAPI"],
                "notes": "",
            }
        )


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, FastAPI, PostgreSQL"}]},
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
                        "text": "Built APIs using FastAPI and PostgreSQL.",
                        "char_count": 43,
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
        "must_have": [{"requirement_id": "req_fastapi", "text": "FastAPI"}],
        "nice_to_have": [],
        "responsibilities": ["Build APIs"],
        "keywords": ["Backend"],
    }


def sample_score():
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def sample_plan():
    return {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "rewrite", "target_keywords": ["FastAPI"]}
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }


def test_rewrite_changes_text_and_preserves_structure():
    resume = sample_resume()
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        DummyProvider(),
    )
    text = tailored["experience"][0]["bullets"][0]["text"]
    assert text != resume["experience"][0]["bullets"][0]["text"]
    assert audit_log["rejected_for_new_terms"] == []
    assert audit_log["bullet_details"][0]["changed"] is True
    assert len(tailored["experience"]) == len(resume["experience"])
    assert len(tailored["projects"]) == len(resume["projects"])
    assert [b["bullet_id"] for b in tailored["experience"][0]["bullets"]] == [
        b["bullet_id"] for b in resume["experience"][0]["bullets"]
    ]
