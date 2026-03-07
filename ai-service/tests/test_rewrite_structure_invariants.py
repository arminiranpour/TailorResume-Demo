import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit
from app.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout_seconds=None):
        raise RuntimeError("provider should not be called")


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Python"},
                {"line_id": "skills_2", "text": "SQL"},
            ],
        },
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
                    },
                    {
                        "bullet_id": "exp_1_b2",
                        "bullet_index": 1,
                        "text": "Wrote tests.",
                        "char_count": 12,
                    },
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
                        "text": "Shipped tooling.",
                        "char_count": 16,
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
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "exp_1_b2", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "proj_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }


def test_structure_invariants_preserved():
    resume = sample_resume()
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        DummyProvider(),
    )
    assert audit_log["rewritten_bullets"] == []
    assert len(tailored["experience"]) == len(resume["experience"])
    assert len(tailored["projects"]) == len(resume["projects"])
    assert [b["bullet_id"] for b in tailored["experience"][0]["bullets"]] == [
        b["bullet_id"] for b in resume["experience"][0]["bullets"]
    ]
    assert [b["bullet_index"] for b in tailored["experience"][0]["bullets"]] == [
        b["bullet_index"] for b in resume["experience"][0]["bullets"]
    ]
