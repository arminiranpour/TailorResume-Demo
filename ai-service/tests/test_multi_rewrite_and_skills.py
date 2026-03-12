import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit
from app.providers.base import LLMProvider


def _extract_payload(messages):
    content = messages[1]["content"]
    start = content.find("BEGIN_UNTRUSTED_TEXT")
    end = content.find("END_UNTRUSTED_TEXT")
    if start == -1 or end == -1:
        return {}
    block = content[start + len("BEGIN_UNTRUSTED_TEXT") : end].strip()
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return {}


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        payload = _extract_payload(messages)
        bullet_id = payload.get("bullet_id", "")
        original_text = payload.get("original_text", "")
        rewritten = f"Refined {original_text}".strip()
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": rewritten,
                "keywords_used": payload.get("target_keywords", []),
                "notes": "",
            }
        )


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer focused on APIs."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Python, FastAPI, PostgreSQL"},
                {"line_id": "skills_2", "text": "Docker, AWS"},
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
                        "text": "Built FastAPI services with PostgreSQL.",
                        "char_count": 43,
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
                        "text": "Deployed services to AWS with Docker.",
                        "char_count": 41,
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


def sample_job():
    return {
        "must_have": [{"requirement_id": "req_fastapi", "text": "FastAPI"}],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "AWS"}],
        "responsibilities": ["Build APIs", "Deploy services"],
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
            {"bullet_id": "exp_1_b1", "rewrite_intent": "rewrite", "target_keywords": ["FastAPI"]},
            {"bullet_id": "exp_2_b1", "rewrite_intent": "rewrite", "target_keywords": ["AWS"]},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["FastAPI", "AWS"],
    }


def test_multiple_bullets_and_skills_tailored():
    resume = sample_resume()
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        DummyProvider(),
    )

    exp1_text = tailored["experience"][0]["bullets"][0]["text"]
    exp2_text = tailored["experience"][1]["bullets"][0]["text"]
    assert exp1_text != resume["experience"][0]["bullets"][0]["text"]
    assert exp2_text != resume["experience"][1]["bullets"][0]["text"]

    changed = [item for item in audit_log["bullet_details"] if item.get("changed") is True]
    assert len(changed) >= 2

    skills_lines = tailored["skills"]["lines"]
    assert len(skills_lines) == len(resume["skills"]["lines"])
    assert [line["line_id"] for line in skills_lines] == [line["line_id"] for line in resume["skills"]["lines"]]
    assert skills_lines[1]["text"].startswith("AWS")
