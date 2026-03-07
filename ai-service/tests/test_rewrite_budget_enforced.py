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
    def generate(self, messages, *, timeout_seconds=None):
        system_prompt = messages[0]["content"]
        if "compressed_text" in system_prompt:
            payload = _extract_payload(messages)
            max_chars = payload.get("max_chars")
            candidate = payload.get("candidate_text", "")
            if isinstance(max_chars, int) and max_chars > 0:
                compressed = candidate[:max_chars]
            else:
                compressed = candidate
            return json.dumps({"compressed_text": compressed})
        return json.dumps(
            {
                "bullet_id": "exp_1_b1",
                "rewritten_text": "Built APIs using FastAPI and PostgreSQL for backend services with reliable delivery.",
                "keywords_used": ["FastAPI"],
                "notes": "",
            }
        )


def sample_resume():
    return {
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


def test_budget_enforced():
    resume = sample_resume()
    budgets = {"bullets": {"exp_1_b1": 40}}
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        DummyProvider(),
        character_budgets=budgets,
    )
    text = tailored["experience"][0]["bullets"][0]["text"]
    assert len(text) <= 40
    assert "exp_1_b1" in audit_log["compressed"]
