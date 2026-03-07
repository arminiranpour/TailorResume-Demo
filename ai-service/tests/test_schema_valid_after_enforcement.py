import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.allowed_vocab import build_allowed_vocab
from app.pipelines.budget_enforcement import enforce_budgets
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json


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
            if isinstance(max_chars, int) and max_chars >= 0:
                return json.dumps({"compressed_text": candidate[:max_chars]})
            return json.dumps({"compressed_text": candidate})
        raise RuntimeError("Unexpected prompt")


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
                        "text": "Built APIs using FastAPI.",
                        "char_count": 27,
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


def test_schema_valid_after_enforcement():
    original = sample_resume()
    tailored = sample_resume()
    tailored["summary"]["text"] = "Backend engineer focused on APIs and systems."
    allowed_vocab = build_allowed_vocab(original)
    final_resume, _, _ = enforce_budgets(
        original,
        tailored,
        DummyProvider(),
        allowed_vocab,
        budgets_override=None,
    )
    ok, errors = validate_json("resume", final_resume)
    assert ok, errors
