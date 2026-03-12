import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.allowed_vocab import build_allowed_vocab
from app.pipelines.budget_enforcement import BudgetEnforcementError, enforce_budgets
from app.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
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


def test_integrity_invariants_fail():
    original = sample_resume()
    tailored = sample_resume()
    tailored["experience"][0]["bullets"] = tailored["experience"][0]["bullets"][:1]
    allowed_vocab = build_allowed_vocab(original)
    try:
        enforce_budgets(original, tailored, DummyProvider(), allowed_vocab, budgets_override=None)
    except BudgetEnforcementError as exc:
        assert any("bullets" in message for message in exc.details)
        return
    raise AssertionError("Expected BudgetEnforcementError")


def test_skills_line_reorder_fails():
    original = sample_resume()
    tailored = sample_resume()
    original["skills"]["lines"].append({"line_id": "skills_2", "text": "SQL"})
    tailored["skills"]["lines"] = [
        {"line_id": "skills_2", "text": "SQL"},
        {"line_id": "skills_1", "text": "Python"},
    ]
    allowed_vocab = build_allowed_vocab(original)
    try:
        enforce_budgets(original, tailored, DummyProvider(), allowed_vocab, budgets_override=None)
    except BudgetEnforcementError as exc:
        assert any("skills.lines order or ids changed" in message for message in exc.details)
        return
    raise AssertionError("Expected BudgetEnforcementError")
