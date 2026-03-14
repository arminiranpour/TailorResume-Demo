import json
import os
import sys
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.resume_signals import extract_resume_signals  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_extract_resume_signals_preserves_evidence_and_recency():
    signals = extract_resume_signals(_load_fixture("ats_resume.json"))

    assert signals.recent_experience_order == ("exp_recent", "exp_older")
    assert "python" in signals.skill_terms
    assert "software engineer" in signals.title_like_terms
    assert "machine learning engineer" in signals.title_like_terms
    assert "computer science" in signals.section_terms["education"]

    python_evidence = {entry.source_id: entry for entry in signals.evidence_map["python"]}
    assert set(python_evidence) == {"summary", "skills_1", "exp_recent_b1"}
    assert python_evidence["skills_1"].section == "skills"
    assert python_evidence["exp_recent_b1"].exp_id == "exp_recent"
    assert python_evidence["exp_recent_b1"].bullet_index == 0
    assert python_evidence["exp_recent_b1"].experience_order == 0


def test_resume_signals_preserve_input_order_when_dates_are_unclear():
    resume = _load_fixture("ats_resume.json")
    modified = deepcopy(resume)
    modified["experience"][0]["end_date"] = ""

    signals = extract_resume_signals(modified)

    assert signals.recent_experience_order == ("exp_older", "exp_recent")
    assert signals.term_source_ids["node.js"] == ("summary", "exp_recent_b2")
