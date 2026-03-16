import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.coverage import build_coverage_model  # noqa: E402
from app.ats.evidence_linking import build_evidence_links  # noqa: E402
from app.ats.job_signals import extract_job_signals  # noqa: E402
from app.ats.resume_signals import extract_resume_signals  # noqa: E402
from app.ats.title_alignment import build_title_alignment  # noqa: E402
from app.ats.weighting import build_job_weights  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_alignment(job_json: dict, resume_json: dict):
    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    return build_title_alignment(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )


def test_build_title_alignment_extracts_job_title_tokens_and_phrases():
    job_json = {
        "title": "Senior Backend Software Engineer",
        "must_have": [],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Platform builder."},
        "skills": {"id": "skills", "lines": []},
        "experience": [],
    }

    alignment = _build_alignment(job_json, resume_json)

    assert alignment.job_title_tokens == ("senior", "backend", "software", "engineer")
    assert "software engineer" in alignment.job_title_phrases
    assert "backend engineer" in alignment.job_title_phrases
    assert "senior engineer" in alignment.job_title_phrases


def test_build_title_alignment_extracts_resume_title_tokens_and_phrases():
    job_json = {
        "title": "Backend Software Engineer",
        "must_have": [],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Backend engineer with Python."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Senior Software Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [],
            }
        ],
    }

    alignment = _build_alignment(job_json, resume_json)

    assert alignment.resume_title_tokens == ("backend", "engineer", "senior", "software")
    assert alignment.resume_title_phrases == (
        "backend engineer",
        "senior software engineer",
        "software engineer",
        "senior engineer",
    )
    assert alignment.overlapping_tokens == ("backend", "software", "engineer")
    assert alignment.overlapping_phrases == ("software engineer", "backend engineer")


def test_build_title_alignment_scores_support_and_missing_tokens_deterministically():
    job_json = {
        "title": "Backend Software Engineer",
        "must_have": [],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Platform engineer focused on Python."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Software Engineer",
                "start_date": "2021-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built backend services as a software engineer.",
                    }
                ],
            }
        ],
    }

    alignment = _build_alignment(job_json, resume_json)

    assert alignment.resume_title_tokens == ("platform", "engineer", "software")
    assert alignment.resume_title_phrases == ("platform engineer", "software engineer")
    assert alignment.overlapping_tokens == ("software", "engineer")
    assert alignment.overlapping_phrases == ("software engineer",)
    assert alignment.supporting_experience_ids == ("exp_1",)
    assert alignment.supporting_bullet_ids == ("exp_1_b1",)
    assert alignment.title_alignment_score == 12
    assert alignment.alignment_strength == "strong"
    assert alignment.is_title_supported is True
    assert alignment.is_safe_for_summary_alignment is True
    assert alignment.is_safe_for_experience_alignment is False
    assert alignment.missing_title_tokens == ("backend",)


def test_build_title_alignment_marks_full_experience_title_match_as_safe():
    job_json = {
        "title": "Senior Software Engineer",
        "must_have": [],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Backend engineer delivering APIs."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_1",
                "title": "Senior Software Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Worked as a senior software engineer on platform services.",
                    }
                ],
            }
        ],
    }

    alignment = _build_alignment(job_json, resume_json)

    assert alignment.strongest_matching_resume_title == "senior software engineer"
    assert alignment.overlapping_tokens == ("senior", "software", "engineer")
    assert alignment.is_safe_for_experience_alignment is True


def test_build_title_alignment_is_deterministic_for_fixture_inputs():
    job_json = _load_fixture("ats_job.json")
    resume_json = _load_fixture("ats_resume.json")

    first = _build_alignment(job_json, resume_json)
    second = _build_alignment(job_json, resume_json)

    assert first == second
    assert first.job_title_tokens == ("senior", "javascript", "software", "engineer")
    assert first.strongest_matching_resume_title == "senior software engineer"
    assert first.overlapping_tokens == ("senior", "software", "engineer")
    assert first.overlapping_phrases == ("software engineer", "senior engineer")
    assert first.missing_title_tokens == ("javascript",)
