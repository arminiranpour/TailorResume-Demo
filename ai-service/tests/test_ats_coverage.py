import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.coverage import build_coverage_model  # noqa: E402
from app.ats.job_signals import extract_job_signals  # noqa: E402
from app.ats.resume_signals import extract_resume_signals  # noqa: E402
from app.ats.weighting import build_job_weights  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_fixture_coverage():
    job_signals = extract_job_signals(_load_fixture("ats_job.json"))
    resume_signals = extract_resume_signals(_load_fixture("ats_resume.json"))
    job_weights = build_job_weights(job_signals)
    return job_signals, resume_signals, job_weights, build_coverage_model(
        job_signals,
        resume_signals,
        job_weights,
    )


def test_build_coverage_model_tracks_covered_missing_and_section_provenance():
    _, _, _, coverage = _build_fixture_coverage()

    rest_api = coverage.coverage_by_term["rest api"]
    missing = coverage.coverage_by_term["ci/cd pipelines"]

    assert rest_api.is_covered is True
    assert rest_api.is_missing is False
    assert rest_api.coverage_strength == "strong"
    assert rest_api.source_ids == ("exp_recent_b1",)
    assert rest_api.source_sections == ("experience",)
    assert rest_api.source_ids_by_section["experience"] == ("exp_recent_b1",)
    assert rest_api.section_presence.experience is True
    assert rest_api.section_presence.skills is False

    assert missing.is_covered is False
    assert missing.is_missing is True
    assert missing.coverage_strength == "missing"
    assert missing.source_ids == ()
    assert missing.source_sections == ()


def test_build_coverage_model_flags_under_supported_and_cross_section_support():
    _, _, _, coverage = _build_fixture_coverage()

    dot_net = coverage.coverage_by_term[".net"]
    postgresql = coverage.coverage_by_term["postgresql"]

    assert dot_net.is_covered is True
    assert dot_net.coverage_strength == "weak"
    assert dot_net.is_under_supported is True
    assert dot_net.source_sections == ("skills",)
    assert dot_net.has_skills_support is True
    assert dot_net.has_experience_support is False

    assert postgresql.coverage_strength == "strong"
    assert postgresql.is_under_supported is False
    assert postgresql.has_cross_section_support is True
    assert postgresql.source_sections == ("skills", "experience")
    assert postgresql.source_ids_by_section["skills"] == ("skills_1",)
    assert postgresql.source_ids_by_section["experience"] == ("exp_recent_b1",)


def test_build_coverage_model_tracks_required_missing_and_summary_only_required_support():
    job_json = {
        "title": "Platform Engineer",
        "must_have": [
            {"requirement_id": "req_1", "text": "Python"},
            {"requirement_id": "req_2", "text": "K8s"},
        ],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Platform engineer with Python."},
        "skills": {"id": "skills", "lines": []},
        "experience": [],
    }

    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)

    python = coverage.coverage_by_term["python"]
    kubernetes = coverage.coverage_by_term["kubernetes"]

    assert python.is_required is True
    assert python.coverage_strength == "medium"
    assert python.is_under_supported is True
    assert python.source_sections == ("summary",)

    assert kubernetes.is_missing is True
    assert coverage.required_missing_terms == ("kubernetes",)
    assert coverage.required_coverage == 0.5


def test_build_coverage_model_tracks_title_missing_and_aggregate_ratios():
    _, _, _, coverage = _build_fixture_coverage()

    assert coverage.title_terms_missing == ("javascript software engineer",)
    assert coverage.covered_terms[0] == "javascript"
    assert coverage.missing_terms[:3] == (
        "ci/cd pipelines",
        "javascript software engineer",
        "own ci/cd automation",
    )
    assert coverage.summary.total_terms == 33
    assert coverage.summary.covered_terms == 23
    assert coverage.summary.missing_terms == 10
    assert coverage.summary.under_supported_terms == 7
    assert coverage.under_supported_terms == (
        "javascript",
        ".net",
        "c#",
        "c",
        "dot",
        "net",
        "sharp",
    )
    assert coverage.overall_distinct_coverage == 23 / 33
    assert coverage.high_priority_coverage == 1.0
    assert coverage.required_coverage == 1.0
    assert coverage.title_coverage == 5 / 6


def test_build_coverage_model_is_deterministic_and_preserves_weight_order():
    job_signals, resume_signals, job_weights, first = _build_fixture_coverage()

    second = build_coverage_model(job_signals, resume_signals, job_weights)

    assert first == second
    assert first.coverage_ordered_terms == job_weights.ordered_terms
    assert first.cross_section_supported_terms == (
        "ci/cd",
        "node.js",
        "engineer",
        "postgresql",
        "react",
    )
