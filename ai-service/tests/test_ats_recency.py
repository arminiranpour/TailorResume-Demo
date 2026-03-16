import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.coverage import build_coverage_model  # noqa: E402
from app.ats.evidence_linking import build_evidence_links  # noqa: E402
from app.ats.job_signals import extract_job_signals  # noqa: E402
from app.ats.recency import build_recency_priorities  # noqa: E402
from app.ats.resume_signals import extract_resume_signals  # noqa: E402
from app.ats.title_alignment import build_title_alignment  # noqa: E402
from app.ats.weighting import build_job_weights  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_recency(job_json: dict, resume_json: dict):
    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    title_alignment = build_title_alignment(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    priorities = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )
    return (
        job_signals,
        resume_signals,
        job_weights,
        coverage,
        evidence_links,
        title_alignment,
        priorities,
    )


def _fixture_recency():
    return _build_recency(_load_fixture("ats_job.json"), _load_fixture("ats_resume.json"))


def test_build_recency_priorities_creates_entry_for_every_weighted_term_and_is_deterministic():
    (
        job_signals,
        resume_signals,
        job_weights,
        coverage,
        evidence_links,
        title_alignment,
        first,
    ) = _fixture_recency()

    second = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )

    assert tuple(first.priorities_by_term) == job_weights.ordered_terms
    assert set(first.priorities_by_term) == set(job_weights.ordered_terms)
    assert first == second
    assert first.recency_ordered_terms == second.recency_ordered_terms
    assert first.prioritized_terms == tuple(
        term
        for term in first.recency_ordered_terms
        if first.priorities_by_term[term].strongest_overall_candidate is not None
    )


def test_build_recency_priorities_prefers_recent_experience_over_older_experience_when_comparable():
    job_json = {
        "title": "Platform Engineer",
        "must_have": [{"requirement_id": "req_1", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build Python services"],
        "keywords": ["Python"],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Platform engineer."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_oldest",
                "title": "Platform Engineer",
                "start_date": "2018-01",
                "end_date": "2020-12",
                "bullets": [
                    {
                        "bullet_id": "exp_oldest_b1",
                        "bullet_index": 0,
                        "text": "Built Python ETL jobs.",
                    }
                ],
            },
            {
                "exp_id": "exp_middle",
                "title": "Systems Engineer",
                "start_date": "2021-01",
                "end_date": "2021-12",
                "bullets": [
                    {
                        "bullet_id": "exp_middle_b1",
                        "bullet_index": 0,
                        "text": "Built internal tooling.",
                    }
                ],
            },
            {
                "exp_id": "exp_recent",
                "title": "Senior Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs.",
                    }
                ],
            },
        ],
    }

    *_, priorities = _build_recency(job_json, resume_json)
    python = priorities.priorities_by_term["python"]

    assert python.strongest_overall_candidate is not None
    assert python.strongest_overall_candidate.source_id == "exp_recent_b1"
    assert python.strongest_recent_candidate is not None
    assert python.strongest_recent_candidate.source_id == "exp_recent_b1"
    assert python.strongest_recent_experience_candidate is not None
    assert python.strongest_recent_experience_candidate.source_id == "exp_recent_b1"
    assert python.recent_source_ids == ("exp_recent_b1",)
    assert python.stale_source_ids == ("exp_oldest_b1",)
    assert python.has_recent_experience_backing is True
    assert python.recency_boost_applied is True


def test_build_recency_priorities_does_not_treat_skills_only_terms_as_recent_backing():
    *_, priorities = _fixture_recency()
    dot_net = priorities.priorities_by_term[".net"]

    assert dot_net.has_recent_backing is False
    assert dot_net.has_recent_experience_backing is False
    assert dot_net.has_recent_project_backing is False
    assert dot_net.has_only_stale_backing is False
    assert dot_net.recent_source_ids == ()
    assert dot_net.stale_source_ids == ()
    assert dot_net.is_recent_and_bullet_safe is False
    assert dot_net.is_recent_and_summary_safe is False


def test_build_recency_priorities_groups_recent_safe_terms_and_stale_only_terms_conservatively():
    job_json = {
        "title": "Senior Platform Engineer",
        "must_have": [
            {"requirement_id": "req_1", "text": "Python"},
            {"requirement_id": "req_2", "text": "Kubernetes"},
            {"requirement_id": "req_3", "text": "React"},
            {"requirement_id": "req_4", "text": "GraphQL"},
        ],
        "nice_to_have": [],
        "responsibilities": [
            "Build Python APIs and React interfaces",
            "Maintain Kubernetes workloads",
        ],
        "keywords": ["Python", "Kubernetes", "React", "GraphQL"],
    }
    resume_json = {
        "summary": {
            "id": "summary",
            "text": "Platform engineer with GraphQL delivery and internal tooling.",
        },
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "React, GraphQL, Kubernetes"}],
        },
        "experience": [
            {
                "exp_id": "exp_older",
                "title": "Software Engineer",
                "start_date": "2018-01",
                "end_date": "2020-12",
                "bullets": [
                    {
                        "bullet_id": "exp_older_b1",
                        "bullet_index": 0,
                        "text": "Managed Kubernetes clusters.",
                    }
                ],
            },
            {
                "exp_id": "exp_middle",
                "title": "Platform Engineer",
                "start_date": "2021-01",
                "end_date": "2021-12",
                "bullets": [
                    {
                        "bullet_id": "exp_middle_b1",
                        "bullet_index": 0,
                        "text": "Improved internal reporting workflows.",
                    }
                ],
            },
            {
                "exp_id": "exp_recent",
                "title": "Senior Platform Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs and platform services.",
                    }
                ],
            },
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Operations Portal",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Built React dashboard for operations workflows.",
                    }
                ],
            }
        ],
    }

    *_, priorities = _build_recency(job_json, resume_json)

    python = priorities.priorities_by_term["python"]
    react = priorities.priorities_by_term["react"]
    kubernetes = priorities.priorities_by_term["kubernetes"]
    graphql = priorities.priorities_by_term["graphql"]

    assert python.has_recent_experience_backing is True
    assert python.is_recent_and_bullet_safe is True
    assert python.is_recent_and_summary_safe is True

    assert react.has_recent_project_backing is True
    assert react.is_recent_and_bullet_safe is True
    assert react.is_recent_and_summary_safe is True

    assert kubernetes.has_recent_backing is False
    assert kubernetes.has_only_stale_backing is True
    assert kubernetes.stale_source_ids == ("exp_older_b1",)
    assert "kubernetes" in priorities.stale_only_terms

    assert graphql.has_recent_backing is False
    assert graphql.has_only_stale_backing is False
    assert graphql.is_recent_and_summary_safe is False

    assert "python" in priorities.recent_bullet_safe_terms
    assert "react" in priorities.recent_bullet_safe_terms
    assert "python" in priorities.recent_summary_safe_terms
    assert "react" in priorities.recent_summary_safe_terms
    assert "graphql" not in priorities.recent_summary_safe_terms

    assert "python" in priorities.recent_high_priority_terms
    assert "react" in priorities.recent_high_priority_terms
    assert "kubernetes" in priorities.stale_high_priority_terms
    assert priorities.recency_ordered_terms.index("python") < priorities.recency_ordered_terms.index(
        "kubernetes"
    )
