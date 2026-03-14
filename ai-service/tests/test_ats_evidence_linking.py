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
from app.ats.weighting import build_job_weights  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_fixture_links():
    job_signals = extract_job_signals(_load_fixture("ats_job.json"))
    resume_signals = extract_resume_signals(_load_fixture("ats_resume.json"))
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    return job_signals, resume_signals, job_weights, coverage, links


def test_build_evidence_links_creates_entry_for_every_weighted_term():
    _, _, job_weights, _, links = _build_fixture_links()

    assert tuple(links.links_by_term) == job_weights.ordered_terms
    assert set(links.links_by_term) == set(job_weights.ordered_terms)

    rest_api = links.links_by_term["rest api"]
    missing = links.links_by_term["ci/cd pipelines"]

    assert rest_api.strongest_candidate is not None
    assert rest_api.strongest_candidate.source_id == "exp_recent_b1"
    assert missing.strongest_candidate is None
    assert missing.all_candidates == ()


def test_build_evidence_links_prefers_experience_over_skills_only_support():
    _, _, _, _, links = _build_fixture_links()

    postgresql = links.links_by_term["postgresql"]

    assert tuple(candidate.source_id for candidate in postgresql.all_candidates) == (
        "skills_1",
        "exp_recent_b1",
    )
    assert tuple(candidate.source_id for candidate in postgresql.ranked_candidates) == (
        "exp_recent_b1",
        "skills_1",
    )
    assert postgresql.strongest_candidate == postgresql.strongest_experience_candidate
    assert postgresql.strongest_skills_candidate is not None


def test_build_evidence_links_prefers_more_recent_experience_when_section_strength_matches():
    job_json = {
        "title": "Platform Engineer",
        "must_have": [{"requirement_id": "req_1", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build Python services"],
        "keywords": [],
    }
    resume_json = {
        "summary": {"id": "summary", "text": "Platform engineer."},
        "skills": {"id": "skills", "lines": []},
        "experience": [
            {
                "exp_id": "exp_older",
                "title": "Engineer",
                "start_date": "2019-01",
                "end_date": "2021-12",
                "bullets": [
                    {"bullet_id": "exp_older_b1", "bullet_index": 0, "text": "Built Python ETL jobs."}
                ],
            },
            {
                "exp_id": "exp_recent",
                "title": "Senior Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {"bullet_id": "exp_recent_b1", "bullet_index": 0, "text": "Built Python APIs."}
                ],
            },
        ],
    }

    job_signals = extract_job_signals(job_json)
    resume_signals = extract_resume_signals(resume_json)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)

    python = links.links_by_term["python"]

    assert tuple(candidate.source_id for candidate in python.ranked_candidates[:2]) == (
        "exp_recent_b1",
        "exp_older_b1",
    )
    assert python.strongest_experience_candidate is not None
    assert python.strongest_experience_candidate.source_id == "exp_recent_b1"
    assert python.has_recent_backing is True


def test_build_evidence_links_groups_skills_only_terms_and_missing_experience_terms():
    _, _, _, _, links = _build_fixture_links()

    javascript = links.links_by_term["javascript"]
    dot_net = links.links_by_term[".net"]

    assert ".net" in links.skills_only_terms
    assert dot_net.has_skills_backing is True
    assert dot_net.has_experience_backing is False
    assert dot_net.is_safe_for_bullets is False

    assert "javascript" in links.missing_experience_terms
    assert javascript.has_skills_backing is True
    assert javascript.has_experience_backing is False


def test_build_evidence_links_computes_conservative_surface_safety_flags():
    _, _, _, _, links = _build_fixture_links()

    rest_api = links.links_by_term["rest api"]
    react = links.links_by_term["react"]
    javascript = links.links_by_term["javascript"]

    assert rest_api.is_safe_for_bullets is True
    assert rest_api.is_safe_for_summary is True
    assert rest_api.is_safe_for_skills is True

    assert react.is_safe_for_bullets is True
    assert react.strongest_project_candidate is not None
    assert react.strongest_project_candidate.source_id == "proj_1_b1"

    assert javascript.is_safe_for_bullets is False
    assert javascript.is_safe_for_summary is False
    assert javascript.is_safe_for_skills is True


def test_build_evidence_links_is_deterministic_and_returns_stable_ordering():
    job_signals, resume_signals, job_weights, coverage, first = _build_fixture_links()
    second = build_evidence_links(job_signals, resume_signals, job_weights, coverage)

    assert first == second
    assert first.linked_terms == tuple(
        term for term in job_weights.ordered_terms if first.links_by_term[term].all_candidates
    )
    assert first.evidence_ordered_terms.index("rest api") < first.evidence_ordered_terms.index(
        "ci/cd pipelines"
    )
    assert first.evidence_ordered_terms.index("react") < first.evidence_ordered_terms.index(
        "javascript"
    )
    assert first.links_by_term["rest api"].ranked_candidates[0].support_reasons == (
        "section:experience_bullet",
        "section_strength:very_strong",
        "section_score:500",
        "match_type:canonicalized_variant",
        "occurrence_count:1",
        "experience_recency_rank:0",
    )
