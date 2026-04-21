import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit
from app.providers.base import LLMProvider


class NoOpProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        raise RuntimeError("provider should not be called for deterministic skills optimization tests")


def _score() -> dict:
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def test_skills_optimization_uses_plan_priority_terms_and_preferred_line_targets():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Leadership, Teamwork"},
                {"line_id": "skills_2", "text": "Python"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python services with PostgreSQL.",
                        "char_count": 38,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Backend Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_postgresql", "text": "PostgreSQL"}],
        "responsibilities": ["Build PostgreSQL-backed APIs"],
        "keywords": ["Python", "PostgreSQL"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["PostgreSQL", "Python"],
        "supported_priority_terms": ["postgresql", "python"],
        "skill_priority_terms": ["postgresql", "python"],
        "recent_priority_terms": ["python"],
        "skills_reorder_plan": ["skills_2", "skills_1"],
    }

    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
        character_budgets={"skills_line_max_chars": {"skills_2": 24}},
    )

    assert tailored["skills"]["lines"][0]["text"] == "Leadership, Teamwork"
    assert tailored["skills"]["lines"][1]["text"] == "Python, PostgreSQL"
    assert [line["line_id"] for line in tailored["skills"]["lines"]] == ["skills_1", "skills_2"]
    assert any(detail["skip_reason"] == "priority_skill_surface" for detail in audit_log["skills_details"])


def test_blocked_skills_are_not_surfaced_even_with_resume_evidence():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Platform engineer."},
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "Python, Teamwork"}],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Platform Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python services and maintained Kubernetes clusters.",
                        "char_count": 56,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Platform Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_kubernetes", "text": "Kubernetes"}],
        "responsibilities": ["Operate Kubernetes services"],
        "keywords": ["Python", "Kubernetes"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["Python", "Kubernetes"],
        "supported_priority_terms": ["python", "kubernetes"],
        "skill_priority_terms": ["python", "kubernetes"],
        "blocked_terms": [
            {
                "term": "kubernetes",
                "priority_bucket": "high",
                "blocked_for": ["skills"],
                "reason": "manually_blocked_for_skills",
            }
        ],
    }

    tailored, _ = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
        character_budgets={"skills_line_max_chars": {"skills_1": 28}},
    )

    assert tailored["skills"]["lines"][0]["text"] == "Python, Teamwork"
    assert "Kubernetes" not in tailored["skills"]["lines"][0]["text"]


def test_unsupported_summary_only_technical_terms_are_rejected_for_skills():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Engineer with GraphQL expertise."},
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "Python, Teamwork"}],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs for internal tools.",
                        "char_count": 37,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "API Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_graphql", "text": "GraphQL"}],
        "responsibilities": ["Build GraphQL APIs"],
        "keywords": ["Python", "GraphQL"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["GraphQL", "Python"],
        "supported_priority_terms": ["graphql", "python"],
        "skill_priority_terms": ["graphql", "python"],
    }

    tailored, _ = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
        character_budgets={"skills_line_max_chars": {"skills_1": 28}},
    )

    assert tailored["skills"]["lines"][0]["text"] == "Python, Teamwork"
    assert "GraphQL" not in tailored["skills"]["lines"][0]["text"]


def test_canonical_supported_skill_variants_are_preferred_safely():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Frontend engineer."},
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "JS, ReactJS, Python"}],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Frontend Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built JavaScript applications with React.",
                        "char_count": 41,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Frontend Engineer",
        "must_have": [{"requirement_id": "req_js", "text": "JavaScript"}],
        "nice_to_have": [{"requirement_id": "req_react", "text": "React"}],
        "responsibilities": ["Build React interfaces"],
        "keywords": ["JavaScript", "React", "Python"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["JavaScript", "React", "Python"],
        "supported_priority_terms": ["javascript", "react", "python"],
        "skill_priority_terms": ["javascript", "react", "python"],
    }

    tailored, _ = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
        character_budgets={"skills_line_max_chars": {"skills_1": 28}},
    )

    assert tailored["skills"]["lines"][0]["text"] == "JavaScript, React, Python"


def test_dense_skills_lines_replace_weaker_terms_instead_of_adding_title_terms():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Engineer delivering platform services."},
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "JS, ReactJS, AWS"}],
        },
        "experience": [
            {
                "exp_id": "exp_recent",
                "company": "Acme",
                "title": "Platform Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python APIs with PostgreSQL.",
                        "char_count": 33,
                    }
                ],
            }
        ],
        "projects": [
            {
                "project_id": "proj_ui",
                "name": "Operations Portal",
                "bullets": [
                    {
                        "bullet_id": "proj_ui_b1",
                        "bullet_index": 0,
                        "text": "Built ReactJS dashboard for operations.",
                        "char_count": 39,
                    }
                ],
            }
        ],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Senior Platform Engineer",
        "must_have": [
            {"requirement_id": "req_python", "text": "Python"},
            {"requirement_id": "req_postgresql", "text": "PostgreSQL"},
            {"requirement_id": "req_react", "text": "React"},
        ],
        "nice_to_have": [
            {"requirement_id": "req_aws", "text": "AWS"},
            {"requirement_id": "req_js", "text": "JavaScript"},
        ],
        "responsibilities": [
            "Build Python APIs with PostgreSQL for platform systems.",
            "Ship React interfaces for operations workflows.",
        ],
        "keywords": ["Senior Platform Engineer", "Python", "PostgreSQL", "React", "JavaScript", "AWS"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_recent_b1", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "proj_ui_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["platform", "postgresql", "python", "react", "javascript"],
        "supported_priority_terms": ["platform", "postgresql", "python", "react", "engineer"],
        "skill_priority_terms": ["platform", "postgresql", "python", "react", "aws", "javascript"],
        "recent_priority_terms": ["postgresql", "python", "react", "platform"],
        "summary_alignment_terms": ["platform engineer", "platform", "engineer"],
        "under_supported_terms": [
            {"term": "aws", "priority_bucket": "medium", "safe_for": ["skills"], "reason": "under_supported_resume_evidence"},
            {"term": "javascript", "priority_bucket": "medium", "safe_for": ["skills"], "reason": "under_supported_resume_evidence"},
        ],
        "title_alignment_status": {
            "is_title_supported": True,
            "is_safe_for_summary_alignment": True,
            "alignment_strength": "strong",
            "supported_terms": ["platform engineer", "platform", "engineer"],
            "missing_tokens": ["senior"],
            "strongest_matching_resume_title": "Platform Engineer",
        },
    }

    tailored, _ = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
        character_budgets={"skills_line_max_chars": {"skills_1": 40}},
    )

    assert tailored["skills"]["lines"][0]["text"] == "PostgreSQL, React, JavaScript"


def test_recent_supported_skills_outrank_weaker_stale_skills():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Analytics engineer."},
        "skills": {
            "id": "skills",
            "lines": [{"line_id": "skills_1", "text": "Tableau, Python, Leadership"}],
        },
        "experience": [
            {
                "exp_id": "exp_old",
                "company": "Acme",
                "title": "Analyst",
                "start_date": "2019-01",
                "end_date": "2021-12",
                "bullets": [
                    {
                        "bullet_id": "exp_old_b1",
                        "bullet_index": 0,
                        "text": "Built Tableau dashboards for reporting.",
                        "char_count": 39,
                    }
                ],
            },
            {
                "exp_id": "exp_recent",
                "company": "Beta",
                "title": "Analytics Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python data services and automation.",
                        "char_count": 42,
                    }
                ],
            },
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Analytics Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_tableau", "text": "Tableau"}],
        "responsibilities": ["Build Python automation and dashboards"],
        "keywords": ["Tableau", "Python"],
    }
    plan = {
        "bullet_actions": [
            {"bullet_id": "exp_old_b1", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "exp_recent_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["Tableau", "Python"],
        "supported_priority_terms": ["tableau", "python"],
        "skill_priority_terms": ["tableau", "python"],
        "recent_priority_terms": ["python"],
    }

    tailored, _ = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        plan,
        NoOpProvider(),
    )

    assert tailored["skills"]["lines"][0]["text"].startswith("Python, Tableau")


def test_skills_structure_and_repeated_runs_remain_deterministic_with_legacy_plan_shape():
    resume = {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Backend engineer."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "Leadership, Teamwork"},
                {"line_id": "skills_2", "text": "Python"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built Python services with PostgreSQL.",
                        "char_count": 38,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [{"edu_id": "edu_1", "school": "Uni", "degree": "BS", "start_date": "2016", "end_date": "2020"}],
    }
    job = {
        "title": "Backend Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [{"requirement_id": "req_postgresql", "text": "PostgreSQL"}],
        "responsibilities": ["Build PostgreSQL-backed APIs"],
        "keywords": ["Python", "PostgreSQL"],
    }
    legacy_plan = {
        "bullet_actions": [
            {"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }

    tailored_one, audit_one = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        legacy_plan,
        NoOpProvider(),
    )
    tailored_two, audit_two = rewrite_resume_text_with_audit(
        resume,
        job,
        _score(),
        legacy_plan,
        NoOpProvider(),
    )

    assert tailored_one == tailored_two
    assert audit_one == audit_two
    assert [line["line_id"] for line in tailored_one["skills"]["lines"]] == ["skills_1", "skills_2"]
    assert len(tailored_one["skills"]["lines"]) == 2
