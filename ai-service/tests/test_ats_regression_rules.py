import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats import (  # noqa: E402
    build_coverage_model,
    build_evidence_links,
    build_frequency_balance,
    build_job_signals,
    build_job_weights,
    build_recency_priorities,
    build_resume_signals,
    build_title_alignment,
    validate_frequency_balance,
)
from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit  # noqa: E402
from app.pipelines.tailoring_plan import build_tailoring_plan  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402


class NullProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        raise RuntimeError("provider should not be called")


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


class ScenarioProvider(LLMProvider):
    def __init__(self, *, summary_text=None, bullet_text_by_id=None):
        self.summary_text = summary_text
        self.bullet_text_by_id = bullet_text_by_id or {}
        self.summary_payloads = []
        self.bullet_payloads = []
        self.compress_payloads = []

    def generate(self, messages, *, timeout=None, **kwargs):
        system_prompt = messages[0]["content"]
        payload = _extract_payload(messages)
        if "text compression engine" in system_prompt:
            self.compress_payloads.append(payload)
            return json.dumps({"compressed_text": payload.get("candidate_text", "")[: payload.get("max_chars", 0)]})

        bullet_id = payload.get("bullet_id")
        if isinstance(bullet_id, str):
            self.bullet_payloads.append(payload)
            rewritten_text = self.bullet_text_by_id.get(bullet_id, payload.get("original_text", ""))
            return json.dumps(
                {
                    "bullet_id": bullet_id,
                    "rewritten_text": rewritten_text,
                    "keywords_used": payload.get("preferred_surface_terms", []),
                    "notes": "",
                }
            )

        self.summary_payloads.append(payload)
        return json.dumps(
            {
                "rewritten_text": self.summary_text if isinstance(self.summary_text, str) else payload.get("original_text", ""),
                "keywords_used": payload.get("preferred_surface_terms", []),
            }
        )


def regression_score():
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def regression_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "ats-regression"},
        "summary": {
            "id": "summary",
            "text": "Backend engineer building Python APIs for platform services.",
        },
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_primary", "text": "Python, REST API, JS, ReactJS, Postgres"},
                {"line_id": "skills_secondary", "text": "Communication, Leadership"},
            ],
        },
        "experience": [
            {
                "exp_id": "exp_old",
                "company": "OldCo",
                "title": "Software Engineer",
                "start_date": "2019-01",
                "end_date": "2021-12",
                "bullets": [
                    {
                        "bullet_id": "exp_old_b1",
                        "bullet_index": 0,
                        "text": "Built Python ETL jobs for internal reporting.",
                        "char_count": 45,
                    }
                ],
            },
            {
                "exp_id": "exp_recent",
                "company": "NewCo",
                "title": "Platform Engineer",
                "start_date": "2022-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python REST APIs with PostgreSQL for platform services.",
                        "char_count": 62,
                    },
                    {
                        "bullet_id": "exp_recent_b2",
                        "bullet_index": 1,
                        "text": "Built React dashboards with JavaScript for internal tooling.",
                        "char_count": 60,
                    },
                ],
            },
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Developer Portal",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Built React dashboard backed by PostgreSQL analytics.",
                        "char_count": 54,
                    }
                ],
            }
        ],
        "education": [
            {
                "edu_id": "edu_1",
                "school": "State University",
                "degree": "BS Computer Science",
                "start_date": "2014",
                "end_date": "2018",
            }
        ],
    }


def regression_job(*, title="Senior Platform Engineer"):
    return {
        "title": title,
        "must_have": [
            {"requirement_id": "req_python", "text": "Python"},
            {"requirement_id": "req_rest", "text": "REST API"},
            {"requirement_id": "req_react", "text": "React"},
            {"requirement_id": "req_postgres", "text": "PostgreSQL"},
        ],
        "nice_to_have": [
            {"requirement_id": "req_js", "text": "JavaScript"},
            {"requirement_id": "req_aws", "text": "AWS"},
            {"requirement_id": "req_kubernetes", "text": "Kubernetes"},
        ],
        "responsibilities": [
            "Build Python platform APIs",
            "Ship React interfaces backed by PostgreSQL",
            "Partner on cloud delivery",
        ],
        "keywords": [
            "Platform Engineer",
            "Python",
            "REST API",
            "React",
            "JavaScript",
            "PostgreSQL",
            "AWS",
            "Kubernetes",
        ],
    }


def _build_ats_context(job_json=None, resume_json=None):
    job = job_json or regression_job()
    resume = resume_json or regression_resume()
    job_signals = build_job_signals(job)
    resume_signals = build_resume_signals(resume)
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
    recency = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )
    return {
        "job": job,
        "resume": resume,
        "job_signals": job_signals,
        "resume_signals": resume_signals,
        "job_weights": job_weights,
        "coverage": coverage,
        "evidence_links": evidence_links,
        "title_alignment": title_alignment,
        "recency": recency,
    }


def _build_plan(job_json=None, resume_json=None):
    resume = resume_json or regression_resume()
    job = job_json or regression_job()
    return build_tailoring_plan(resume, job, regression_score(), NullProvider())


def _build_frequency_snapshot(source_resume, tailored_resume, plan, job_json=None):
    context = _build_ats_context(job_json=job_json, resume_json=source_resume)
    return build_frequency_balance(
        source_resume_json=source_resume,
        tailored_resume_json=tailored_resume,
        tailoring_plan=plan,
        job_weights=context["job_weights"],
        coverage=context["coverage"],
        evidence_links=context["evidence_links"],
        recency=context["recency"],
        title_alignment=context["title_alignment"],
    )


def _bullet_detail(audit_log, bullet_id):
    return next(item for item in audit_log["bullet_details"] if item.get("bullet_id") == bullet_id)


def test_required_skills_outrank_preferred_skills_across_weighting_and_planning():
    context = _build_ats_context()
    plan = _build_plan()
    weights = context["job_weights"]

    assert weights.weights_by_term["python"].total_weight > weights.weights_by_term["aws"].total_weight
    assert weights.weights_by_term["rest api"].total_weight > weights.weights_by_term["javascript"].total_weight
    assert weights.weights_by_term["react"].total_weight > weights.weights_by_term["javascript"].total_weight
    assert weights.ordered_terms.index("python") < weights.ordered_terms.index("aws")
    assert weights.ordered_terms.index("rest api") < weights.ordered_terms.index("javascript")
    assert plan["prioritized_keywords"].index("python") < plan["prioritized_keywords"].index("javascript")
    assert plan["prioritized_keywords"].index("rest api") < plan["prioritized_keywords"].index("javascript")
    assert plan["supported_priority_terms"].index("python") < plan["supported_priority_terms"].index("javascript")
    assert any(item["term"] == "aws" for item in plan["blocked_terms"])
    assert "kubernetes" not in plan["supported_priority_terms"]


def test_truthful_cross_section_coverage_is_preserved_without_stuffing():
    source_resume = regression_resume()
    plan = _build_plan(resume_json=source_resume)
    context = _build_ats_context(resume_json=source_resume)
    python_coverage = context["coverage"].coverage_by_term["python"]

    provider = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )
    tailored, audit_log = rewrite_resume_text_with_audit(
        source_resume,
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    balance = _build_frequency_snapshot(source_resume, tailored, plan)
    python_status = balance.frequency_by_term["python"]

    assert python_coverage.has_cross_section_support is True
    assert python_coverage.source_sections == ("summary", "skills", "experience")
    assert "platform engineer" in plan["summary_rewrite"]["target_keywords"]
    assert "rest api" in plan["summary_rewrite"]["target_keywords"]
    assert python_status.section_counts["summary"] == 1
    assert python_status.section_counts["skills"] == 1
    assert python_status.section_counts["experience"] == 2
    assert python_status.status == "within_target"
    assert all(action["term"] != "python" for action in audit_log["frequency_actions"])


def test_unsupported_terms_are_blocked_and_do_not_survive_final_resume():
    source_resume = regression_resume()
    plan = _build_plan(resume_json=source_resume)
    blocked = {item["term"]: item for item in plan["blocked_terms"]}
    provider = ScenarioProvider(
        summary_text="Platform engineer building Python APIs with Kubernetes.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL and Kubernetes for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )

    tailored, audit_log = rewrite_resume_text_with_audit(
        source_resume,
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    bullet_detail = _bullet_detail(audit_log, "exp_recent_b1")

    assert blocked["kubernetes"]["blocked_for"] == ["bullets", "summary", "skills"]
    assert tailored["experience"][1]["bullets"][0]["text"] == source_resume["experience"][1]["bullets"][0]["text"]
    assert "kubernetes" not in json.dumps(tailored).lower()
    assert audit_log["summary_detail"]["reject_reason"] == "blocked_terms"
    assert audit_log["summary_detail"]["fallback_used"] is True
    assert bullet_detail["reject_reason"] == "blocked_terms"


def test_recent_evidence_is_preferred_across_evidence_recency_and_planning():
    context = _build_ats_context()
    plan = _build_plan()
    python_link = context["evidence_links"].links_by_term["python"]
    python_priority = context["recency"].priorities_by_term["python"]
    actions = {item["bullet_id"]: item for item in plan["bullet_actions"]}

    provider = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )
    rewrite_resume_text_with_audit(
        regression_resume(),
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    recent_payload = next(payload for payload in provider.bullet_payloads if payload.get("bullet_id") == "exp_recent_b1")

    assert python_link.strongest_candidate is not None
    assert python_link.strongest_candidate.source_id == "exp_recent_b1"
    assert python_priority.strongest_recent_experience_candidate is not None
    assert python_priority.strongest_recent_experience_candidate.source_id == "exp_recent_b1"
    assert tuple(candidate.source_id for candidate in python_link.ranked_candidates[:2]) == (
        "exp_recent_b1",
        "exp_old_b1",
    )
    assert python_priority.recent_source_ids == ("exp_old_b1", "exp_recent_b1")
    assert python_priority.stale_source_ids == ()
    assert actions["exp_recent_b1"]["rewrite_intent"] == "rewrite"
    assert actions["exp_recent_b1"]["is_recent"] is True
    assert actions["exp_old_b1"]["rewrite_intent"] == "keep"
    assert recent_payload["ats_emphasis"] == "strong"
    assert recent_payload["evidence_terms"][:3] == ["rest api", "api", "postgresql"]
    assert "python" in recent_payload["evidence_terms"]


def test_summary_alignment_improves_safely_and_unsafe_title_inflation_is_rejected():
    safe_resume = {
        "meta": {"total_pages": 1, "structure_hash": "title-safe"},
        "summary": {"id": "summary", "text": "Engineer building Python services."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, REST API"}]},
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
                        "text": "Built Python APIs for internal services.",
                        "char_count": 40,
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
                "start_date": "2018",
                "end_date": "2022",
            }
        ],
    }
    safe_job = {
        "title": "Senior Backend Engineer",
        "must_have": [{"requirement_id": "req_python", "text": "Python"}],
        "nice_to_have": [],
        "responsibilities": ["Build backend services"],
        "keywords": ["Backend Engineer", "Python"],
    }
    safe_plan = _build_plan(job_json=safe_job, resume_json=safe_resume)
    safe_provider = ScenarioProvider(
        summary_text="Backend engineer building Python services."
    )
    safe_tailored, safe_audit = rewrite_resume_text_with_audit(
        safe_resume,
        safe_job,
        regression_score(),
        safe_plan,
        safe_provider,
    )

    unsafe_resume = regression_resume()
    unsafe_job = regression_job(title="Senior Platform Architect")
    unsafe_plan = _build_plan(job_json=unsafe_job, resume_json=unsafe_resume)
    unsafe_provider = ScenarioProvider(
        summary_text="Senior Platform Architect building Python REST APIs for platform services."
    )
    unsafe_tailored, unsafe_audit = rewrite_resume_text_with_audit(
        unsafe_resume,
        unsafe_job,
        regression_score(),
        unsafe_plan,
        unsafe_provider,
    )

    assert safe_plan["summary_rewrite"]["title_alignment_safe"] is True
    assert safe_plan["summary_alignment_terms"][0] == "backend engineer"
    assert safe_tailored["summary"]["text"] == "Backend engineer building Python services."
    assert safe_audit["summary_detail"]["changed"] is True
    assert safe_audit["summary_detail"]["reject_reason"] is None
    assert "architect" in unsafe_plan["summary_rewrite"]["blocked_terms"]
    assert "senior" in unsafe_plan["summary_rewrite"]["blocked_terms"]
    assert "architect" not in unsafe_tailored["summary"]["text"].lower()
    assert "senior" not in unsafe_tailored["summary"]["text"].lower()
    assert unsafe_audit["summary_detail"]["reject_reason"] == "blocked_terms"
    assert "architect" in unsafe_audit["summary_detail"]["disallowed_terms"]


def test_frequency_balance_rolls_back_stuffed_summary_deterministically():
    source_resume = regression_resume()
    plan = _build_plan(resume_json=source_resume)
    provider = ScenarioProvider(
        summary_text="Platform engineer building Python Python Python REST APIs with Python services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )

    tailored, audit_log = rewrite_resume_text_with_audit(
        source_resume,
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    balance = _build_frequency_snapshot(source_resume, tailored, plan)

    assert tailored["summary"]["text"] == source_resume["summary"]["text"]
    assert audit_log["summary_detail"]["reject_reason"] == "frequency_balance"
    assert audit_log["summary_detail"]["skip_reason"] == "frequency_balance_rollback"
    assert audit_log["frequency_actions"] == [
        {
            "term": "python",
            "action": "rollback_surface",
            "section": "summary",
            "surface_id": "summary",
            "reason": "summary_cap_exceeded:4>1",
            "previous_text": "Platform engineer building Python Python Python REST APIs with Python services.",
            "final_text": source_resume["summary"]["text"],
        }
    ]
    assert balance.validation_errors == ()
    assert validate_frequency_balance(balance) == []


def test_canonical_terms_are_preferred_and_duplicate_variants_are_flagged():
    source_resume = regression_resume()
    plan = _build_plan(resume_json=source_resume)
    provider = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )

    rewrite_resume_text_with_audit(
        source_resume,
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    recent_payload = next(payload for payload in provider.bullet_payloads if payload.get("bullet_id") == "exp_recent_b1")
    duplicate_resume = copy.deepcopy(source_resume)
    duplicate_resume["skills"]["lines"][0]["text"] = (
        "Python, JavaScript, JS, React, ReactJS, PostgreSQL, Postgres"
    )
    duplicate_balance = _build_frequency_snapshot(source_resume, duplicate_resume, plan)
    duplicate_errors = validate_frequency_balance(duplicate_balance)

    assert {"javascript", "react", "postgresql"}.issubset(set(plan["skill_priority_terms"]))
    assert recent_payload["preferred_surface_terms"][:4] == ["rest api", "api", "postgresql", "python"]
    assert "term 'javascript' exceeds total cap 2 with count 3" in duplicate_errors
    assert "term 'react' exceeds skills cap 1 with count 2" in duplicate_errors
    assert "term 'postgresql' exceeds skills cap 1 with count 2" in duplicate_errors


def test_bullet_rewrite_preserves_action_and_supported_skill_context():
    source_resume = regression_resume()
    plan = _build_plan(resume_json=source_resume)
    provider = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Worked on backend improvements.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )

    tailored, audit_log = rewrite_resume_text_with_audit(
        source_resume,
        regression_job(),
        regression_score(),
        plan,
        provider,
    )
    bullet_detail = _bullet_detail(audit_log, "exp_recent_b1")

    assert tailored["experience"][1]["bullets"][0]["text"] == source_resume["experience"][1]["bullets"][0]["text"]
    assert bullet_detail["reject_reason"] == "missing_required_evidence_terms"
    assert "python" in bullet_detail["disallowed_terms"]
    assert tailored["experience"][1]["bullets"][0]["text"].startswith("Built Python REST APIs")


def test_repeated_identical_runs_produce_identical_plan_rewrite_and_frequency_outputs():
    source_resume = regression_resume()
    first_plan = _build_plan(resume_json=source_resume)
    second_plan = _build_plan(resume_json=copy.deepcopy(source_resume))

    provider_one = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )
    tailored_one, audit_one = rewrite_resume_text_with_audit(
        copy.deepcopy(source_resume),
        regression_job(),
        regression_score(),
        first_plan,
        provider_one,
    )
    balance_one = _build_frequency_snapshot(source_resume, tailored_one, first_plan)

    provider_two = ScenarioProvider(
        summary_text="Platform engineer building Python REST APIs for platform services.",
        bullet_text_by_id={
            "exp_recent_b1": "Built Python REST APIs with PostgreSQL for platform services.",
            "exp_recent_b2": "Built React dashboards with JavaScript for internal tooling.",
        },
    )
    tailored_two, audit_two = rewrite_resume_text_with_audit(
        copy.deepcopy(source_resume),
        regression_job(),
        regression_score(),
        second_plan,
        provider_two,
    )
    balance_two = _build_frequency_snapshot(source_resume, tailored_two, second_plan)

    assert first_plan == second_plan
    assert tailored_one == tailored_two
    assert audit_one == audit_two
    assert balance_one == balance_two
