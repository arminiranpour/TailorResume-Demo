import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats import (
    build_coverage_model,
    build_evidence_links,
    build_frequency_balance,
    build_job_weights,
    build_recency_priorities,
    build_title_alignment,
    extract_job_signals,
    extract_resume_signals,
    validate_frequency_balance,
)
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


class SummaryOnlyProvider(LLMProvider):
    def __init__(self, rewritten_text):
        self.rewritten_text = rewritten_text

    def generate(self, messages, *, timeout=None, **kwargs):
        system_prompt = messages[0]["content"]
        payload = _extract_payload(messages)
        if "text compression engine" in system_prompt:
            return json.dumps({"compressed_text": payload.get("candidate_text", "")})
        return json.dumps(
            {
                "rewritten_text": self.rewritten_text,
                "keywords_used": payload.get("preferred_surface_terms", []),
            }
        )


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "freq"},
        "summary": {"id": "summary", "text": "Backend engineer building Python services."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, JavaScript, REST API"}]},
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
                        "text": "Built Python REST API services.",
                        "char_count": 31,
                    }
                ],
            }
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Internal Tooling",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Created reporting dashboards.",
                        "char_count": 29,
                    }
                ],
            }
        ],
        "education": [
            {
                "edu_id": "edu_1",
                "school": "Uni",
                "degree": "BS Computer Science",
                "start_date": "2016",
                "end_date": "2020",
            }
        ],
    }


def sample_job():
    return {
        "title": "Senior Backend Engineer",
        "must_have": [
            {"requirement_id": "req_python", "text": "Python"},
            {"requirement_id": "req_rest", "text": "REST API"},
        ],
        "nice_to_have": [
            {"requirement_id": "req_js", "text": "JavaScript"},
            {"requirement_id": "req_kubernetes", "text": "Kubernetes"},
        ],
        "responsibilities": ["Build backend Python APIs", "Ship reliable services"],
        "keywords": ["Backend Engineer", "Python", "REST API", "JavaScript", "Kubernetes"],
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
        "bullet_actions": [{"bullet_id": "exp_1_b1", "rewrite_intent": "keep", "target_keywords": []}],
        "missing_requirements": [],
        "prioritized_keywords": ["python", "rest api", "javascript"],
        "summary_rewrite": {
            "rewrite_intent": "rewrite",
            "target_keywords": ["backend engineer", "python", "rest api"],
            "title_alignment_safe": True,
            "title_terms": ["backend engineer"],
        },
        "supported_priority_terms": ["python", "rest api", "javascript", "backend engineer"],
        "recent_priority_terms": ["python", "rest api"],
        "skill_priority_terms": ["python", "javascript", "rest api"],
        "summary_alignment_terms": ["backend engineer"],
        "under_supported_terms": [
            {
                "term": "kubernetes",
                "priority_bucket": "medium",
                "safe_for": [],
                "reason": "under_supported_resume_evidence",
            }
        ],
        "blocked_terms": [
            {
                "term": "kubernetes",
                "priority_bucket": "medium",
                "blocked_for": ["summary", "skills", "bullets"],
                "reason": "unsupported_for_tailoring",
            }
        ],
        "title_alignment_status": {
            "is_title_supported": True,
            "is_safe_for_summary_alignment": True,
            "alignment_strength": "strong",
            "supported_terms": ["backend engineer"],
            "missing_tokens": ["senior"],
            "strongest_matching_resume_title": "Backend Engineer",
        },
    }


def _build_balance(source_resume, tailored_resume, plan=None):
    job = sample_job()
    job_signals = extract_job_signals(job)
    source_signals = extract_resume_signals(source_resume)
    job_weights = build_job_weights(job_signals)
    coverage = build_coverage_model(job_signals, source_signals, job_weights)
    evidence_links = build_evidence_links(job_signals, source_signals, job_weights, coverage)
    title_alignment = build_title_alignment(
        job_signals=job_signals,
        resume_signals=source_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
    )
    recency = build_recency_priorities(
        job_signals=job_signals,
        resume_signals=source_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=title_alignment,
    )
    return build_frequency_balance(
        source_resume_json=source_resume,
        tailored_resume_json=tailored_resume,
        tailoring_plan=plan or sample_plan(),
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        recency=recency,
        title_alignment=title_alignment,
    )


def test_high_priority_technical_terms_are_capped_naturally():
    source_resume = sample_resume()
    tailored_resume = copy.deepcopy(source_resume)
    tailored_resume["summary"]["text"] = "Backend engineer building Python Python Python services."
    tailored_resume["experience"][0]["bullets"].append(
        {
            "bullet_id": "exp_1_b2",
            "bullet_index": 1,
            "text": "Scaled Python services for internal tooling.",
            "char_count": 43,
        }
    )

    balance = _build_balance(source_resume, tailored_resume)
    python_status = balance.frequency_by_term["python"]

    assert python_status.target_max_total == 3
    assert python_status.total_count == 6
    assert python_status.status == "stuffed"
    assert python_status.is_overused is True


def test_summary_stuffing_is_detected_against_summary_cap():
    source_resume = sample_resume()
    tailored_resume = copy.deepcopy(source_resume)
    tailored_resume["summary"]["text"] = "Backend engineer building Python Python Python services."

    balance = _build_balance(source_resume, tailored_resume)
    python_status = balance.frequency_by_term["python"]

    assert python_status.section_counts["summary"] == 3
    assert python_status.target_section_caps["summary"] == 1
    assert any("summary_cap_exceeded" in reason for reason in python_status.balancing_reasons)


def test_skills_canonical_duplicates_are_suppressed_by_frequency_caps():
    source_resume = sample_resume()
    tailored_resume = copy.deepcopy(source_resume)
    tailored_resume["skills"]["lines"][0]["text"] = "Python, JS, JavaScript"

    balance = _build_balance(source_resume, tailored_resume)
    javascript_status = balance.frequency_by_term["javascript"]

    assert javascript_status.section_counts["skills"] == 2
    assert javascript_status.target_section_caps["skills"] == 1
    assert javascript_status.is_overused is True


def test_blocked_or_unsupported_terms_are_hard_capped_at_zero():
    source_resume = sample_resume()
    tailored_resume = copy.deepcopy(source_resume)
    tailored_resume["summary"]["text"] = "Backend engineer building Python services with Kubernetes."

    balance = _build_balance(source_resume, tailored_resume)
    kubernetes_status = balance.frequency_by_term["kubernetes"]

    assert kubernetes_status.target_max_total == 0
    assert kubernetes_status.status == "hard_capped"
    assert "term 'kubernetes' exceeds hard cap 0 with count 1" in validate_frequency_balance(balance)


def test_section_distribution_is_measured_per_section():
    source_resume = sample_resume()
    tailored_resume = copy.deepcopy(source_resume)
    tailored_resume["experience"][0]["bullets"].append(
        {
            "bullet_id": "exp_1_b2",
            "bullet_index": 1,
            "text": "Scaled Python deployments.",
            "char_count": 26,
        }
    )
    tailored_resume["projects"][0]["bullets"][0]["text"] = "Created Python reporting dashboards."

    balance = _build_balance(source_resume, tailored_resume)
    python_status = balance.frequency_by_term["python"]

    assert python_status.section_counts == {
        "summary": 1,
        "skills": 1,
        "experience": 2,
        "projects": 1,
        "education": 0,
    }


def test_overused_terms_trigger_deterministic_summary_rollback():
    provider = SummaryOnlyProvider("Backend engineer building Python Python Python services.")

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    assert tailored["summary"]["text"] == "Backend engineer building Python services."
    assert audit_log["summary_detail"]["reject_reason"] == "frequency_balance"
    assert audit_log["summary_detail"]["skip_reason"] == "frequency_balance_rollback"
    assert audit_log["frequency_actions"] == [
        {
            "term": "python",
            "action": "rollback_surface",
            "section": "summary",
            "surface_id": "summary",
            "reason": "summary_cap_exceeded:3>1",
            "previous_text": "Backend engineer building Python Python Python services.",
            "final_text": "Backend engineer building Python services.",
        }
    ]
    assert audit_log["frequency_balance"]["validation_errors"] == ()


def test_within_range_terms_pass_without_frequency_rollback():
    provider = SummaryOnlyProvider("Backend engineer building Python and REST API services.")

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    assert tailored["summary"]["text"] == "Backend engineer building Python and REST API services."
    assert audit_log["frequency_actions"] == []
    assert audit_log["frequency_balance"]["validation_errors"] == ()


def test_repeated_identical_runs_produce_identical_frequency_results():
    provider_one = SummaryOnlyProvider("Backend engineer building Python Python Python services.")
    tailored_one, audit_one = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider_one,
    )

    provider_two = SummaryOnlyProvider("Backend engineer building Python Python Python services.")
    tailored_two, audit_two = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider_two,
    )

    assert tailored_one == tailored_two
    assert audit_one["frequency_actions"] == audit_two["frequency_actions"]
    assert audit_one["frequency_balance"] == audit_two["frequency_balance"]
