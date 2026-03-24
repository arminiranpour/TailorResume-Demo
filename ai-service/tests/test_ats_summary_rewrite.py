import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


class SummaryProvider(LLMProvider):
    def __init__(self, rewritten_text, compressed_text=None):
        self.rewritten_text = rewritten_text
        self.compressed_text = compressed_text
        self.summary_payloads = []
        self.compress_payload = None

    def generate(self, messages, *, timeout=None, **kwargs):
        system_prompt = messages[0]["content"]
        payload = _extract_payload(messages)
        if "text compression engine" in system_prompt:
            self.compress_payload = payload
            compressed = self.compressed_text
            if compressed is None:
                compressed = payload.get("candidate_text", "")[: payload.get("max_chars", 0)]
            return json.dumps({"compressed_text": compressed})

        self.summary_payloads.append(payload)
        return json.dumps(
            {
                "rewritten_text": self.rewritten_text,
                "keywords_used": payload.get("preferred_surface_terms", []),
            }
        )


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "Engineer building Python services for APIs."},
        "skills": {"id": "skills", "lines": [{"line_id": "skills_1", "text": "Python, REST API, AWS, Kubernetes"}]},
        "experience": [
            {
                "exp_id": "exp_recent",
                "company": "Acme",
                "title": "Backend Engineer",
                "start_date": "2023-01",
                "end_date": "Present",
                "bullets": [
                    {
                        "bullet_id": "exp_recent_b1",
                        "bullet_index": 0,
                        "text": "Built Python REST API services with FastAPI.",
                        "char_count": 45,
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


def sample_job():
    return {
        "title": "Senior Backend Engineer",
        "must_have": [
            {"requirement_id": "req_python", "text": "python"},
            {"requirement_id": "req_rest_api", "text": "rest api"},
        ],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "aws"}],
        "responsibilities": ["Build backend APIs", "Ship reliable Python services"],
        "keywords": ["backend engineer", "python", "rest api", "aws", "kubernetes"],
    }


def sample_score():
    return {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }


def sample_plan(*, title_alignment_safe=True):
    return {
        "bullet_actions": [
            {"bullet_id": "exp_recent_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
        "summary_rewrite": {
            "rewrite_intent": "rewrite",
            "target_keywords": ["backend engineer", "rest api", "python"],
            "title_alignment_safe": title_alignment_safe,
            "title_terms": ["backend engineer"],
            "blocked_terms": ["kubernetes"],
        },
        "supported_priority_terms": ["python", "rest api"],
        "under_supported_terms": [
            {
                "term": "aws",
                "priority_bucket": "medium",
                "safe_for": ["skills"],
                "reason": "missing_summary_support",
            }
        ],
        "blocked_terms": [
            {
                "term": "kubernetes",
                "priority_bucket": "medium",
                "blocked_for": ["summary"],
                "reason": "unsupported_for_summary",
            }
        ],
        "recent_priority_terms": ["rest api"],
        "summary_alignment_terms": ["backend engineer"] if title_alignment_safe else [],
        "title_alignment_status": {
            "is_title_supported": True,
            "is_safe_for_summary_alignment": title_alignment_safe,
            "alignment_strength": "strong" if title_alignment_safe else "medium",
            "supported_terms": ["backend engineer"],
            "missing_tokens": ["senior"],
            "strongest_matching_resume_title": "Backend Engineer",
        },
    }


def test_summary_payload_uses_ats_plan_guidance_and_prefers_recent_supported_terms():
    provider = SummaryProvider("Backend engineer building Python REST APIs.")

    tailored, _ = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    payload = provider.summary_payloads[0]
    assert payload["target_keywords"] == ["backend engineer", "rest api", "python"]
    assert payload["allowed_title_terms"] == ["backend engineer"]
    assert payload["preferred_surface_terms"][:3] == ["backend engineer", "rest api", "python"]
    assert payload["recent_preferred_terms"] == ["rest api"]
    assert payload["required_supported_terms"] == ["python"]
    assert tailored["summary"]["text"] == "Backend engineer building Python REST APIs."


def test_safe_title_terms_are_rejected_when_summary_title_alignment_is_unsafe():
    provider = SummaryProvider("Backend engineer building Python REST APIs.")

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(title_alignment_safe=False),
        provider,
    )

    assert "backend engineer" not in tailored["summary"]["text"].lower()
    assert audit_log["summary_detail"]["fallback_used"] is True
    assert audit_log["summary_detail"]["reject_reason"] == "unsafe_title_alignment"


def test_blocked_summary_terms_are_rejected_deterministically():
    provider = SummaryProvider("Engineer building Python APIs with Kubernetes.")

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    assert "kubernetes" not in tailored["summary"]["text"].lower()
    assert audit_log["summary_detail"]["fallback_used"] is True
    assert audit_log["summary_detail"]["reject_reason"] == "blocked_terms"


def test_unsupported_summary_terms_are_rejected_even_when_globally_allowed():
    provider = SummaryProvider("Engineer building Python APIs with AWS.")

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    assert "aws" not in tailored["summary"]["text"].lower()
    assert audit_log["summary_detail"]["fallback_used"] is True
    assert audit_log["summary_detail"]["reject_reason"] == "unsupported_ats_terms"
    assert "aws" in audit_log["summary_detail"]["disallowed_terms"]


def test_summary_compression_revalidates_ats_rules_and_preserves_required_signal():
    provider = SummaryProvider(
        "Backend engineer building Python REST APIs for Python services.",
        compressed_text="Backend engineer building Python APIs.",
    )

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
        character_budgets={"summary": 40},
    )

    assert tailored["summary"]["text"] == "Backend engineer building Python APIs."
    assert len(tailored["summary"]["text"]) <= 40
    assert provider.compress_payload["preserve_terms"] == ["python"]
    assert provider.compress_payload["blocked_terms"] == ["kubernetes"]
    assert provider.compress_payload["avoid_terms"] == ["aws", "senior"]
    assert "summary" in audit_log["compressed"]


def test_pipeline_remains_compatible_with_legacy_summary_plan_and_repeated_runs_are_stable():
    legacy_plan = {
        "bullet_actions": [{"bullet_id": "exp_recent_b1", "rewrite_intent": "keep", "target_keywords": []}],
        "missing_requirements": [],
        "prioritized_keywords": [],
        "summary_rewrite": {"rewrite_intent": "rewrite", "target_keywords": ["python"]},
    }

    provider_one = SummaryProvider("Python engineer building API services.")
    tailored_one, audit_one = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        legacy_plan,
        provider_one,
    )

    provider_two = SummaryProvider("Python engineer building API services.")
    tailored_two, audit_two = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        legacy_plan,
        provider_two,
    )

    assert tailored_one == tailored_two
    assert audit_one == audit_two
    assert tailored_one["summary"]["text"] == "Python engineer building API services."
