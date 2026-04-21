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


def sample_resume():
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": "backend engineer focused on api delivery."},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": "skills_1", "text": "python, fastapi, postgresql, aws, kubernetes"},
            ],
        },
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
                        "text": "Built api services with fastapi and postgresql.",
                        "char_count": 46,
                    }
                ],
            },
            {
                "exp_id": "exp_old",
                "company": "Beta",
                "title": "Engineer",
                "start_date": "2020-01",
                "end_date": "2022-12",
                "bullets": [
                    {
                        "bullet_id": "exp_old_b1",
                        "bullet_index": 0,
                        "text": "Maintained internal tooling for reporting.",
                        "char_count": 40,
                    }
                ],
            },
        ],
        "projects": [
            {
                "project_id": "proj_1",
                "name": "Tooling",
                "bullets": [
                    {
                        "bullet_id": "proj_1_b1",
                        "bullet_index": 0,
                        "text": "Automated data exports.",
                        "char_count": 23,
                    }
                ],
            }
        ],
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
            {"requirement_id": "req_fastapi", "text": "fastapi"},
        ],
        "nice_to_have": [{"requirement_id": "req_aws", "text": "aws"}],
        "responsibilities": ["Build backend apis", "Ship reliable services"],
        "keywords": ["backend", "api", "fastapi", "postgresql", "aws"],
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
        "bullet_actions": [
            {
                "bullet_id": "exp_recent_b1",
                "rewrite_intent": "rewrite",
                "target_keywords": ["fastapi", "postgresql"],
                "evidence_terms": ["fastapi", "postgresql"],
                "source_section": "experience_bullet",
                "is_recent": True,
                "is_primary_evidence": True,
                "is_safe_for_ats": True,
            },
            {
                "bullet_id": "exp_old_b1",
                "rewrite_intent": "rewrite",
                "target_keywords": ["fastapi"],
                "evidence_terms": [],
                "source_section": "experience_bullet",
                "is_recent": False,
                "is_primary_evidence": False,
                "is_safe_for_ats": False,
            },
            {
                "bullet_id": "proj_1_b1",
                "rewrite_intent": "keep",
                "target_keywords": [],
                "evidence_terms": [],
                "source_section": "project_bullet",
                "is_recent": False,
                "is_primary_evidence": False,
                "is_safe_for_ats": False,
            },
        ],
        "missing_requirements": [],
        "prioritized_keywords": ["fastapi", "postgresql"],
        "supported_priority_terms": ["fastapi", "postgresql", "backend"],
        "under_supported_terms": [{"term": "kubernetes", "priority_bucket": "medium", "safe_for": [], "reason": "missing_evidence"}],
        "blocked_terms": [
            {"term": "kubernetes", "priority_bucket": "medium", "blocked_for": ["bullets"], "reason": "unsupported_for_bullets"}
        ],
        "recent_priority_terms": ["fastapi"],
        "summary_alignment_terms": ["backend engineer"],
        "skill_priority_terms": ["python", "aws", "fastapi"],
        "title_alignment_status": {
            "is_title_supported": True,
            "is_safe_for_summary_alignment": True,
            "alignment_strength": "strong",
            "supported_terms": ["backend engineer"],
            "missing_tokens": ["senior"],
            "strongest_matching_resume_title": "Backend Engineer",
        },
    }


class RecordingProvider(LLMProvider):
    def __init__(self):
        self.payloads = []

    def generate(self, messages, *, timeout=None, **kwargs):
        system_prompt = messages[0]["content"]
        payload = _extract_payload(messages)
        self.payloads.append(payload)
        if "compressed_text" in system_prompt:
            return json.dumps({"compressed_text": payload.get("candidate_text", "")[: payload.get("max_chars", 0)]})

        bullet_id = payload.get("bullet_id")
        if bullet_id == "exp_recent_b1":
            return json.dumps(
                {
                    "bullet_id": bullet_id,
                    "rewritten_text": "Built fastapi api services with postgresql for reliable delivery.",
                    "keywords_used": payload.get("preferred_surface_terms", []),
                    "notes": "",
                }
            )
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": "Maintained internal reporting tools.",
                "keywords_used": [],
                "notes": "",
            }
        )


class BlockingProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        payload = _extract_payload(messages)
        bullet_id = payload.get("bullet_id")
        rewritten_text = payload.get("original_text", "")
        keywords_used = []
        if bullet_id == "exp_recent_b1":
            rewritten_text = "Built kubernetes services for backend delivery."
            keywords_used = ["kubernetes"]
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": rewritten_text,
                "keywords_used": keywords_used,
                "notes": "",
            }
        )


class UnsupportedProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        payload = _extract_payload(messages)
        bullet_id = payload.get("bullet_id")
        rewritten_text = payload.get("original_text", "")
        keywords_used = []
        if bullet_id == "exp_recent_b1":
            rewritten_text = "Built fastapi api services with aws and postgresql."
            keywords_used = ["fastapi", "aws"]
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": rewritten_text,
                "keywords_used": keywords_used,
                "notes": "",
            }
        )


class CompressionProvider(LLMProvider):
    def __init__(self):
        self.compress_payload = None

    def generate(self, messages, *, timeout=None, **kwargs):
        system_prompt = messages[0]["content"]
        payload = _extract_payload(messages)
        if "compressed_text" in system_prompt:
            self.compress_payload = payload
            return json.dumps({"compressed_text": "Built fastapi apis with postgresql."})
        bullet_id = payload.get("bullet_id")
        rewritten_text = payload.get("original_text", "")
        keywords_used = []
        if bullet_id == "exp_recent_b1":
            rewritten_text = (
                "Built fastapi api services with postgresql for reliable delivery across distributed systems."
            )
            keywords_used = ["fastapi", "postgresql"]
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": rewritten_text,
                "keywords_used": keywords_used,
                "notes": "",
            }
        )


class StableProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, timeout=None, **kwargs):
        payload = _extract_payload(messages)
        system_prompt = messages[0]["content"]
        if "compressed_text" in system_prompt:
            return json.dumps({"compressed_text": payload.get("candidate_text", "")})
        self.calls += 1
        if payload.get("bullet_id") == "exp_recent_b1":
            return json.dumps(
                {
                    "bullet_id": "exp_recent_b1",
                    "rewritten_text": "Built fastapi api services with postgresql.",
                    "keywords_used": ["fastapi", "postgresql"],
                    "notes": "",
                }
            )
        return json.dumps(
            {
                "bullet_id": payload.get("bullet_id"),
                "rewritten_text": payload.get("original_text", ""),
                "keywords_used": [],
                "notes": "",
            }
        )


class IncidentalBlockedTermProvider(LLMProvider):
    def generate(self, messages, *, timeout=None, **kwargs):
        payload = _extract_payload(messages)
        bullet_id = payload.get("bullet_id")
        if bullet_id == "exp_recent_b1":
            return json.dumps(
                {
                    "bullet_id": bullet_id,
                    "rewritten_text": "Built fastapi api services with postgresql for platform services.",
                    "keywords_used": ["fastapi", "postgresql"],
                    "notes": "",
                }
            )
        return json.dumps(
            {
                "bullet_id": bullet_id,
                "rewritten_text": payload.get("original_text", ""),
                "keywords_used": [],
                "notes": "",
            }
        )


def test_ats_rewrite_payload_prefers_evidence_terms_and_preserves_structure():
    provider = RecordingProvider()
    resume = sample_resume()

    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
    )

    bullet_payload = next(payload for payload in provider.payloads if payload.get("bullet_id") == "exp_recent_b1")
    assert bullet_payload["evidence_terms"] == ["fastapi", "postgresql"]
    assert bullet_payload["preferred_surface_terms"][:2] == ["fastapi", "postgresql"]
    assert bullet_payload["ats_emphasis"] == "strong"
    assert tailored["experience"][0]["bullets"][0]["text"] == "Built fastapi api services with postgresql for reliable delivery."
    assert [b["bullet_id"] for b in tailored["experience"][0]["bullets"]] == [
        b["bullet_id"] for b in resume["experience"][0]["bullets"]
    ]
    assert [b["bullet_id"] for b in tailored["projects"][0]["bullets"]] == [
        b["bullet_id"] for b in resume["projects"][0]["bullets"]
    ]
    assert audit_log["rejected_for_new_terms"] == []


def test_blocked_terms_are_rejected_deterministically():
    resume = sample_resume()
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        BlockingProvider(),
    )

    assert tailored["experience"][0]["bullets"][0]["text"] == resume["experience"][0]["bullets"][0]["text"]
    assert audit_log["rejected_for_new_terms"] == ["exp_recent_b1"]
    assert audit_log["bullet_details"][0]["reject_reason"] == "blocked_terms"


def test_incidental_blocked_surface_can_be_removed_when_required_evidence_survives():
    resume = sample_resume()
    plan = sample_plan()
    plan["summary_alignment_terms"] = ["platform engineer", "platform", "engineer"]
    plan["title_alignment_status"] = {
        "is_title_supported": True,
        "is_safe_for_summary_alignment": True,
        "alignment_strength": "strong",
        "supported_terms": ["platform engineer", "platform", "engineer"],
        "missing_tokens": ["senior"],
        "strongest_matching_resume_title": "Platform Engineer",
    }
    plan["blocked_terms"] = [
        {"term": "platform", "priority_bucket": "high", "blocked_for": ["bullets"], "reason": "surface_not_safe"}
    ]

    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        plan,
        IncidentalBlockedTermProvider(),
    )

    assert tailored["experience"][0]["bullets"][0]["text"] == "Built fastapi api services with postgresql for services."
    assert audit_log["rejected_for_new_terms"] == []
    assert audit_log["bullet_details"][0]["changed"] is True


def test_unsupported_terms_are_rejected_even_when_globally_allowed():
    resume = sample_resume()
    tailored, audit_log = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        sample_plan(),
        UnsupportedProvider(),
    )

    assert tailored["experience"][0]["bullets"][0]["text"] == resume["experience"][0]["bullets"][0]["text"]
    assert audit_log["rejected_for_new_terms"] == ["exp_recent_b1"]
    assert audit_log["bullet_details"][0]["reject_reason"] == "unsupported_ats_terms"
    assert "aws" in audit_log["bullet_details"][0]["disallowed_terms"]


def test_recent_primary_rewrite_uses_strong_emphasis_and_compression_preserves_evidence():
    provider = CompressionProvider()
    budgets = {"bullets": {"exp_recent_b1": 36, "exp_old_b1": 80, "proj_1_b1": 80}}

    tailored, audit_log = rewrite_resume_text_with_audit(
        sample_resume(),
        sample_job(),
        sample_score(),
        sample_plan(),
        provider,
        character_budgets=budgets,
    )

    text = tailored["experience"][0]["bullets"][0]["text"]
    assert text == "Built fastapi apis with postgresql."
    assert len(text) <= 36
    assert "fastapi" in text
    assert provider.compress_payload["preserve_terms"] == ["fastapi", "postgresql"]
    assert provider.compress_payload["blocked_terms"] == ["kubernetes"]
    assert "exp_recent_b1" in audit_log["compressed"]


def test_pipeline_remains_compatible_with_legacy_plan_shape_and_repeated_runs_are_stable():
    resume = sample_resume()
    legacy_plan = {
        "bullet_actions": [
            {"bullet_id": "exp_recent_b1", "rewrite_intent": "rewrite", "target_keywords": ["fastapi"]},
            {"bullet_id": "exp_old_b1", "rewrite_intent": "keep", "target_keywords": []},
            {"bullet_id": "proj_1_b1", "rewrite_intent": "keep", "target_keywords": []},
        ],
        "missing_requirements": [],
        "prioritized_keywords": [],
    }

    provider_one = StableProvider()
    tailored_one, audit_one = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        legacy_plan,
        provider_one,
    )
    provider_two = StableProvider()
    tailored_two, audit_two = rewrite_resume_text_with_audit(
        resume,
        sample_job(),
        sample_score(),
        legacy_plan,
        provider_two,
    )

    assert tailored_one == tailored_two
    assert audit_one == audit_two
    assert tailored_one["experience"][0]["bullets"][0]["text"] == "Built fastapi api services with postgresql."
