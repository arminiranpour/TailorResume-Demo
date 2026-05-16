from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.pipelines.llm_scoring import (  # noqa: E402
    build_llm_prompt_packet,
    build_scoring_packet,
    run_llm_adjudicated_scoring,
)
from app.pipelines.scoring import run_ats_scoring, run_ats_scoring_analysis  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402


SCENARIO_DIR = Path(__file__).resolve().parent / "fixtures" / "ats_scenarios"
DIAGNOSTIC_DIR = Path(__file__).resolve().parent / "fixtures" / "ats_diagnostics"


class StubProvider(LLMProvider):
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response

    def generate(self, messages, **kwargs) -> str:
        _ = messages, kwargs
        return self.raw_response


class FailingProvider(LLMProvider):
    def generate(self, messages, **kwargs) -> str:
        _ = messages, kwargs
        raise RuntimeError("provider unavailable")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_pair(resume_name: str) -> tuple[dict, dict]:
    job = _load_json(SCENARIO_DIR / "shared_job.json")
    resume = _load_json(SCENARIO_DIR / resume_name)
    return resume, job


def _retail_pair() -> tuple[dict, dict]:
    return (
        _load_json(DIAGNOSTIC_DIR / "retail_transferable_but_missing_hard_gates_resume.json"),
        _load_json(DIAGNOSTIC_DIR / "retail_transferable_but_missing_hard_gates_job.json"),
    )


def _valid_llm_response() -> dict:
    return {
        "score_total": 94,
        "decision": "PROCEED",
        "confidence": "high",
        "fit_summary": "Strong platform-engineering fit with direct experience evidence for nearly all core requirements.",
        "matched_requirements": [
            {"requirement_id": "req_python_backend", "evidence_source_ids": ["exp_1_b1"], "rationale": "Direct recent backend evidence."},
            {"requirement_id": "req_fastapi_rest", "evidence_source_ids": ["exp_1_b1"], "rationale": "Direct recent FastAPI and REST evidence."},
            {"requirement_id": "req_postgresql_modeling", "evidence_source_ids": ["exp_1_b2"], "rationale": "Recent data-modeling evidence."},
            {"requirement_id": "req_aws_cicd", "evidence_source_ids": ["exp_1_b4"], "rationale": "Recent AWS and CI/CD ownership."},
            {"requirement_id": "req_react_typescript", "evidence_source_ids": ["exp_1_b3"], "rationale": "Recent React and TypeScript dashboard evidence."},
            {"requirement_id": "req_kafka_events", "evidence_source_ids": ["exp_2_b1"], "rationale": "Prior direct Kafka evidence."},
            {"requirement_id": "req_graphql_integrations", "evidence_source_ids": ["exp_2_b1"], "rationale": "Prior direct GraphQL integration evidence."},
            {"requirement_id": "req_datadog_observability", "evidence_source_ids": ["exp_1_b4"], "rationale": "Recent Datadog monitoring evidence."},
        ],
        "missing_requirements": [],
        "transferable_matches": [
            {
                "requirement_id": "req_kubernetes_containers",
                "evidence_source_ids": ["exp_1_b4"],
                "rationale": "Kubernetes rollout evidence is relevant but only partially satisfies the requirement.",
            }
        ],
        "risk_flags": ["minor_preferred_gap"],
        "reasons": [
            {"code": "DIRECT_MATCH", "message": "Most required skills have strong recent evidence.", "severity": "info"}
        ],
        "score_breakdown": {
            "must_have": 58,
            "nice_to_have": 8,
            "transferable_fit": 2,
            "title_seniority": 10,
            "evidence_quality": 16,
            "risk_penalty": 0,
        },
    }


def _prompt_requirement_by_id(prompt_packet: dict, requirement_id: str) -> dict:
    for item in prompt_packet["job"]["must_have"] + prompt_packet["job"]["nice_to_have"]:
        if item["requirement_id"] == requirement_id:
            return item
    raise AssertionError(f"Requirement not found in prompt packet: {requirement_id}")


def _scoring_requirement_by_id(scoring_packet: dict, requirement_id: str) -> dict:
    for item in scoring_packet["job"]["must_have"] + scoring_packet["job"]["nice_to_have"]:
        if item["requirement_id"] == requirement_id:
            return item
    raise AssertionError(f"Requirement not found in scoring packet: {requirement_id}")


def test_llm_scorer_accepts_valid_strict_json_response():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider(json.dumps(_valid_llm_response()))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "PROCEED"
    assert result["score_total"] == 94
    assert result["confidence"] == "high"
    assert any(item["requirement_id"] == "req_python_backend" for item in result["matched_requirements"])


def test_llm_scorer_accepts_markdown_fenced_json_response():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider(f"```json\n{json.dumps(_valid_llm_response(), indent=2)}\n```")

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "PROCEED"


def test_llm_scorer_accepts_prefixed_json_response():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider(f"Here is the JSON result:\n{json.dumps(_valid_llm_response())}\nDone.")

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "PROCEED"


def test_llm_scorer_rejects_fabricated_matched_requirement():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 88,
                "decision": "PROCEED",
                "confidence": "medium",
                "fit_summary": "Invalid fabricated result.",
                "matched_requirements": [
                    {"requirement_id": "fake_requirement", "evidence_source_ids": ["exp_1_b1"], "rationale": "Fabricated."}
                ],
                "missing_requirements": [],
                "transferable_matches": [],
                "risk_flags": [],
                "reasons": [
                    {"code": "INVALID", "message": "Fabricated requirement.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 50,
                    "nice_to_have": 10,
                    "transferable_fit": 5,
                    "title_seniority": 10,
                    "evidence_quality": 13,
                    "risk_penalty": 0
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert result["decision"] == "PROCEED"
    assert result["reasons"][-1]["code"] == "LLM_ADJUDICATION_FALLBACK"


def test_llm_prompt_packet_marks_strong_and_transferable_eligibility():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    analysis = run_ats_scoring_analysis(resume, job)
    prompt_packet = build_llm_prompt_packet(build_scoring_packet(resume, job, analysis))

    strong_req = _prompt_requirement_by_id(prompt_packet, "req_python_backend")
    partial_req = _prompt_requirement_by_id(prompt_packet, "req_kubernetes_containers")

    assert strong_req["strong_match_allowed"] is True
    assert strong_req["transferable_match_allowed"] is True
    assert strong_req["evidence_strength"] == "strong_direct"

    assert partial_req["strong_match_allowed"] is False
    assert partial_req["transferable_match_allowed"] is True
    assert partial_req["evidence_strength"] == "partial_transferable"


def test_llm_scorer_falls_back_when_ineligible_requirement_is_strong_matched():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    bad_response = _valid_llm_response()
    bad_response["matched_requirements"].append(
        {
            "requirement_id": "req_kubernetes_containers",
            "evidence_source_ids": ["exp_1_b4"],
            "rationale": "Kubernetes is mentioned directly.",
        }
    )
    provider = StubProvider(json.dumps(bad_response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert "not eligible for strong match: req_kubernetes_containers" in result["llm_fallback_reason"]


def test_llm_scorer_accepts_same_ineligible_requirement_as_transferable_match():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["transferable_matches"] = [
        {
            "requirement_id": "req_kubernetes_containers",
            "evidence_source_ids": ["exp_1_b4"],
            "rationale": "Kubernetes rollout evidence is relevant but only partially satisfies the requirement.",
        }
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert any(item["requirement_id"] == "req_kubernetes_containers" for item in result["transferable_matches"])


def test_llm_scorer_repairs_partial_requirement_listed_as_missing_to_transferable():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    analysis = run_ats_scoring_analysis(resume, job)
    scoring_packet = build_scoring_packet(resume, job, analysis)
    requirement = _scoring_requirement_by_id(scoring_packet, "req_kubernetes_containers")
    response = _valid_llm_response()
    response["transferable_matches"] = []
    response["missing_requirements"] = [
        {
            "requirement_id": "req_kubernetes_containers",
            "rationale": "Incorrectly treated as missing despite covered partial evidence.",
        }
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert all(item["requirement_id"] != "req_kubernetes_containers" for item in result["missing_requirements"])
    transferable = next(
        item for item in result["transferable_matches"] if item["requirement_id"] == "req_kubernetes_containers"
    )
    assert transferable["evidence_source_ids"] == requirement["allowed_evidence_source_ids"]
    assert transferable["rationale"] == (
        "Requirement has partial deterministic coverage and is treated as transferable rather than missing."
    )


def test_llm_scorer_keeps_fully_missing_requirement_in_missing_requirements():
    resume, job = _scenario_pair("resume_false_positive_risk.json")
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 48,
                "decision": "SKIP",
                "confidence": "medium",
                "fit_summary": "Some overlap exists, but key requirements remain unsupported.",
                "matched_requirements": [],
                "missing_requirements": [
                    {"requirement_id": "req_kafka_events", "rationale": "No deterministic coverage exists."}
                ],
                "transferable_matches": [],
                "risk_flags": ["insufficient_direct_evidence"],
                "reasons": [
                    {"code": "MISSING_CORE", "message": "A core requirement is unsupported.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 28,
                    "nice_to_have": 4,
                    "transferable_fit": 6,
                    "title_seniority": 6,
                    "evidence_quality": 8,
                    "risk_penalty": -4
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert any(item["requirement_id"] == "req_kafka_events" for item in result["missing_requirements"])


def test_llm_scorer_repairs_matched_requirement_listed_as_missing_safely():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["missing_requirements"] = [
        {
            "requirement_id": "req_python_backend",
            "rationale": "Incorrectly treated as missing despite direct support.",
        }
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert any(item["requirement_id"] == "req_python_backend" for item in result["matched_requirements"])
    assert all(item["requirement_id"] != "req_python_backend" for item in result["missing_requirements"])


def test_llm_scorer_repair_does_not_hide_fabricated_transferable_evidence():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["transferable_matches"] = [
        {
            "requirement_id": "req_kubernetes_containers",
            "evidence_source_ids": ["fake_source_id"],
            "rationale": "Invalid fabricated evidence.",
        }
    ]
    response["missing_requirements"] = [
        {
            "requirement_id": "req_kubernetes_containers",
            "rationale": "Incorrectly treated as missing despite covered partial evidence.",
        }
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert "Evidence source_id does not exist for requirement req_kubernetes_containers" in result["llm_fallback_reason"]


def test_llm_scorer_final_output_does_not_auto_mark_covered_requirement_missing():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["transferable_matches"] = []
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert all(item["requirement_id"] != "req_kubernetes_containers" for item in result["missing_requirements"])


def test_llm_scorer_accepts_strong_eligible_requirement_in_matched_requirements():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["matched_requirements"] = [
        {
            "requirement_id": "req_python_backend",
            "evidence_source_ids": ["exp_1_b1"],
            "rationale": "Direct recent backend evidence.",
        }
    ]
    response["missing_requirements"] = []
    response["transferable_matches"] = []
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["matched_requirements"][0]["requirement_id"] == "req_python_backend"


def test_llm_scorer_rejects_proceed_when_hard_gate_missing():
    resume, job = _retail_pair()
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 82,
                "decision": "PROCEED",
                "confidence": "medium",
                "fit_summary": "Transferable retail overlap is meaningful but should not bypass hard gates.",
                "matched_requirements": [
                    {"requirement_id": "must_4", "evidence_source_ids": ["exp_1_b2"], "rationale": "Direct transaction evidence."}
                ],
                "missing_requirements": [
                    {"requirement_id": "must_7", "rationale": "Degree hard gate is missing."}
                ],
                "transferable_matches": [
                    {"requirement_id": "must_1", "evidence_source_ids": ["summary"], "rationale": "Customer consultation evidence is present but only partially transferable."},
                    {"requirement_id": "must_2", "evidence_source_ids": ["summary"], "rationale": "Retail sales evidence is present but not a strong direct match for the target requirement."},
                    {"requirement_id": "must_3", "evidence_source_ids": ["exp_1_b1"], "rationale": "Product-selection evidence is transferable."}
                ],
                "risk_flags": ["hard_gate_missing"],
                "reasons": [
                    {"code": "TRANSFERABLE", "message": "Some transferable retail experience exists.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 42,
                    "nice_to_have": 2,
                    "transferable_fit": 14,
                    "title_seniority": 10,
                    "evidence_quality": 14,
                    "risk_penalty": 0
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "SKIP"
    assert result["score_total"] == 69.0
    assert "hard_gate_missing" in result["risk_flags"]


def test_llm_scorer_rejects_unknown_evidence_ids():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 92,
                "decision": "PROCEED",
                "confidence": "medium",
                "fit_summary": "Invalid evidence ids.",
                "matched_requirements": [
                    {"requirement_id": "req_python_backend", "evidence_source_ids": ["missing_source_id"], "rationale": "Invalid evidence id."}
                ],
                "missing_requirements": [],
                "transferable_matches": [],
                "risk_flags": [],
                "reasons": [
                    {"code": "INVALID", "message": "Invalid evidence ids.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 55,
                    "nice_to_have": 10,
                    "transferable_fit": 0,
                    "title_seniority": 10,
                    "evidence_quality": 17,
                    "risk_penalty": 0
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert result["reasons"][-1]["code"] == "LLM_ADJUDICATION_FALLBACK"


def test_llm_scorer_falls_back_on_invalid_json():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider("{not valid json")

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert result["reasons"][-1]["code"] == "LLM_ADJUDICATION_FALLBACK"


def test_llm_scorer_falls_back_on_empty_response():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    provider = StubProvider("")

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert "empty response" in result["llm_fallback_reason"]


def test_llm_scorer_falls_back_on_schema_invalid_response():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    bad_response = _valid_llm_response()
    bad_response["reasons"] = ["not an object"]
    provider = StubProvider(json.dumps(bad_response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert "Schema validation failed" in result["llm_fallback_reason"]


def test_llm_scorer_normalizes_reason_severity_aliases():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["reasons"] = [
        {"code": "DIRECT_MATCH", "message": "Most required skills have strong recent evidence.", "severity": "low"},
        {"code": "MINOR_GAP", "message": "One requirement is only partially covered.", "severity": "medium"},
        {"code": "HARD_STOP", "message": "A blocking issue exists.", "severity": "critical"},
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    severities = [item["severity"] for item in result["reasons"] if item["code"] in {"DIRECT_MATCH", "MINOR_GAP", "HARD_STOP"}]
    assert severities == ["info", "warning", "blocker"]


def test_llm_scorer_falls_back_on_unrecognized_reason_severity():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    response = _valid_llm_response()
    response["reasons"] = [
        {"code": "DIRECT_MATCH", "message": "Most required skills have strong recent evidence.", "severity": "urgent"}
    ]
    provider = StubProvider(json.dumps(response))

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert "Schema validation failed" in result["llm_fallback_reason"]


def test_transferable_retail_case_can_score_higher_but_still_skip():
    resume, job = _retail_pair()
    deterministic = run_ats_scoring(resume, job)
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 52,
                "decision": "SKIP",
                "confidence": "medium",
                "fit_summary": "Transferable customer-facing retail evidence exists, but hard gates and domain gaps still make this a skip.",
                "matched_requirements": [
                    {"requirement_id": "must_4", "evidence_source_ids": ["exp_1_b2"], "rationale": "Direct POS, returns, and exchange evidence."},
                    {"requirement_id": "must_5", "evidence_source_ids": ["exp_1_b4"], "rationale": "Direct showroom and merchandising evidence."}
                ],
                "missing_requirements": [
                    {"requirement_id": "must_7", "rationale": "Degree hard gate is missing."}
                ],
                "transferable_matches": [
                    {"requirement_id": "must_1", "evidence_source_ids": ["summary"], "rationale": "Consultative customer interaction evidence exists but is only partially transferable."},
                    {"requirement_id": "must_2", "evidence_source_ids": ["summary"], "rationale": "Retail sales activity is present but does not qualify as a strong direct match."},
                    {"requirement_id": "must_3", "evidence_source_ids": ["exp_1_b1"], "rationale": "Customer product-selection evidence is partially transferable."},
                    {"requirement_id": "must_6", "evidence_source_ids": ["skills_3"], "rationale": "Weekend availability is explicitly stated."}
                ],
                "risk_flags": ["hard_gate_missing", "domain_specific_sales_gap"],
                "reasons": [
                    {"code": "TRANSFERABLE_ONLY", "message": "Some requirements are only partially transferable.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 31,
                    "nice_to_have": 1,
                    "transferable_fit": 10,
                    "title_seniority": 5,
                    "evidence_quality": 9,
                    "risk_penalty": -4
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "SKIP"
    assert result["score_total"] == 52
    assert result["score_total"] > deterministic["score_total"]
    assert any(item["requirement_id"] == "must_3" for item in result["transferable_matches"])


def test_false_positive_buzzword_case_remains_skip():
    resume, job = _scenario_pair("resume_false_positive_risk.json")
    provider = StubProvider(
        json.dumps(
            {
                "score_total": 84,
                "decision": "PROCEED",
                "confidence": "low",
                "fit_summary": "Keyword overlap is broad, but this should not bypass seniority safety.",
                "matched_requirements": [],
                "missing_requirements": [
                    {"requirement_id": "req_kafka_events", "rationale": "No supporting evidence was found."},
                    {"requirement_id": "req_datadog_observability", "rationale": "No supporting evidence was found."}
                ],
                "transferable_matches": [
                    {"requirement_id": "req_python_backend", "evidence_source_ids": ["summary"], "rationale": "Summary-only Python overlap is weakly relevant."},
                    {"requirement_id": "req_fastapi_rest", "evidence_source_ids": ["summary"], "rationale": "FastAPI overlap is summary-only."},
                    {"requirement_id": "req_postgresql_modeling", "evidence_source_ids": ["summary"], "rationale": "Data-modeling overlap is present but only through weak summary evidence."}
                ],
                "risk_flags": ["summary_only_overlap", "seniority_mismatch"],
                "reasons": [
                    {"code": "BUZZWORD_RISK", "message": "Overlap is broad but weakly evidenced.", "severity": "warning"}
                ],
                "score_breakdown": {
                    "must_have": 40,
                    "nice_to_have": 4,
                    "transferable_fit": 10,
                    "title_seniority": 18,
                    "evidence_quality": 12,
                    "risk_penalty": 0
                }
            }
        )
    )

    result = run_llm_adjudicated_scoring(resume, job, provider)

    assert result["scoring_mode"] == "llm_adjudicated"
    assert result["decision"] == "SKIP"
    assert result["score_total"] == 69.0
    assert "seniority_mismatch" in result["risk_flags"]


def test_deterministic_fallback_remains_unchanged_if_provider_fails():
    resume, job = _scenario_pair("resume_borderline_fit.json")
    deterministic = run_ats_scoring(resume, job)

    result = run_llm_adjudicated_scoring(resume, job, FailingProvider())

    assert result["scoring_mode"] == "ats_deterministic_fallback"
    assert result["decision"] == deterministic["decision"]
    assert result["score_total"] == deterministic["score_total"]
    assert result["must_have_coverage_percent"] == deterministic["must_have_coverage_percent"]
    assert result["matched_requirements"] == deterministic["matched_requirements"]
    assert result["missing_requirements"] == deterministic["missing_requirements"]


def test_compact_llm_prompt_packet_is_significantly_smaller_than_full_packet():
    resume, job = _scenario_pair("resume_very_high_fit.json")
    analysis = run_ats_scoring_analysis(resume, job)
    full_packet = build_scoring_packet(resume, job, analysis)
    prompt_packet = build_llm_prompt_packet(full_packet)

    full_packet_chars = len(json.dumps(full_packet, ensure_ascii=True))
    prompt_packet_chars = len(json.dumps(prompt_packet, ensure_ascii=True, separators=(",", ":")))

    assert prompt_packet_chars < full_packet_chars
    assert prompt_packet_chars <= 30000
    assert prompt_packet_chars <= full_packet_chars // 3
