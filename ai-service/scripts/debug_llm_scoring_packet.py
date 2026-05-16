from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipelines.llm_scoring import (  # noqa: E402
    LLMScoreError,
    build_llm_adjudication_messages,
    build_llm_prompt_packet,
    build_scoring_packet,
    extract_llm_json_object,
    run_llm_adjudicated_scoring,
    validate_llm_score_result,
)
from app.pipelines.scoring import run_ats_scoring_analysis  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402
from app.providers.factory import get_provider  # noqa: E402
from app.schemas.schema_loader import load_schema  # noqa: E402


class DemoStubProvider(LLMProvider):
    def generate(self, messages, **kwargs) -> str:
        _ = messages, kwargs
        return json.dumps(
            {
                "score_total": 68,
                "decision": "SKIP",
                "confidence": "medium",
                "fit_summary": "Demo stub kept the resume below proceed threshold while preserving evidence-backed partial credit.",
                "matched_requirements": [],
                "missing_requirements": [],
                "transferable_matches": [],
                "risk_flags": ["demo_stub"],
                "reasons": [
                    {
                        "code": "DEMO_STUB",
                        "message": "Returned demo stub result.",
                        "severity": "info",
                    }
                ],
                "score_breakdown": {
                    "must_have": 35,
                    "nice_to_have": 8,
                    "transferable_fit": 8,
                    "title_seniority": 8,
                    "evidence_quality": 9,
                    "risk_penalty": 0
                }
            }
        )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _classify_raw_output(raw: str) -> str:
    if raw == "":
        return "empty response"
    stripped = raw.strip()
    if stripped == "":
        return "whitespace-only response"
    if stripped.startswith("```"):
        return "markdown fenced response"
    if not stripped.startswith("{"):
        return "non-JSON prefixed response"
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return "invalid JSON syntax"
    return "valid JSON"


def _size_summary(full_packet: dict, prompt_packet: dict, messages: list[dict[str, str]]) -> dict:
    full_packet_json = json.dumps(full_packet, ensure_ascii=True)
    prompt_packet_json = json.dumps(prompt_packet, ensure_ascii=True, separators=(",", ":"))
    system_chars = len(messages[0]["content"])
    user_chars = len(messages[1]["content"])
    total_chars = system_chars + user_chars
    return {
        "full_packet_chars": len(full_packet_json),
        "prompt_packet_chars": len(prompt_packet_json),
        "packet_reduction_chars": len(full_packet_json) - len(prompt_packet_json),
        "packet_reduction_percent": round(
            ((len(full_packet_json) - len(prompt_packet_json)) / max(1, len(full_packet_json))) * 100,
            2,
        ),
        "system_prompt_chars": system_chars,
        "user_message_chars": user_chars,
        "total_message_chars": total_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resume_json")
    parser.add_argument("job_json")
    parser.add_argument("--demo-stub", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--real-provider", action="store_true")
    parser.add_argument("--print-raw-llm-output", action="store_true")
    parser.add_argument("--print-message-preview", action="store_true")
    parser.add_argument("--skip-packet", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()

    resume_json = _load_json(Path(args.resume_json))
    job_json = _load_json(Path(args.job_json))

    analysis = run_ats_scoring_analysis(resume_json, job_json)
    full_packet = build_scoring_packet(resume_json, job_json, analysis)
    prompt_packet = build_llm_prompt_packet(full_packet)
    messages = build_llm_adjudication_messages(full_packet)
    schema = load_schema("llm_score_result")

    print("=== DETERMINISTIC ANALYSIS ===")
    print(json.dumps(
        {
            "score_total": analysis["score_total"],
            "decision": analysis["decision"],
            "must_have_coverage_percent": analysis["must_have_coverage_percent"],
            "must_have_strict_match_percent": analysis["must_have_strict_match_percent"],
            "reasons": analysis["reasons"],
        },
        indent=2,
    ))

    print("\n=== PACKET SIZE ===")
    print(json.dumps(_size_summary(full_packet, prompt_packet, messages), indent=2))

    if not args.skip_packet:
        print("\n=== COMPACT LLM PROMPT PACKET ===")
        print(json.dumps(prompt_packet, indent=2))

    provider = None
    if args.demo_stub:
        provider = DemoStubProvider()
    elif args.run_provider or args.real_provider:
        provider = get_provider()

    if provider is not None:
        if args.print_message_preview:
            print("\n=== MESSAGE PREVIEW ===")
            print(
                json.dumps(
                    {
                        "message_count": len(messages),
                        "roles": [message["role"] for message in messages],
                        "schema_top_level_required": schema.get("required", []),
                        "user_content_preview": messages[1]["content"][:500],
                    },
                    indent=2,
                )
            )

        if args.print_raw_llm_output:
            raw = provider.generate(
                messages,
                json_schema=schema,
                temperature=0,
                seed=0,
            )
            print("\n=== RAW LLM OUTPUT ===")
            print(raw)
            print("\n=== RAW OUTPUT CLASSIFICATION ===")
            print(
                json.dumps(
                    {
                        "classification": _classify_raw_output(raw),
                        "char_count": len(raw),
                    },
                    indent=2,
                )
            )
            try:
                llm_raw = extract_llm_json_object(raw)
                validated = validate_llm_score_result(llm_raw, full_packet)
                print("\n=== VALIDATION RESULT ===")
                print(json.dumps({"ok": True, "decision": validated["decision"]}, indent=2))
            except LLMScoreError as exc:
                print("\n=== VALIDATION RESULT ===")
                print(json.dumps({"ok": False, "error": str(exc)}, indent=2))

        if not args.raw_only:
            print("\n=== LLM ADJUDICATED RESULT ===")
            print(json.dumps(run_llm_adjudicated_scoring(resume_json, job_json, provider), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
