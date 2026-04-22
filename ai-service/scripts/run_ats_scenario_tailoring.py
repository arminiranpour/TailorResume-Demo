from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats import (  # noqa: E402
    build_coverage_model,
    build_evidence_links,
    build_job_signals,
    build_job_weights,
    build_recency_priorities,
    build_resume_signals,
    build_title_alignment,
)
from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit  # noqa: E402
from app.pipelines.scoring import run_scoring  # noqa: E402
from app.pipelines.tailoring_plan import build_tailoring_plan  # noqa: E402
from app.providers.base import LLMProvider  # noqa: E402
from app.scoring import decide  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ats_scenarios"


class StubProvider(LLMProvider):
    def generate(
        self,
        messages,
        *,
        json_schema=None,
        temperature: float = 0,
        seed: int = 0,
        timeout=None,
    ) -> str:
        del json_schema, temperature, seed, timeout
        task_label, payload, raw_json_output = self._extract_payload(messages)
        if task_label == "summary_rewrite":
            return json.dumps(
                {
                    "rewritten_text": str(payload.get("original_text", "")),
                    "keywords_used": list(payload.get("preferred_surface_terms", [])),
                }
            )
        if task_label == "bullet_rewrite":
            return json.dumps(
                {
                    "bullet_id": str(payload.get("bullet_id", "")),
                    "rewritten_text": str(payload.get("original_text", "")),
                    "keywords_used": list(payload.get("preferred_surface_terms", [])),
                    "notes": "stub_provider",
                }
            )
        if task_label == "compress_text":
            candidate_text = str(payload.get("candidate_text", ""))
            max_chars = payload.get("max_chars")
            if isinstance(max_chars, int) and max_chars >= 0:
                candidate_text = candidate_text[:max_chars].rstrip()
            return json.dumps({"compressed_text": candidate_text})
        if task_label == "json_repair":
            return raw_json_output
        raise RuntimeError(f"Unsupported task label: {task_label}")

    @staticmethod
    def _extract_payload(messages) -> tuple[str, dict[str, Any], str]:
        if not isinstance(messages, list) or not messages:
            return "", {}, ""
        content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        lines = content.splitlines()
        task_label = ""
        if lines and lines[0].startswith("Task: "):
            task_label = lines[0][len("Task: ") :].strip()
        begin_marker = "BEGIN_UNTRUSTED_TEXT\n"
        end_marker = "\nEND_UNTRUSTED_TEXT"
        start = content.find(begin_marker)
        end = content.rfind(end_marker)
        payload: dict[str, Any] = {}
        if start != -1 and end != -1 and end > start:
            raw_payload = content[start + len(begin_marker) : end]
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {}
        repair_marker = "RAW_JSON_OUTPUT:\n"
        repair_index = content.find(repair_marker)
        raw_json_output = ""
        if repair_index != -1:
            raw_json_output = content[repair_index + len(repair_marker) :].strip()
        return task_label, payload, raw_json_output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest() -> list[dict[str, Any]]:
    return _load_json(FIXTURES_DIR / "scenario_manifest.json")


def _skills_lines(resume_json: dict[str, Any]) -> list[str]:
    skills = resume_json.get("skills", {})
    lines = skills.get("lines", []) if isinstance(skills, dict) else []
    values: list[str] = []
    for line in lines:
        text = line.get("text") if isinstance(line, dict) else None
        if isinstance(text, str):
            values.append(text)
    return values


def _bullet_text_map(resume_json: dict[str, Any]) -> dict[str, str]:
    items: dict[str, str] = {}
    for exp in resume_json.get("experience", []):
        if not isinstance(exp, dict):
            continue
        for bullet in exp.get("bullets", []):
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str):
                items[bullet["bullet_id"]] = str(bullet.get("text", ""))
    for project in resume_json.get("projects", []):
        if not isinstance(project, dict):
            continue
        for bullet in project.get("bullets", []):
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str):
                items[bullet["bullet_id"]] = str(bullet.get("text", ""))
    return items


def _score_resume_against_job(resume_json: dict[str, Any], job_json: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score_result = run_scoring(resume_json, job_json)
    decision_result = decide(score_result, job_json)
    return score_result, decision_result


def _compact_audit_log(audit_log: dict[str, Any]) -> dict[str, Any]:
    summary_detail = audit_log.get("summary_detail")
    skills_details = audit_log.get("skills_details", [])
    bullet_details = audit_log.get("bullet_details", [])
    frequency_balance = audit_log.get("frequency_balance") or {}
    return {
        "rewritten_bullets": audit_log.get("rewritten_bullets", []),
        "kept_bullets": audit_log.get("kept_bullets", []),
        "rejected_for_new_terms": audit_log.get("rejected_for_new_terms", []),
        "compressed": audit_log.get("compressed", []),
        "frequency_actions": audit_log.get("frequency_actions", []),
        "frequency_validation_errors": frequency_balance.get("validation_errors", []),
        "summary_detail": summary_detail,
        "skills_details": skills_details,
        "bullet_details": [
            {
                "bullet_id": detail.get("bullet_id"),
                "rewrite_intent": detail.get("rewrite_intent"),
                "changed": detail.get("changed"),
                "skip_reason": detail.get("skip_reason"),
                "reject_reason": detail.get("reject_reason"),
                "target_keywords": detail.get("target_keywords", []),
            }
            for detail in bullet_details
        ],
    }


def main() -> int:
    job_json = _load_json(FIXTURES_DIR / "shared_job.json")
    manifest = _load_manifest()
    provider = StubProvider()

    for scenario in manifest:
        resume_path = FIXTURES_DIR / scenario["resume_fixture"]
        resume_json = _load_json(resume_path)
        score_result, decision_result = _score_resume_against_job(resume_json, job_json)

        print(f"{scenario['scenario_name']} ({scenario['scenario_id']})")
        print(f"  scoring_decision: {decision_result['decision']}")
        print(f"  score_total: {score_result['score_total']}")

        if decision_result["decision"] != "PROCEED":
            print("  skipped_tailoring: decision != PROCEED")
            print()
            continue

        job_signals = build_job_signals(job_json)
        resume_signals = build_resume_signals(resume_json)
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
        recency_priorities = build_recency_priorities(
            job_signals=job_signals,
            resume_signals=resume_signals,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            title_alignment=title_alignment,
        )

        full_score_result = dict(score_result)
        full_score_result.update(decision_result)

        tailoring_plan = build_tailoring_plan(
            resume_json=resume_json,
            job_json=job_json,
            score_result=full_score_result,
            provider=None,
            job_signals=job_signals,
            resume_signals=resume_signals,
            job_weights=job_weights,
            coverage=coverage,
            evidence_links=evidence_links,
            title_alignment=title_alignment,
            recency_priorities=recency_priorities,
        )

        tailored_resume, audit_log = rewrite_resume_text_with_audit(
            resume_json=deepcopy(resume_json),
            job_json=job_json,
            score_result=full_score_result,
            tailoring_plan=tailoring_plan,
            provider=provider,
        )

        print(f"  summary_before: {resume_json['summary']['text']}")
        print(f"  summary_after:  {tailored_resume['summary']['text']}")
        print(f"  skills_before: {json.dumps(_skills_lines(resume_json), ensure_ascii=True)}")
        print(f"  skills_after:  {json.dumps(_skills_lines(tailored_resume), ensure_ascii=True)}")
        print("  changed_bullets:")

        before_bullets = _bullet_text_map(resume_json)
        after_bullets = _bullet_text_map(tailored_resume)
        changed = False
        for bullet_id, before_text in before_bullets.items():
            after_text = after_bullets.get(bullet_id)
            if after_text != before_text:
                changed = True
                print(f"    {bullet_id}")
                print(f"      before: {before_text}")
                print(f"      after:  {after_text}")
        if not changed:
            print("    <none>")

        print("  audit_log:")
        print(json.dumps(_compact_audit_log(audit_log), indent=2, sort_keys=True))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
