from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from app.pipelines.tailoring_plan import build_tailoring_plan  # noqa: E402
from scripts.ats_diagnostic_scenarios import (  # noqa: E402
    SCENARIOS,
    SCENARIOS_BY_ID,
    DiagnosticScenario,
)


class ScenarioStubProvider:
    def __init__(self, spec: Mapping[str, Any] | None = None):
        self.spec = dict(spec or {})
        self.calls: list[dict[str, Any]] = []
        self.summary_call_count = 0
        self.bullet_call_count = 0

    def generate(self, messages, **kwargs):
        task_label, payload = self._extract_task_payload(messages)
        self.calls.append({"task": task_label, "payload": payload, "kwargs": kwargs})
        if task_label == "summary_rewrite":
            self.summary_call_count += 1
            return json.dumps(
                {
                    "rewritten_text": self._summary_text(payload),
                    "keywords_used": list(payload.get("preferred_surface_terms", [])),
                }
            )
        if task_label == "bullet_rewrite":
            self.bullet_call_count += 1
            bullet_id = str(payload.get("bullet_id", ""))
            rewritten_text = self._bullet_text(bullet_id, payload)
            return json.dumps(
                {
                    "bullet_id": bullet_id,
                    "rewritten_text": rewritten_text,
                    "keywords_used": list(payload.get("preferred_surface_terms", [])),
                    "notes": "ats_diagnostic_stub",
                }
            )
        if task_label == "compress_text":
            return json.dumps(
                {
                    "compressed_text": self._compressed_text(payload),
                }
            )
        if task_label == "json_repair":
            return self._extract_repair_raw_json(messages)
        raise RuntimeError(f"ScenarioStubProvider does not support task: {task_label}")

    def _summary_text(self, payload: Mapping[str, Any]) -> str:
        summary_text = self.spec.get("summary_text")
        if isinstance(summary_text, list):
            index = min(max(self.summary_call_count - 1, 0), len(summary_text) - 1)
            value = summary_text[index]
            if isinstance(value, str):
                return value
        if isinstance(summary_text, str):
            return summary_text
        original = payload.get("original_text")
        return original if isinstance(original, str) else ""

    def _bullet_text(self, bullet_id: str, payload: Mapping[str, Any]) -> str:
        rewritten = self.spec.get("bullet_text_by_id")
        if isinstance(rewritten, Mapping):
            value = rewritten.get(bullet_id)
            if isinstance(value, list):
                index = min(max(self.bullet_call_count - 1, 0), len(value) - 1)
                if isinstance(value[index], str):
                    return value[index]
            if isinstance(value, str):
                return value
        original = payload.get("original_text")
        return original if isinstance(original, str) else ""

    def _compressed_text(self, payload: Mapping[str, Any]) -> str:
        override = self.spec.get("compressed_text")
        if isinstance(override, str):
            return override
        candidate = payload.get("candidate_text")
        max_chars = payload.get("max_chars")
        if isinstance(candidate, str) and isinstance(max_chars, int) and max_chars > 0:
            return candidate[:max_chars].rstrip()
        return candidate if isinstance(candidate, str) else ""

    @staticmethod
    def _extract_task_payload(messages):
        if not isinstance(messages, list) or len(messages) < 2:
            raise RuntimeError("ScenarioStubProvider expected LLM messages")
        user_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        task_prefix = "Task: "
        begin_marker = "BEGIN_UNTRUSTED_TEXT\n"
        end_marker = "\nEND_UNTRUSTED_TEXT"
        if not user_content.startswith(task_prefix):
            raise RuntimeError("ScenarioStubProvider could not identify task label")
        task_label = user_content[len(task_prefix) :].splitlines()[0].strip()
        start = user_content.find(begin_marker)
        end = user_content.rfind(end_marker)
        payload: dict[str, Any] = {}
        if start != -1 and end != -1 and end > start:
            raw_payload = user_content[start + len(begin_marker) : end]
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {}
        return task_label, payload

    @staticmethod
    def _extract_repair_raw_json(messages):
        user_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        marker = "RAW_JSON_OUTPUT:\n"
        start = user_content.find(marker)
        if start == -1:
            raise RuntimeError("ScenarioStubProvider could not extract repair payload")
        return user_content[start + len(marker) :].strip()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def default_score_result(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "decision": "PROCEED",
        "reasons": [],
        "matched_requirements": [],
        "missing_requirements": [],
        "must_have_coverage_percent": 100,
    }
    if overrides:
        result.update(dict(overrides))
    return result


def build_ats_snapshot(job_json: Mapping[str, Any], resume_json: Mapping[str, Any]) -> dict[str, Any]:
    job_signals = build_job_signals(dict(job_json))
    resume_signals = build_resume_signals(dict(resume_json))
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
        "job_signals": job_signals,
        "resume_signals": resume_signals,
        "job_weights": job_weights,
        "coverage": coverage,
        "evidence_links": evidence_links,
        "title_alignment": title_alignment,
        "recency": recency,
    }


def run_scenario(scenario: DiagnosticScenario) -> dict[str, Any]:
    resume_json = load_json(scenario.resume_fixture_path)
    job_json = load_json(scenario.job_fixture_path)
    score_result = default_score_result(scenario.score_result_overrides)
    before_resume = deepcopy(resume_json)
    before_snapshot = build_ats_snapshot(job_json, resume_json)
    plan = build_tailoring_plan(
        resume_json=resume_json,
        job_json=job_json,
        score_result=score_result,
        provider=None,
        job_signals=before_snapshot["job_signals"],
        resume_signals=before_snapshot["resume_signals"],
        job_weights=before_snapshot["job_weights"],
        coverage=before_snapshot["coverage"],
        evidence_links=before_snapshot["evidence_links"],
        title_alignment=before_snapshot["title_alignment"],
        recency_priorities=before_snapshot["recency"],
    )
    provider = ScenarioStubProvider(scenario.provider_spec)
    after_resume, audit_log = rewrite_resume_text_with_audit(
        resume_json=deepcopy(resume_json),
        job_json=job_json,
        score_result=score_result,
        tailoring_plan=plan,
        provider=provider,
        character_budgets=dict(scenario.character_budgets) if scenario.character_budgets else None,
    )
    after_snapshot = build_ats_snapshot(job_json, after_resume)
    changes = compute_section_changes(before_resume, after_resume, audit_log)
    evaluations = evaluate_expectations(scenario, before_snapshot, after_snapshot, plan, before_resume, after_resume, audit_log, changes)
    notes = build_diagnostic_notes(before_snapshot, after_snapshot, plan, audit_log, evaluations)
    overall_status = summarize_scenario_status(evaluations)
    return {
        "scenario": scenario,
        "job_json": job_json,
        "before_resume": before_resume,
        "after_resume": after_resume,
        "score_result": score_result,
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
        "plan": plan,
        "audit_log": audit_log,
        "changes": changes,
        "evaluations": evaluations,
        "notes": notes,
        "overall_status": overall_status,
        "provider_calls": provider.calls,
    }


def compute_section_changes(
    before_resume: Mapping[str, Any],
    after_resume: Mapping[str, Any],
    audit_log: Mapping[str, Any],
) -> dict[str, Any]:
    before_summary = _summary_text(before_resume)
    after_summary = _summary_text(after_resume)
    skills_changes: list[dict[str, Any]] = []
    bullet_changes: list[dict[str, Any]] = []

    before_lines = _skills_line_map(before_resume)
    after_lines = _skills_line_map(after_resume)
    for line_id, before_text in before_lines.items():
        after_text = after_lines.get(line_id, before_text)
        if before_text != after_text:
            skills_changes.append(
                {
                    "line_id": line_id,
                    "before": before_text,
                    "after": after_text,
                    "attribution": attribute_skills_change(line_id, audit_log),
                }
            )

    before_bullets = _bullet_map(before_resume)
    after_bullets = _bullet_map(after_resume)
    for bullet_id, before_text in before_bullets.items():
        after_text = after_bullets.get(bullet_id, before_text)
        if before_text != after_text:
            bullet_changes.append(
                {
                    "bullet_id": bullet_id,
                    "before": before_text,
                    "after": after_text,
                    "attribution": attribute_bullet_change(bullet_id, audit_log),
                }
            )

    summary_change = None
    if before_summary != after_summary:
        summary_change = {
            "before": before_summary,
            "after": after_summary,
            "attribution": attribute_summary_change(audit_log),
        }

    changed_sections = []
    if summary_change is not None:
        changed_sections.append("summary")
    if skills_changes:
        changed_sections.append("skills")
    if bullet_changes:
        changed_sections.append("bullets")

    return {
        "summary": summary_change,
        "skills": skills_changes,
        "bullets": bullet_changes,
        "changed_sections": tuple(changed_sections),
    }


def attribute_summary_change(audit_log: Mapping[str, Any]) -> list[str]:
    detail = audit_log.get("summary_detail") if isinstance(audit_log, Mapping) else None
    steps = ["Step 10 summary rewrite"]
    if isinstance(detail, Mapping):
        if detail.get("allowed_title_terms") or detail.get("blocked_title_terms"):
            steps.append("Step 5 title alignment")
        if detail.get("recent_priority_terms"):
            steps.append("Step 6 recency")
        if detail.get("reject_reason") == "frequency_balance" or detail.get("skip_reason") == "frequency_balance_rollback":
            steps.append("Step 11 frequency balance rollback")
    return steps


def attribute_skills_change(line_id: str, audit_log: Mapping[str, Any]) -> list[str]:
    steps = ["Step 9 skills optimization"]
    details = audit_log.get("skills_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, Mapping) and detail.get("line_id") == line_id:
                if detail.get("target_keywords"):
                    steps.append("Step 2 weighting")
                break
    for action in audit_log.get("frequency_actions", []) if isinstance(audit_log, Mapping) else []:
        if isinstance(action, Mapping) and action.get("surface_id") == line_id:
            steps.append("Step 11 frequency balance rollback")
    return steps


def attribute_bullet_change(bullet_id: str, audit_log: Mapping[str, Any]) -> list[str]:
    steps = ["Step 8 bullet rewrite", "Step 7 tailoring plan"]
    details = audit_log.get("bullet_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, Mapping) and detail.get("bullet_id") == bullet_id:
                if detail.get("source_section") == "experience_bullet":
                    steps.append("Step 4 evidence linking")
                break
    for action in audit_log.get("frequency_actions", []) if isinstance(audit_log, Mapping) else []:
        if isinstance(action, Mapping) and action.get("surface_id") == bullet_id:
            steps.append("Step 11 frequency balance rollback")
    return steps


def evaluate_expectations(
    scenario: DiagnosticScenario,
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    plan: Mapping[str, Any],
    before_resume: Mapping[str, Any],
    after_resume: Mapping[str, Any],
    audit_log: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for expectation in scenario.expectations:
        result = evaluate_expectation(expectation, before_snapshot, after_snapshot, plan, before_resume, after_resume, audit_log, changes)
        evaluations.append(result)
    return evaluations


def evaluate_expectation(
    expectation: Mapping[str, Any],
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    plan: Mapping[str, Any],
    before_resume: Mapping[str, Any],
    after_resume: Mapping[str, Any],
    audit_log: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    kind = expectation["id"]
    label = expectation["label"]
    likely_steps = tuple(expectation.get("likely_steps", ()))
    focus_section = expectation.get("focus_section")
    passed = False
    details = ""

    if kind == "weight_order":
        weights = before_snapshot["job_weights"]
        higher = expectation["higher"]
        lower = expectation["lower"]
        higher_weight = weights.weights_by_term[higher].total_weight
        lower_weight = weights.weights_by_term[lower].total_weight
        higher_rank = weights.ordered_terms.index(higher)
        lower_rank = weights.ordered_terms.index(lower)
        passed = higher_weight > lower_weight and higher_rank < lower_rank
        details = f"{higher}={higher_weight} (rank {higher_rank + 1}), {lower}={lower_weight} (rank {lower_rank + 1})"
    elif kind == "coverage_cross_section_improves":
        term = expectation["term"]
        before_term = before_snapshot["coverage"].coverage_by_term[term]
        after_term = after_snapshot["coverage"].coverage_by_term[term]
        passed = (not before_term.has_cross_section_support) and after_term.has_cross_section_support and not after_term.is_missing
        details = (
            f"before cross-section={before_term.has_cross_section_support}, "
            f"after cross-section={after_term.has_cross_section_support}, "
            f"after strength={after_term.coverage_strength}"
        )
    elif kind == "evidence_distinction":
        strong_term = expectation["strong_term"]
        weak_term = expectation["weak_term"]
        strong_link = before_snapshot["evidence_links"].links_by_term[strong_term]
        weak_link = before_snapshot["evidence_links"].links_by_term[weak_term]
        passed = strong_link.has_experience_backing and weak_term in before_snapshot["evidence_links"].skills_only_terms and not weak_link.is_safe_for_bullets
        details = (
            f"{strong_term}: experience={strong_link.has_experience_backing}; "
            f"{weak_term}: skills_only={weak_term in before_snapshot['evidence_links'].skills_only_terms}, "
            f"bullet_safe={weak_link.is_safe_for_bullets}"
        )
    elif kind == "title_alignment_safe":
        status = plan.get("title_alignment_status", {})
        summary_terms = plan.get("summary_alignment_terms", [])
        passed = bool(status.get("is_safe_for_summary_alignment")) and bool(summary_terms)
        details = (
            f"safe={status.get('is_safe_for_summary_alignment')}, "
            f"alignment_strength={status.get('alignment_strength')}, "
            f"summary_terms={summary_terms}"
        )
    elif kind == "recent_candidate_preferred":
        term = expectation["term"]
        expected_source = expectation["expected_source_id"]
        recency = before_snapshot["recency"].priorities_by_term[term]
        actual = recency.strongest_recent_experience_candidate.source_id if recency.strongest_recent_experience_candidate else None
        passed = actual == expected_source
        details = f"expected={expected_source}, actual={actual}"
    elif kind == "bullet_action_intent":
        action = _plan_bullet_action(plan, expectation["bullet_id"])
        intent = action.get("rewrite_intent") if isinstance(action, Mapping) else None
        passed = intent == expectation["intent"]
        details = f"expected={expectation['intent']}, actual={intent}"
    elif kind == "plan_supported_term":
        term = expectation["term"]
        passed = term in _lowered_terms(plan.get("supported_priority_terms"))
        details = f"supported_priority_terms={plan.get('supported_priority_terms', [])}"
    elif kind == "plan_under_supported_term":
        term = expectation["term"]
        passed = term in {item.get("term") for item in plan.get("under_supported_terms", []) if isinstance(item, Mapping)}
        details = f"under_supported_terms={plan.get('under_supported_terms', [])}"
    elif kind == "plan_blocks_term_scope":
        term = expectation["term"]
        expected_scopes = {expectation["scope"], *expectation.get("extra_scopes", ())}
        blocked = _blocked_scopes_for_term(plan, term)
        passed = expected_scopes.issubset(blocked)
        details = f"expected_scopes={sorted(expected_scopes)}, actual_scopes={sorted(blocked)}"
    elif kind == "bullet_changed":
        detail = _bullet_detail(audit_log, expectation["bullet_id"])
        passed = bool(detail.get("changed"))
        details = f"changed={detail.get('changed')}, skip_reason={detail.get('skip_reason')}, reject_reason={detail.get('reject_reason')}"
    elif kind == "bullet_contains":
        bullet_text = _bullet_map(after_resume).get(expectation["bullet_id"], "")
        terms = tuple(expectation["terms"])
        passed = all(term.lower() in bullet_text.lower() for term in terms)
        details = f"final_bullet={bullet_text}"
    elif kind == "bullet_rejects_term":
        detail = _bullet_detail(audit_log, expectation["bullet_id"])
        term = expectation["term"]
        disallowed = _lowered_terms(detail.get("disallowed_terms"))
        passed = term in disallowed or term in _lowered_text(detail.get("candidate_text"))
        details = f"reject_reason={detail.get('reject_reason')}, disallowed_terms={detail.get('disallowed_terms')}"
    elif kind == "skills_contains_any_line":
        term = expectation["term"]
        passed = term in _lowered_text(" | ".join(_skills_line_map(after_resume).values()))
        details = f"skills_lines={list(_skills_line_map(after_resume).values())}"
    elif kind == "skills_not_contains":
        term = expectation["term"]
        passed = term not in _lowered_text(" | ".join(_skills_line_map(after_resume).values()))
        details = f"skills_lines={list(_skills_line_map(after_resume).values())}"
    elif kind == "summary_reject_reason_in":
        summary_detail = audit_log.get("summary_detail", {})
        reason = summary_detail.get("reject_reason")
        passed = reason in set(expectation["reasons"])
        details = f"reject_reason={reason}"
    elif kind == "summary_contains":
        term = expectation["term"]
        summary = _summary_text(after_resume)
        passed = term in summary.lower()
        details = f"summary={summary}"
    elif kind == "summary_not_contains":
        term = expectation["term"]
        summary = _summary_text(after_resume)
        passed = term not in summary.lower()
        details = f"summary={summary}"
    elif kind == "frequency_rollback":
        actions = audit_log.get("frequency_actions", [])
        term = expectation["term"]
        section = expectation["section"]
        passed = any(
            isinstance(action, Mapping) and action.get("term") == term and action.get("section") == section
            for action in actions
        )
        details = f"frequency_actions={actions}"
    elif kind == "summary_skip_reason":
        summary_detail = audit_log.get("summary_detail", {})
        actual = summary_detail.get("skip_reason")
        passed = actual == expectation["skip_reason"]
        details = f"skip_reason={actual}"
    elif kind == "summary_max_term_count":
        term = expectation["term"]
        summary = _summary_text(after_resume).lower()
        actual_count = summary.count(term)
        passed = actual_count <= int(expectation["max_count"])
        details = f"{term}_count={actual_count}, summary={summary}"
    elif kind == "section_changed":
        section = expectation["section"]
        passed = section in changes.get("changed_sections", ())
        alternate = expectation.get("allow_alternate_section")
        if not passed and isinstance(alternate, str):
            passed = alternate in changes.get("changed_sections", ())
        details = f"changed_sections={changes.get('changed_sections', ())}"
    else:
        raise ValueError(f"Unsupported expectation kind: {kind}")

    status = "PASS" if passed else "FAIL"
    return {
        "label": label,
        "status": status,
        "details": details,
        "likely_steps": likely_steps,
        "focus_section": focus_section,
        "kind": kind,
    }


def build_diagnostic_notes(
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    plan: Mapping[str, Any],
    audit_log: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> list[str]:
    notes: list[str] = []
    if before_snapshot["coverage"].overall_distinct_coverage > after_snapshot["coverage"].overall_distinct_coverage:
        notes.append("Coverage dropped after tailoring, which is suspicious for a deterministic ATS improvement path.")
    if audit_log.get("rejected_for_new_terms"):
        notes.append(f"Rejected bullet rewrites: {audit_log['rejected_for_new_terms']}.")
    summary_detail = audit_log.get("summary_detail")
    if isinstance(summary_detail, Mapping) and summary_detail.get("reject_reason"):
        notes.append(f"Summary rewrite rejection: {summary_detail['reject_reason']}.")
    failed = [item for item in evaluations if item["status"] != "PASS"]
    if failed:
        failed_labels = ", ".join(item["label"] for item in failed)
        notes.append(f"Heuristic misses detected: {failed_labels}.")
    if not notes:
        notes.append("No immediate ATS regressions surfaced for this scenario.")
    if not plan.get("blocked_terms"):
        notes.append("Planner emitted no blocked terms for this scenario; confirm that is expected.")
    return notes


def summarize_scenario_status(evaluations: Sequence[Mapping[str, Any]]) -> str:
    failures = sum(1 for item in evaluations if item["status"] == "FAIL")
    if failures == 0:
        return "PASS"
    if failures <= max(1, len(evaluations) // 3):
        return "WARNING"
    return "FAIL"


def render_scenario_report(report: Mapping[str, Any]) -> str:
    scenario = report["scenario"]
    before_snapshot = report["before_snapshot"]
    after_snapshot = report["after_snapshot"]
    plan = report["plan"]
    audit_log = report["audit_log"]
    changes = report["changes"]
    evaluations = report["evaluations"]
    summary_detail = audit_log.get("summary_detail") if isinstance(audit_log, Mapping) else {}
    lines = [
        "=" * 88,
        f"SCENARIO: {scenario.scenario_name} ({scenario.scenario_id})",
        "=" * 88,
        f"Target ATS step(s): {', '.join(f'Step {step}' for step in scenario.target_steps)}",
        f"Description: {scenario.description}",
        f"Overall assessment: {report['overall_status']}",
        "Key ATS inputs/conditions:",
    ]
    lines.extend(f"  - {item}" for item in scenario.key_conditions)
    lines.extend(
        [
            f"Score decision: {report['score_result']['decision']}",
            f"Top weighted terms: {_format_weighted_terms(before_snapshot['job_weights'])}",
            (
                "Coverage summary: "
                f"before covered={len(before_snapshot['coverage'].covered_terms)} / {len(before_snapshot['coverage'].coverage_ordered_terms)}, "
                f"after covered={len(after_snapshot['coverage'].covered_terms)} / {len(after_snapshot['coverage'].coverage_ordered_terms)}, "
                f"before cross-section={len(before_snapshot['coverage'].cross_section_supported_terms)}, "
                f"after cross-section={len(after_snapshot['coverage'].cross_section_supported_terms)}"
            ),
            (
                "Evidence summary: "
                f"linked={len(before_snapshot['evidence_links'].linked_terms)}, "
                f"skills_only={list(before_snapshot['evidence_links'].skills_only_terms[:5])}, "
                f"missing_experience={list(before_snapshot['evidence_links'].missing_experience_terms[:5])}"
            ),
            (
                "Title alignment summary: "
                f"before safe={before_snapshot['title_alignment'].is_safe_for_summary_alignment}, "
                f"after safe={after_snapshot['title_alignment'].is_safe_for_summary_alignment}, "
                f"before score={before_snapshot['title_alignment'].title_alignment_score}, "
                f"after score={after_snapshot['title_alignment'].title_alignment_score}, "
                f"strongest_resume_title={before_snapshot['title_alignment'].strongest_matching_resume_title}"
            ),
            (
                "Recency summary: "
                f"recent_high_priority={list(before_snapshot['recency'].recent_high_priority_terms[:5])}, "
                f"recent_bullet_safe={list(before_snapshot['recency'].recent_bullet_safe_terms[:5])}, "
                f"stale_only={list(before_snapshot['recency'].stale_only_terms[:5])}"
            ),
            (
                "Planner summary: "
                f"supported={list(plan.get('supported_priority_terms', [])[:5])}, "
                f"under_supported={list(plan.get('under_supported_terms', [])[:3])}, "
                f"blocked={list(plan.get('blocked_terms', [])[:3])}, "
                f"bullet_actions={_format_bullet_actions(plan)}"
            ),
            "Before/after summary text:",
            f"  BEFORE: {_summary_text(report['before_resume'])}",
            f"  AFTER : {_summary_text(report['after_resume'])}",
            "Before/after skills lines:",
        ]
    )
    for line_id, before_text in _skills_line_map(report["before_resume"]).items():
        after_text = _skills_line_map(report["after_resume"]).get(line_id, before_text)
        lines.append(f"  - {line_id}: {before_text} -> {after_text}")
    lines.append("Before/after bullets:")
    for bullet_id, before_text in _bullet_map(report["before_resume"]).items():
        after_text = _bullet_map(report["after_resume"]).get(bullet_id, before_text)
        lines.append(f"  - {bullet_id}: {before_text} -> {after_text}")
    lines.extend(
        [
            (
                "Frequency balance summary: "
                f"overused={audit_log.get('frequency_balance', {}).get('overused_terms') if isinstance(audit_log.get('frequency_balance'), Mapping) else None}, "
                f"actions={audit_log.get('frequency_actions', [])}"
            ),
            f"Which sections changed: {changes.get('changed_sections', ())}",
            "Section-level change attribution:",
        ]
    )
    if changes.get("summary"):
        lines.append(f"  - summary: {', '.join(changes['summary']['attribution'])}")
    for item in changes.get("skills", []):
        lines.append(f"  - skills/{item['line_id']}: {', '.join(item['attribution'])}")
    for item in changes.get("bullets", []):
        lines.append(f"  - bullets/{item['bullet_id']}: {', '.join(item['attribution'])}")
    lines.extend(
        [
            "Heuristic evaluation:",
        ]
    )
    for item in evaluations:
        lines.append(f"  - {item['status']}: {item['label']} [{item['details']}]")
    lines.extend(
        [
            "Diagnostic notes:",
        ]
    )
    lines.extend(f"  - {note}" for note in report["notes"])
    if isinstance(summary_detail, Mapping):
        lines.append(f"Summary rewrite audit: {dict(summary_detail)}")
    return "\n".join(lines)


def render_global_summary(reports: Sequence[Mapping[str, Any]]) -> str:
    strong = [report["scenario"].scenario_id for report in reports if report["overall_status"] == "PASS"]
    partial = [report["scenario"].scenario_id for report in reports if report["overall_status"] == "WARNING"]
    weak = [report["scenario"].scenario_id for report in reports if report["overall_status"] == "FAIL"]
    weak_steps: dict[int, int] = {}
    weak_sections: dict[str, int] = {}
    for report in reports:
        for evaluation in report["evaluations"]:
            if evaluation["status"] == "PASS":
                continue
            for step in evaluation.get("likely_steps", ()):
                weak_steps[int(step)] = weak_steps.get(int(step), 0) + 1
            focus = evaluation.get("focus_section")
            if isinstance(focus, str):
                weak_sections[focus] = weak_sections.get(focus, 0) + 1
    sorted_steps = sorted(weak_steps.items(), key=lambda item: (-item[1], item[0]))
    sorted_sections = sorted(weak_sections.items(), key=lambda item: (-item[1], item[0]))
    return "\n".join(
        [
            "=" * 88,
            "GLOBAL SUMMARY",
            "=" * 88,
            f"Scenarios passed strongly: {strong}",
            f"Scenarios partially passed: {partial}",
            f"Scenarios revealing likely weak spots: {weak}",
            f"ATS steps needing more work: {sorted_steps}",
            f"Sections most often failing to improve correctly: {sorted_sections}",
        ]
    )


def run_suite(selected_scenarios: Sequence[str] | None = None) -> list[dict[str, Any]]:
    scenarios = resolve_scenarios(selected_scenarios)
    return [run_scenario(scenario) for scenario in scenarios]


def resolve_scenarios(selected_scenarios: Sequence[str] | None = None) -> list[DiagnosticScenario]:
    if not selected_scenarios:
        return list(SCENARIOS)
    resolved = []
    for scenario_id in selected_scenarios:
        scenario = SCENARIOS_BY_ID.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        resolved.append(scenario)
    return resolved


def _summary_text(resume_json: Mapping[str, Any]) -> str:
    summary = resume_json.get("summary")
    if isinstance(summary, Mapping):
        text = summary.get("text")
        if isinstance(text, str):
            return text
    return ""


def _skills_line_map(resume_json: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    skills = resume_json.get("skills")
    if not isinstance(skills, Mapping):
        return result
    lines = skills.get("lines")
    if not isinstance(lines, list):
        return result
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if isinstance(line_id, str) and isinstance(text, str):
            result[line_id] = text
    return result


def _bullet_map(resume_json: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for exp in resume_json.get("experience", []) if isinstance(resume_json.get("experience"), list) else []:
        if not isinstance(exp, Mapping):
            continue
        for bullet in exp.get("bullets", []) if isinstance(exp.get("bullets"), list) else []:
            if not isinstance(bullet, Mapping):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                result[bullet_id] = text
    for project in resume_json.get("projects", []) if isinstance(resume_json.get("projects"), list) else []:
        if not isinstance(project, Mapping):
            continue
        for bullet in project.get("bullets", []) if isinstance(project.get("bullets"), list) else []:
            if not isinstance(bullet, Mapping):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                result[bullet_id] = text
    return result


def _lowered_terms(values: Any) -> set[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    result = set()
    for value in values:
        if isinstance(value, str):
            result.add(value.lower())
    return result


def _lowered_text(value: Any) -> str:
    return value.lower() if isinstance(value, str) else ""


def _plan_bullet_action(plan: Mapping[str, Any], bullet_id: str) -> Mapping[str, Any] | None:
    actions = plan.get("bullet_actions")
    if not isinstance(actions, list):
        return None
    for action in actions:
        if isinstance(action, Mapping) and action.get("bullet_id") == bullet_id:
            return action
    return None


def _blocked_scopes_for_term(plan: Mapping[str, Any], term: str) -> set[str]:
    blocked_scopes: set[str] = set()
    for item in plan.get("blocked_terms", []) if isinstance(plan.get("blocked_terms"), list) else []:
        if not isinstance(item, Mapping) or item.get("term") != term:
            continue
        scopes = item.get("blocked_for")
        if isinstance(scopes, list):
            blocked_scopes.update(scope for scope in scopes if isinstance(scope, str))
    return blocked_scopes


def _bullet_detail(audit_log: Mapping[str, Any], bullet_id: str) -> Mapping[str, Any]:
    details = audit_log.get("bullet_details")
    if not isinstance(details, list):
        return {}
    for detail in details:
        if isinstance(detail, Mapping) and detail.get("bullet_id") == bullet_id:
            return detail
    return {}


def _format_weighted_terms(job_weights, limit: int = 6) -> list[dict[str, Any]]:
    items = []
    for term in job_weights.ordered_terms[:limit]:
        weight = job_weights.weights_by_term[term]
        items.append({"term": term, "weight": weight.total_weight, "reasons": weight.reasons})
    return items


def _format_bullet_actions(plan: Mapping[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    actions = plan.get("bullet_actions")
    if not isinstance(actions, list):
        return []
    formatted = []
    for action in actions[:limit]:
        if not isinstance(action, Mapping):
            continue
        formatted.append(
            {
                "bullet_id": action.get("bullet_id"),
                "rewrite_intent": action.get("rewrite_intent"),
                "target_keywords": action.get("target_keywords"),
            }
        )
    return formatted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic ATS diagnostic suite.")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Run only the named scenario_id. Repeat to run multiple scenarios.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenario ids and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.scenario_id}: {scenario.scenario_name}")
        return 0

    reports = run_suite(args.scenarios)
    for report in reports:
        print(render_scenario_report(report))
    print(render_global_summary(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
