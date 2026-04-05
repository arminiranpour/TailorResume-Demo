import json
from copy import deepcopy
from pprint import pprint

from app.ats import (
    build_job_signals,
    build_resume_signals,
    build_job_weights,
    build_coverage_model,
    build_evidence_links,
    build_title_alignment,
    build_recency_priorities,
)

from app.pipelines.tailoring_plan import build_tailoring_plan
from app.pipelines.bullet_rewrite import rewrite_resume_text_with_audit


class StubProvider:
    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        task_label, payload = self._extract_task_payload(messages)
        self.calls.append({"task": task_label, "payload": payload, "kwargs": kwargs})
        if task_label == "summary_rewrite":
            return json.dumps(
                {
                    "rewritten_text": "Senior software engineer with Node.js delivery.",
                    "keywords_used": ["senior", "software engineer", "node.js"],
                }
            )
        if task_label == "bullet_rewrite":
            bullet_id = payload.get("bullet_id")
            rewritten_by_bullet = {
                "exp_older_b1": "Shipped TypeScript data pipelines at scale.",
                "exp_recent_b1": "Built Python APIs with PostgreSQL.",
                "exp_recent_b2": "Led CI/CD automation for NodeJS services.",
                "proj_1_b1": "Built ReactJS dashboard for resume tailoring.",
            }
            rewritten_text = rewritten_by_bullet.get(bullet_id, payload.get("original_text", ""))
            return json.dumps(
                {
                    "bullet_id": bullet_id,
                    "rewritten_text": rewritten_text,
                    "keywords_used": list(payload.get("target_keywords", [])),
                    "notes": "witness_stub",
                }
            )
        if task_label == "compress_text":
            candidate_text = str(payload.get("candidate_text", ""))
            max_chars = payload.get("max_chars")
            if isinstance(max_chars, int) and max_chars > 0:
                candidate_text = candidate_text[:max_chars].rstrip()
            return json.dumps({"compressed_text": candidate_text})
        if task_label == "json_repair":
            raw_json_text = self._extract_repair_raw_json(messages)
            json.loads(raw_json_text)
            return raw_json_text
        raise RuntimeError(f"StubProvider does not support task: {task_label}")

    def _extract_task_payload(self, messages):
        if not isinstance(messages, list) or len(messages) < 2:
            raise RuntimeError("StubProvider expected LLM messages")
        user_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        task_prefix = "Task: "
        begin_marker = "BEGIN_UNTRUSTED_TEXT\n"
        end_marker = "\nEND_UNTRUSTED_TEXT"
        if not user_content.startswith(task_prefix):
            raise RuntimeError("StubProvider could not identify task label")
        task_label = user_content[len(task_prefix) :].splitlines()[0].strip()
        start = user_content.find(begin_marker)
        end = user_content.rfind(end_marker)
        payload = {}
        if start != -1 and end != -1 and end > start:
            raw_payload = user_content[start + len(begin_marker) : end]
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {}
        return task_label, payload

    def _extract_repair_raw_json(self, messages):
        user_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        marker = "RAW_JSON_OUTPUT:\n"
        start = user_content.find(marker)
        if start == -1:
            raise RuntimeError("StubProvider could not extract repair payload")
        return user_content[start + len(marker) :].strip()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_top_terms(label: str, terms, limit=10):
    print(f"{label}:")
    for term in list(terms)[:limit]:
        print(f"  - {term}")


def summarize_coverage(coverage):
    print("Covered terms:", len(getattr(coverage, "covered_terms", ())))
    print("Missing terms:", len(getattr(coverage, "missing_terms", ())))
    print("Under-supported terms:", len(getattr(coverage, "under_supported_terms", ())))
    print("Overall distinct coverage:", getattr(coverage, "overall_distinct_coverage", None))
    print("High-priority coverage:", getattr(coverage, "high_priority_coverage", None))
    print("Required coverage:", getattr(coverage, "required_coverage", None))


def summarize_alignment(alignment):
    print("Job title tokens:", getattr(alignment, "job_title_tokens", ()))
    print("Resume title tokens:", getattr(alignment, "resume_title_tokens", ()))
    print("Overlapping tokens:", getattr(alignment, "overlapping_tokens", ()))
    print("Overlapping phrases:", getattr(alignment, "overlapping_phrases", ()))
    print("Strongest resume title:", getattr(alignment, "strongest_matching_resume_title", None))
    print("Alignment score:", getattr(alignment, "title_alignment_score", None))
    print("Alignment strength:", getattr(alignment, "alignment_strength", None))
    print("Title supported:", getattr(alignment, "is_title_supported", None))
    print("Summary-safe title alignment:", getattr(alignment, "is_safe_for_summary_alignment", None))
    print("Missing title tokens:", getattr(alignment, "missing_title_tokens", ()))


def summarize_recency(recency):
    print_top_terms("Recent high-priority terms", getattr(recency, "recent_high_priority_terms", ()))
    print_top_terms("Recent bullet-safe terms", getattr(recency, "recent_bullet_safe_terms", ()))
    print_top_terms("Recent summary-safe terms", getattr(recency, "recent_summary_safe_terms", ()))
    print_top_terms("Stale-only terms", getattr(recency, "stale_only_terms", ()))


def summarize_plan(plan):
    print_top_terms("Prioritized keywords", plan.get("prioritized_keywords", ()))
    print_top_terms("Supported priority terms", plan.get("supported_priority_terms", ()))
    print_top_terms("Under-supported terms", plan.get("under_supported_terms", ()))
    print_top_terms("Recent priority terms", plan.get("recent_priority_terms", ()))
    print("Blocked terms:")
    for item in plan.get("blocked_terms", [])[:10]:
        print("  -", item)
    print("Summary rewrite:")
    pprint(plan.get("summary_rewrite"))
    print("Skills reorder plan:", plan.get("skills_reorder_plan"))
    print("Bullet actions:")
    for action in plan.get("bullet_actions", [])[:10]:
        pprint(action)


def print_resume_diff(before_resume, after_resume):
    print_header("SUMMARY BEFORE / AFTER")
    before_summary = before_resume.get("summary", {}).get("text", "")
    after_summary = after_resume.get("summary", {}).get("text", "")
    print("BEFORE:", before_summary)
    print("AFTER :", after_summary)

    print_header("SKILLS BEFORE / AFTER")
    before_lines = before_resume.get("skills", {}).get("lines", [])
    after_lines = after_resume.get("skills", {}).get("lines", [])
    for idx, (b, a) in enumerate(zip(before_lines, after_lines), start=1):
        print(f"Line {idx} [{b.get('line_id')}]:")
        print("  BEFORE:", b.get("text"))
        print("  AFTER :", a.get("text"))

    print_header("BULLETS BEFORE / AFTER")
    before_exps = before_resume.get("experience", [])
    after_exps = after_resume.get("experience", [])
    for bexp, aexp in zip(before_exps, after_exps):
        before_bullets = bexp.get("bullets", [])
        after_bullets = aexp.get("bullets", [])
        for bb, ab in zip(before_bullets, after_bullets):
            if bb.get("text") != ab.get("text"):
                print(f"Bullet {bb.get('bullet_id')}:")
                print("  BEFORE:", bb.get("text"))
                print("  AFTER :", ab.get("text"))


def summarize_frequency(audit_log):
    print_header("FREQUENCY BALANCE")
    freq_balance = audit_log.get("frequency_balance")
    freq_actions = audit_log.get("frequency_actions")

    if not freq_balance:
        print("No frequency_balance found in audit log.")
        return

    if isinstance(freq_balance, dict):
        print("Overused terms:", freq_balance.get("overused_terms"))
        print("Underused terms:", freq_balance.get("underused_terms"))
        print("Within-range terms:", freq_balance.get("within_range_terms"))
        print("Capped terms:", freq_balance.get("capped_terms"))
        print("Validation errors:")
        pprint(freq_balance.get("validation_errors"))
    else:
        pprint(freq_balance)

    print("\nFrequency actions:")
    pprint(freq_actions)


def main():
    resume_json = load_json("tests/fixtures/ats_resume.json")
    job_json = load_json("tests/fixtures/ats_job.json")

    score_result = {
        "decision": "PROCEED",
        "missing_requirements": []
    }

    original_resume = deepcopy(resume_json)

    print_header("STEP 1 — SIGNAL EXTRACTION")
    job_signals = build_job_signals(job_json)
    resume_signals = build_resume_signals(resume_json)
    print_top_terms("Job canonical terms", getattr(job_signals, "canonical_terms", ()))
    print_top_terms("Resume all terms", getattr(resume_signals, "all_terms", ()))
    print_top_terms("Resume title-like terms", getattr(resume_signals, "title_like_terms", ()))

    print_header("STEP 2 — KEYWORD WEIGHTING")
    job_weights = build_job_weights(job_signals)
    print_top_terms("Ordered weighted terms", getattr(job_weights, "ordered_terms", ()))
    print_top_terms("High priority terms", getattr(job_weights, "high_priority_terms", ()))
    print_top_terms("Required priority terms", getattr(job_weights, "required_priority_terms", ()))

    print_header("STEP 3 — COVERAGE MODEL")
    coverage = build_coverage_model(job_signals, resume_signals, job_weights)
    summarize_coverage(coverage)

    print_header("STEP 4 — EVIDENCE LINKING")
    evidence_links = build_evidence_links(job_signals, resume_signals, job_weights, coverage)
    print_top_terms("Linked terms", getattr(evidence_links, "linked_terms", ()))
    print_top_terms("Skills-only terms", getattr(evidence_links, "skills_only_terms", ()))
    print_top_terms("Missing experience terms", getattr(evidence_links, "missing_experience_terms", ()))

    print_header("STEP 5 — TITLE ALIGNMENT")
    alignment = build_title_alignment(job_signals, resume_signals, job_weights, coverage, evidence_links)
    summarize_alignment(alignment)

    print_header("STEP 6 — RECENCY PRIORITIES")
    recency = build_recency_priorities(
        job_signals,
        resume_signals,
        job_weights,
        coverage,
        evidence_links,
        alignment,
    )
    summarize_recency(recency)

    print_header("STEP 7 — ATS TAILORING PLAN")
    plan = build_tailoring_plan(
        resume_json=resume_json,
        job_json=job_json,
        score_result=score_result,
        provider=None,
        job_signals=job_signals,
        resume_signals=resume_signals,
        job_weights=job_weights,
        coverage=coverage,
        evidence_links=evidence_links,
        title_alignment=alignment,
        recency_priorities=recency,
    )
    summarize_plan(plan)

    print_header("STEPS 8–11 — REWRITE / SKILLS / SUMMARY / FREQUENCY")
    provider = StubProvider()

    rewritten_resume, audit_log = rewrite_resume_text_with_audit(
        resume_json=deepcopy(resume_json),
        job_json=job_json,
        score_result=score_result,
        tailoring_plan=plan,
        provider=provider,
    )

    print_resume_diff(original_resume, rewritten_resume)

    summarize_frequency(audit_log)

    print_header("DONE")
    print("If the before/after sections and ATS summaries above look correct, you have a witness run for Steps 1–11.")
    print("Step 12 is your regression test suite and should be run separately with pytest.")


if __name__ == "__main__":
    main()
