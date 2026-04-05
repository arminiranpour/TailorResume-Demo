from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ats_diagnostics"


@dataclass(frozen=True)
class DiagnosticScenario:
    scenario_id: str
    scenario_name: str
    target_steps: tuple[int, ...]
    description: str
    key_conditions: tuple[str, ...]
    resume_fixture_path: Path
    job_fixture_path: Path
    expectations: tuple[Mapping[str, Any], ...] = ()
    provider_spec: Mapping[str, Any] = field(default_factory=dict)
    character_budgets: Mapping[str, Any] = field(default_factory=dict)
    score_result_overrides: Mapping[str, Any] = field(default_factory=dict)


def _scenario(
    *,
    scenario_id: str,
    scenario_name: str,
    target_steps: tuple[int, ...],
    description: str,
    key_conditions: tuple[str, ...],
    resume_fixture: str,
    job_fixture: str,
    expectations: tuple[Mapping[str, Any], ...],
    provider_spec: Mapping[str, Any] | None = None,
    character_budgets: Mapping[str, Any] | None = None,
    score_result_overrides: Mapping[str, Any] | None = None,
) -> DiagnosticScenario:
    return DiagnosticScenario(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        target_steps=target_steps,
        description=description,
        key_conditions=key_conditions,
        resume_fixture_path=FIXTURES_DIR / resume_fixture,
        job_fixture_path=FIXTURES_DIR / job_fixture,
        expectations=expectations,
        provider_spec=provider_spec or {},
        character_budgets=character_budgets or {},
        score_result_overrides=score_result_overrides or {},
    )


SCENARIOS: tuple[DiagnosticScenario, ...] = (
    _scenario(
        scenario_id="weighting_required_over_preferred",
        scenario_name="Required Weighting Beats Preferred",
        target_steps=(2,),
        description="Verifies that must-have ATS terms outrank preferred-only terms in deterministic weighting.",
        key_conditions=(
            "Python and REST API are must-have requirements.",
            "GraphQL and Kafka are preferred-only terms.",
            "Resume carries both strong required evidence and a preferred term.",
        ),
        resume_fixture="weighting_resume.json",
        job_fixture="weighting_job.json",
        expectations=(
            {
                "id": "weight_order",
                "label": "Required Python should outrank preferred GraphQL.",
                "higher": "python",
                "lower": "graphql",
                "likely_steps": (2,),
            },
            {
                "id": "weight_order",
                "label": "Required REST API should outrank preferred Kafka.",
                "higher": "rest api",
                "lower": "kafka",
                "likely_steps": (2,),
            },
        ),
    ),
    _scenario(
        scenario_id="coverage_cross_section_growth",
        scenario_name="Coverage Cross-Section Growth",
        target_steps=(3, 9, 10),
        description="Verifies that truthful ATS coverage can expand across summary and skills without inventing new evidence.",
        key_conditions=(
            "Python and PostgreSQL exist in experience only before tailoring.",
            "Summary is generic and skills omit the supported technical terms.",
            "Deterministic rewrites should improve cross-section support after tailoring.",
        ),
        resume_fixture="coverage_resume.json",
        job_fixture="coverage_job.json",
        expectations=(
            {
                "id": "coverage_cross_section_improves",
                "label": "Python should gain truthful cross-section support.",
                "term": "python",
                "likely_steps": (3, 9, 10),
                "focus_section": "summary",
            },
            {
                "id": "coverage_cross_section_improves",
                "label": "PostgreSQL should gain truthful cross-section support.",
                "term": "postgresql",
                "likely_steps": (3, 9, 10),
                "focus_section": "skills",
            },
            {
                "id": "section_changed",
                "label": "Coverage scenario should change the summary or skills section.",
                "section": "skills",
                "allow_alternate_section": "summary",
                "likely_steps": (9, 10),
            },
        ),
        provider_spec={
            "summary_text": "Backend engineer with Python and PostgreSQL platform delivery.",
        },
        character_budgets={
            "skills_line_max_chars": {
                "skills_1": 48,
            }
        },
    ),
    _scenario(
        scenario_id="evidence_linking_strength",
        scenario_name="Evidence Linking Distinguishes Skills-Only Support",
        target_steps=(4, 8),
        description="Verifies that skills-only support is treated differently from strong experience evidence during ATS rewrites.",
        key_conditions=(
            "Python has recent experience evidence.",
            "Kubernetes appears only in the skills section.",
            "A deterministic bullet rewrite intentionally attempts to inject the skills-only term.",
        ),
        resume_fixture="evidence_resume.json",
        job_fixture="evidence_job.json",
        expectations=(
            {
                "id": "evidence_distinction",
                "label": "Python should be strong experience-backed while Kubernetes remains skills-only.",
                "strong_term": "python",
                "weak_term": "kubernetes",
                "likely_steps": (4,),
            },
            {
                "id": "plan_blocks_term_scope",
                "label": "Kubernetes should be blocked for bullets.",
                "term": "kubernetes",
                "scope": "bullets",
                "likely_steps": (4, 7),
            },
            {
                "id": "bullet_rejects_term",
                "label": "Bullet rewrite should reject Kubernetes injection.",
                "bullet_id": "exp_recent_b1",
                "term": "kubernetes",
                "likely_steps": (4, 8),
                "focus_section": "bullets",
            },
        ),
        provider_spec={
            "bullet_text_by_id": {
                "exp_recent_b1": "Built Python APIs with Kubernetes for platform services.",
            }
        },
    ),
    _scenario(
        scenario_id="title_alignment_safe_only",
        scenario_name="Safe Title Alignment Only",
        target_steps=(5, 10),
        description="Verifies that the summary aligns to the job title only when ATS title support is safely established.",
        key_conditions=(
            "Resume already has matching platform-engineering title evidence.",
            "Job title asks for Senior Platform Engineer.",
            "Summary rewrite candidate includes the aligned title phrase.",
        ),
        resume_fixture="title_resume.json",
        job_fixture="title_job.json",
        expectations=(
            {
                "id": "title_alignment_safe",
                "label": "Title alignment should be safe and summary-ready.",
                "likely_steps": (5,),
            },
            {
                "id": "summary_contains",
                "label": "Summary should include the supported platform engineer title phrase.",
                "term": "platform engineer",
                "likely_steps": (5, 10),
                "focus_section": "summary",
            },
        ),
        provider_spec={
            "summary_text": "Platform engineer building Python APIs.",
        },
    ),
    _scenario(
        scenario_id="recency_prefers_recent_evidence",
        scenario_name="Recency Prefers Recent Evidence",
        target_steps=(6, 7, 8),
        description="Verifies that recent comparable evidence is preferred over stale evidence in bullet targeting and rewrite selection.",
        key_conditions=(
            "Python appears in both older and recent experience bullets.",
            "Recent experience is the stronger ATS source to prefer.",
            "Only the recent bullet should be rewritten.",
        ),
        resume_fixture="recency_resume.json",
        job_fixture="recency_job.json",
        expectations=(
            {
                "id": "recent_candidate_preferred",
                "label": "Python should prefer the recent experience bullet.",
                "term": "python",
                "expected_source_id": "exp_recent_b1",
                "likely_steps": (6,),
            },
            {
                "id": "bullet_action_intent",
                "label": "Recent bullet should be targeted for rewrite.",
                "bullet_id": "exp_recent_b1",
                "intent": "rewrite",
                "likely_steps": (6, 7),
            },
            {
                "id": "bullet_action_intent",
                "label": "Older bullet should remain untouched.",
                "bullet_id": "exp_old_b1",
                "intent": "keep",
                "likely_steps": (6, 7),
            },
        ),
        provider_spec={
            "bullet_text_by_id": {
                "exp_recent_b1": "Built Python APIs for platform services.",
            }
        },
    ),
    _scenario(
        scenario_id="tailoring_plan_controls",
        scenario_name="Tailoring Plan Control Surface",
        target_steps=(7,),
        description="Verifies that the deterministic plan emits supported, under-supported, blocked, and bullet-action guidance correctly.",
        key_conditions=(
            "Python and FastAPI have strong evidence.",
            "AWS is skills-only support and should be under-supported.",
            "Kubernetes is unsupported and should be blocked.",
        ),
        resume_fixture="plan_resume.json",
        job_fixture="plan_job.json",
        expectations=(
            {
                "id": "plan_supported_term",
                "label": "FastAPI should appear as a supported priority term.",
                "term": "fastapi",
                "likely_steps": (7,),
            },
            {
                "id": "plan_under_supported_term",
                "label": "AWS should appear as an under-supported term.",
                "term": "aws",
                "likely_steps": (7,),
            },
            {
                "id": "plan_blocks_term_scope",
                "label": "Kubernetes should be blocked for summary, skills, and bullets.",
                "term": "kubernetes",
                "scope": "summary",
                "extra_scopes": ("skills", "bullets"),
                "likely_steps": (7,),
            },
            {
                "id": "bullet_action_intent",
                "label": "Primary experience bullet should be targeted for rewrite.",
                "bullet_id": "exp_recent_b1",
                "intent": "rewrite",
                "likely_steps": (7,),
            },
        ),
    ),
    _scenario(
        scenario_id="bullet_rewrite_enforces_support",
        scenario_name="Bullet Rewrite Enforces Evidence Support",
        target_steps=(8,),
        description="Verifies that evidence-backed bullet rewrites succeed while blocked or unsupported ATS terms are rejected.",
        key_conditions=(
            "FastAPI and PostgreSQL have bullet-safe evidence.",
            "AWS exists only as skills support and should not appear in bullets.",
            "Two bullet rewrites are attempted: one safe and one unsupported.",
        ),
        resume_fixture="bullet_resume.json",
        job_fixture="bullet_job.json",
        expectations=(
            {
                "id": "bullet_changed",
                "label": "Primary API bullet should be safely rewritten.",
                "bullet_id": "exp_recent_b1",
                "likely_steps": (8,),
                "focus_section": "bullets",
            },
            {
                "id": "bullet_contains",
                "label": "Primary API bullet should retain FastAPI and PostgreSQL context.",
                "bullet_id": "exp_recent_b1",
                "terms": ("fastapi", "postgresql"),
                "likely_steps": (8,),
                "focus_section": "bullets",
            },
            {
                "id": "bullet_rejects_term",
                "label": "Secondary bullet should reject AWS injection.",
                "bullet_id": "exp_recent_b2",
                "term": "aws",
                "likely_steps": (8,),
                "focus_section": "bullets",
            },
        ),
        provider_spec={
            "bullet_text_by_id": {
                "exp_recent_b1": "Built FastAPI services with PostgreSQL for internal systems.",
                "exp_recent_b2": "Improved Python services with AWS deployment workflows.",
            }
        },
    ),
    _scenario(
        scenario_id="skills_optimization_canonical",
        scenario_name="Skills Optimization Uses Canonical Supported Terms",
        target_steps=(9,),
        description="Verifies that supported canonical skill surfaces are promoted while unsupported skills stay out of the final skills lines.",
        key_conditions=(
            "Resume uses JS and ReactJS variants.",
            "Job prefers canonical JavaScript and React wording.",
            "AWS is requested by the job but unsupported by resume evidence.",
        ),
        resume_fixture="skills_resume.json",
        job_fixture="skills_job.json",
        expectations=(
            {
                "id": "skills_contains_any_line",
                "label": "Skills should surface canonical JavaScript.",
                "term": "javascript",
                "likely_steps": (9,),
                "focus_section": "skills",
            },
            {
                "id": "skills_contains_any_line",
                "label": "Skills should surface canonical React.",
                "term": "react",
                "likely_steps": (9,),
                "focus_section": "skills",
            },
            {
                "id": "skills_not_contains",
                "label": "Unsupported AWS should not be surfaced into skills.",
                "term": "aws",
                "likely_steps": (9,),
                "focus_section": "skills",
            },
        ),
        character_budgets={
            "skills_line_max_chars": {
                "skills_1": 34,
            }
        },
    ),
    _scenario(
        scenario_id="summary_rewrite_blocks_unsafe_terms",
        scenario_name="Summary Rewrite Blocks Unsafe Terms",
        target_steps=(5, 10),
        description="Verifies that summary rewriting preserves safe ATS terms while rejecting unsafe title and unsupported tool/domain phrasing.",
        key_conditions=(
            "Original summary already contains safe Python/API wording.",
            "Rewrite candidate intentionally injects architect and AWS phrasing.",
            "Final summary must stay safe and ATS-supported.",
        ),
        resume_fixture="summary_resume.json",
        job_fixture="summary_job.json",
        expectations=(
            {
                "id": "summary_reject_reason_in",
                "label": "Summary rewrite should be rejected for unsafe ATS phrasing.",
                "reasons": ("unsafe_title_alignment", "unsupported_ats_terms", "blocked_terms"),
                "likely_steps": (5, 10),
                "focus_section": "summary",
            },
            {
                "id": "summary_not_contains",
                "label": "Unsafe architect phrasing must stay out of the final summary.",
                "term": "architect",
                "likely_steps": (5, 10),
                "focus_section": "summary",
            },
            {
                "id": "summary_not_contains",
                "label": "Unsupported AWS should stay out of the final summary.",
                "term": "aws",
                "likely_steps": (10,),
                "focus_section": "summary",
            },
            {
                "id": "summary_contains",
                "label": "Safe Python signal should remain in the final summary.",
                "term": "python",
                "likely_steps": (10,),
                "focus_section": "summary",
            },
        ),
        provider_spec={
            "summary_text": "Staff platform architect building Python REST APIs with AWS.",
        },
    ),
    _scenario(
        scenario_id="frequency_balance_rolls_back_stuffing",
        scenario_name="Frequency Balance Rolls Back Stuffing",
        target_steps=(11,),
        description="Verifies that ATS stuffing is detected after rewrite and deterministically rolled back.",
        key_conditions=(
            "Summary rewrite candidate repeats Python excessively.",
            "Frequency balance should rollback the stuffed summary.",
            "Final summary should avoid repeated-term inflation.",
        ),
        resume_fixture="frequency_resume.json",
        job_fixture="frequency_job.json",
        expectations=(
            {
                "id": "frequency_rollback",
                "label": "Frequency balance should rollback the stuffed summary.",
                "term": "python",
                "section": "summary",
                "likely_steps": (11,),
                "focus_section": "summary",
            },
            {
                "id": "summary_skip_reason",
                "label": "Summary detail should record the frequency rollback.",
                "skip_reason": "frequency_balance_rollback",
                "likely_steps": (11,),
                "focus_section": "summary",
            },
            {
                "id": "summary_max_term_count",
                "label": "Final summary should not exceed the Python repetition cap.",
                "term": "python",
                "max_count": 3,
                "likely_steps": (11,),
                "focus_section": "summary",
            },
        ),
        provider_spec={
            "summary_text": "Python engineer building Python API for Python teams.",
        },
    ),
    _scenario(
        scenario_id="integrated_multi_step_signal",
        scenario_name="Integrated Multi-Step ATS Interaction",
        target_steps=(2, 4, 5, 6, 7, 8, 9, 10, 11),
        description="Exercises multiple ATS steps together so weak interactions are visible across summary, skills, bullets, and frequency controls.",
        key_conditions=(
            "Job asks for Senior Platform Engineer with Python, React, and PostgreSQL.",
            "Resume mixes recent API evidence, old Kubernetes evidence, and JS/React variants.",
            "Scenario should trigger summary, skills, and bullet changes without unsafe term introduction.",
        ),
        resume_fixture="integrated_resume.json",
        job_fixture="integrated_job.json",
        expectations=(
            {
                "id": "recent_candidate_preferred",
                "label": "Integrated scenario should still prefer recent Python evidence.",
                "term": "python",
                "expected_source_id": "exp_recent_b1",
                "likely_steps": (6,),
            },
            {
                "id": "summary_contains",
                "label": "Summary should align toward the supported platform engineer title.",
                "term": "platform engineer",
                "likely_steps": (5, 10),
                "focus_section": "summary",
            },
            {
                "id": "skills_contains_any_line",
                "label": "Skills should surface canonical JavaScript.",
                "term": "javascript",
                "likely_steps": (2, 9),
                "focus_section": "skills",
            },
            {
                "id": "bullet_changed",
                "label": "Recent API bullet should be rewritten.",
                "bullet_id": "exp_recent_b1",
                "likely_steps": (7, 8),
                "focus_section": "bullets",
            },
            {
                "id": "summary_not_contains",
                "label": "Unsupported Kubernetes should stay out of the summary.",
                "term": "kubernetes",
                "likely_steps": (4, 10),
                "focus_section": "summary",
            },
            {
                "id": "section_changed",
                "label": "Integrated scenario should change the summary section.",
                "section": "summary",
                "likely_steps": (5, 10),
                "focus_section": "summary",
            },
            {
                "id": "section_changed",
                "label": "Integrated scenario should change the skills section.",
                "section": "skills",
                "likely_steps": (9,),
                "focus_section": "skills",
            },
            {
                "id": "section_changed",
                "label": "Integrated scenario should change bullets.",
                "section": "bullets",
                "likely_steps": (7, 8),
                "focus_section": "bullets",
            },
        ),
        provider_spec={
            "summary_text": "Senior platform engineer building Python APIs with PostgreSQL.",
            "bullet_text_by_id": {
                "exp_recent_b1": "Built Python APIs with PostgreSQL for platform services.",
                "proj_ui_b1": "Built React dashboard for platform operations.",
            },
        },
        character_budgets={
            "skills_line_max_chars": {
                "skills_1": 40,
            }
        },
    ),
)


SCENARIOS_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
