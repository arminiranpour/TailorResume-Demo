from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.config import get_config
from app.pipelines.allowed_vocab import build_allowed_vocab, normalize_terms
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages
from shared.scoring.normalize import generate_ngrams, normalize_text, tokenize


@dataclass
class BulletRewriteError(Exception):
    details: List[str]
    raw_preview: str = ""


@dataclass
class BulletRewriteNotAllowed(Exception):
    decision: str
    reasons: List[Dict[str, Any]]

    def __str__(self) -> str:
        return f"Bullet rewrite not allowed for decision={self.decision}"


_REWRITE_KEEP = {"keep", "skip", "none"}
_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9#.+-]*")
_TRAILING_PERIODS = "."
_MIN_BUDGET = 80

_SOFT_SKILL_TERMS = {
    "communication",
    "communication skills",
    "collaboration",
    "collaborative",
    "teamwork",
    "team player",
    "leadership",
    "problem solving",
    "problem-solving",
    "customer service",
    "time management",
    "adaptability",
    "organization",
    "organizational",
    "interpersonal",
    "interpersonal skills",
    "cross functional",
    "cross-functional",
    "stakeholder management",
    "mentorship",
    "coaching",
    "initiative",
    "ownership",
    "strategic thinking",
    "critical thinking",
    "attention to detail",
    "detail oriented",
    "presentation",
    "writing",
    "verbal communication",
    "conflict resolution",
    "negotiation",
    "prioritization",
}

_TECHNICAL_INDICATORS = {
    "api",
    "apis",
    "backend",
    "frontend",
    "full stack",
    "full-stack",
    "database",
    "databases",
    "data",
    "sql",
    "etl",
    "pipeline",
    "pipelines",
    "cloud",
    "infrastructure",
    "devops",
    "ci cd",
    "ci/cd",
    "microservices",
    "distributed systems",
    "systems",
    "architecture",
    "security",
    "network",
    "performance",
    "scalability",
    "automation",
    "testing",
}


def rewrite_resume_text(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    character_budgets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tailored, _ = rewrite_resume_text_with_audit(
        resume_json,
        job_json,
        score_result,
        tailoring_plan,
        provider,
        character_budgets=character_budgets,
    )
    return tailored


def rewrite_resume_text_with_audit(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    character_budgets: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(resume_json, dict) or not isinstance(job_json, dict) or not isinstance(score_result, dict):
        raise ValueError("resume_json, job_json, and score_result must be objects")
    if not isinstance(tailoring_plan, dict):
        raise ValueError("tailoring_plan must be an object")

    decision = score_result.get("decision")
    if decision != "PROCEED":
        reasons = score_result.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        raise BulletRewriteNotAllowed(decision=str(decision), reasons=reasons)

    _validate_schema_or_raise("resume", resume_json)
    _validate_schema_or_raise("tailoring_plan", tailoring_plan)

    allowed_vocab = build_allowed_vocab(resume_json)
    budgets = _derive_budgets(resume_json, character_budgets)

    tailored = _clone_resume(resume_json)
    audit_log = {
        "rewritten_bullets": [],
        "kept_bullets": [],
        "rejected_for_new_terms": [],
        "compressed": [],
        "bullet_details": [],
        "summary_detail": None,
        "skills_details": [],
    }

    _rewrite_summary(tailored, job_json, tailoring_plan, provider, allowed_vocab, budgets, audit_log)
    _reorder_skills(tailored, tailoring_plan)
    _tailor_skill_lines(tailored, job_json, tailoring_plan, allowed_vocab, audit_log)
    _rewrite_bullets(
        tailored,
        job_json,
        tailoring_plan,
        provider,
        allowed_vocab,
        budgets,
        audit_log,
    )

    ok, errors = validate_json("resume", tailored)
    if not ok:
        raise BulletRewriteError(details=errors, raw_preview="")

    invariant_errors = _check_invariants(resume_json, tailored)
    if invariant_errors:
        raise BulletRewriteError(details=invariant_errors, raw_preview="")
    return tailored, audit_log


def _validate_schema_or_raise(schema_name: str, obj: Dict[str, Any]) -> None:
    ok, errors = validate_json(schema_name, obj)
    if not ok:
        raise BulletRewriteError(details=errors, raw_preview="")


def _clone_resume(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(resume_json))


def _derive_budgets(resume_json: Dict[str, Any], character_budgets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary_budget = None
    bullet_budgets: Dict[str, int] = {}
    if isinstance(character_budgets, dict):
        summary_value = character_budgets.get("summary")
        if isinstance(summary_value, int) and summary_value > 0:
            summary_budget = summary_value
        bullets_value = character_budgets.get("bullets")
        if isinstance(bullets_value, dict):
            for bullet_id, value in bullets_value.items():
                if isinstance(bullet_id, str) and isinstance(value, int) and value > 0:
                    bullet_budgets[bullet_id] = value

    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    summary_text = summary.get("text") if isinstance(summary.get("text"), str) else ""
    if summary_budget is None and summary_text:
        summary_budget = max(len(summary_text), _MIN_BUDGET)

    for bullet_id, text in _iter_bullet_texts(resume_json):
        if bullet_id in bullet_budgets:
            continue
        budget = max(len(text), _MIN_BUDGET)
        bullet_budgets[bullet_id] = budget

    return {"summary": summary_budget, "bullets": bullet_budgets}


def _iter_bullet_texts(resume_json: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                yield bullet_id, text

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            text = bullet.get("text")
            if isinstance(bullet_id, str) and isinstance(text, str):
                yield bullet_id, text


def _rewrite_summary(
    tailored: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    summary_detail = {
        "original_text": "",
        "rewrite_intent": None,
        "target_keywords": [],
        "candidate_text": None,
        "final_text": None,
        "skip_reason": None,
        "reject_reason": None,
        "changed": False,
        "fallback_used": False,
        "fallback_reason": None,
    }
    summary_plan = tailoring_plan.get("summary_rewrite")
    if not isinstance(summary_plan, dict):
        summary_detail["skip_reason"] = "no_summary_plan"
        audit_log["summary_detail"] = summary_detail
        return
    rewrite_intent = summary_plan.get("rewrite_intent")
    if isinstance(rewrite_intent, str):
        summary_detail["rewrite_intent"] = rewrite_intent
    target_keywords = summary_plan.get("target_keywords") if isinstance(summary_plan.get("target_keywords"), list) else []

    summary = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else None
    if summary is None:
        summary_detail["skip_reason"] = "missing_summary"
        audit_log["summary_detail"] = summary_detail
        return
    original_text = summary.get("text")
    if not isinstance(original_text, str) or original_text.strip() == "":
        summary_detail["skip_reason"] = "missing_summary_text"
        audit_log["summary_detail"] = summary_detail
        return
    summary_detail["original_text"] = original_text
    summary_detail["target_keywords"] = list(target_keywords)

    budget = budgets.get("summary")
    rewritten_text, reject_reason = _call_summary_rewrite(
        original_text,
        job_json,
        target_keywords,
        allowed_vocab,
        provider,
        budget,
        force_change=True,
    )
    summary_detail["candidate_text"] = rewritten_text
    if reject_reason:
        summary_detail["reject_reason"] = reject_reason
    if _needs_summary_fallback(original_text, rewritten_text, reject_reason):
        summary_detail["fallback_used"] = True
        summary_detail["fallback_reason"] = _fallback_reason(original_text, rewritten_text, reject_reason)
        fallback_text = _fallback_summary_rewrite(original_text)
        if fallback_text is None or _is_effectively_same(original_text, fallback_text):
            summary_detail["skip_reason"] = "fallback_failed"
            audit_log["summary_detail"] = summary_detail
            return
        rewritten_text = fallback_text
        summary_detail["candidate_text"] = rewritten_text

    if rewritten_text is None:
        summary_detail["skip_reason"] = "llm_no_rewrite_or_rejected"
        audit_log["summary_detail"] = summary_detail
        return
    if budget is not None and len(rewritten_text) > budget:
        compressed = _compress_text(original_text, rewritten_text, budget, provider)
        if compressed is not None:
            rewritten_text = compressed
            disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
            if disallowed_after:
                summary_detail["reject_reason"] = "disallowed_terms_after_compress"
                summary_detail["disallowed_terms"] = disallowed_after
                audit_log["summary_detail"] = summary_detail
                return
        if budget is not None and len(rewritten_text) > budget:
            rewritten_text = _truncate_to_budget(rewritten_text, budget)
    if _is_effectively_same(original_text, rewritten_text):
        summary_detail["skip_reason"] = "final_identical"
        audit_log["summary_detail"] = summary_detail
        return
    summary["text"] = rewritten_text
    summary_detail["final_text"] = rewritten_text
    summary_detail["changed"] = rewritten_text != original_text
    audit_log["summary_detail"] = summary_detail


def _call_summary_rewrite(
    original_text: str,
    job_json: Dict[str, Any],
    target_keywords: Sequence[str],
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
    force_change: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    allowed_target, missing_target = _split_target_keywords(target_keywords, allowed_terms)
    payload = {
        "original_text": original_text,
        "job_title": job_json.get("title"),
        "target_keywords": list(target_keywords),
        "allowed_target_keywords": allowed_target,
        "missing_target_keywords": missing_target,
        "allowed_terms": _select_allowed_terms(allowed_terms, original_text, target_keywords),
        "allowed_proper_nouns": _select_allowed_proper_nouns(allowed_proper, original_text),
        "force_change": force_change,
    }
    if budget is not None:
        payload["max_chars"] = budget

    system_prompt = load_system_prompt("summary_rewrite")
    messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="summary_rewrite")
    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    obj = _parse_llm_json(raw, "summary_rewrite", provider)
    if obj is None:
        return None, "invalid_response"
    rewritten_text = obj.get("rewritten_text")
    if not isinstance(rewritten_text, str):
        return None, "invalid_text"

    disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
    if disallowed:
        payload["disallowed_terms"] = disallowed
        payload["retry_instruction"] = "Remove the disallowed terms. Do not replace them with new terms."
        raw_retry = provider.generate(
            build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="summary_rewrite"),
            timeout=config.llm_timeout_seconds,
        )
        obj_retry = _parse_llm_json(raw_retry, "summary_rewrite", provider)
        if obj_retry and isinstance(obj_retry.get("rewritten_text"), str):
            rewritten_text = obj_retry["rewritten_text"]
            disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
        if disallowed:
            return None, "disallowed_terms"

    return rewritten_text, None


def _reorder_skills(tailored: Dict[str, Any], tailoring_plan: Dict[str, Any]) -> None:
    # Skills line order must remain unchanged for DOCX mapping integrity.
    # Tailoring should only adjust text within each line (handled elsewhere).
    _ = tailoring_plan
    return


def _tailor_skill_lines(
    tailored: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    allowed_vocab: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else None
    if skills is None:
        return
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    if not lines:
        return

    candidates = _collect_candidate_keywords(job_json, tailoring_plan, allowed_vocab)
    if not candidates:
        for line in lines:
            if not isinstance(line, dict):
                continue
            line_id = line.get("line_id")
            original_text = line.get("text")
            if not isinstance(line_id, str) or not isinstance(original_text, str):
                continue
            audit_log["skills_details"].append(
                {
                    "line_id": line_id,
                    "original_text": original_text,
                    "final_text": original_text,
                    "changed": False,
                    "skip_reason": "no_candidate_keywords",
                    "target_keywords": [],
                }
            )
        return

    resume_texts = list(_iter_resume_texts(tailored))
    resume_normalized = normalize_text(" ".join(resume_texts))
    soft_skill_terms = set(_SOFT_SKILL_TERMS)
    hard_skill_terms, hard_skill_display = _build_resume_hard_skill_inventory(
        tailored,
        candidates,
        resume_texts,
        soft_skill_terms,
        resume_normalized,
    )

    matched_hard_terms = _collect_matched_hard_terms(candidates, hard_skill_terms)
    matched_soft_terms = _collect_matched_soft_terms(candidates, soft_skill_terms, resume_normalized)

    existing_skill_terms = _collect_existing_skill_terms(lines)
    missing_hard_terms = [
        term for term in matched_hard_terms if term not in existing_skill_terms and term in hard_skill_display
    ]

    line_states: List[Dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        original_text = line.get("text")
        if not isinstance(line_id, str) or not isinstance(original_text, str):
            continue
        segments, separator = _split_skills_line(original_text)
        if len(segments) <= 1:
            audit_log["skills_details"].append(
                {
                    "line_id": line_id,
                    "original_text": original_text,
                    "target_keywords": [],
                    "final_text": original_text,
                    "changed": False,
                    "skip_reason": "single_segment",
                }
            )
            line_states.append(
                {
                    "line": line,
                    "line_id": line_id,
                    "original_text": original_text,
                    "segments": segments,
                    "separator": separator,
                    "classifications": [_classify_skill_segment(seg, hard_skill_terms, soft_skill_terms) for seg in segments],
                    "budget": len(original_text),
                }
            )
            continue

        detail = {
            "line_id": line_id,
            "original_text": original_text,
            "target_keywords": [],
            "final_text": original_text,
            "changed": False,
            "skip_reason": None,
        }
        classifications = [_classify_skill_segment(seg, hard_skill_terms, soft_skill_terms) for seg in segments]
        reordered = _reorder_skill_segments(
            segments,
            classifications,
            matched_hard_terms,
            matched_soft_terms,
        )
        if reordered == segments:
            detail["skip_reason"] = "no_reorder_needed"
        else:
            detail["target_keywords"] = _extract_matching_keywords(reordered, candidates)
        new_text = separator.join(reordered)
        line["text"] = new_text
        detail["final_text"] = new_text
        detail["changed"] = new_text != original_text
        audit_log["skills_details"].append(detail)
        line_states.append(
            {
                "line": line,
                "line_id": line_id,
                "original_text": original_text,
                "segments": reordered,
                "separator": separator,
                "classifications": [
                    _classify_skill_segment(seg, hard_skill_terms, soft_skill_terms) for seg in reordered
                ],
                "budget": len(original_text),
            }
        )

    if missing_hard_terms:
        _surface_missing_hard_skills(line_states, missing_hard_terms, hard_skill_display, soft_skill_terms, audit_log)


def _rewrite_bullets(
    tailored: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, Any],
) -> None:
    action_map = _build_action_map(tailoring_plan)
    bullet_budgets = budgets.get("bullets", {})

    experience = tailored.get("experience") if isinstance(tailored.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            _rewrite_bullet(
                bullet,
                action_map,
                job_json,
                provider,
                allowed_vocab,
                bullet_budgets,
                audit_log,
            )

    projects = tailored.get("projects") if isinstance(tailored.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            _rewrite_bullet(
                bullet,
                action_map,
                job_json,
                provider,
                allowed_vocab,
                bullet_budgets,
                audit_log,
            )


def _rewrite_bullet(
    bullet: Dict[str, Any],
    action_map: Dict[str, Dict[str, Any]],
    job_json: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    bullet_budgets: Dict[str, int],
    audit_log: Dict[str, Any],
) -> None:
    bullet_id = bullet.get("bullet_id")
    original_text = bullet.get("text")
    if not isinstance(bullet_id, str) or not isinstance(original_text, str):
        return
    detail = {
        "bullet_id": bullet_id,
        "original_text": original_text,
        "rewrite_intent": None,
        "target_keywords": [],
        "candidate_text": None,
        "final_text": None,
        "skip_reason": None,
        "reject_reason": None,
        "changed": False,
    }

    action = action_map.get(bullet_id)
    if not isinstance(action, dict):
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "missing_action"
        _append_bullet_detail(audit_log, detail)
        return

    rewrite_intent = action.get("rewrite_intent")
    if isinstance(rewrite_intent, str):
        detail["rewrite_intent"] = rewrite_intent
    if isinstance(rewrite_intent, str) and rewrite_intent.strip().lower() in _REWRITE_KEEP:
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "rewrite_intent_keep"
        _append_bullet_detail(audit_log, detail)
        return

    target_keywords = action.get("target_keywords") if isinstance(action.get("target_keywords"), list) else []
    detail["target_keywords"] = list(target_keywords)
    if not target_keywords:
        audit_log["kept_bullets"].append(bullet_id)
        detail["skip_reason"] = "empty_target_keywords"
        _append_bullet_detail(audit_log, detail)
        return

    budget = bullet_budgets.get(bullet_id)

    rewritten_text, rejected, disallowed_terms = _call_bullet_rewrite(
        bullet_id,
        original_text,
        job_json,
        target_keywords,
        allowed_vocab,
        provider,
        budget,
    )
    if rejected:
        audit_log["rejected_for_new_terms"].append(bullet_id)
        detail["reject_reason"] = "disallowed_terms"
        if disallowed_terms:
            detail["disallowed_terms"] = disallowed_terms
        _append_bullet_detail(audit_log, detail)
    if rewritten_text is None:
        if not rejected:
            audit_log["kept_bullets"].append(bullet_id)
            detail["skip_reason"] = "llm_no_rewrite_or_rejected"
            _append_bullet_detail(audit_log, detail)
        return

    provider_len = len(rewritten_text)
    did_shorten = False
    detail["candidate_text"] = rewritten_text
    compressed = _compress_text(original_text, rewritten_text, budget, provider)
    if compressed is not None:
        if len(compressed) < len(rewritten_text):
            did_shorten = True
        rewritten_text = compressed

        disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
        if disallowed_after:
            audit_log["rejected_for_new_terms"].append(bullet_id)
            detail["reject_reason"] = "disallowed_terms_after_compress"
            detail["disallowed_terms"] = disallowed_after
            _append_bullet_detail(audit_log, detail)
            return

    if len(rewritten_text) > budget:
        truncated = _truncate_to_budget(rewritten_text, budget)
        if len(truncated) < len(rewritten_text):
            did_shorten = True
        rewritten_text = truncated

    if did_shorten:
        _append_unique(audit_log["compressed"], bullet_id)
    bullet["text"] = rewritten_text
    audit_log["rewritten_bullets"].append(bullet_id)
    detail["final_text"] = rewritten_text
    detail["changed"] = rewritten_text != original_text
    _append_bullet_detail(audit_log, detail)

def _call_bullet_rewrite(
    bullet_id: str,
    original_text: str,
    job_json: Dict[str, Any],
    target_keywords: Sequence[str],
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
) -> Tuple[Optional[str], bool, Optional[List[str]]]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    allowed_target, missing_target = _split_target_keywords(target_keywords, allowed_terms)

    payload = {
        "bullet_id": bullet_id,
        "original_text": original_text,
        "job_title": job_json.get("title"),
        "target_keywords": list(target_keywords),
        "allowed_target_keywords": allowed_target,
        "missing_target_keywords": missing_target,
        "allowed_terms": _select_allowed_terms(allowed_terms, original_text, target_keywords),
        "allowed_proper_nouns": _select_allowed_proper_nouns(allowed_proper, original_text),
    }
    if budget is not None:
        payload["max_chars"] = budget

    system_prompt = load_system_prompt("bullet_rewrite")
    messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="bullet_rewrite")
    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    obj = _parse_llm_json(raw, "bullet_rewrite", provider)
    if obj is None:
        return None, False, None

    if obj.get("bullet_id") != bullet_id:
        return None, False, None
    rewritten_text = obj.get("rewritten_text")
    if not isinstance(rewritten_text, str):
        return None, False, None

    disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
    if disallowed:
        payload["disallowed_terms"] = disallowed
        payload["retry_instruction"] = "Remove the disallowed terms. Do not replace them with new terms."
        raw_retry = provider.generate(
            build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="bullet_rewrite"),
            timeout=config.llm_timeout_seconds,
        )
        obj_retry = _parse_llm_json(raw_retry, "bullet_rewrite", provider)
        if obj_retry and obj_retry.get("bullet_id") == bullet_id and isinstance(obj_retry.get("rewritten_text"), str):
            rewritten_text = obj_retry["rewritten_text"]
            disallowed = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
        if disallowed:
            return None, True, disallowed

    return rewritten_text, False, None


def _compress_text(
    original_text: str,
    candidate_text: str,
    max_chars: int,
    provider: LLMProvider,
) -> Optional[str]:
    payload = {
        "original_text": original_text,
        "candidate_text": candidate_text,
        "max_chars": max_chars,
    }
    system_prompt = load_system_prompt("compress_text")
    messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="compress_text")
    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    obj = _parse_llm_json(raw, "compress_text", provider)
    if obj is None:
        return None
    compressed_text = obj.get("compressed_text")
    if not isinstance(compressed_text, str):
        return None
    return compressed_text


def _parse_llm_json(raw: str, schema_name: str, provider: LLMProvider) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors = [f"JSON parse error: {exc.msg} at pos {exc.pos}"]
        try:
            obj = repair_json_to_schema(raw, schema_name, errors, provider)
        except Exception:
            return None
    ok, errors = validate_json(schema_name, obj)
    if not ok:
        return None
    return obj


def _select_allowed_terms(allowed_terms: Iterable[str], original_text: str, target_keywords: Sequence[str]) -> List[str]:
    tokens = tokenize(original_text)
    phrases = generate_ngrams(tokens, 3)
    bullet_terms = set(tokens) | phrases
    keyword_terms = set(normalize_terms(target_keywords))
    candidates = [term for term in allowed_terms if term in bullet_terms or term in keyword_terms]
    candidates = sorted(set(candidates))
    return candidates[:50]


def _select_allowed_proper_nouns(allowed_proper: Iterable[str], original_text: str) -> List[str]:
    normalized = normalize_text(original_text)
    candidates = [term for term in allowed_proper if term and term in normalized]
    candidates = sorted(set(candidates))
    return candidates[:30]


def _split_target_keywords(target_keywords: Sequence[str], allowed_terms: Iterable[str]) -> Tuple[List[str], List[str]]:
    allowed_set = set(allowed_terms)
    allowed: List[str] = []
    missing: List[str] = []
    for keyword in target_keywords:
        if not isinstance(keyword, str):
            continue
        normalized = normalize_text(keyword)
        if not normalized:
            continue
        if normalized in allowed_set:
            allowed.append(keyword)
        else:
            missing.append(keyword)
    return allowed, missing


def _find_disallowed_terms(
    rewritten_text: str,
    allowed_vocab: Dict[str, Any],
    target_keywords: Sequence[str],
) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())

    disallowed = set()
    normalized_text = normalize_text(rewritten_text)

    for keyword in target_keywords:
        if not isinstance(keyword, str):
            continue
        kw_norm = normalize_text(keyword)
        if not kw_norm or kw_norm in allowed_terms or kw_norm in allowed_proper:
            continue
        if _contains_phrase(normalized_text, kw_norm):
            disallowed.add(kw_norm)

    for term in _extract_tool_like_terms(rewritten_text):
        if term not in allowed_terms and term not in allowed_proper:
            disallowed.add(term)

    for term in _extract_proper_noun_candidates(rewritten_text):
        if term not in allowed_proper and term not in allowed_terms:
            disallowed.add(term)

    return sorted(disallowed)


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    if not phrase:
        return False
    haystack = f" {normalized_text} "
    needle = f" {phrase} "
    if " " in phrase:
        return needle in haystack
    return phrase in normalized_text.split()


def _extract_tool_like_terms(text: str) -> List[str]:
    terms: List[str] = []
    for raw in _RAW_TOKEN_PATTERN.findall(text):
        cleaned = _strip_trailing_periods(raw)
        if not cleaned:
            continue
        if _is_tool_like_raw(cleaned):
            normalized = _normalize_tool_token(cleaned)
            if normalized:
                terms.append(normalized)
    return terms


def _is_tool_like_raw(raw: str) -> bool:
    if any(ch.isdigit() for ch in raw):
        return True
    if any(ch in "#+." for ch in raw):
        return True
    if raw.isupper() and len(raw) >= 2:
        return True
    if any(ch.isupper() for ch in raw[1:]):
        return True
    return False


def _normalize_tool_token(raw: str) -> str:
    tokens = tokenize(raw)
    if tokens:
        return tokens[0]
    return normalize_text(raw)


def _extract_proper_noun_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    tokens = _RAW_TOKEN_PATTERN.findall(text)
    for idx, raw in enumerate(tokens):
        if idx == 0 and raw[:1].isupper() and raw[1:].islower():
            continue
        if any(ch.isupper() for ch in raw):
            candidates.append(_normalize_proper_token(_strip_trailing_periods(raw)))
    return candidates


def _normalize_proper_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip()


def _strip_trailing_periods(token: str) -> str:
    if not token:
        return token
    return token.rstrip(_TRAILING_PERIODS)


def _truncate_to_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space].rstrip()
    return truncated


def _is_effectively_same(original_text: str, candidate_text: Optional[str]) -> bool:
    if not candidate_text:
        return True
    return normalize_text(original_text) == normalize_text(candidate_text)


def _needs_summary_fallback(
    original_text: str,
    candidate_text: Optional[str],
    reject_reason: Optional[str],
) -> bool:
    if reject_reason:
        return True
    if not candidate_text or not candidate_text.strip():
        return True
    if _is_effectively_same(original_text, candidate_text):
        return True
    return False


def _fallback_reason(
    original_text: str,
    candidate_text: Optional[str],
    reject_reason: Optional[str],
) -> str:
    if reject_reason:
        return reject_reason
    if not candidate_text or not candidate_text.strip():
        return "empty_or_invalid"
    if _is_effectively_same(original_text, candidate_text):
        return "identical_to_original"
    return "unknown"


def _fallback_summary_rewrite(original_text: str) -> Optional[str]:
    stripped = original_text.strip()
    if not stripped:
        return None
    clauses, joiner, suffix = _split_summary_clauses(stripped)
    if len(clauses) >= 2:
        reordered = [clauses[-1]] + clauses[:-1]
        candidate = joiner.join(reordered).strip()
        if suffix:
            candidate = candidate.rstrip() + suffix
        if not _is_effectively_same(original_text, candidate):
            return candidate
    swapped = _swap_conjunction_phrases(stripped, " and ")
    if swapped and not _is_effectively_same(original_text, swapped):
        return swapped
    pivoted = _rephrase_by_pivot(
        stripped,
        [
            "focused on",
            "specializing in",
            "experienced in",
            "expertise in",
            "skilled in",
            "proficient in",
        ],
    )
    if pivoted and not _is_effectively_same(original_text, pivoted):
        return pivoted
    rotated = _rotate_summary_tokens(stripped)
    if rotated and not _is_effectively_same(original_text, rotated):
        return rotated
    return None


def _split_summary_clauses(text: str) -> Tuple[List[str], str, str]:
    core, suffix = _strip_summary_suffix(text)
    separators = ["; ", " | ", " / ", " - ", " — ", ": ", ", "]
    for sep in separators:
        if sep in core:
            parts = [part.strip() for part in core.split(sep) if part.strip()]
            if len(parts) >= 2:
                return parts, sep, suffix
    for sep in [";", ",", ":"]:
        if sep in core:
            parts = [part.strip() for part in core.split(sep) if part.strip()]
            if len(parts) >= 2:
                return parts, f"{sep} ", suffix
    return [core], " ", suffix


def _strip_summary_suffix(text: str) -> Tuple[str, str]:
    stripped = text.strip()
    if stripped and stripped[-1] in ".!?":
        return stripped[:-1].rstrip(), stripped[-1]
    return stripped, ""


def _swap_conjunction_phrases(text: str, conjunction: str) -> Optional[str]:
    lower = text.lower()
    conj_lower = conjunction.lower()
    idx = lower.find(conj_lower)
    if idx == -1:
        return None
    before = text[:idx].strip()
    after = text[idx + len(conjunction) :].strip()
    after_core, suffix = _strip_summary_suffix(after)
    if not before or not after_core:
        return None
    candidate = f"{after_core}{conjunction}{before}".strip()
    if suffix:
        candidate = candidate.rstrip() + suffix
    return candidate


def _rephrase_by_pivot(text: str, pivots: Sequence[str]) -> Optional[str]:
    lower = text.lower()
    for pivot in pivots:
        marker = f" {pivot} "
        idx = lower.find(marker)
        if idx == -1:
            continue
        before = text[:idx].strip()
        after = text[idx + len(marker) :].strip()
        after_core, suffix = _strip_summary_suffix(after)
        if not before or not after_core:
            continue
        candidate = f"{pivot.capitalize()} {after_core}, {before}".strip()
        if suffix:
            candidate = candidate.rstrip() + suffix
        return candidate
    return None


def _rotate_summary_tokens(text: str) -> Optional[str]:
    core, suffix = _strip_summary_suffix(text)
    tokens = [tok for tok in core.split() if tok]
    if len(tokens) < 2:
        return None
    rotated = [tokens[-1]] + tokens[:-1]
    if tokens[0][:1].isupper():
        rotated[0] = rotated[0][:1].upper() + rotated[0][1:]
    candidate = " ".join(rotated).strip()
    if suffix:
        candidate = candidate.rstrip() + suffix
    return candidate


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _append_bullet_detail(audit_log: Dict[str, Any], detail: Dict[str, Any]) -> None:
    details = audit_log.get("bullet_details")
    if isinstance(details, list):
        details.append(detail)


def _collect_candidate_keywords(
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    allowed_vocab: Dict[str, Any],
) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    candidates: List[str] = []

    prioritized = tailoring_plan.get("prioritized_keywords") if isinstance(tailoring_plan.get("prioritized_keywords"), list) else []
    for keyword in prioritized:
        normalized = normalize_text(keyword)
        if normalized and (normalized in allowed_terms or normalized in allowed_proper):
            _append_unique(candidates, normalized)

    for text in _iter_job_texts(job_json):
        for term in _extract_job_terms(text):
            if term in allowed_terms or term in allowed_proper:
                _append_unique(candidates, term)
        if len(candidates) >= 50:
            break
    return candidates


def _iter_job_texts(job_json: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    title = job_json.get("title")
    if isinstance(title, str):
        texts.append(title)
    keywords = job_json.get("keywords")
    if isinstance(keywords, list):
        texts.extend([kw for kw in keywords if isinstance(kw, str)])
    responsibilities = job_json.get("responsibilities")
    if isinstance(responsibilities, list):
        texts.extend([item for item in responsibilities if isinstance(item, str)])
    for field in ("must_have", "nice_to_have"):
        items = job_json.get(field)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
    return texts


def _extract_job_terms(text: str) -> List[str]:
    tokens = tokenize(text)
    ordered: List[str] = []
    for token in tokens:
        _append_unique(ordered, token)
    for phrase in _ordered_ngrams(tokens, 3):
        _append_unique(ordered, phrase)
    return ordered


def _ordered_ngrams(tokens: List[str], max_n: int) -> List[str]:
    ordered: List[str] = []
    if not tokens or max_n < 2:
        return ordered
    n_max = min(max_n, len(tokens))
    for n in range(2, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            ordered.append(" ".join(tokens[i : i + n]))
    return ordered


def _split_skills_line(text: str) -> Tuple[List[str], str]:
    if " | " in text:
        parts = [part.strip() for part in text.split(" | ")]
        return parts, " | "
    if " / " in text:
        parts = [part.strip() for part in text.split(" / ")]
        return parts, " / "
    if ";" in text:
        parts = [part.strip() for part in text.split(";") if part.strip() != ""]
        return parts, "; "
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip() != ""]
        return parts, ", "
    return [text.strip()], ", "


def _iter_resume_texts(resume_json: Dict[str, Any]) -> Iterable[str]:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    if isinstance(summary.get("text"), str):
        yield summary.get("text")

    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if isinstance(line, dict) and isinstance(line.get("text"), str):
            yield line.get("text")

    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        for key in ("company", "title"):
            if isinstance(exp.get(key), str):
                yield exp.get(key)
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                yield bullet.get("text")

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        for key in ("name", "text"):
            if isinstance(proj.get(key), str):
                yield proj.get(key)
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in bullets:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                yield bullet.get("text")

    education = resume_json.get("education") if isinstance(resume_json.get("education"), list) else []
    for edu in education:
        if not isinstance(edu, dict):
            continue
        for key in ("school", "degree"):
            if isinstance(edu.get(key), str):
                yield edu.get(key)


def _extract_tool_like_terms_with_display(text: str) -> List[Tuple[str, str]]:
    terms: List[Tuple[str, str]] = []
    for raw in _RAW_TOKEN_PATTERN.findall(text):
        cleaned = _strip_trailing_periods(raw)
        if not cleaned:
            continue
        if _is_tool_like_raw(cleaned):
            normalized = _normalize_tool_token(cleaned)
            if normalized:
                terms.append((normalized, cleaned))
    return terms


def _find_display_in_texts(term: str, resume_texts: Sequence[str]) -> Optional[str]:
    if not term:
        return None
    has_special = any(ch in "#+." for ch in term)
    if has_special:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
    else:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for text in resume_texts:
        if not isinstance(text, str) or not text:
            continue
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _segment_has_tool_like_token(segment: str) -> bool:
    for raw in _RAW_TOKEN_PATTERN.findall(segment):
        cleaned = _strip_trailing_periods(raw)
        if cleaned and _is_tool_like_raw(cleaned):
            return True
    return False


def _term_is_tool_like(term: str) -> bool:
    if not term:
        return False
    return any(ch.isdigit() for ch in term) or any(ch in "#+." for ch in term)


def _segment_has_technical_indicator(segment: str) -> bool:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    return any(indicator in terms for indicator in _TECHNICAL_INDICATORS)


def _term_has_technical_indicator(term: str) -> bool:
    tokens = tokenize(term)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    return any(indicator in terms for indicator in _TECHNICAL_INDICATORS)


def _build_resume_hard_skill_inventory(
    resume_json: Dict[str, Any],
    candidate_keywords: Sequence[str],
    resume_texts: Sequence[str],
    soft_skill_terms: Set[str],
    resume_normalized: str,
) -> Tuple[Set[str], Dict[str, str]]:
    hard_terms: Set[str] = set()
    display_map: Dict[str, str] = {}

    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if not isinstance(text, str):
            continue
        segments, _ = _split_skills_line(text)
        for segment in segments:
            normalized = normalize_text(segment)
            if not normalized or normalized in soft_skill_terms:
                continue
            if _segment_has_tool_like_token(segment) or _segment_has_technical_indicator(segment):
                for term in _segment_term_set(segment) | {normalized}:
                    if term in soft_skill_terms:
                        continue
                    if term not in hard_terms:
                        hard_terms.add(term)
                        display_map[term] = segment

    for text in resume_texts:
        if not isinstance(text, str):
            continue
        for normalized, display in _extract_tool_like_terms_with_display(text):
            if normalized not in hard_terms:
                hard_terms.add(normalized)
                display_map[normalized] = display

    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized in soft_skill_terms:
            continue
        if normalized in hard_terms:
            continue
        if normalized and resume_normalized and normalized not in resume_normalized:
            continue
        display = _find_display_in_texts(normalized, resume_texts) or keyword
        hard_terms.add(normalized)
        display_map.setdefault(normalized, display)

    return hard_terms, display_map


def _collect_existing_skill_terms(lines: List[Any]) -> Set[str]:
    existing: Set[str] = set()
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if not isinstance(text, str):
            continue
        segments, _ = _split_skills_line(text)
        for segment in segments:
            normalized = normalize_text(segment)
            if normalized:
                existing.add(normalized)
            existing.update(_segment_term_set(segment))
    return existing


def _collect_matched_hard_terms(candidate_keywords: Sequence[str], hard_skill_terms: Set[str]) -> List[str]:
    matched: List[str] = []
    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized:
            continue
        if normalized in hard_skill_terms and normalized not in matched:
            matched.append(normalized)
    return matched


def _collect_matched_soft_terms(
    candidate_keywords: Sequence[str],
    soft_skill_terms: Set[str],
    resume_normalized: str,
) -> List[str]:
    matched: List[str] = []
    for keyword in candidate_keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized not in soft_skill_terms:
            continue
        if resume_normalized and normalized not in resume_normalized:
            continue
        if normalized not in matched:
            matched.append(normalized)
    return matched


def _classify_skill_segment(segment: str, hard_skill_terms: Set[str], soft_skill_terms: Set[str]) -> str:
    normalized = normalize_text(segment)
    if normalized in soft_skill_terms:
        return "soft"
    if normalized in hard_skill_terms:
        return "hard"
    if _segment_has_tool_like_token(segment) or _segment_has_technical_indicator(segment):
        return "hard"
    return "unknown"


def _segment_term_set(segment: str) -> Set[str]:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    return set(tokens) | phrases


def _reorder_skill_segments(
    segments: List[str],
    classifications: List[str],
    matched_hard_terms: Sequence[str],
    matched_soft_terms: Sequence[str],
) -> List[str]:
    matched_hard_set = set(matched_hard_terms)
    matched_soft_set = set(matched_soft_terms)
    buckets: Dict[str, List[Tuple[int, str]]] = {
        "hard_matched": [],
        "hard": [],
        "soft_matched": [],
        "soft": [],
        "unknown": [],
    }
    for idx, (segment, classification) in enumerate(zip(segments, classifications)):
        terms = _segment_term_set(segment)
        if classification == "hard":
            if matched_hard_set and terms.intersection(matched_hard_set):
                buckets["hard_matched"].append((idx, segment))
            else:
                buckets["hard"].append((idx, segment))
        elif classification == "soft":
            if matched_soft_set and terms.intersection(matched_soft_set):
                buckets["soft_matched"].append((idx, segment))
            else:
                buckets["soft"].append((idx, segment))
        else:
            buckets["unknown"].append((idx, segment))

    ordered: List[str] = []
    for key in ("hard_matched", "hard", "soft_matched", "soft", "unknown"):
        ordered.extend([segment for _, segment in sorted(buckets[key], key=lambda item: item[0])])
    return ordered


def _surface_missing_hard_skills(
    line_states: List[Dict[str, Any]],
    missing_hard_terms: Sequence[str],
    hard_skill_display: Dict[str, str],
    soft_skill_terms: Set[str],
    audit_log: Dict[str, Any],
) -> None:
    if not line_states:
        return

    for term in missing_hard_terms:
        display = hard_skill_display.get(term)
        if not display:
            continue
        inserted = False
        for state in line_states:
            if inserted:
                break
            segments = state["segments"]
            separator = state["separator"]
            budget = state["budget"]
            current_text = separator.join(segments)
            current_len = len(current_text)
            normalized_segments = {normalize_text(seg) for seg in segments if normalize_text(seg)}
            if term in normalized_segments:
                continue

            classifications = state["classifications"]
            insert_index = _preferred_hard_insert_index(classifications)
            candidate_segments = list(segments)
            candidate_segments.insert(insert_index, display)
            candidate_len = len(separator.join(candidate_segments))
            if candidate_len <= budget:
                segments.insert(insert_index, display)
                classifications.insert(insert_index, "hard")
                state["segments"] = segments
                state["classifications"] = classifications
                state["line"]["text"] = separator.join(segments)
                _update_skills_detail(audit_log, state["line_id"], state["line"]["text"], "hard_skill_surface")
                inserted = True
                continue

            replacement_index = _find_replacement_index(
                segments,
                classifications,
                display,
                budget,
                separator,
                soft_skill_terms,
            )
            if replacement_index is not None:
                segments[replacement_index] = display
                classifications[replacement_index] = "hard"
                state["segments"] = segments
                state["classifications"] = classifications
                state["line"]["text"] = separator.join(segments)
                _update_skills_detail(audit_log, state["line_id"], state["line"]["text"], "hard_skill_surface")
                inserted = True


def _preferred_hard_insert_index(classifications: Sequence[str]) -> int:
    last_hard_index = -1
    for idx, classification in enumerate(classifications):
        if classification == "hard":
            last_hard_index = idx
    return last_hard_index + 1 if last_hard_index >= 0 else 0


def _find_replacement_index(
    segments: Sequence[str],
    classifications: Sequence[str],
    replacement: str,
    budget: int,
    separator: str,
    soft_skill_terms: Set[str],
) -> Optional[int]:
    candidates: List[int] = []
    for idx, classification in enumerate(classifications):
        if classification == "hard":
            continue
        candidates.append(idx)

    def candidate_rank(index: int) -> Tuple[int, int]:
        normalized = normalize_text(segments[index])
        is_soft = normalized in soft_skill_terms
        classification = classifications[index]
        if classification == "soft" and is_soft:
            return (0, index)
        if classification == "unknown":
            return (1, index)
        return (2, index)

    for idx in sorted(candidates, key=candidate_rank):
        candidate_segments = list(segments)
        candidate_segments[idx] = replacement
        if len(separator.join(candidate_segments)) <= budget:
            return idx
    return None


def _update_skills_detail(
    audit_log: Dict[str, Any],
    line_id: str,
    final_text: str,
    reason: str,
) -> None:
    details = audit_log.get("skills_details")
    if not isinstance(details, list):
        return
    for detail in reversed(details):
        if isinstance(detail, dict) and detail.get("line_id") == line_id:
            detail["final_text"] = final_text
            detail["changed"] = True
            detail["skip_reason"] = reason
            return


def _score_skill_segments(segments: List[str], candidates: List[str]) -> List[Tuple[str, int, int]]:
    scored: List[Tuple[str, int, int]] = []
    for idx, segment in enumerate(segments):
        score = _count_segment_overlap(segment, candidates)
        scored.append((segment, score, idx))
    return scored


def _reorder_segments(scored: List[Tuple[str, int, int]]) -> List[str]:
    ordered = sorted(scored, key=lambda item: (-item[1], item[2]))
    return [item[0] for item in ordered]


def _count_segment_overlap(segment: str, candidates: List[str]) -> int:
    tokens = tokenize(segment)
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    score = 0
    for keyword in candidates:
        if keyword in terms:
            score += 1
    return score


def _extract_matching_keywords(segments: List[str], candidates: List[str]) -> List[str]:
    matched: List[str] = []
    for segment in segments:
        tokens = tokenize(segment)
        phrases = generate_ngrams(tokens, 3)
        terms = set(tokens) | phrases
        for keyword in candidates:
            if keyword in terms:
                _append_unique(matched, keyword)
    return matched


def _build_action_map(tailoring_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    actions = tailoring_plan.get("bullet_actions") if isinstance(tailoring_plan.get("bullet_actions"), list) else []
    action_map: Dict[str, Dict[str, Any]] = {}
    for item in actions:
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if not isinstance(bullet_id, str) or bullet_id.strip() == "":
            continue
        if bullet_id not in action_map:
            action_map[bullet_id] = item
    return action_map


def _sorted_bullets(bullets: List[Any]) -> List[Dict[str, Any]]:
    indexed: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, bullet in enumerate(bullets):
        if not isinstance(bullet, dict):
            continue
        bi = bullet.get("bullet_index")
        bullet_index = bi if isinstance(bi, int) else idx
        indexed.append((bullet_index, idx, bullet))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def _check_invariants(original: Dict[str, Any], tailored: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    _compare_summary(original, tailored, errors)
    _compare_skills(original, tailored, errors)
    _compare_experience(original, tailored, errors)
    _compare_projects(original, tailored, errors)

    return errors


def _compare_summary(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_summary = original.get("summary") if isinstance(original.get("summary"), dict) else {}
    tailored_summary = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else {}
    if original_summary.get("id") != tailored_summary.get("id"):
        errors.append("summary.id changed")


def _compare_skills(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_skills = original.get("skills") if isinstance(original.get("skills"), dict) else {}
    tailored_skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else {}
    original_lines = original_skills.get("lines") if isinstance(original_skills.get("lines"), list) else []
    tailored_lines = tailored_skills.get("lines") if isinstance(tailored_skills.get("lines"), list) else []
    if len(original_lines) != len(tailored_lines):
        errors.append("skills.lines count changed")
        return
    original_ids = [line.get("line_id") for line in original_lines if isinstance(line, dict)]
    tailored_ids = [line.get("line_id") for line in tailored_lines if isinstance(line, dict)]
    if sorted(original_ids) != sorted(tailored_ids):
        errors.append("skills.lines ids changed")


def _compare_experience(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_exps = original.get("experience") if isinstance(original.get("experience"), list) else []
    tailored_exps = tailored.get("experience") if isinstance(tailored.get("experience"), list) else []
    if len(original_exps) != len(tailored_exps):
        errors.append("experience count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_exps, tailored_exps)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("exp_id") != new.get("exp_id"):
            errors.append(f"experience[{idx}].exp_id changed")
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"experience[{idx}].bullets")


def _compare_projects(original: Dict[str, Any], tailored: Dict[str, Any], errors: List[str]) -> None:
    original_projects = original.get("projects") if isinstance(original.get("projects"), list) else []
    tailored_projects = tailored.get("projects") if isinstance(tailored.get("projects"), list) else []
    if len(original_projects) != len(tailored_projects):
        errors.append("projects count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_projects, tailored_projects)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("project_id") != new.get("project_id"):
            errors.append(f"projects[{idx}].project_id changed")
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"projects[{idx}].bullets")


def _compare_bullets(original_bullets: Any, tailored_bullets: Any, errors: List[str], prefix: str) -> None:
    original_list = original_bullets if isinstance(original_bullets, list) else []
    tailored_list = tailored_bullets if isinstance(tailored_bullets, list) else []
    if len(original_list) != len(tailored_list):
        errors.append(f"{prefix} count changed")
        return
    for idx, (orig, new) in enumerate(zip(original_list, tailored_list)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        if orig.get("bullet_id") != new.get("bullet_id"):
            errors.append(f"{prefix}[{idx}].bullet_id changed")
        if orig.get("bullet_index") != new.get("bullet_index"):
            errors.append(f"{prefix}[{idx}].bullet_index changed")
