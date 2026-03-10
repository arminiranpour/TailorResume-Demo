from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config import get_config
from app.pipelines.allowed_vocab import build_allowed_vocab
from app.pipelines.integrity import check_structure_invariants
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages
from shared.scoring.normalize import normalize_text, tokenize


@dataclass
class BudgetEnforcementError(Exception):
    details: List[str]
    raw_preview: str = ""


_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9#.+-]*")
_TRAILING_PERIODS = "."


def compute_baseline_budgets(original_resume_json: Dict[str, Any]) -> Dict[str, Any]:
    summary_text = _get_summary_text(original_resume_json)
    summary_len = len(summary_text)

    skills_line_max_chars: Dict[str, int] = {}
    for line_id, text in _iter_skill_lines(original_resume_json):
        skills_line_max_chars[line_id] = len(text)

    bullet_max_chars: Dict[str, int] = {}
    for bullet_id, text in _iter_bullet_texts(original_resume_json):
        if bullet_id not in bullet_max_chars:
            bullet_max_chars[bullet_id] = len(text)

    total_max_chars = summary_len
    for value in skills_line_max_chars.values():
        total_max_chars += value
    for value in bullet_max_chars.values():
        total_max_chars += value

    return {
        "summary_max_chars": summary_len,
        "skills_line_max_chars": skills_line_max_chars,
        "bullet_max_chars": bullet_max_chars,
        "total_max_chars": total_max_chars,
    }


def measure_resume_size(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    summary_text = _get_summary_text(resume_json)
    summary_len = len(summary_text)

    bullets_len_by_id: Dict[str, int] = {}
    bullets_total_len = 0
    for bullet_id, text in _iter_bullet_texts(resume_json):
        if bullet_id not in bullets_len_by_id:
            length = len(text)
            bullets_len_by_id[bullet_id] = length
            bullets_total_len += length

    skills_len_by_id: Dict[str, int] = {}
    skills_total_len = 0
    for line_id, text in _iter_skill_lines(resume_json):
        if line_id not in skills_len_by_id:
            length = len(text)
            skills_len_by_id[line_id] = length
            skills_total_len += length

    total_len = summary_len + bullets_total_len + skills_total_len

    longest_bullets = sorted(
        [{"bullet_id": bid, "length": length} for bid, length in bullets_len_by_id.items()],
        key=lambda item: (-item["length"], item["bullet_id"]),
    )

    return {
        "summary_len": summary_len,
        "bullets_total_len": bullets_total_len,
        "bullets_len_by_id": bullets_len_by_id,
        "skills_total_len": skills_total_len,
        "skills_len_by_id": skills_len_by_id,
        "total_len": total_len,
        "longest_bullets": longest_bullets,
    }


def enforce_budgets(
    original_resume_json: Dict[str, Any],
    tailored_resume_json: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Optional[Dict[str, Any]],
    budgets_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[str]]]:
    if not isinstance(original_resume_json, dict) or not isinstance(tailored_resume_json, dict):
        raise ValueError("original_resume_json and tailored_resume_json must be objects")

    _validate_schema_or_raise("resume", original_resume_json)
    _validate_schema_or_raise("resume", tailored_resume_json)

    if not isinstance(allowed_vocab, dict):
        allowed_vocab = build_allowed_vocab(original_resume_json)

    baseline_budgets = compute_baseline_budgets(original_resume_json)
    effective_budgets, overrides_used = _apply_budget_overrides(baseline_budgets, budgets_override)

    initial_size = measure_resume_size(tailored_resume_json)
    overflow_before = _compute_overflow(initial_size, effective_budgets)

    final_resume = _clone_resume(tailored_resume_json)
    audit_additions = {"compressed_fields": [], "truncated_fields": [], "fallbacks": []}

    _restore_skill_line_texts(original_resume_json, final_resume, audit_additions)

    summary_original = _get_summary_text(original_resume_json)
    summary_rewritten = summary_original != _get_summary_text(final_resume)

    summary_budget = effective_budgets.get("summary_max_chars")
    if isinstance(summary_budget, int):
        _enforce_summary_budget(
            final_resume,
            summary_original,
            summary_budget,
            provider,
            allowed_vocab,
            audit_additions,
            allow_llm=summary_rewritten,
        )

    bullet_original_map = _build_bullet_text_map(original_resume_json)
    bullet_budget_map = effective_budgets.get("bullet_max_chars", {})
    _enforce_bullet_budgets(
        final_resume,
        bullet_original_map,
        bullet_budget_map,
        provider,
        allowed_vocab,
        audit_additions,
    )

    _enforce_global_budget(
        final_resume,
        summary_original,
        bullet_original_map,
        effective_budgets,
        provider,
        allowed_vocab,
        audit_additions,
    )

    final_size = measure_resume_size(final_resume)
    overflow_after = _compute_overflow(final_size, effective_budgets)

    overflow_errors: List[str] = []
    if overflow_after.get("summary_over", 0) > 0:
        overflow_errors.append("summary exceeds budget")
    for bullet_id, over in overflow_after.get("bullets_over", {}).items():
        if over > 0:
            overflow_errors.append(f"bullet {bullet_id} exceeds budget")
    for line_id, over in overflow_after.get("skills_lines_over", {}).items():
        if over > 0:
            overflow_errors.append(f"skills line {line_id} exceeds budget")
    if overflow_after.get("total_over", 0) > 0:
        overflow_errors.append("total length exceeds budget")

    ok, errors = validate_json("resume", final_resume)
    if not ok:
        raise BudgetEnforcementError(details=errors, raw_preview="")

    invariant_errors = check_structure_invariants(original_resume_json, final_resume)
    if invariant_errors:
        raise BudgetEnforcementError(details=invariant_errors, raw_preview="")

    if overflow_errors:
        raise BudgetEnforcementError(details=overflow_errors, raw_preview="")

    budget_report = {
        "budgets": {
            "baseline": baseline_budgets,
            "overrides": overrides_used,
            "effective": effective_budgets,
        },
        "size_report": {
            "initial": initial_size,
            "final": final_size,
            "overflow_before": overflow_before,
            "overflow_after": overflow_after,
        },
    }

    return final_resume, budget_report, audit_additions


def compress_to_budget(
    original_text: str,
    candidate_text: str,
    max_chars: int,
    provider: LLMProvider,
) -> Optional[str]:
    if max_chars <= 0:
        return ""
    if len(candidate_text) <= max_chars:
        return candidate_text

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
    if len(compressed_text) > max_chars:
        compressed_text = _truncate_to_budget(compressed_text, max_chars)
    return compressed_text


def _apply_budget_overrides(
    baseline: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    effective = json.loads(json.dumps(baseline))
    overrides_used: Dict[str, Any] = {}
    if not isinstance(overrides, dict):
        return effective, overrides_used

    summary_value = overrides.get("summary")
    if isinstance(summary_value, int) and summary_value >= 0:
        effective["summary_max_chars"] = summary_value
        overrides_used["summary_max_chars"] = summary_value

    summary_value = overrides.get("summary_max_chars")
    if isinstance(summary_value, int) and summary_value >= 0:
        effective["summary_max_chars"] = summary_value
        overrides_used["summary_max_chars"] = summary_value

    total_value = overrides.get("total")
    if isinstance(total_value, int) and total_value >= 0:
        effective["total_max_chars"] = total_value
        overrides_used["total_max_chars"] = total_value

    total_value = overrides.get("total_max_chars")
    if isinstance(total_value, int) and total_value >= 0:
        effective["total_max_chars"] = total_value
        overrides_used["total_max_chars"] = total_value

    bullet_overrides = overrides.get("bullets")
    if isinstance(bullet_overrides, dict):
        for bullet_id, value in bullet_overrides.items():
            if isinstance(bullet_id, str) and isinstance(value, int) and value >= 0:
                effective["bullet_max_chars"][bullet_id] = value
                overrides_used.setdefault("bullet_max_chars", {})[bullet_id] = value

    bullet_overrides = overrides.get("bullet_max_chars")
    if isinstance(bullet_overrides, dict):
        for bullet_id, value in bullet_overrides.items():
            if isinstance(bullet_id, str) and isinstance(value, int) and value >= 0:
                effective["bullet_max_chars"][bullet_id] = value
                overrides_used.setdefault("bullet_max_chars", {})[bullet_id] = value

    skills_overrides = overrides.get("skills_lines")
    if isinstance(skills_overrides, dict):
        for line_id, value in skills_overrides.items():
            if isinstance(line_id, str) and isinstance(value, int) and value >= 0:
                effective["skills_line_max_chars"][line_id] = value
                overrides_used.setdefault("skills_line_max_chars", {})[line_id] = value

    skills_overrides = overrides.get("skills_line_max_chars")
    if isinstance(skills_overrides, dict):
        for line_id, value in skills_overrides.items():
            if isinstance(line_id, str) and isinstance(value, int) and value >= 0:
                effective["skills_line_max_chars"][line_id] = value
                overrides_used.setdefault("skills_line_max_chars", {})[line_id] = value

    if effective.get("total_max_chars", 0) > baseline.get("total_max_chars", 0):
        effective["total_max_chars"] = baseline.get("total_max_chars", 0)

    if effective.get("summary_max_chars", 0) < 0:
        effective["summary_max_chars"] = 0

    if effective.get("total_max_chars", 0) < 0:
        effective["total_max_chars"] = 0

    return effective, overrides_used


def _compute_overflow(size: Dict[str, Any], budgets: Dict[str, Any]) -> Dict[str, Any]:
    summary_budget = budgets.get("summary_max_chars", 0)
    summary_len = size.get("summary_len", 0)
    summary_over = max(0, summary_len - summary_budget) if isinstance(summary_budget, int) else 0

    bullet_over: Dict[str, int] = {}
    bullet_budgets = budgets.get("bullet_max_chars", {})
    for bullet_id, length in size.get("bullets_len_by_id", {}).items():
        budget = bullet_budgets.get(bullet_id)
        if isinstance(budget, int) and length > budget:
            bullet_over[bullet_id] = length - budget

    skills_over: Dict[str, int] = {}
    skills_budgets = budgets.get("skills_line_max_chars", {})
    if isinstance(skills_budgets, dict):
        for line_id, length in size.get("skills_len_by_id", {}).items():
            budget = skills_budgets.get(line_id)
            if isinstance(budget, int) and length > budget:
                skills_over[line_id] = length - budget

    total_budget = budgets.get("total_max_chars", 0)
    total_len = size.get("total_len", 0)
    total_over = max(0, total_len - total_budget) if isinstance(total_budget, int) else 0

    return {
        "summary_over": summary_over,
        "bullets_over": _sorted_int_dict(bullet_over),
        "skills_lines_over": _sorted_int_dict(skills_over),
        "total_over": total_over,
    }


def _restore_skill_line_texts(
    original_resume_json: Dict[str, Any],
    final_resume: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
) -> None:
    original_map: Dict[str, str] = {}
    for line_id, text in _iter_skill_lines(original_resume_json):
        if line_id not in original_map:
            original_map[line_id] = text

    skills = final_resume.get("skills") if isinstance(final_resume.get("skills"), dict) else None
    if skills is None:
        return
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if not isinstance(line_id, str):
            continue
        original_text = original_map.get(line_id)
        if isinstance(original_text, str) and line.get("text") != original_text:
            line["text"] = original_text
            _append_unique(audit_additions["fallbacks"], line_id)


def _enforce_summary_budget(
    final_resume: Dict[str, Any],
    original_text: str,
    max_chars: int,
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
    allow_llm: bool,
) -> None:
    summary = final_resume.get("summary") if isinstance(final_resume.get("summary"), dict) else None
    if summary is None:
        return
    candidate = summary.get("text")
    if not isinstance(candidate, str):
        return
    new_text = _compress_or_truncate(
        "summary",
        original_text,
        candidate,
        max_chars,
        provider,
        allowed_vocab,
        audit_additions,
        allow_llm=allow_llm,
    )
    summary["text"] = new_text


def _enforce_bullet_budgets(
    final_resume: Dict[str, Any],
    original_map: Dict[str, str],
    budgets: Dict[str, int],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
) -> None:
    for bullet_id, bullet in _iter_bullet_nodes(final_resume):
        if bullet_id not in original_map:
            continue
        budget = budgets.get(bullet_id)
        if not isinstance(budget, int):
            continue
        candidate = bullet.get("text")
        if not isinstance(candidate, str):
            continue
        new_text = _compress_or_truncate(
            bullet_id,
            original_map[bullet_id],
            candidate,
            budget,
            provider,
            allowed_vocab,
            audit_additions,
            allow_llm=True,
        )
        bullet["text"] = new_text


def _enforce_global_budget(
    final_resume: Dict[str, Any],
    summary_original: str,
    bullet_original_map: Dict[str, str],
    budgets: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
) -> None:
    total_budget = budgets.get("total_max_chars")
    if not isinstance(total_budget, int):
        return
    size = measure_resume_size(final_resume)
    overflow = size.get("total_len", 0) - total_budget
    if overflow <= 0:
        return

    step_size = 20
    summary_text = _get_summary_text(final_resume)
    if summary_text:
        reduction = min(overflow, step_size)
        new_max = max(len(summary_text) - reduction, 0)
        summary_rewritten = summary_text != summary_original
        new_text = _compress_or_truncate(
            "summary",
            summary_original,
            summary_text,
            new_max,
            provider,
            allowed_vocab,
            audit_additions,
            allow_llm=summary_rewritten,
        )
        summary_obj = final_resume.get("summary") if isinstance(final_resume.get("summary"), dict) else None
        if summary_obj is not None:
            summary_obj["text"] = new_text
        overflow -= max(0, len(summary_text) - len(new_text))

    if overflow <= 0:
        return

    bullet_order = _sorted_bullets_by_length(final_resume)
    overflow = _reduce_bullets_to_overflow(
        final_resume,
        bullet_original_map,
        bullet_order,
        overflow,
        step_size,
        provider,
        allowed_vocab,
        audit_additions,
    )

    if overflow <= 0:
        return

    step_size_second = 40
    while overflow > 0:
        previous = overflow
        overflow = _reduce_bullets_to_overflow(
            final_resume,
            bullet_original_map,
            bullet_order,
            overflow,
            step_size_second,
            provider,
            allowed_vocab,
            audit_additions,
        )
        if overflow == previous:
            break


def _reduce_bullets_to_overflow(
    final_resume: Dict[str, Any],
    bullet_original_map: Dict[str, str],
    bullet_order: List[str],
    overflow: int,
    step_size: int,
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
) -> int:
    if overflow <= 0:
        return overflow
    bullet_map = {bullet_id: bullet for bullet_id, bullet in _iter_bullet_nodes(final_resume)}
    for bullet_id in bullet_order:
        if overflow <= 0:
            break
        bullet = bullet_map.get(bullet_id)
        if bullet is None:
            continue
        candidate = bullet.get("text")
        if not isinstance(candidate, str):
            continue
        current_len = len(candidate)
        if current_len <= 0:
            continue
        reduction = min(overflow, step_size)
        new_max = max(current_len - reduction, 0)
        original_text = bullet_original_map.get(bullet_id, "")
        new_text = _compress_or_truncate(
            bullet_id,
            original_text,
            candidate,
            new_max,
            provider,
            allowed_vocab,
            audit_additions,
            allow_llm=True,
        )
        bullet["text"] = new_text
        overflow -= max(0, current_len - len(new_text))
    return overflow


def _compress_or_truncate(
    field_id: str,
    original_text: str,
    candidate_text: str,
    max_chars: int,
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    audit_additions: Dict[str, List[str]],
    allow_llm: bool,
) -> str:
    if max_chars <= 0:
        if candidate_text:
            _append_unique(audit_additions["truncated_fields"], field_id)
        return ""
    if len(candidate_text) <= max_chars:
        return candidate_text

    truncated = None
    used_llm = False
    truncated_used = False
    candidate_original = candidate_text
    candidate_len = len(candidate_text)
    if allow_llm:
        compressed = compress_to_budget(original_text, candidate_text, max_chars, provider)
        if compressed is not None:
            disallowed = _find_disallowed_terms(compressed, allowed_vocab)
            if disallowed:
                _append_unique(audit_additions["fallbacks"], field_id)
                truncated = _truncate_to_budget(candidate_original, max_chars)
                truncated_used = True
            else:
                used_llm = True
                candidate_text = compressed
        else:
            _append_unique(audit_additions["fallbacks"], field_id)
            truncated = _truncate_to_budget(candidate_original, max_chars)
            truncated_used = True
    else:
        truncated = _truncate_to_budget(candidate_text, max_chars)
        truncated_used = True

    if truncated is None:
        truncated = candidate_text

    if truncated_used and len(truncated) < candidate_len:
        _append_unique(audit_additions["truncated_fields"], field_id)
    if used_llm and len(truncated) < candidate_len:
        _append_unique(audit_additions["compressed_fields"], field_id)
    return truncated


def _sorted_bullets_by_length(resume_json: Dict[str, Any]) -> List[str]:
    items: List[Tuple[str, int]] = []
    for bullet_id, text in _iter_bullet_texts(resume_json):
        items.append((bullet_id, len(text)))
    items.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in items]


def _build_bullet_text_map(resume_json: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for bullet_id, text in _iter_bullet_texts(resume_json):
        if bullet_id not in mapping:
            mapping[bullet_id] = text
    return mapping


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


def _iter_bullet_nodes(resume_json: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if isinstance(bullet_id, str):
                yield bullet_id, bullet

    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if isinstance(bullet_id, str):
                yield bullet_id, bullet


def _iter_skill_lines(resume_json: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if isinstance(line_id, str) and isinstance(text, str):
            yield line_id, text


def _get_summary_text(resume_json: Dict[str, Any]) -> str:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    text = summary.get("text")
    if isinstance(text, str):
        return text
    return ""


def _clone_resume(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(resume_json))


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


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _sorted_int_dict(values: Dict[str, int]) -> Dict[str, int]:
    return {key: values[key] for key in sorted(values)}


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


def _validate_schema_or_raise(schema_name: str, obj: Dict[str, Any]) -> None:
    ok, errors = validate_json(schema_name, obj)
    if not ok:
        raise BudgetEnforcementError(details=errors, raw_preview="")


def _find_disallowed_terms(text: str, allowed_vocab: Dict[str, Any]) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())
    disallowed = set()

    for term in _extract_tool_like_terms(text):
        if term not in allowed_terms and term not in allowed_proper:
            disallowed.add(term)

    for term in _extract_proper_noun_candidates(text):
        if term not in allowed_proper:
            disallowed.add(term)

    return sorted(disallowed)


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
            candidates.append(_normalize_proper_token(raw))
    return candidates


def _normalize_proper_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip()


def _strip_trailing_periods(token: str) -> str:
    if not token:
        return token
    return token.rstrip(_TRAILING_PERIODS)
