from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
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
    }

    _rewrite_summary(tailored, job_json, tailoring_plan, provider, allowed_vocab, budgets)
    _reorder_skills(tailored, tailoring_plan)
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
) -> None:
    summary_plan = tailoring_plan.get("summary_rewrite")
    if not isinstance(summary_plan, dict):
        return
    rewrite_intent = summary_plan.get("rewrite_intent")
    if isinstance(rewrite_intent, str) and rewrite_intent.strip().lower() in _REWRITE_KEEP:
        return
    target_keywords = summary_plan.get("target_keywords") if isinstance(summary_plan.get("target_keywords"), list) else []
    if not target_keywords:
        return

    summary = tailored.get("summary") if isinstance(tailored.get("summary"), dict) else None
    if summary is None:
        return
    original_text = summary.get("text")
    if not isinstance(original_text, str) or original_text.strip() == "":
        return

    budget = budgets.get("summary")
    rewritten_text = _call_summary_rewrite(
        original_text,
        job_json,
        target_keywords,
        allowed_vocab,
        provider,
        budget,
    )
    if rewritten_text is None:
        return
    if budget is not None and len(rewritten_text) > budget:
        compressed = _compress_text(original_text, rewritten_text, budget, provider)
        if compressed is not None:
            rewritten_text = compressed
            disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
            if disallowed_after:
                return
        if budget is not None and len(rewritten_text) > budget:
            rewritten_text = _truncate_to_budget(rewritten_text, budget)
    summary["text"] = rewritten_text


def _call_summary_rewrite(
    original_text: str,
    job_json: Dict[str, Any],
    target_keywords: Sequence[str],
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
) -> Optional[str]:
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
    }
    if budget is not None:
        payload["max_chars"] = budget

    system_prompt = load_system_prompt("summary_rewrite")
    messages = build_llm_messages(system_prompt, json.dumps(payload, ensure_ascii=True), task_label="summary_rewrite")
    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    obj = _parse_llm_json(raw, "summary_rewrite", provider)
    if obj is None:
        return None
    rewritten_text = obj.get("rewritten_text")
    if not isinstance(rewritten_text, str):
        return None

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
            return None

    return rewritten_text


def _reorder_skills(tailored: Dict[str, Any], tailoring_plan: Dict[str, Any]) -> None:
    plan = tailoring_plan.get("skills_reorder_plan")
    if not isinstance(plan, list):
        return
    skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else None
    if skills is None:
        return
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    if not lines:
        return

    line_map: Dict[str, Dict[str, Any]] = {}
    original_order: List[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if isinstance(line_id, str):
            line_map[line_id] = line
            original_order.append(line_id)

    reordered: List[Dict[str, Any]] = []
    seen = set()
    for line_id in plan:
        if not isinstance(line_id, str):
            continue
        item = line_map.get(line_id)
        if item is None:
            continue
        if line_id in seen:
            continue
        seen.add(line_id)
        reordered.append(item)

    for line_id in original_order:
        if line_id in seen:
            continue
        item = line_map.get(line_id)
        if item is not None:
            reordered.append(item)

    skills["lines"] = reordered


def _rewrite_bullets(
    tailored: Dict[str, Any],
    job_json: Dict[str, Any],
    tailoring_plan: Dict[str, Any],
    provider: LLMProvider,
    allowed_vocab: Dict[str, Any],
    budgets: Dict[str, Any],
    audit_log: Dict[str, List[str]],
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
    audit_log: Dict[str, List[str]],
) -> None:
    bullet_id = bullet.get("bullet_id")
    original_text = bullet.get("text")
    if not isinstance(bullet_id, str) or not isinstance(original_text, str):
        return

    action = action_map.get(bullet_id)
    if not isinstance(action, dict):
        audit_log["kept_bullets"].append(bullet_id)
        return

    rewrite_intent = action.get("rewrite_intent")
    if isinstance(rewrite_intent, str) and rewrite_intent.strip().lower() in _REWRITE_KEEP:
        audit_log["kept_bullets"].append(bullet_id)
        return

    target_keywords = action.get("target_keywords") if isinstance(action.get("target_keywords"), list) else []
    if not target_keywords:
        audit_log["kept_bullets"].append(bullet_id)
        return

    budget = bullet_budgets.get(bullet_id)

    rewritten_text, rejected = _call_bullet_rewrite(
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
    if rewritten_text is None:
        if not rejected:
            audit_log["kept_bullets"].append(bullet_id)
        return

    provider_len = len(rewritten_text)
    did_shorten = False
    compressed = _compress_text(original_text, rewritten_text, budget, provider)
    if compressed is not None:
        if len(compressed) < len(rewritten_text):
            did_shorten = True
        rewritten_text = compressed

        disallowed_after = _find_disallowed_terms(rewritten_text, allowed_vocab, target_keywords)
        if disallowed_after:
            audit_log["rejected_for_new_terms"].append(bullet_id)
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

def _call_bullet_rewrite(
    bullet_id: str,
    original_text: str,
    job_json: Dict[str, Any],
    target_keywords: Sequence[str],
    allowed_vocab: Dict[str, Any],
    provider: LLMProvider,
    budget: Optional[int],
) -> Tuple[Optional[str], bool]:
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
        return None, False

    if obj.get("bullet_id") != bullet_id:
        return None, False
    rewritten_text = obj.get("rewritten_text")
    if not isinstance(rewritten_text, str):
        return None, False

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
            return None, True

    return rewritten_text, False


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
        if not kw_norm or kw_norm in allowed_terms:
            continue
        if _contains_phrase(normalized_text, kw_norm):
            disallowed.add(kw_norm)

    for term in _extract_tool_like_terms(rewritten_text):
        if term not in allowed_terms and term not in allowed_proper:
            disallowed.add(term)

    for term in _extract_proper_noun_candidates(rewritten_text):
        if term not in allowed_proper:
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
            candidates.append(_normalize_proper_token(raw))
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


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


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
