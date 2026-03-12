from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import get_config
from app.pipelines.allowed_vocab import build_allowed_vocab
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages
from shared.scoring.normalize import generate_ngrams, normalize_text, tokenize


@dataclass
class TailoringPlanError(Exception):
    details: List[str]
    raw_preview: str


@dataclass
class TailorNotAllowed(Exception):
    decision: str
    reasons: List[Dict[str, Any]]

    def __str__(self) -> str:
        return f"Tailoring plan not allowed for decision={self.decision}"


def generate_tailoring_plan(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    provider: LLMProvider,
) -> Dict[str, Any]:
    if not isinstance(resume_json, dict) or not isinstance(job_json, dict) or not isinstance(score_result, dict):
        raise ValueError("resume_json, job_json, and score_result must be objects")

    decision = score_result.get("decision")
    if decision != "PROCEED":
        reasons = score_result.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        raise TailorNotAllowed(decision=str(decision), reasons=reasons)

    bullet_order = _collect_bullet_order(resume_json)
    allowed_bullet_ids = _collect_allowed_bullet_ids(resume_json)
    skills_line_ids = _collect_skills_line_ids(resume_json)
    allowed_skills = set(skills_line_ids)

    prompt_payload = _build_prompt_payload(resume_json, job_json, score_result, bullet_order, skills_line_ids)
    system_prompt = load_system_prompt("tailor_plan")
    messages = build_llm_messages(system_prompt, prompt_payload, task_label="tailor_plan")

    config = get_config()
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)

    ok, obj, errors = _parse_and_validate(raw)
    if not ok or obj is None:
        repair_errors = errors
        try:
            repaired = repair_json_to_schema(raw, "tailoring_plan", repair_errors, provider)
        except Exception as exc:
            raise TailoringPlanError(
                details=[f"Repair failed: {exc}"],
                raw_preview=_preview_raw(raw),
            ) from exc
        ok, obj, errors = _validate_object(repaired)
        if not ok or obj is None:
            raise TailoringPlanError(details=errors, raw_preview=_preview_raw(raw))

    plan = _enforce_invariants(
        obj,
        bullet_order,
        allowed_bullet_ids,
        allowed_skills,
        score_result,
        _preview_raw(json.dumps(obj, ensure_ascii=True)),
    )

    plan = _ensure_actionable_rewrites(plan, resume_json, job_json)

    ok, errors = validate_json("tailoring_plan", plan)
    if not ok:
        raise TailoringPlanError(details=errors, raw_preview=_preview_raw(raw))

    return plan


def _build_prompt_payload(
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
    score_result: Dict[str, Any],
    bullet_ids: List[str],
    skills_line_ids: List[str],
) -> str:
    payload = {
        "resume": _minimize_resume(resume_json),
        "job": _minimize_job(job_json),
        "score_result": _minimize_score(score_result),
        "allowed_ids": {
            "bullet_ids": bullet_ids,
            "skills_line_ids": skills_line_ids,
        },
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _minimize_resume(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    skills_lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []

    return {
        "summary": {
            "id": summary.get("id"),
            "text": summary.get("text"),
        },
        "skills": [
            {"line_id": line.get("line_id"), "text": line.get("text")}
            for line in skills_lines
            if isinstance(line, dict)
        ],
        "experience": [_minimize_experience(exp) for exp in experience if isinstance(exp, dict)],
        "projects": [_minimize_project(proj) for proj in projects if isinstance(proj, dict)],
    }


def _minimize_experience(exp: Dict[str, Any]) -> Dict[str, Any]:
    bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
    return {
        "exp_id": exp.get("exp_id"),
        "company": exp.get("company"),
        "title": exp.get("title"),
        "bullets": [
            {
                "bullet_id": bullet.get("bullet_id"),
                "bullet_index": bullet.get("bullet_index"),
                "text": bullet.get("text"),
            }
            for bullet in bullets
            if isinstance(bullet, dict)
        ],
    }


def _minimize_project(project: Dict[str, Any]) -> Dict[str, Any]:
    bullets = project.get("bullets") if isinstance(project.get("bullets"), list) else []
    return {
        "project_id": project.get("project_id"),
        "name": project.get("name"),
        "bullets": [
            {
                "bullet_id": bullet.get("bullet_id"),
                "bullet_index": bullet.get("bullet_index"),
                "text": bullet.get("text"),
            }
            for bullet in bullets
            if isinstance(bullet, dict)
        ],
    }


def _minimize_job(job_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": job_json.get("title"),
        "seniority": job_json.get("seniority"),
        "must_have": _minimize_requirements(job_json.get("must_have")),
        "nice_to_have": _minimize_requirements(job_json.get("nice_to_have")),
        "responsibilities": job_json.get("responsibilities"),
        "keywords": job_json.get("keywords"),
    }


def _minimize_requirements(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    minimized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            minimized.append({"requirement_id": item.get("requirement_id"), "text": item.get("text")})
    return minimized


def _minimize_score(score_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "must_have_coverage_percent": score_result.get("must_have_coverage_percent"),
        "matched_requirements": score_result.get("matched_requirements"),
        "missing_requirements": score_result.get("missing_requirements"),
    }


def _collect_bullet_order(resume_json: Dict[str, Any]) -> List[str]:
    order: List[str] = []
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            bullet_id = bullet.get("bullet_id") if isinstance(bullet, dict) else None
            if isinstance(bullet_id, str) and bullet_id.strip() != "":
                order.append(bullet_id)
    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        bullets = proj.get("bullets") if isinstance(proj.get("bullets"), list) else []
        for bullet in _sorted_bullets(bullets):
            bullet_id = bullet.get("bullet_id") if isinstance(bullet, dict) else None
            if isinstance(bullet_id, str) and bullet_id.strip() != "":
                order.append(bullet_id)
    return order


def _collect_allowed_bullet_ids(resume_json: Dict[str, Any]) -> Set[str]:
    return set(_collect_bullet_order(resume_json))


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


def _collect_skills_line_ids(resume_json: Dict[str, Any]) -> List[str]:
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    ids: List[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if isinstance(line_id, str) and line_id.strip() != "":
            ids.append(line_id)
    return ids


def _parse_and_validate(raw: str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, None, [f"JSON parse error: {exc.msg} at pos {exc.pos}"]
    return _validate_object(obj)


def _validate_object(obj: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    ok, errors = validate_json("tailoring_plan", obj)
    if not ok:
        return False, obj, errors
    return True, obj, []


def _enforce_invariants(
    plan: Dict[str, Any],
    bullet_order: List[str],
    allowed_bullet_ids: Set[str],
    allowed_skill_ids: Set[str],
    score_result: Dict[str, Any],
    raw_preview: str,
) -> Dict[str, Any]:
    errors: List[str] = []

    actions = plan.get("bullet_actions")
    if not isinstance(actions, list):
        errors.append("bullet_actions must be a list")
        actions = []

    plan_bullet_ids: List[str] = []
    seen_ids: Set[str] = set()
    duplicates: Set[str] = set()
    for item in actions:
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if not isinstance(bullet_id, str) or bullet_id.strip() == "":
            continue
        plan_bullet_ids.append(bullet_id)
        if bullet_id in seen_ids:
            duplicates.add(bullet_id)
        else:
            seen_ids.add(bullet_id)

    unknown = sorted(set(plan_bullet_ids) - allowed_bullet_ids)
    if unknown:
        errors.append(f"unknown_bullet_ids: {unknown}")
    if duplicates:
        errors.append(f"duplicate_bullet_ids: {sorted(duplicates)}")

    action_map: Dict[str, Dict[str, Any]] = {}
    for item in actions:
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if not isinstance(bullet_id, str) or bullet_id.strip() == "":
            continue
        if bullet_id not in action_map:
            action_map[bullet_id] = item

    for bullet_id in bullet_order:
        if bullet_id not in action_map:
            action_map[bullet_id] = {
                "bullet_id": bullet_id,
                "rewrite_intent": "keep",
                "target_keywords": [],
            }

    ordered_actions = [action_map[bullet_id] for bullet_id in bullet_order]
    plan["bullet_actions"] = ordered_actions

    skills_plan = plan.get("skills_reorder_plan")
    if skills_plan is not None:
        if not isinstance(skills_plan, list):
            errors.append("skills_reorder_plan must be a list")
        else:
            deduped: List[str] = []
            seen: Set[str] = set()
            for item in skills_plan:
                if not isinstance(item, str):
                    continue
                if item not in allowed_skill_ids:
                    errors.append(f"skills_reorder_plan line_id not in resume: {item}")
                    continue
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            plan["skills_reorder_plan"] = deduped

    prioritized = plan.get("prioritized_keywords")
    if isinstance(prioritized, list):
        deduped_kw: List[str] = []
        seen_kw: Set[str] = set()
        for item in prioritized:
            if not isinstance(item, str):
                continue
            if item not in seen_kw:
                seen_kw.add(item)
                deduped_kw.append(item)
        plan["prioritized_keywords"] = deduped_kw

    missing_req = score_result.get("missing_requirements")
    if not isinstance(missing_req, list):
        errors.append("score_result.missing_requirements must be a list")
    else:
        plan["missing_requirements"] = missing_req

    if errors:
        raise TailoringPlanError(details=errors, raw_preview=raw_preview)

    return plan


def _preview_raw(raw: str) -> str:
    if raw is None:
        return ""
    return raw[:500]


_REWRITE_KEEP = {"keep", "skip", "none"}
_MAX_AUTO_REWRITES = 6
_MAX_TARGET_KEYWORDS = 3
_TRAILING_PERIODS = "."


def _ensure_actionable_rewrites(
    plan: Dict[str, Any],
    resume_json: Dict[str, Any],
    job_json: Dict[str, Any],
) -> Dict[str, Any]:
    actions = plan.get("bullet_actions")
    if not isinstance(actions, list):
        return plan

    allowed_vocab = build_allowed_vocab(resume_json)
    candidate_keywords = _collect_candidate_keywords(plan, job_json, allowed_vocab)
    if not candidate_keywords:
        return plan

    bullet_map = _build_bullet_text_map(resume_json)
    actionable_count = 0
    total_bullets = len(actions)
    for item in actions:
        if not isinstance(item, dict):
            continue
        if _is_rewrite_intent(item.get("rewrite_intent")) and _has_target_keywords(item):
            actionable_count += 1

    scored_bullets = _score_bullets(bullet_map, candidate_keywords)
    order_map = _build_action_order(actions)
    desired_count = _determine_rewrite_target(scored_bullets, total_bullets)
    auto_rewrites_used = 0
    for item in actions:
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if not isinstance(bullet_id, str):
            continue
        bullet_text = bullet_map.get(bullet_id, "")
        if not bullet_text:
            continue

        rewrite_intent = item.get("rewrite_intent")
        if _is_rewrite_intent(rewrite_intent):
            if not _has_target_keywords(item):
                selected = _select_targets_for_bullet(bullet_text, candidate_keywords)
                if selected:
                    item["target_keywords"] = selected
                else:
                    item["rewrite_intent"] = "keep"
                    item["target_keywords"] = []
            continue

    if actionable_count < desired_count:
        for bullet_id in _ranked_bullets(scored_bullets, order_map):
            if actionable_count >= desired_count or auto_rewrites_used >= _MAX_AUTO_REWRITES:
                break
            action = _find_action(actions, bullet_id)
            if action is None:
                continue
            if _is_rewrite_intent(action.get("rewrite_intent")) and _has_target_keywords(action):
                continue
            bullet_text = bullet_map.get(bullet_id, "")
            if not bullet_text:
                continue
            selected = _select_targets_for_bullet(bullet_text, candidate_keywords)
            if not selected:
                continue
            action["rewrite_intent"] = "rewrite"
            action["target_keywords"] = selected
            actionable_count += 1
            auto_rewrites_used += 1

    _maybe_enable_summary_rewrite(plan, resume_json, candidate_keywords)
    _ensure_skills_reorder_plan(plan, resume_json, candidate_keywords)

    return plan


def _maybe_enable_summary_rewrite(
    plan: Dict[str, Any],
    resume_json: Dict[str, Any],
    candidate_keywords: List[str],
) -> None:
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else None
    if summary is None:
        return
    summary_text = summary.get("text")
    if not isinstance(summary_text, str) or not summary_text.strip():
        return
    summary_targets = _select_targets_for_bullet(summary_text, candidate_keywords)
    if not summary_targets:
        return
    summary_plan = plan.get("summary_rewrite")
    if isinstance(summary_plan, dict):
        intent = summary_plan.get("rewrite_intent")
        if isinstance(intent, str) and intent.strip().lower() in _REWRITE_KEEP:
            return
        if _has_target_keywords(summary_plan):
            return
        summary_plan["rewrite_intent"] = summary_plan.get("rewrite_intent") or "rewrite"
        summary_plan["target_keywords"] = summary_targets
        return
    plan["summary_rewrite"] = {
        "rewrite_intent": "rewrite",
        "target_keywords": summary_targets,
    }


def _ensure_skills_reorder_plan(
    plan: Dict[str, Any],
    resume_json: Dict[str, Any],
    candidate_keywords: List[str],
) -> None:
    if not candidate_keywords:
        return
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else None
    if skills is None:
        return
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    if not lines:
        return
    line_map, order = _build_skills_line_map(lines)
    scores = _score_skill_lines(line_map, candidate_keywords, order)
    reordered = [line_id for line_id, _ in scores if line_id in order]
    if reordered == order:
        return
    plan["skills_reorder_plan"] = reordered


def _collect_candidate_keywords(
    plan: Dict[str, Any],
    job_json: Dict[str, Any],
    allowed_vocab: Dict[str, Any],
) -> List[str]:
    allowed_terms = allowed_vocab.get("terms", set())
    allowed_proper = allowed_vocab.get("proper_nouns", set())

    candidates: List[str] = []
    prioritized = plan.get("prioritized_keywords") if isinstance(plan.get("prioritized_keywords"), list) else []
    for keyword in prioritized:
        normalized = _normalize_candidate(keyword)
        if not normalized:
            continue
        if normalized in allowed_terms or normalized in allowed_proper:
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
    tokens = _clean_tokens(tokenize(text))
    ordered: List[str] = []
    for token in tokens:
        _append_unique(ordered, token)
    for phrase in _ordered_ngrams(tokens, 3):
        _append_unique(ordered, phrase)
    return ordered


def _normalize_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_text(value)


def _build_bullet_text_map(resume_json: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
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
            if isinstance(bullet_id, str) and isinstance(text, str) and bullet_id not in mapping:
                mapping[bullet_id] = text
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
            if isinstance(bullet_id, str) and isinstance(text, str) and bullet_id not in mapping:
                mapping[bullet_id] = text
    return mapping


def _select_targets_for_bullet(text: str, candidate_keywords: List[str]) -> List[str]:
    tokens = _clean_tokens(tokenize(text))
    phrases = generate_ngrams(tokens, 3)
    bullet_terms = set(tokens) | phrases
    selected: List[str] = []
    for term in candidate_keywords:
        if term in bullet_terms:
            _append_unique(selected, term)
            if len(selected) >= _MAX_TARGET_KEYWORDS:
                break
    return selected


def _is_rewrite_intent(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() not in _REWRITE_KEEP


def _has_target_keywords(action: Dict[str, Any]) -> bool:
    targets = action.get("target_keywords") if isinstance(action.get("target_keywords"), list) else []
    return bool(targets)


def _append_unique(values: List[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _score_bullets(
    bullet_map: Dict[str, str],
    candidate_keywords: List[str],
) -> List[Tuple[str, int]]:
    scores: List[Tuple[str, int]] = []
    for bullet_id, text in bullet_map.items():
        score = _count_keyword_overlap(text, candidate_keywords)
        if score > 0:
            scores.append((bullet_id, score))
    return scores


def _ranked_bullets(scored: List[Tuple[str, int]], order_map: Dict[str, int]) -> List[str]:
    return [
        bullet_id
        for bullet_id, _ in sorted(scored, key=lambda item: (-item[1], order_map.get(item[0], 0)))
    ]


def _determine_rewrite_target(scored: List[Tuple[str, int]], total_bullets: int) -> int:
    eligible = len(scored)
    if eligible <= 0:
        return 0
    baseline = max(2, int(math.ceil(eligible * 0.6)))
    cap = min(_MAX_AUTO_REWRITES, eligible)
    target = min(baseline, cap)
    if total_bullets > 0:
        target = min(target, total_bullets)
    return target


def _count_keyword_overlap(text: str, candidate_keywords: List[str]) -> int:
    if not text or not candidate_keywords:
        return 0
    tokens = _clean_tokens(tokenize(text))
    phrases = generate_ngrams(tokens, 3)
    terms = set(tokens) | phrases
    count = 0
    for keyword in candidate_keywords:
        if keyword in terms:
            count += 1
    return count


def _find_action(actions: List[Any], bullet_id: str) -> Optional[Dict[str, Any]]:
    for item in actions:
        if isinstance(item, dict) and item.get("bullet_id") == bullet_id:
            return item
    return None


def _build_action_order(actions: List[Any]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    for idx, item in enumerate(actions):
        if not isinstance(item, dict):
            continue
        bullet_id = item.get("bullet_id")
        if isinstance(bullet_id, str) and bullet_id not in order:
            order[bullet_id] = idx
    return order


def _ordered_ngrams(tokens: List[str], max_n: int) -> List[str]:
    ordered: List[str] = []
    if not tokens or max_n < 2:
        return ordered
    n_max = min(max_n, len(tokens))
    for n in range(2, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            ordered.append(" ".join(tokens[i : i + n]))
    return ordered


def _clean_tokens(tokens: List[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        stripped = _strip_trailing_periods(token)
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _strip_trailing_periods(token: str) -> str:
    if not token:
        return token
    return token.rstrip(_TRAILING_PERIODS)


def _build_skills_line_map(lines: List[Any]) -> Tuple[Dict[str, str], List[str]]:
    mapping: Dict[str, str] = {}
    order: List[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        text = line.get("text")
        if isinstance(line_id, str):
            order.append(line_id)
            if isinstance(text, str):
                mapping[line_id] = text
    return mapping, order


def _score_skill_lines(
    line_map: Dict[str, str],
    candidate_keywords: List[str],
    order: List[str],
) -> List[Tuple[str, int]]:
    scored: List[Tuple[str, int]] = []
    index_map = {line_id: idx for idx, line_id in enumerate(order)}
    for line_id in order:
        text = line_map.get(line_id, "")
        score = _count_keyword_overlap(text, candidate_keywords)
        scored.append((line_id, score))
    return sorted(scored, key=lambda item: (-item[1], index_map.get(item[0], 0)))
