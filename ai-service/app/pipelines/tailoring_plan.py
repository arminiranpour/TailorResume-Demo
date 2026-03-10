from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import get_config
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages


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
