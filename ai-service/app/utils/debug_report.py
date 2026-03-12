import json
from typing import Any, Dict, List, Optional

from shared.scoring import build_resume_index

_REPORT_DIVIDER = "=" * 50


def print_tailoring_debug_report(
    *,
    job_json: Optional[Dict[str, Any]],
    resume_json: Optional[Dict[str, Any]],
    tailored_resume_json: Optional[Dict[str, Any]],
    score_result: Optional[Dict[str, Any]] = None,
) -> None:
    header_lines = [_REPORT_DIVIDER, "TAILORING DEBUG REPORT"]
    job_title = _safe_str(job_json, "title")
    if job_title:
        header_lines.append(f"JOB: {job_title}")
    decision = _safe_str(score_result, "decision")
    if decision:
        header_lines.append(f"DECISION: {decision}")
    header_lines.append(_REPORT_DIVIDER)
    print("\n".join(header_lines))

    print("JOB KEYWORDS JSON")
    job_payload = _build_job_keywords_payload(job_json)
    if job_payload is None:
        print("Job keywords not available at this stage")
    else:
        print(_dump_json(job_payload))
    print("")

    print("RESUME KEYWORDS JSON")
    resume_payload = _build_resume_signals_payload(resume_json)
    if resume_payload is None:
        print("Resume signals not available at this stage")
    else:
        print(_dump_json(resume_payload))
    print("")

    print("RESUME CHANGES JSON")
    changes_payload = _build_resume_changes_payload(resume_json, tailored_resume_json)
    if changes_payload is None:
        print("Resume changes not available at this stage")
    elif _is_no_changes(changes_payload):
        print("No changes detected")
    else:
        print(_dump_json(changes_payload))


def _safe_str(obj: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    value = obj.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _dump_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _build_job_keywords_payload(job_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(job_json, dict):
        return None
    fields = [
        "title",
        "company",
        "keywords",
        "must_have",
        "nice_to_have",
        "responsibilities",
    ]
    payload: Dict[str, Any] = {}
    for field in fields:
        if field in job_json:
            payload[field] = job_json.get(field)
    return payload


def _build_resume_signals_payload(resume_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(resume_json, dict):
        return None
    try:
        index = build_resume_index(resume_json)
    except Exception:
        return None

    summary_payload = None
    summary = index.get("summary") if isinstance(index.get("summary"), dict) else None
    if isinstance(summary, dict):
        summary_payload = {
            "text": summary.get("original", ""),
            "signals": _jsonify_signals(summary.get("signals")),
        }

    skills_payload: List[Dict[str, Any]] = []
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    lines = skills.get("lines") if isinstance(skills.get("lines"), list) else []
    skills_index = index.get("skills") if isinstance(index.get("skills"), dict) else {}
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if not isinstance(line_id, str):
            continue
        entry = skills_index.get(line_id) if isinstance(skills_index.get(line_id), dict) else {}
        skills_payload.append(
            {
                "line_id": line_id,
                "text": line.get("text", ""),
                "signals": _jsonify_signals(entry.get("signals")),
            }
        )

    experience_payload: List[Dict[str, Any]] = []
    experience = resume_json.get("experience") if isinstance(resume_json.get("experience"), list) else []
    exp_index = index.get("experience") if isinstance(index.get("experience"), dict) else {}
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        exp_id = exp.get("exp_id")
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if not isinstance(bullet_id, str):
                continue
            entry = exp_index.get(bullet_id) if isinstance(exp_index.get(bullet_id), dict) else {}
            experience_payload.append(
                {
                    "exp_id": exp_id,
                    "bullet_id": bullet_id,
                    "text": bullet.get("text", ""),
                    "signals": _jsonify_signals(entry.get("signals")),
                }
            )

    projects_payload: List[Dict[str, Any]] = []
    projects = resume_json.get("projects") if isinstance(resume_json.get("projects"), list) else []
    projects_index = index.get("projects") if isinstance(index.get("projects"), dict) else {}
    for project in projects:
        if not isinstance(project, dict):
            continue
        project_id = project.get("project_id")
        bullets = project.get("bullets") if isinstance(project.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if not isinstance(bullet_id, str):
                continue
            entry = projects_index.get(bullet_id) if isinstance(projects_index.get(bullet_id), dict) else {}
            projects_payload.append(
                {
                    "project_id": project_id,
                    "bullet_id": bullet_id,
                    "text": bullet.get("text", ""),
                    "signals": _jsonify_signals(entry.get("signals")),
                }
            )

    all_tokens = index.get("all_tokens")
    all_phrases = index.get("all_phrases")
    payload = {
        "summary": summary_payload,
        "skills_lines": skills_payload,
        "experience_bullets": experience_payload,
        "all_tokens": _sorted_list(all_tokens),
        "all_phrases": _sorted_list(all_phrases),
    }
    if projects_payload:
        payload["project_bullets"] = projects_payload
    return payload


def _build_resume_changes_payload(
    original: Optional[Dict[str, Any]],
    tailored: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(original, dict) or not isinstance(tailored, dict):
        return None

    summary_original = _safe_nested_text(original, "summary")
    summary_final = _safe_nested_text(tailored, "summary")
    summary_changed = summary_original != summary_final
    payload: Dict[str, Any] = {
        "summary_changed": summary_changed,
        "skills_lines_changed": [],
        "bullets_changed": [],
    }
    if summary_changed:
        payload["summary"] = {"original": summary_original, "final": summary_final}

    skills_changes: List[Dict[str, Any]] = []
    original_skills = original.get("skills") if isinstance(original.get("skills"), dict) else {}
    tailored_skills = tailored.get("skills") if isinstance(tailored.get("skills"), dict) else {}
    original_lines = original_skills.get("lines") if isinstance(original_skills.get("lines"), list) else []
    tailored_lines = tailored_skills.get("lines") if isinstance(tailored_skills.get("lines"), list) else []
    tailored_line_map = {
        line.get("line_id"): line.get("text", "")
        for line in tailored_lines
        if isinstance(line, dict) and isinstance(line.get("line_id"), str)
    }
    for line in original_lines:
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if not isinstance(line_id, str):
            continue
        original_text = line.get("text", "")
        final_text = tailored_line_map.get(line_id)
        if final_text is None:
            continue
        if final_text != original_text:
            skills_changes.append(
                {"line_id": line_id, "original": original_text, "final": final_text}
            )
    payload["skills_lines_changed"] = skills_changes

    bullet_changes: List[Dict[str, Any]] = []
    original_exps = original.get("experience") if isinstance(original.get("experience"), list) else []
    tailored_exps = tailored.get("experience") if isinstance(tailored.get("experience"), list) else []
    tailored_bullet_map = {}
    for exp in tailored_exps:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if isinstance(bullet_id, str):
                tailored_bullet_map[bullet_id] = bullet.get("text", "")
    for exp in original_exps:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets") if isinstance(exp.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if not isinstance(bullet_id, str):
                continue
            original_text = bullet.get("text", "")
            final_text = tailored_bullet_map.get(bullet_id)
            if final_text is None:
                continue
            if final_text != original_text:
                bullet_changes.append(
                    {"bullet_id": bullet_id, "original": original_text, "final": final_text}
                )
    payload["bullets_changed"] = bullet_changes

    project_changes: List[Dict[str, Any]] = []
    original_projects = original.get("projects") if isinstance(original.get("projects"), list) else []
    tailored_projects = tailored.get("projects") if isinstance(tailored.get("projects"), list) else []
    tailored_project_bullets = {}
    for project in tailored_projects:
        if not isinstance(project, dict):
            continue
        bullets = project.get("bullets") if isinstance(project.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if isinstance(bullet_id, str):
                tailored_project_bullets[bullet_id] = bullet.get("text", "")
    for project in original_projects:
        if not isinstance(project, dict):
            continue
        bullets = project.get("bullets") if isinstance(project.get("bullets"), list) else []
        for bullet in bullets:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get("bullet_id")
            if not isinstance(bullet_id, str):
                continue
            original_text = bullet.get("text", "")
            final_text = tailored_project_bullets.get(bullet_id)
            if final_text is None:
                continue
            if final_text != original_text:
                project_changes.append(
                    {"bullet_id": bullet_id, "original": original_text, "final": final_text}
                )
    if project_changes:
        payload["project_bullets_changed"] = project_changes

    return payload


def _safe_nested_text(obj: Dict[str, Any], key: str) -> str:
    section = obj.get(key) if isinstance(obj.get(key), dict) else {}
    text = section.get("text")
    return text if isinstance(text, str) else ""


def _jsonify_signals(signals: Any) -> Dict[str, Any]:
    if not isinstance(signals, dict):
        return {"normalized": "", "tokens": [], "phrases": []}
    tokens = signals.get("tokens")
    phrases = signals.get("phrases")
    return {
        "normalized": signals.get("normalized", ""),
        "tokens": tokens if isinstance(tokens, list) else [],
        "phrases": _sorted_list(phrases),
    }


def _sorted_list(value: Any) -> List[str]:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _is_no_changes(payload: Dict[str, Any]) -> bool:
    if payload.get("summary_changed"):
        return False
    skills = payload.get("skills_lines_changed")
    bullets = payload.get("bullets_changed")
    projects = payload.get("project_bullets_changed")
    return (
        (not skills)
        and (not bullets)
        and (not projects)
    )
