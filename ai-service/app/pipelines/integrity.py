from __future__ import annotations

from typing import Any, Dict, List

from app.schemas.schema_loader import load_schema


def check_structure_invariants(original_resume_json: Dict[str, Any], final_resume_json: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    _check_required_keys(original_resume_json, final_resume_json, errors)
    _check_sections(original_resume_json, final_resume_json, errors)
    _check_skills(original_resume_json, final_resume_json, errors)
    _check_experience(original_resume_json, final_resume_json, errors)
    _check_projects(original_resume_json, final_resume_json, errors)
    _check_education(original_resume_json, final_resume_json, errors)

    return errors


def _check_required_keys(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    try:
        schema = load_schema("resume")
        required = schema.get("required") if isinstance(schema, dict) else []
    except Exception:
        required = ["meta", "summary", "skills", "experience", "education", "projects"]
    if not isinstance(required, list):
        required = ["meta", "summary", "skills", "experience", "education", "projects"]

    for key in required:
        if key not in original:
            errors.append(f"original missing required key: {key}")
        if key not in final:
            errors.append(f"final missing required key: {key}")

    if set(original.keys()) != set(final.keys()):
        errors.append("top-level keys changed")


def _check_sections(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    original_sections = original.get("sections") if isinstance(original.get("sections"), list) else None
    final_sections = final.get("sections") if isinstance(final.get("sections"), list) else None
    if original_sections is None and final_sections is None:
        return
    if not isinstance(original_sections, list) or not isinstance(final_sections, list):
        errors.append("sections missing or not a list")
        return
    if len(original_sections) != len(final_sections):
        errors.append("sections count changed")
        return
    original_ids = [item.get("section_id") for item in original_sections if isinstance(item, dict)]
    final_ids = [item.get("section_id") for item in final_sections if isinstance(item, dict)]
    if original_ids != final_ids:
        errors.append("sections order or ids changed")


def _check_skills(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    original_skills = original.get("skills") if isinstance(original.get("skills"), dict) else {}
    final_skills = final.get("skills") if isinstance(final.get("skills"), dict) else {}
    original_lines = original_skills.get("lines") if isinstance(original_skills.get("lines"), list) else []
    final_lines = final_skills.get("lines") if isinstance(final_skills.get("lines"), list) else []
    if len(original_lines) != len(final_lines):
        errors.append("skills.lines count changed")
        return
    original_ids = [line.get("line_id") for line in original_lines if isinstance(line, dict)]
    final_ids = [line.get("line_id") for line in final_lines if isinstance(line, dict)]
    if sorted(original_ids) != sorted(final_ids):
        errors.append("skills.lines ids changed")


def _check_experience(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    original_exps = original.get("experience") if isinstance(original.get("experience"), list) else []
    final_exps = final.get("experience") if isinstance(final.get("experience"), list) else []
    if len(original_exps) != len(final_exps):
        errors.append("experience count changed")
        return
    original_ids = [exp.get("exp_id") for exp in original_exps if isinstance(exp, dict)]
    final_ids = [exp.get("exp_id") for exp in final_exps if isinstance(exp, dict)]
    if original_ids != final_ids:
        errors.append("experience exp_id sequence changed")
    for idx, (orig, new) in enumerate(zip(original_exps, final_exps)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"experience[{idx}].bullets")


def _check_projects(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    original_projects = original.get("projects") if isinstance(original.get("projects"), list) else []
    final_projects = final.get("projects") if isinstance(final.get("projects"), list) else []
    if len(original_projects) != len(final_projects):
        errors.append("projects count changed")
        return
    original_ids = [proj.get("project_id") for proj in original_projects if isinstance(proj, dict)]
    final_ids = [proj.get("project_id") for proj in final_projects if isinstance(proj, dict)]
    if original_ids != final_ids:
        errors.append("projects project_id sequence changed")
    for idx, (orig, new) in enumerate(zip(original_projects, final_projects)):
        if not isinstance(orig, dict) or not isinstance(new, dict):
            continue
        _compare_bullets(orig.get("bullets"), new.get("bullets"), errors, f"projects[{idx}].bullets")


def _check_education(original: Dict[str, Any], final: Dict[str, Any], errors: List[str]) -> None:
    original_edu = original.get("education") if isinstance(original.get("education"), list) else []
    final_edu = final.get("education") if isinstance(final.get("education"), list) else []
    if len(original_edu) != len(final_edu):
        errors.append("education count changed")
        return
    original_ids = [edu.get("edu_id") for edu in original_edu if isinstance(edu, dict)]
    final_ids = [edu.get("edu_id") for edu in final_edu if isinstance(edu, dict)]
    if original_ids != final_ids:
        errors.append("education edu_id sequence changed")


def _compare_bullets(original_bullets: Any, final_bullets: Any, errors: List[str], prefix: str) -> None:
    original_list = original_bullets if isinstance(original_bullets, list) else []
    final_list = final_bullets if isinstance(final_bullets, list) else []
    if len(original_list) != len(final_list):
        errors.append(f"{prefix} count changed")
        return

    original_ids = [bullet.get("bullet_id") for bullet in original_list if isinstance(bullet, dict)]
    final_ids = [bullet.get("bullet_id") for bullet in final_list if isinstance(bullet, dict)]
    if original_ids != final_ids:
        if sorted(original_ids) != sorted(final_ids):
            errors.append(f"{prefix} bullet ids changed")
        else:
            original_index_map = _bullet_index_map(original_list)
            final_index_map = _bullet_index_map(final_list)
            for bullet_id, index in original_index_map.items():
                if final_index_map.get(bullet_id) != index:
                    errors.append(f"{prefix} bullet_index changed for {bullet_id}")
    else:
        original_index_map = _bullet_index_map(original_list)
        final_index_map = _bullet_index_map(final_list)
        for bullet_id, index in original_index_map.items():
            if final_index_map.get(bullet_id) != index:
                errors.append(f"{prefix} bullet_index changed for {bullet_id}")


def _bullet_index_map(bullets: List[Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for bullet in bullets:
        if not isinstance(bullet, dict):
            continue
        bullet_id = bullet.get("bullet_id")
        bullet_index = bullet.get("bullet_index")
        if isinstance(bullet_id, str) and isinstance(bullet_index, int):
            mapping[bullet_id] = bullet_index
    return mapping
