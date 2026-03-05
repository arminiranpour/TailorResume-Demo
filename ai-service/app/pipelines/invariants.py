from __future__ import annotations

from typing import Dict, List, Set


def enforce_resume_invariants(obj: Dict[str, object]) -> tuple[bool, List[str]]:
    errors: List[str] = []
    _check_meta_structure_hash(obj, errors)
    _check_sections(obj, errors)
    _check_experiences(obj, errors)
    _check_skills(obj, errors)
    return (len(errors) == 0), errors


def _check_meta_structure_hash(obj: Dict[str, object], errors: List[str]) -> None:
    meta = obj.get("meta")
    if isinstance(meta, dict) and "structure_hash" in meta:
        value = meta.get("structure_hash")
        if not isinstance(value, str) or value.strip() == "":
            errors.append("meta.structure_hash must be a non-empty string")


def _check_sections(obj: Dict[str, object], errors: List[str]) -> None:
    sections = obj.get("sections")
    if not isinstance(sections, list):
        return
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or section_id.strip() == "":
            errors.append(f"sections[{idx}].section_id missing or empty")


def _check_experiences(obj: Dict[str, object], errors: List[str]) -> None:
    for key in ("experience", "experiences", "work_experience"):
        experiences = obj.get(key)
        if not isinstance(experiences, list):
            continue
        for idx, exp in enumerate(experiences):
            if not isinstance(exp, dict):
                continue
            exp_id = exp.get("exp_id")
            if not isinstance(exp_id, str) or exp_id.strip() == "":
                errors.append(f"{key}[{idx}].exp_id missing or empty")
            bullets = exp.get("bullets")
            if not isinstance(bullets, list):
                continue
            _check_bullets(bullets, errors, f"{key}[{idx}].bullets")


def _check_bullets(bullets: List[object], errors: List[str], prefix: str) -> None:
    seen: Set[int] = set()
    for idx, bullet in enumerate(bullets):
        if not isinstance(bullet, dict):
            continue
        bullet_id = bullet.get("bullet_id")
        if not isinstance(bullet_id, str) or bullet_id.strip() == "":
            errors.append(f"{prefix}[{idx}].bullet_id missing or empty")
        bullet_index = bullet.get("bullet_index")
        if not isinstance(bullet_index, int):
            errors.append(f"{prefix}[{idx}].bullet_index missing or not int")
        else:
            if bullet_index < 0 or bullet_index >= len(bullets):
                errors.append(f"{prefix}[{idx}].bullet_index out of range")
            if bullet_index in seen:
                errors.append(f"{prefix}[{idx}].bullet_index duplicate")
            seen.add(bullet_index)
    if len(bullets) > 0 and len(seen) == len(bullets):
        expected = set(range(len(bullets)))
        if seen != expected:
            errors.append(f"{prefix} bullet_index must cover 0..{len(bullets) - 1}")


def _check_skills(obj: Dict[str, object], errors: List[str]) -> None:
    skills = obj.get("skills")
    if isinstance(skills, dict):
        lines = skills.get("lines")
    elif isinstance(skills, list):
        lines = skills
    else:
        return
    if not isinstance(lines, list):
        return
    for idx, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        line_id = line.get("line_id")
        if not isinstance(line_id, str) or line_id.strip() == "":
            errors.append(f"skills.lines[{idx}].line_id missing or empty")
        if "line_index" in line:
            line_index = line.get("line_index")
            if not isinstance(line_index, int) or line_index != idx:
                errors.append(f"skills.lines[{idx}].line_index must equal list position")
