from __future__ import annotations

from typing import Dict, List


def enforce_job_invariants(obj: Dict[str, object], job_text: str) -> tuple[bool, List[str]]:
    errors: List[str] = []
    _check_arrays(obj, errors)
    _check_requirement_entries(obj, errors)
    _check_enums(obj, errors)
    _check_keywords(obj, errors)
    _check_evidence(obj, job_text, errors)
    return (len(errors) == 0), errors


def _check_arrays(obj: Dict[str, object], errors: List[str]) -> None:
    for key in ("must_have", "nice_to_have", "responsibilities"):
        if key not in obj:
            errors.append(f"{key} missing")
            continue
        if not isinstance(obj.get(key), list):
            errors.append(f"{key} must be an array")


def _check_requirement_entries(obj: Dict[str, object], errors: List[str]) -> None:
    for key in ("must_have", "nice_to_have"):
        entries = obj.get(key)
        if not isinstance(entries, list):
            continue
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{key}[{idx}] must be an object")
                continue
            req_id = entry.get("requirement_id")
            if not isinstance(req_id, str) or req_id.strip() == "":
                errors.append(f"{key}[{idx}].requirement_id missing or empty")
            text = entry.get("text")
            if not isinstance(text, str) or text.strip() == "":
                errors.append(f"{key}[{idx}].text missing or empty")


def _check_enums(obj: Dict[str, object], errors: List[str]) -> None:
    if "seniority" in obj:
        value = obj.get("seniority")
        if value is not None and value not in ("junior", "mid", "senior", "unknown"):
            errors.append("seniority must be one of: junior, mid, senior, unknown")
    if "remote" in obj:
        value = obj.get("remote")
        if value is not None and not isinstance(value, bool):
            errors.append("remote must be boolean or null")


def _check_keywords(obj: Dict[str, object], errors: List[str]) -> None:
    if "keywords" not in obj:
        return
    keywords = obj.get("keywords")
    if keywords is None:
        return
    if not isinstance(keywords, list):
        errors.append("keywords must be an array of strings")
        return
    for idx, keyword in enumerate(keywords):
        if not isinstance(keyword, str):
            errors.append(f"keywords[{idx}] must be a string")


def _check_evidence(obj: Dict[str, object], job_text: str, errors: List[str]) -> None:
    haystack = job_text.lower()
    for field in ("company", "title"):
        value = obj.get(field)
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if candidate == "" or candidate.lower() == "unknown":
            continue
        if candidate.lower() not in haystack:
            errors.append(f"field_not_evidenced:{field}")
