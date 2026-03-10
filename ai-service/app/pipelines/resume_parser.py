from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from app.config import get_config
from app.pipelines.invariants import enforce_resume_invariants
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages


@dataclass
class ResumeParseError(Exception):
    details: List[str]
    raw_preview: str


def normalize_resume_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    collapsed: List[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append("")
            continue
        blank_run = 0
        collapsed.append(line)
    normalized = "\n".join(collapsed)
    max_chars = get_config().max_input_chars
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    separator = "\n\n"
    if max_chars <= len(separator):
        return normalized[:max_chars]
    head = (max_chars - len(separator)) // 2
    tail = max_chars - len(separator) - head
    return normalized[:head] + separator + normalized[-tail:]


def extract_resume_json(resume_text: str, provider: LLMProvider) -> Dict[str, object]:
    config = get_config()
    normalized = normalize_resume_text(resume_text)
    system_prompt = load_system_prompt("resume_to_json")
    messages = build_llm_messages(system_prompt, normalized, task_label="resume_to_json")
    raw = provider.generate(messages, timeout=config.llm_timeout_seconds)
    ok, obj, errors = _parse_validate_invariants(raw)
    if ok and obj is not None:
        return obj
    repair_errors = errors
    try:
        repaired = repair_json_to_schema(raw, "resume", repair_errors, provider)
    except Exception as exc:
        raise ResumeParseError(
            details=[f"Repair failed: {exc}"],
            raw_preview=_preview_raw(raw),
        ) from exc
    ok, obj, errors = _validate_object(repaired)
    if ok and obj is not None:
        return obj
    raise ResumeParseError(details=errors, raw_preview=_preview_raw(raw))


def _parse_validate_invariants(raw: str) -> tuple[bool, Dict[str, object] | None, List[str]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, None, [f"JSON parse error: {exc.msg} at pos {exc.pos}"]
    return _validate_object(obj)


def _validate_object(obj: Dict[str, object]) -> tuple[bool, Dict[str, object] | None, List[str]]:
    ok, errors = validate_json("resume", obj)
    if not ok:
        return False, obj, errors
    ok, inv_errors = enforce_resume_invariants(obj)
    if not ok:
        return False, obj, inv_errors
    return True, obj, []


def _preview_raw(raw: str) -> str:
    if raw is None:
        return ""
    return raw[:500]
