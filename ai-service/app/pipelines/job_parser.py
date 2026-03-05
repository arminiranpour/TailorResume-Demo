from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import get_config
from app.pipelines.job_invariants import enforce_job_invariants
from app.pipelines.repair import repair_json_to_schema
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.validator import validate_json
from app.security.untrusted import build_llm_messages


@dataclass
class JobParseError(Exception):
    details: List[str]
    raw_preview: str


def normalize_job_text(text: str, *, max_chars: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", normalized)
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


def extract_job_json(
    job_text: str, provider: LLMProvider, *, url: Optional[str] = None
) -> Dict[str, object]:
    config = get_config()
    normalized = normalize_job_text(job_text, max_chars=config.max_input_chars)
    prompt_text = normalized
    if url:
        prompt_text = f"SOURCE_URL: {url}\n\n{normalized}"
    system_prompt = load_system_prompt("job_to_json")
    messages = build_llm_messages(system_prompt, prompt_text, task_label="job_to_json")
    raw = provider.generate(messages, timeout_seconds=config.llm_timeout_seconds)
    ok, obj, errors = _parse_validate_invariants(raw, normalized)
    if ok and obj is not None:
        return obj
    repair_errors = errors
    try:
        repaired = repair_json_to_schema(raw, "job", repair_errors, provider)
    except Exception as exc:
        raise JobParseError(
            details=[f"Repair failed: {exc}"],
            raw_preview=_preview_raw(raw),
        ) from exc
    ok, obj, errors = _validate_object(repaired, normalized)
    if ok and obj is not None:
        return obj
    raise JobParseError(details=errors, raw_preview=_preview_raw(raw))


def _parse_validate_invariants(
    raw: str, job_text: str
) -> tuple[bool, Dict[str, object] | None, List[str]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, None, [f"JSON parse error: {exc.msg} at pos {exc.pos}"]
    return _validate_object(obj, job_text)


def _validate_object(
    obj: Dict[str, object], job_text: str
) -> tuple[bool, Dict[str, object] | None, List[str]]:
    ok, errors = validate_json("job", obj)
    if not ok:
        return False, obj, errors
    ok, inv_errors = enforce_job_invariants(obj, job_text)
    if not ok:
        return False, obj, inv_errors
    return True, obj, []


def _preview_raw(raw: str) -> str:
    if raw is None:
        return ""
    return raw[:500]
