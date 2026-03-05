from __future__ import annotations

import json
from typing import Dict, List

from app.config import get_config
from app.prompts.loader import load_system_prompt
from app.providers.base import LLMProvider
from app.schemas.schema_loader import load_schema
from app.security.untrusted import build_llm_messages


def repair_json_to_schema(
    raw_json_text: str, schema_name: str, errors: List[str], provider: LLMProvider
) -> Dict[str, object]:
    config = get_config()
    schema = load_schema(schema_name)
    minified_schema = json.dumps(schema, separators=(",", ":"))
    error_lines = "\n".join(f"- {err}" for err in errors) if errors else "- (none)"
    untrusted_block = (
        f"SCHEMA:{minified_schema}\n"
        f"VALIDATION_ERRORS:\n{error_lines}\n"
        "RAW_JSON_OUTPUT:\n"
        f"{raw_json_text}"
    )
    system_prompt = load_system_prompt("json_repair")
    messages = build_llm_messages(system_prompt, untrusted_block, task_label="json_repair")
    raw = provider.generate(messages, timeout_seconds=config.llm_timeout_seconds)
    return json.loads(raw)


def repair_schema(payload: Dict[str, object]) -> Dict[str, object]:
    raise NotImplementedError("repair_schema is not implemented")
