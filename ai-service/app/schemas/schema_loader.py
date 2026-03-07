import json
from typing import Any, Dict

from app.schemas.paths import shared_schema_path

_SCHEMA_MAP = {
    "resume": "resume.schema.json",
    "job": "job.schema.json",
    "tailoring_plan": "tailoring_plan.schema.json",
    "bullet_rewrite": "bullet_rewrite.schema.json",
    "summary_rewrite": "summary_rewrite.schema.json",
    "compress_text": "compress_text.schema.json",
}


def _resolve_schema_filename(schema_name: str) -> str:
    normalized = schema_name.strip().lower()
    if normalized.endswith(".schema.json") or normalized.endswith(".json"):
        return normalized
    if normalized in _SCHEMA_MAP:
        return _SCHEMA_MAP[normalized]
    raise ValueError(f"Unknown schema name: {schema_name}")


def load_schema(schema_name: str) -> Dict[str, Any]:
    filename = _resolve_schema_filename(schema_name)
    path = shared_schema_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Shared schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
