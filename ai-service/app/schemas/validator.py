from __future__ import annotations

from typing import Any, Tuple, List

from jsonschema import Draft202012Validator

from app.schemas.schema_loader import load_schema


def validate_json(schema_name: str, obj: Any) -> Tuple[bool, List[str]]:
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))

    if not errors:
        return True, []

    messages: List[str] = []
    for e in errors:
        path = "$"
        if e.path:
            path = "$." + ".".join(str(p) for p in e.path)
        messages.append(f"{path}: {e.message}")
    return False, messages


def validate_or_raise(schema_name: str, obj: Any) -> None:
    ok, errs = validate_json(schema_name, obj)
    if not ok:
        raise ValueError("Schema validation failed: " + " | ".join(errs))