from typing import Any, Dict

from app.schemas.schema_loader import load_schema


def load_shared_schema(filename: str) -> Dict[str, Any]:
    return load_schema(filename)


__all__ = ["load_schema", "load_shared_schema"]
