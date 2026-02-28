import json
from typing import Any, Dict

from .paths import shared_schema_path


def load_shared_schema(filename: str) -> Dict[str, Any]:
    """
    Load a JSON schema from the canonical shared/schemas directory.
    Phase-0: load only, no validation.
    """
    path = shared_schema_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Shared schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
