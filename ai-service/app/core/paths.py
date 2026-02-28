from pathlib import Path


def repo_root_path() -> Path:
    """
    Resolve the repository root path regardless of current working directory.
    Location is derived from this file: ai-service/app/core/paths.py
    """
    return Path(__file__).resolve().parents[3]


def shared_schema_path(filename: str) -> Path:
    """
    Resolve a schema path under <repo_root>/shared/schemas/<filename>.
    """
    return repo_root_path() / "shared" / "schemas" / filename
