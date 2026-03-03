from pathlib import Path


def repo_root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def shared_schema_dir() -> Path:
    return repo_root_path() / "shared" / "schemas"


def shared_schema_path(filename: str) -> Path:
    return shared_schema_dir() / filename
