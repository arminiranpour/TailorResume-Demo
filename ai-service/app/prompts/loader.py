from pathlib import Path


def load_system_prompt(name: str) -> str:
    filename = _resolve_prompt_filename(name)
    path = _shared_prompts_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Shared prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _resolve_prompt_filename(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith(".txt"):
        return normalized
    return f"{normalized}.system.txt"


def _shared_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "shared" / "prompts"
