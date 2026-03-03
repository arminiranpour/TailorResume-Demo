from dataclasses import dataclass
from functools import lru_cache
import os


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    llm_timeout_seconds: int
    max_input_chars: int


@lru_cache
def get_config() -> AppConfig:
    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "local"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        llm_timeout_seconds=_get_env_int("LLM_TIMEOUT_SECONDS", 60),
        max_input_chars=_get_env_int("MAX_INPUT_CHARS", 30000),
    )
