import time
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_config
from app.providers.base import LLMProvider
from app.providers.factory import get_provider
from app.schemas.schema_loader import load_schema

app = FastAPI(title="TailorResume AI Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    return {"ok": True}


def get_llm_provider() -> LLMProvider:
    return get_provider()


def get_schema_loader() -> Callable[[str], Dict[str, Any]]:
    return load_schema


@app.get("/health/llm")
def health_llm() -> dict:
    config = get_config()
    provider = get_provider()
    start = time.perf_counter()
    error = None
    ok = False
    try:
        content = provider.generate(
            [
                {
                    "role": "system",
                    "content": "You are a health check. Reply with exactly {\"ok\":true}.",
                },
                {"role": "user", "content": "Return exactly {\"ok\":true}."},
            ],
            timeout_seconds=config.llm_timeout_seconds,
        )
        if not isinstance(content, str) or content.strip() == "":
            raise RuntimeError("Empty response from provider")
        ok = True
    except Exception as exc:
        error = str(exc)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "ok": ok,
        "provider": config.llm_provider,
        "model": config.ollama_model,
        "latency_ms": latency_ms,
        "error": error,
    }


class ParseResumeRequest(BaseModel):
    resume_text: str


class ParseJobRequest(BaseModel):
    job_text: str
    url: Optional[str] = None


class RepairJsonRequest(BaseModel):
    raw: str
    schema_name: str


class TailorRequest(BaseModel):
    resume_json: Dict[str, Any]
    job_json: Dict[str, Any]


def _not_implemented(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "ok": False,
            "error": "Not Implemented - Phase 0 stub",
            "detail": detail,
        },
    )


@app.post("/parse-resume")
def parse_resume(
    _: ParseResumeRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = provider, schema_loader
    return _not_implemented("parse-resume endpoint is a Phase 0 stub")


@app.post("/parse-job")
def parse_job(
    _: ParseJobRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = provider, schema_loader
    return _not_implemented("parse-job endpoint is a Phase 0 stub")


@app.post("/repair-json")
def repair_json(
    _: RepairJsonRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = provider, schema_loader
    return _not_implemented("repair-json endpoint is a Phase 0 stub")


@app.post("/tailor")
def tailor(
    _: TailorRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = provider, schema_loader
    return _not_implemented("tailor endpoint is a Phase 0 stub")
