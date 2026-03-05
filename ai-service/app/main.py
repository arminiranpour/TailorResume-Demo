import time
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_config
from app.providers.base import LLMProvider
from app.providers.factory import get_provider
from app.pipelines.job_parser import JobParseError, extract_job_json
from app.pipelines.resume_parser import ResumeParseError, extract_resume_json
from app.pipelines.scoring import run_scoring
from shared.scoring import decide
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


class ScoreRequest(BaseModel):
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
    payload: ParseResumeRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = schema_loader
    try:
        resume_json = extract_resume_json(payload.resume_text, provider)
    except ResumeParseError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "resume_json_invalid",
                "details": exc.details,
                "raw_preview": exc.raw_preview,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "resume_parse_failed",
                "detail": str(exc),
            },
        )
    return JSONResponse(status_code=200, content={"resume_json": resume_json})


@app.post("/parse-job")
def parse_job(
    payload: ParseJobRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = schema_loader
    try:
        job_json = extract_job_json(payload.job_text, provider, url=payload.url)
    except JobParseError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "job_json_invalid",
                "details": exc.details,
                "raw_preview": exc.raw_preview,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "job_parse_failed",
                "detail": str(exc),
            },
        )
    return JSONResponse(status_code=200, content={"job_json": job_json})


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


@app.post("/score")
def score(payload: ScoreRequest) -> JSONResponse:
    if not isinstance(payload.resume_json, dict) or not isinstance(payload.job_json, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": "resume_json and job_json must be objects"},
        )
    try:
        step2_result = run_scoring(payload.resume_json, payload.job_json)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "score_failed", "detail": str(exc)},
        )
    step3_result = decide(step2_result, payload.job_json)
    return JSONResponse(
        status_code=200,
        content={
            "decision": step3_result["decision"],
            "score_total": step2_result["score_total"],
            "score_breakdown": step2_result["score_breakdown"],
            "must_have_coverage_percent": step2_result["must_have"]["coverage_percent"],
            "seniority_ok": step2_result["seniority"]["seniority_ok"],
            "reasons": step3_result["reasons"],
            "matched_requirements": step3_result["matched_requirements"],
            "missing_requirements": step3_result["missing_requirements"],
        },
    )
