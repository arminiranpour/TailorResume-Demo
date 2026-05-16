import base64
import tempfile
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
from app.pipelines.llm_scoring import run_llm_adjudicated_scoring
from app.pipelines.resume_parser import ResumeParseError, extract_resume_json
from app.pipelines.scoring import run_ats_scoring
from app.pipelines.tailoring_plan import TailorNotAllowed, TailoringPlanError, generate_tailoring_plan
from app.pipelines.allowed_vocab import build_allowed_vocab
from app.pipelines.bullet_rewrite import (
    BulletRewriteError,
    BulletRewriteNotAllowed,
    rewrite_resume_text_with_audit,
)
from app.pipelines.budget_enforcement import BudgetEnforcementError, enforce_budgets
from app.docx_engine.editor import apply_tailored_text_to_docx
from app.docx_engine.mapping import build_docx_mapping
from app.docx_engine.types import DocxReplacementError
from app.utils.debug_report import print_tailoring_debug_report
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
            timeout=config.llm_timeout_seconds,
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


class TailorPlanRequest(BaseModel):
    resume_json: Dict[str, Any]
    job_json: Dict[str, Any]
    score_result: Dict[str, Any]


class ScoreRequest(BaseModel):
    resume_json: Dict[str, Any]
    job_json: Dict[str, Any]


class RewriteBulletsRequest(BaseModel):
    resume_json: Dict[str, Any]
    job_json: Dict[str, Any]
    score_result: Dict[str, Any]
    tailoring_plan: Dict[str, Any]
    character_budgets: Optional[Dict[str, Any]] = None


class EnforceBudgetsRequest(BaseModel):
    original_resume_json: Dict[str, Any]
    tailored_resume_json: Dict[str, Any]
    score_result: Dict[str, Any]
    character_budgets: Optional[Dict[str, Any]] = None


class RenderDocxRequest(BaseModel):
    original_resume_json: Dict[str, Any]
    final_resume_json: Dict[str, Any]
    original_docx_base64: Optional[str] = None
    original_docx_name: Optional[str] = None


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


@app.post("/tailor-plan")
def tailor_plan(
    payload: TailorPlanRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = schema_loader
    try:
        plan = generate_tailoring_plan(
            payload.resume_json,
            payload.job_json,
            payload.score_result,
            provider,
        )
    except TailorNotAllowed as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": "tailor_not_allowed",
                "detail": str(exc),
                "decision": exc.decision,
                "reasons": exc.reasons,
            },
        )
    except TailoringPlanError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "tailoring_plan_invalid",
                "details": exc.details,
                "raw_preview": exc.raw_preview,
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "tailoring_plan_failed", "detail": str(exc)},
        )
    return JSONResponse(status_code=200, content={"tailoring_plan": plan})


@app.post("/rewrite-bullets")
def rewrite_bullets(
    payload: RewriteBulletsRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = schema_loader
    try:
        tailored_resume_json, audit_log = rewrite_resume_text_with_audit(
            payload.resume_json,
            payload.job_json,
            payload.score_result,
            payload.tailoring_plan,
            provider,
            character_budgets=payload.character_budgets,
        )
    except BulletRewriteNotAllowed as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": "rewrite_not_allowed",
                "detail": str(exc),
                "decision": exc.decision,
                "reasons": exc.reasons,
            },
        )
    except BulletRewriteError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "bullet_rewrite_invalid",
                "details": exc.details,
                "raw_preview": exc.raw_preview,
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "bullet_rewrite_failed", "detail": str(exc)},
        )
    print_tailoring_debug_report(
        job_json=payload.job_json,
        resume_json=payload.resume_json,
        tailored_resume_json=tailored_resume_json,
        score_result=payload.score_result,
    )
    return JSONResponse(
        status_code=200,
        content={"tailored_resume_json": tailored_resume_json, "audit_log": audit_log},
    )


@app.post("/enforce-budgets")
def enforce_budgets_endpoint(
    payload: EnforceBudgetsRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    schema_loader: Callable[[str], Dict[str, Any]] = Depends(get_schema_loader),
) -> JSONResponse:
    _ = schema_loader
    decision = payload.score_result.get("decision") if isinstance(payload.score_result, dict) else None
    if decision != "PROCEED":
        reasons = payload.score_result.get("reasons") if isinstance(payload.score_result, dict) else []
        if not isinstance(reasons, list):
            reasons = []
        return JSONResponse(
            status_code=409,
            content={
                "error": "budget_enforcement_not_allowed",
                "detail": f"Budget enforcement not allowed for decision={decision}",
                "decision": decision,
                "reasons": reasons,
            },
        )
    try:
        final_resume_json, budget_report, audit_log = enforce_budgets(
            payload.original_resume_json,
            payload.tailored_resume_json,
            provider,
            build_allowed_vocab(payload.original_resume_json),
            budgets_override=payload.character_budgets,
        )
    except BudgetEnforcementError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "budget_enforcement_invalid",
                "details": exc.details,
                "raw_preview": exc.raw_preview,
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "budget_enforcement_failed", "detail": str(exc)},
        )
    return JSONResponse(
        status_code=200,
        content={
            "final_resume_json": final_resume_json,
            "budgets": budget_report["budgets"],
            "size_report": budget_report["size_report"],
            "audit_log": audit_log,
        },
    )


@app.post("/render-docx")
def render_docx(payload: RenderDocxRequest) -> JSONResponse:
    if not payload.original_docx_base64:
        return JSONResponse(
            status_code=422,
            content={
                "error": "missing_docx_base64",
                "detail": "original_docx_base64 is required",
            },
        )

    try:
        original_bytes = base64.b64decode(payload.original_docx_base64)
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_docx_base64", "detail": str(exc)},
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=".docx") as docx_file:
            docx_file.write(original_bytes)
            docx_file.flush()
            mapping = build_docx_mapping(docx_file.name, payload.original_resume_json)
            result = apply_tailored_text_to_docx(
                docx_file.name,
                payload.original_resume_json,
                payload.final_resume_json,
                mapping,
            )
    except DocxReplacementError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "docx_render_invalid",
                "detail": str(exc),
                "details": exc.details,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "docx_render_failed", "detail": str(exc)},
        )

    docx_bytes = result.get("docx_bytes")
    if not isinstance(docx_bytes, (bytes, bytearray)):
        return JSONResponse(
            status_code=500,
            content={
                "error": "docx_render_failed",
                "detail": "DOCX bytes missing from render output",
            },
        )

    file_name = payload.original_docx_name or "tailored"
    if not file_name.endswith(".docx"):
        file_name = f"{file_name}.docx"

    return JSONResponse(
        status_code=200,
        content={
            "docx_base64": base64.b64encode(docx_bytes).decode("utf-8"),
            "file_name": file_name,
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
    )


@app.post("/score")
def score(payload: ScoreRequest) -> JSONResponse:
    if not isinstance(payload.resume_json, dict) or not isinstance(payload.job_json, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": "resume_json and job_json must be objects"},
        )
    try:
        result = run_ats_scoring(payload.resume_json, payload.job_json)
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
    return JSONResponse(status_code=200, content=result)


@app.post("/score/llm")
def score_llm(payload: ScoreRequest) -> JSONResponse:
    if not isinstance(payload.resume_json, dict) or not isinstance(payload.job_json, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": "resume_json and job_json must be objects"},
        )

    try:
        provider = get_llm_provider()
    except Exception:
        provider = None

    try:
        result = run_llm_adjudicated_scoring(payload.resume_json, payload.job_json, provider)
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
    return JSONResponse(status_code=200, content=result)
