from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

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
def parse_resume(_: ParseResumeRequest) -> JSONResponse:
    return _not_implemented("parse-resume endpoint is a Phase 0 stub")


@app.post("/parse-job")
def parse_job(_: ParseJobRequest) -> JSONResponse:
    return _not_implemented("parse-job endpoint is a Phase 0 stub")


@app.post("/repair-json")
def repair_json(_: RepairJsonRequest) -> JSONResponse:
    return _not_implemented("repair-json endpoint is a Phase 0 stub")


@app.post("/tailor")
def tailor(_: TailorRequest) -> JSONResponse:
    return _not_implemented("tailor endpoint is a Phase 0 stub")
