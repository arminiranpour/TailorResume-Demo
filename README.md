# TailorResume Demo – Deterministic Resume-to-Job Pipeline

## What This Is
A deterministic, multi-step AI pipeline that matches resumes to jobs and tailors resumes *without* altering layout or inventing facts. This repo scaffolds the architecture only; business logic will be implemented in later phases.

## Why Two Services
- **frontend/** (Next.js) handles UI + API routes.
- **ai-service/** (FastAPI) runs the deterministic pipeline and provider abstraction.
This keeps UI concerns isolated from model and document handling, and allows swapping model providers without rewriting the pipeline.

## Injection Safety Stance
All job pages are untrusted input. Extraction, parsing, and prompt construction must guard against prompt injection. Only validated, schema-conformant content may enter the pipeline.

## No-Fabrication Policy
The system must never invent experience, education, employers, dates, certifications, tools, or metrics. Missing requirements must be flagged explicitly as **Missing Requirement**.

## Strict DOCX Preservation
Output must preserve fonts, font sizes, margins, layout, bullet counts, section order, and page count. If the input is 1 page, the output must remain 1 page. No new bullets or sections.

## High-Level Pipeline
1. Ingest resume (DOCX or text).
2. Ingest job URLs (company sites only) and extract job descriptions (fallback: pasted text).
3. Convert resume + job into strict, schema-validated JSON.
4. Deterministic fit scoring in code.
5. Decision: **PROCEED** (tailor) or **SKIP**.
6. If PROCEED: tailor existing resume text only.
7. Output DOCX that preserves formatting and a summary report.

## Provider Abstraction Plan
The pipeline uses a provider interface so the same prompts and schemas work with:
- **LocalProvider** (Ollama for demo)
- **ApiProvider** (OpenAI/Anthropic/etc. after demo)

## Local Dev Quickstart (Placeholders)
- Frontend (Next.js):
  - `cd frontend`
  - `pnpm install`
  - `pnpm dev`
- AI Service (FastAPI):
  - `cd ai-service`
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
  - `uvicorn app.main:app --reload`

## Repository Layout
```
.
├─ frontend/             # Next.js UI + API routes
├─ ai-service/           # FastAPI service + provider abstraction
├─ shared/               # Schemas, prompts, and deterministic scoring docs
├─ README.md
├─ LICENSE
└─ .gitignore
```

## Phase Milestones
1. **Architecture** (this phase)
2. **Extraction** (job/resume ingestion + safety)
3. **Scoring** (deterministic fit scoring)
4. **Tailoring** (rewrite existing text only)
5. **DOCX Engine** (preserve formatting/layout/page count)
6. **UI Polish**
7. **Provider Switch** (Local → API without rewriting)

## Testing Philosophy (Later Phases)
Tests will focus on determinism, schema validation, DOCX preservation, and regression checks against prompt-injection attempts.

## License
MIT. See `LICENSE`.
