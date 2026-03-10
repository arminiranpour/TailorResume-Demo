"use client";

import type { JobEntry } from "../../types/intake";
import type { JobRunState, JobProcessingStatus } from "../../types/pipeline";

const statusLabels: Record<JobProcessingStatus, string> = {
  queued: "Queued",
  parsing_job: "Parsing job",
  scoring: "Scoring",
  skipped: "Skipped",
  tailoring: "Tailoring",
  enforcing_budgets: "Enforcing budgets",
  rendering_docx: "Rendering DOCX",
  complete: "Complete",
  failed: "Failed",
};

const getStatusTone = (status: JobProcessingStatus): string => {
  if (status === "failed") {
    return "badge danger";
  }
  if (status === "complete") {
    return "badge success";
  }
  if (status === "skipped") {
    return "badge warn";
  }
  return "badge";
};

type JobRunStatusCardProps = {
  job: JobEntry;
  run: JobRunState;
  isActive: boolean;
};

export function JobRunStatusCard({
  job,
  run,
  isActive,
}: JobRunStatusCardProps) {
  const label = run.parsedJob?.title || job.url || job.id;
  const company = run.parsedJob?.company || "";
  const score = run.scoreResult?.score_total;
  const decision = run.scoreResult?.decision;
  const docxReady = Boolean(run.renderResult?.objectUrl || run.renderResult?.blob || run.renderResult?.base64);

  return (
    <div className="job-card">
      <div className="job-header">
        <div className="stack">
          <h3>{label}</h3>
          {company ? <p className="helper">{company}</p> : null}
        </div>
        <span className={getStatusTone(run.status)}>{statusLabels[run.status]}</span>
      </div>
      <div className="status-grid">
        <p className="helper">{isActive ? "Processing..." : ""}</p>
        <p className="helper">Decision: {decision ?? "—"}</p>
        <p className="helper">Score: {score ?? "—"}</p>
        <p className="helper">DOCX: {docxReady ? "Ready" : "—"}</p>
      </div>
      {run.error ? <p className="error">{run.error}</p> : null}
      {run.scoreResult?.reasons?.length ? (
        <p className="helper">{run.scoreResult.reasons.join(" ")}</p>
      ) : null}
    </div>
  );
}
