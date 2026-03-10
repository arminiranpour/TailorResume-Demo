import type { JobEntry } from "../types/intake";

export type ValidationResult = {
  ok: boolean;
  error?: string;
};

export type JobParsePayload = {
  jobText: string;
  url?: string;
};

const JOB_TEXT_REQUIRED_MESSAGE =
  "Job description text is required until URL extraction is implemented.";

export function validateJobInputs(jobs: JobEntry[]): ValidationResult {
  const invalid = jobs.find((job) => !job.url.trim() && !job.fallbackText.trim());
  if (invalid) {
    return {
      ok: false,
      error: "Each job needs a URL or fallback description.",
    };
  }
  return { ok: true };
}

export function resolveJobParsePayload(job: JobEntry): JobParsePayload {
  const jobText = job.fallbackText.trim();
  const url = job.url.trim();
  if (jobText) {
    return { jobText, url: url || undefined };
  }
  if (url) {
    throw new Error(JOB_TEXT_REQUIRED_MESSAGE);
  }
  throw new Error("Job description text is required.");
}
