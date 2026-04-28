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
  "Paste the job description text. Job URL is optional metadata only right now.";

export function validateJobInputs(jobs: JobEntry[]): ValidationResult {
  const invalid = jobs.find((job) => !job.fallbackText.trim());
  if (invalid) {
    return {
      ok: false,
      error: "Each job needs pasted job description text. Job URL is optional.",
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
  throw new Error(JOB_TEXT_REQUIRED_MESSAGE);
}
