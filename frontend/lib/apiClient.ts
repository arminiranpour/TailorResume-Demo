import type { JobJSON, ResumeJSON } from "../types";

const baseUrl =
  process.env.NEXT_PUBLIC_AI_SERVICE_URL ?? "http://localhost:8000";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (response.status === 501) {
    throw new Error("Not implemented");
  }

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function parseResume(resumeText: string): Promise<ResumeJSON> {
  return postJson<ResumeJSON>("/parse-resume", { resume_text: resumeText });
}

export async function parseJob(
  jobText: string,
  url?: string
): Promise<JobJSON> {
  return postJson<JobJSON>("/parse-job", { job_text: jobText, url });
}

export async function repairJson(
  raw: string,
  schemaName: string
): Promise<unknown> {
  return postJson<unknown>("/repair-json", {
    raw,
    schema_name: schemaName,
  });
}

export async function tailor(
  resumeJson: ResumeJSON,
  jobJson: JobJSON
): Promise<Blob> {
  const response = await fetch(`${baseUrl}/tailor`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_json: resumeJson, job_json: jobJson }),
  });

  if (response.status === 501) {
    throw new Error("Not implemented");
  }

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.blob();
}

export const apiClient = {
  parseResume,
  parseJob,
  repairJson,
  tailor,
};
