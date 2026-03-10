import type { JobJSON, ResumeJSON, TailoringPlan } from "../types";
import type { ScoreResult } from "../types/pipeline";
import { toErrorMessage } from "./errors";

const baseUrl =
  process.env.NEXT_PUBLIC_AI_SERVICE_URL ?? "http://localhost:8000";

type ApiErrorPayload = {
  error?: string;
  detail?: string;
  details?: unknown;
  raw_preview?: string;
  decision?: string;
  reasons?: string[];
};

export class ApiError extends Error {
  status: number;
  payload?: ApiErrorPayload;

  constructor(message: string, status: number, payload?: ApiErrorPayload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

type ParseResumeRequest = {
  resume_text: string;
  resume_docx_base64?: string;
};

type ParseResumeResponse = {
  resume_json: ResumeJSON;
};

type ParseJobResponse = {
  job_json: JobJSON;
};

type ScoreResponse = ScoreResult;

type TailorPlanResponse = {
  tailoring_plan: TailoringPlan;
};

type RewriteBulletsResponse = {
  tailored_resume_json: ResumeJSON;
  audit_log?: unknown;
};

type EnforceBudgetsResponse = {
  final_resume_json: ResumeJSON;
  budgets?: unknown;
  size_report?: unknown;
  audit_log?: unknown;
};

export type RenderDocxResponse = {
  blob?: Blob;
  base64?: string;
  bytes?: number[];
  fileName?: string;
  mimeType?: string;
};

type RenderDocxJsonResponse = {
  docx_base64?: string;
  docx_bytes?: number[];
  file_name?: string;
  mime_type?: string;
};

async function readJsonSafe(response: Response): Promise<ApiErrorPayload | null> {
  try {
    return (await response.json()) as ApiErrorPayload;
  } catch {
    return null;
  }
}

async function buildError(response: Response): Promise<ApiError> {
  const payload = await readJsonSafe(response);
  const rawMessage = payload?.detail ?? payload?.error;
  const message =
    rawMessage !== undefined && rawMessage !== null
      ? toErrorMessage(rawMessage)
      : `Request failed: ${response.status}`;
  return new ApiError(message, response.status, payload ?? undefined);
}

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
    throw await buildError(response);
  }

  return (await response.json()) as T;
}

export async function parseResume(
  resumeText: string,
  resumeDocxBase64?: string
): Promise<ResumeJSON> {
  const payload: ParseResumeRequest = {
    resume_text: resumeText,
    resume_docx_base64: resumeDocxBase64,
  };
  const data = await postJson<ParseResumeResponse>("/parse-resume", payload);
  if (!data || typeof data.resume_json !== "object" || data.resume_json === null) {
    throw new Error("Malformed parse-resume response.");
  }
  return data.resume_json;
}

export async function parseJob(
  jobText: string,
  url?: string
): Promise<JobJSON> {
  const data = await postJson<ParseJobResponse>("/parse-job", {
    job_text: jobText,
    url,
  });
  if (!data || typeof data.job_json !== "object" || data.job_json === null) {
    throw new Error("Malformed parse-job response.");
  }
  return data.job_json;
}

export async function scoreJob(
  resumeJson: ResumeJSON,
  jobJson: JobJSON
): Promise<ScoreResult> {
  const data = await postJson<ScoreResponse>("/score", {
    resume_json: resumeJson,
    job_json: jobJson,
  });
  if (data.decision !== "PROCEED" && data.decision !== "SKIP") {
    throw new Error("Malformed score response.");
  }
  return data;
}

export async function createTailoringPlan(
  resumeJson: ResumeJSON,
  jobJson: JobJSON,
  scoreResult: ScoreResult
): Promise<TailoringPlan> {
  const data = await postJson<TailorPlanResponse>("/tailor-plan", {
    resume_json: resumeJson,
    job_json: jobJson,
    score_result: scoreResult,
  });
  if (!data || typeof data.tailoring_plan !== "object" || data.tailoring_plan === null) {
    throw new Error("Malformed tailor-plan response.");
  }
  return data.tailoring_plan;
}

export async function rewriteBullets(
  resumeJson: ResumeJSON,
  jobJson: JobJSON,
  scoreResult: ScoreResult,
  tailoringPlan: TailoringPlan
): Promise<RewriteBulletsResponse> {
  const data = await postJson<RewriteBulletsResponse>("/rewrite-bullets", {
    resume_json: resumeJson,
    job_json: jobJson,
    score_result: scoreResult,
    tailoring_plan: tailoringPlan,
  });
  if (!data || typeof data.tailored_resume_json !== "object" || data.tailored_resume_json === null) {
    throw new Error("Malformed rewrite-bullets response.");
  }
  return data;
}

export async function enforceBudgets(
  originalResumeJson: ResumeJSON,
  tailoredResumeJson: ResumeJSON,
  scoreResult: ScoreResult
): Promise<EnforceBudgetsResponse> {
  const data = await postJson<EnforceBudgetsResponse>("/enforce-budgets", {
    original_resume_json: originalResumeJson,
    tailored_resume_json: tailoredResumeJson,
    score_result: scoreResult,
  });
  if (!data || typeof data.final_resume_json !== "object" || data.final_resume_json === null) {
    throw new Error("Malformed enforce-budgets response.");
  }
  return data;
}

export async function renderDocx(payload: {
  original_resume_json: ResumeJSON;
  final_resume_json: ResumeJSON;
  original_docx_base64?: string;
  original_docx_name?: string;
}): Promise<RenderDocxResponse> {
  const response = await fetch(`${baseUrl}/render-docx`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (response.status === 501) {
    throw new Error("Not implemented");
  }

  if (!response.ok) {
    throw await buildError(response);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const data = (await response.json()) as RenderDocxJsonResponse;
    return {
      base64: data.docx_base64,
      bytes: data.docx_bytes,
      fileName: data.file_name,
      mimeType: data.mime_type,
    };
  }

  return {
    blob: await response.blob(),
    mimeType: contentType || undefined,
  };
}

export const apiClient = {
  parseResume,
  parseJob,
  scoreJob,
  createTailoringPlan,
  rewriteBullets,
  enforceBudgets,
  renderDocx,
};
