import type { JobJSON } from "./job";
import type { ResumeJSON } from "./resume";
import type { TailoringPlan } from "./tailoring";

export type ScoreDecision = "PROCEED" | "SKIP";

export type ScoreReason = {
  code: string;
  message: string;
  details?: unknown;
};

export type ScoreRequirement = {
  requirement_id: string;
  text?: string | null;
  evidence?: unknown;
  hard_gate?: boolean;
};

export type ScoreResult = {
  decision: ScoreDecision;
  score_total: number;
  score_breakdown: Record<string, number>;
  must_have_coverage_percent?: number;
  seniority_ok?: boolean;
  reasons?: ScoreReason[];
  matched_requirements?: ScoreRequirement[];
  missing_requirements?: ScoreRequirement[];
  scoring_mode?: string;
};

export type TailoredResumeJSON = ResumeJSON;

export type RenderDocxResult = {
  blob?: Blob;
  base64?: string;
  bytes?: number[];
  fileName?: string;
  mimeType?: string;
  objectUrl?: string | null;
};

export type JobProcessingStatus =
  | "queued"
  | "parsing_job"
  | "scoring"
  | "skipped"
  | "tailoring"
  | "enforcing_budgets"
  | "rendering_docx"
  | "complete"
  | "failed";

export type JobRunState = {
  jobId: string;
  status: JobProcessingStatus;
  parsedJob: JobJSON | null;
  scoreResult: ScoreResult | null;
  tailoringPlan: TailoringPlan | null;
  tailoredResume: TailoredResumeJSON | null;
  finalResume: TailoredResumeJSON | null;
  renderResult: RenderDocxResult | null;
  error: string | null;
  note: string | null;
};
