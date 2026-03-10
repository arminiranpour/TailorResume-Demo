import { apiClient } from "./apiClient";
import { normalizeError } from "./errors";
import type { JobEntry } from "../types/intake";
import type {
  JobProcessingStatus,
  JobRunState,
  RenderDocxResult,
  ScoreResult,
} from "../types/pipeline";
import type { ResumeJSON } from "../types";
import { buildObjectUrlFromDocx } from "./file";
import { resolveJobParsePayload } from "./pipelineUtils";

export type PipelineCallbacks = {
  onJobUpdate: (jobId: string, patch: Partial<JobRunState>) => void;
  onActiveJob: (jobId: string | null) => void;
};

const initialJobRun = (jobId: string): JobRunState => ({
  jobId,
  status: "queued",
  parsedJob: null,
  scoreResult: null,
  tailoringPlan: null,
  tailoredResume: null,
  finalResume: null,
  renderResult: null,
  error: null,
});

export function initializeJobRuns(jobs: JobEntry[]): JobRunState[] {
  return jobs.map((job) => initialJobRun(job.id));
}

const safeErrorMessage = (error: unknown): string => normalizeError(error);

const mapRenderResult = (result: RenderDocxResult): RenderDocxResult => {
  const objectUrl = buildObjectUrlFromDocx(result);
  return {
    ...result,
    objectUrl,
  };
};

const isProceedDecision = (scoreResult: ScoreResult | null): boolean => {
  return scoreResult?.decision === "PROCEED";
};

export async function processJobsSequentially(
  jobs: JobEntry[],
  resumeJson: ResumeJSON,
  docxBase64: string | undefined,
  callbacks: PipelineCallbacks
): Promise<void> {
  for (const job of jobs) {
    callbacks.onActiveJob(job.id);
    callbacks.onJobUpdate(job.id, { status: "parsing_job", error: null });

    try {
      const payload = resolveJobParsePayload(job);
      const parsedJob = await apiClient.parseJob(payload.jobText, payload.url);
      callbacks.onJobUpdate(job.id, { parsedJob });

      callbacks.onJobUpdate(job.id, { status: "scoring" });
      const scoreResult = await apiClient.scoreJob(resumeJson, parsedJob);
      callbacks.onJobUpdate(job.id, { scoreResult });

      if (!isProceedDecision(scoreResult)) {
        callbacks.onJobUpdate(job.id, {
          status: "skipped",
        });
        continue;
      }

      callbacks.onJobUpdate(job.id, { status: "tailoring" });
      const tailoringPlan = await apiClient.createTailoringPlan(
        resumeJson,
        parsedJob,
        scoreResult
      );
      callbacks.onJobUpdate(job.id, { tailoringPlan });

      const rewriteResult = await apiClient.rewriteBullets(
        resumeJson,
        parsedJob,
        scoreResult,
        tailoringPlan
      );
      callbacks.onJobUpdate(job.id, { tailoredResume: rewriteResult.tailored_resume_json });

      callbacks.onJobUpdate(job.id, { status: "enforcing_budgets" });
      const budgetResult = await apiClient.enforceBudgets(
        resumeJson,
        rewriteResult.tailored_resume_json,
        scoreResult
      );
      callbacks.onJobUpdate(job.id, { finalResume: budgetResult.final_resume_json });

      callbacks.onJobUpdate(job.id, { status: "rendering_docx" });
      const renderResult = await apiClient.renderDocx({
        original_resume_json: resumeJson,
        final_resume_json: budgetResult.final_resume_json,
        original_docx_base64: docxBase64,
        original_docx_name: job.id,
      });

      const mapped = mapRenderResult(renderResult);

      callbacks.onJobUpdate(job.id, {
        renderResult: mapped,
        status: mapped.objectUrl ? "complete" : "failed",
        error: mapped.objectUrl ? null : "DOCX render did not return a file.",
      });
      if (!mapped.objectUrl) {
        continue;
      }
    } catch (error) {
      callbacks.onJobUpdate(job.id, {
        status: "failed",
        error: safeErrorMessage(error),
      });
      continue;
    }
  }

  callbacks.onActiveJob(null);
}
