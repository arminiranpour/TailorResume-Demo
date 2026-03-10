"use client";

import { useMemo, useRef, useState } from "react";
import { JobInputsSection } from "../components/jobs/JobInputsSection";
import { JobRunStatusCard } from "../components/jobs/JobRunStatusCard";
import { ResumeUpload } from "../components/resume/ResumeUpload";
import {
  ResultsDashboard,
  type JobResult,
} from "../components/dashboard/ResultsDashboard";
import { apiClient } from "../lib/apiClient";
import { readResumeInput } from "../lib/file";
import { processJobsSequentially, initializeJobRuns } from "../lib/pipeline";
import { validateJobInputs } from "../lib/pipelineUtils";
import { MAX_JOBS, MIN_JOBS } from "../lib/constants";
import { getJobsCountError, validateResumeFile } from "../lib/validation";
import type { JobEntry } from "../types/intake";
import type { JobProcessingStatus, JobRunState } from "../types/pipeline";
import type { ResumeJSON } from "../types";

const createJobEntry = (id: string): JobEntry => ({
  id,
  url: "",
  fallbackText: "",
});

export default function HomePage() {
  const nextJobId = useRef(2);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobEntry[]>(() => [createJobEntry("job-1")]);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [parsedResume, setParsedResume] = useState<ResumeJSON | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobRuns, setJobRuns] = useState<JobRunState[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [resumeDocxBase64, setResumeDocxBase64] = useState<string | null>(null);

  const handleSelectResume = (file: File) => {
    const error = validateResumeFile(file);
    if (error) {
      setResumeFile(null);
      setResumeError(error);
      return;
    }
    setResumeFile(file);
    setResumeError(null);
  };

  const handleRemoveResume = () => {
    setResumeFile(null);
    setResumeError(null);
  };

  const handleAddJob = () => {
    setJobs((prev) => {
      if (prev.length >= MAX_JOBS) {
        return prev;
      }
      const next = [...prev, createJobEntry(`job-${nextJobId.current}`)];
      nextJobId.current += 1;
      return next;
    });
    setJobsError(null);
  };

  const handleRemoveJob = (id: string) => {
    setJobs((prev) => {
      if (prev.length <= MIN_JOBS) {
        return prev;
      }
      return prev.filter((job) => job.id !== id);
    });
    setJobsError(null);
  };

  const handleChangeJob = (id: string, next: Partial<JobEntry>) => {
    setJobs((prev) =>
      prev.map((job) => (job.id === id ? { ...job, ...next } : job))
    );
  };

  const jobRunsById = useMemo(() => {
    return new Map(jobRuns.map((run) => [run.jobId, run]));
  }, [jobRuns]);

  const mapRunStatus = (
    status: JobProcessingStatus
  ): JobResult["status"] => {
    if (status === "complete" || status === "failed" || status === "skipped") {
      return status;
    }
    return "processing";
  };

  const dashboardJobs = useMemo<JobResult[]>(() => {
    if (jobRuns.length === 0) {
      return [];
    }

    return jobs
      .map((job) => {
        const run = jobRunsById.get(job.id);
        if (!run) {
          return null;
        }

        return {
          id: job.id,
          url: job.url,
          title: run.parsedJob?.title ?? undefined,
          company: run.parsedJob?.company ?? undefined,
          score: run.scoreResult?.score_total ?? undefined,
          coverage: run.scoreResult?.must_have_coverage_percent ?? undefined,
          decision: run.scoreResult?.decision ?? undefined,
          status: mapRunStatus(run.status),
          docxUrl: run.renderResult?.objectUrl ?? undefined,
        };
      })
      .filter((job): job is JobResult => Boolean(job));
  }, [jobRuns.length, jobRunsById, jobs]);

  const handleAnalyze = async () => {
    let hasError = false;

    if (!resumeFile) {
      setResumeError("Resume file is required.");
      hasError = true;
    } else {
      const error = validateResumeFile(resumeFile);
      if (error) {
        setResumeError(error);
        hasError = true;
      } else {
        setResumeError(null);
      }
    }

    const countError = getJobsCountError(jobs.length);
    if (countError) {
      setJobsError(countError);
      hasError = true;
    } else {
      setJobsError(null);
    }

    const jobInputValidation = validateJobInputs(jobs);
    if (!jobInputValidation.ok) {
      setJobsError(jobInputValidation.error ?? "Job inputs are incomplete.");
      hasError = true;
    }

    if (hasError) {
      return;
    }

    if (!resumeFile) {
      setResumeError("Resume file is required.");
      return;
    }

    setIsProcessing(true);
    setRunError(null);
    setJobRuns(initializeJobRuns(jobs));

    try {
      const resumeInput = await readResumeInput(resumeFile);
      setResumeDocxBase64(resumeInput.docxBase64 ?? null);

      const resumeJson = await apiClient.parseResume(
        resumeInput.text,
        resumeInput.docxBase64
      );
      setParsedResume(resumeJson);

      await processJobsSequentially(jobs, resumeJson, resumeInput.docxBase64, {
        onActiveJob: setActiveJobId,
        onJobUpdate: (jobId, patch) => {
          setJobRuns((prev) =>
            prev.map((run) => (run.jobId === jobId ? { ...run, ...patch } : run))
          );
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Resume parse failed.";
      setRunError(message);
      setResumeError(message);
    } finally {
      setIsProcessing(false);
      setActiveJobId(null);
    }
  };

  return (
    <main>
      <div className="page">
        <header className="header">
          <h1>Step 1 — Resume Intake + Job Input</h1>
          <p className="muted">
            Upload a resume and add up to three job targets. This step runs
            client-side validation only.
          </p>
        </header>

        <ResumeUpload
          file={resumeFile}
          error={resumeError}
          onSelect={handleSelectResume}
          onRemove={handleRemoveResume}
        />

        <JobInputsSection
          jobs={jobs}
          error={jobsError}
          onAdd={handleAddJob}
          onRemove={handleRemoveJob}
          onChange={handleChangeJob}
        />

        <section className="card">
          <div className="section-header">
            <h2>Analyze</h2>
            <p className="muted">
              Ready to run the deterministic pipeline once inputs are ready.
            </p>
          </div>
          <div className="actions">
            <button
              type="button"
              className="button"
              onClick={handleAnalyze}
              disabled={isProcessing}
            >
              {isProcessing ? "Processing jobs..." : "Analyze Jobs"}
            </button>
            <p className="helper">
              {isProcessing
                ? "Pipeline running sequentially."
                : "Runs the deterministic pipeline in order."}
            </p>
          </div>
          {runError ? <p className="error">{runError}</p> : null}
        </section>

        {jobRuns.length ? (
          <section className="card">
            <div className="section-header">
              <h2>Step 2 — Pipeline Execution</h2>
              <p className="muted">Live status for each job in the batch.</p>
            </div>
            <div className="job-grid">
              {jobs.map((job) => {
                const run = jobRunsById.get(job.id);
                if (!run) {
                  return null;
                }
                return (
                  <JobRunStatusCard
                    key={job.id}
                    job={job}
                    run={run}
                    isActive={activeJobId === job.id}
                  />
                );
              })}
            </div>
            {parsedResume ? (
              <p className="helper" style={{ marginTop: 12 }}>
                Resume parsed once and reused for all jobs.
              </p>
            ) : null}
            {resumeDocxBase64 ? (
              <p className="helper">DOCX payload ready for rendering.</p>
            ) : null}
          </section>
        ) : null}

        <ResultsDashboard jobs={dashboardJobs} />
      </div>
    </main>
  );
}
