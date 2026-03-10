"use client";

import { MAX_JOBS, MIN_JOBS } from "../../lib/constants";
import { getUrlError } from "../../lib/validation";
import type { JobEntry } from "../../types/intake";
import { JobInputCard } from "./JobInputCard";

type JobInputsSectionProps = {
  jobs: JobEntry[];
  error: string | null;
  onAdd: () => void;
  onRemove: (id: string) => void;
  onChange: (id: string, next: Partial<JobEntry>) => void;
};

export function JobInputsSection({
  jobs,
  error,
  onAdd,
  onRemove,
  onChange,
}: JobInputsSectionProps) {
  const canAdd = jobs.length < MAX_JOBS;
  const canRemove = jobs.length > MIN_JOBS;

  return (
    <section className="card">
      <div className="section-header">
        <h2>Job Inputs</h2>
        <p className="muted">Add up to {MAX_JOBS} job targets.</p>
      </div>
      <div className="job-grid">
        {jobs.map((job, index) => (
          <JobInputCard
            key={job.id}
            index={index}
            job={job}
            urlError={getUrlError(job.url)}
            disableRemove={!canRemove}
            onChange={onChange}
            onRemove={onRemove}
          />
        ))}
      </div>
      <div className="actions" style={{ marginTop: 16 }}>
        <button
          type="button"
          className="button secondary"
          onClick={onAdd}
          disabled={!canAdd}
        >
          Add Job
        </button>
        <p className="helper">{`Minimum ${MIN_JOBS}. Maximum ${MAX_JOBS}.`}</p>
      </div>
      {error ? <p className="error" style={{ marginTop: 12 }}>{error}</p> : null}
    </section>
  );
}
