"use client";

import type { JobEntry } from "../../types/intake";

type JobInputCardProps = {
  index: number;
  job: JobEntry;
  urlError: string | null;
  disableRemove: boolean;
  onChange: (id: string, next: Partial<JobEntry>) => void;
  onRemove: (id: string) => void;
};

export function JobInputCard({
  index,
  job,
  urlError,
  disableRemove,
  onChange,
  onRemove,
}: JobInputCardProps) {
  return (
    <div className="job-card">
      <div className="job-header">
        <h3>{`Job ${index + 1}`}</h3>
        <button
          type="button"
          className="button ghost"
          onClick={() => onRemove(job.id)}
          disabled={disableRemove}
        >
          Remove
        </button>
      </div>
      <div className="field">
        <label htmlFor={`job-url-${job.id}`}>Job URL</label>
        <input
          id={`job-url-${job.id}`}
          type="url"
          placeholder="https://company.com/jobs/role"
          value={job.url}
          onChange={(event) => onChange(job.id, { url: event.target.value })}
        />
        {urlError ? <p className="error">{urlError}</p> : null}
        <p className="helper">Optional source URL for reference.</p>
      </div>
      <div className="field">
        <label htmlFor={`job-fallback-${job.id}`}>Job description</label>
        <textarea
          id={`job-fallback-${job.id}`}
          placeholder="Paste the job description text"
          value={job.fallbackText}
          onChange={(event) =>
            onChange(job.id, { fallbackText: event.target.value })
          }
        />
        <p className="helper">Required. URL-only intake is not enabled in the current frontend flow.</p>
      </div>
    </div>
  );
}
