"use client";

import { useId, useRef } from "react";
import type { ChangeEvent } from "react";
import {
  ACCEPTED_RESUME_EXTENSIONS,
  MAX_RESUME_FILE_BYTES,
} from "../../lib/constants";
import { formatBytes } from "../../lib/validation";

type ResumeUploadProps = {
  file: File | null;
  error: string | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
};

export function ResumeUpload({
  file,
  error,
  onSelect,
  onRemove,
}: ResumeUploadProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    if (selected) {
      onSelect(selected);
    }
    event.target.value = "";
  };

  const handleRemove = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onRemove();
  };

  return (
    <section className="card">
      <div className="section-header">
        <h2>Resume Upload</h2>
        <p className="muted">
          {`Accepted formats: ${ACCEPTED_RESUME_EXTENSIONS.join(
            ", "
          )}. Max size ${formatBytes(MAX_RESUME_FILE_BYTES)}. Use .docx for downloadable output; .txt supports analysis only.`}
        </p>
      </div>
      <div className="field">
        <label htmlFor={inputId}>Resume file</label>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPTED_RESUME_EXTENSIONS.join(",")}
          onChange={handleChange}
        />
        {file ? (
          <div className="file-row">
            <span className="file-pill">{file.name}</span>
            <span className="helper">{formatBytes(file.size)}</span>
            <button
              type="button"
              className="button ghost"
              onClick={handleRemove}
            >
              Remove
            </button>
          </div>
        ) : (
          <p className="helper">Choose a single resume file to continue.</p>
        )}
        {error ? <p className="error">{error}</p> : null}
      </div>
    </section>
  );
}
