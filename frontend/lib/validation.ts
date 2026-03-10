import {
  ACCEPTED_RESUME_EXTENSIONS,
  ACCEPTED_RESUME_MIME_TYPES,
  MAX_RESUME_FILE_BYTES,
  MIN_JOBS,
  MAX_JOBS,
} from "./constants";

export function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) {
    return "";
  }
  return filename.slice(lastDot).toLowerCase();
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kb = bytes / 1024;
  if (kb < 1024) {
    return `${kb.toFixed(1)} KB`;
  }
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

export function validateResumeFile(file: File): string | null {
  const extension = getFileExtension(file.name);
  if (!ACCEPTED_RESUME_EXTENSIONS.includes(extension)) {
    return `File must be ${ACCEPTED_RESUME_EXTENSIONS.join(", ")}.`;
  }

  if (file.type && !ACCEPTED_RESUME_MIME_TYPES.includes(file.type)) {
    return "File type is not supported.";
  }

  if (file.size > MAX_RESUME_FILE_BYTES) {
    return `File exceeds ${formatBytes(MAX_RESUME_FILE_BYTES)}.`;
  }

  return null;
}

export function isProbablyUrl(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) {
    return true;
  }
  const normalized = /^https?:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
  try {
    const parsed = new URL(normalized);
    return Boolean(parsed.hostname && parsed.hostname.includes("."));
  } catch {
    return false;
  }
}

export function getUrlError(value: string): string | null {
  if (!value.trim()) {
    return null;
  }
  return isProbablyUrl(value) ? null : "Enter a valid job URL.";
}

export function getJobsCountError(count: number): string | null {
  if (count < MIN_JOBS) {
    return `Add at least ${MIN_JOBS} job.`;
  }
  if (count > MAX_JOBS) {
    return `Only ${MAX_JOBS} jobs are allowed.`;
  }
  return null;
}
