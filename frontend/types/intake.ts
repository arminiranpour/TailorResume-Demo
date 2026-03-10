export type JobEntry = {
  id: string;
  url: string;
  fallbackText: string;
};

export type ResumeFileMeta = {
  name: string;
  size: number;
  type: string;
  lastModified: number;
};

export type ResumeIntakePayload = {
  resumeFile: ResumeFileMeta | null;
  jobs: JobEntry[];
};
