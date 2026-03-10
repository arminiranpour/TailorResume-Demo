import type { ReactNode } from "react";

export type JobResult = {
  id: string;
  url: string;
  title?: string;
  company?: string;
  score?: number;
  coverage?: number;
  decision?: "PROCEED" | "SKIP";
  status: "processing" | "complete" | "failed" | "skipped";
  docxUrl?: string;
};

export type ResultsDashboardProps = {
  jobs: JobResult[];
};

const formatScore = (score?: number): ReactNode => {
  if (score === undefined || score === null || Number.isNaN(score)) {
    return "-";
  }
  return score.toFixed(1);
};

const formatCoverage = (coverage?: number): ReactNode => {
  if (coverage === undefined || coverage === null || Number.isNaN(coverage)) {
    return "-";
  }
  const normalized = coverage <= 1 ? coverage * 100 : coverage;
  return `${Math.round(normalized)}%`;
};

const decisionClasses: Record<"PROCEED" | "SKIP", string> = {
  PROCEED: "bg-green-100 text-green-700",
  SKIP: "bg-gray-200 text-gray-700",
};

export function ResultsDashboard({ jobs }: ResultsDashboardProps) {
  return (
    <section className="rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium">Results Dashboard</h2>
      </div>

      {jobs.length === 0 ? (
        <p className="mt-3 text-sm text-gray-500">
          No results yet. Run the analysis to see matches.
        </p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b text-xs uppercase tracking-wide text-gray-500">
              <tr>
                <th className="px-3 py-2 font-medium">Job</th>
                <th className="px-3 py-2 font-medium">Score</th>
                <th className="px-3 py-2 font-medium">Coverage</th>
                <th className="px-3 py-2 font-medium">Decision</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Download</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const title = job.title ?? job.url ?? "Job";
                return (
                  <tr key={job.id} className="border-b last:border-b-0">
                    <td className="px-3 py-3">
                      <div className="flex flex-col gap-1">
                        <span className="font-medium text-gray-900">
                          {title}
                        </span>
                        {job.company ? (
                          <span className="text-xs text-gray-500">
                            {job.company}
                          </span>
                        ) : null}
                      </div>
                    </td>
                    <td className="px-3 py-3">{formatScore(job.score)}</td>
                    <td className="px-3 py-3">
                      {formatCoverage(job.coverage)}
                    </td>
                    <td className="px-3 py-3">
                      {job.decision ? (
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${decisionClasses[job.decision]}`}
                        >
                          {job.decision}
                        </span>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td className="px-3 py-3 capitalize">{job.status}</td>
                    <td className="px-3 py-3">
                      {job.docxUrl ? (
                        <a
                          href={job.docxUrl}
                          download
                          className="inline-flex items-center rounded-md border px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50"
                        >
                          Download Resume
                        </a>
                      ) : (
                        "-"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
