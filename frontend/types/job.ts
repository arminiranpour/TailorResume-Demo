export type JobJSON = {
  title?: string;
  company?: string;
  location?: string;
  remote?: boolean | null;
  seniority?: string;
  must_have?: string[];
  nice_to_have?: string[];
  responsibilities?: string[];
  keywords?: string[];
  source_url?: string;
};
