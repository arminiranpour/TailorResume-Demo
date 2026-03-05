export type JobRequirement = {
  requirement_id: string;
  text: string;
};

export type JobJSON = {
  title?: string | null;
  company?: string | null;
  location?: string | null;
  remote?: boolean | null;
  seniority?: string | null;
  must_have?: JobRequirement[];
  nice_to_have?: JobRequirement[];
  responsibilities?: string[];
  keywords?: string[] | null;
  source_url?: string | null;
};
