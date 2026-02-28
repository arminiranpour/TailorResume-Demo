export type ResumeJSON = {
  meta?: {
    name?: string;
    email?: string;
    phone?: string;
    location?: string;
    links?: string[];
  };
  summary?: string;
  skills?: string[];
  experience?: Array<{
    company?: string;
    title?: string;
    start_date?: string;
    end_date?: string;
    bullets?: string[];
  }>;
  education?: Array<{
    school?: string;
    degree?: string;
    start_date?: string;
    end_date?: string;
    bullets?: string[];
  }>;
  projects?: Array<{
    name?: string;
    description?: string;
    bullets?: string[];
    tech?: string[];
  }>;
};
