from typing import Any, Dict

from app.scoring import score_job


def run_scoring(resume_json: Dict[str, Any], job_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    if not isinstance(job_json, dict):
        raise ValueError("job_json must be a dict")
    if "summary" not in resume_json or resume_json["summary"] is None:
        raise ValueError("resume_json missing required field: summary")
    if "skills" not in resume_json or resume_json["skills"] is None:
        raise ValueError("resume_json missing required field: skills")
    if "experience" not in resume_json or resume_json["experience"] is None:
        raise ValueError("resume_json missing required field: experience")
    if "must_have" not in job_json or job_json["must_have"] is None:
        raise ValueError("job_json missing required field: must_have")
    if "nice_to_have" not in job_json or job_json["nice_to_have"] is None:
        raise ValueError("job_json missing required field: nice_to_have")
    return score_job(resume_json, job_json)


def score_fit(resume: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    return run_scoring(resume, job)
