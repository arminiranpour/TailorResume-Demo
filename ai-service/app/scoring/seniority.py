from typing import Any, Dict, List

from .normalize import normalize_text

JUNIOR_TOKENS = ["intern", "internship", "student", "co-op", "coop", "junior", "jr"]
SENIOR_TOKENS = [
    "senior",
    "sr",
    "lead",
    "staff",
    "principal",
    "architect",
    "manager",
    "director",
    "head",
]


def _contains_token(normalized_text: str, token: str) -> bool:
    token_norm = normalize_text(token)
    if not token_norm:
        return False
    haystack = f" {normalized_text} "
    needle = f" {token_norm} "
    if " " in token_norm:
        return needle in haystack
    return token_norm in normalized_text.split()


def _has_any_token(texts: List[str], tokens: List[str]) -> bool:
    for text in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        for token in tokens:
            if _contains_token(normalized, token):
                return True
    return False


def detect_resume_seniority(resume_json: Dict[str, Any]) -> str:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    summary = resume_json.get("summary")
    if summary is None:
        raise ValueError("resume_json missing required field: summary")
    if "text" not in summary:
        raise ValueError("summary missing required field: text")
    experience = resume_json.get("experience")
    if experience is None:
        raise ValueError("resume_json missing required field: experience")

    texts: List[str] = []
    summary_text = summary.get("text")
    if summary_text:
        texts.append(summary_text)
    for exp in experience:
        if "title" not in exp:
            raise ValueError("experience entry missing required field: title")
        title = exp.get("title")
        if title:
            texts.append(title)

    if _has_any_token(texts, JUNIOR_TOKENS):
        return "junior"
    if _has_any_token(texts, SENIOR_TOKENS):
        return "senior"
    return "mid"


def normalize_job_seniority(job_json: Dict[str, Any]) -> str:
    if not isinstance(job_json, dict):
        raise ValueError("job_json must be a dict")
    value = job_json.get("seniority")
    if value is None:
        return "unknown"
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    if normalized in {"junior", "mid", "senior", "unknown"}:
        return normalized
    return "unknown"


def compute_seniority_alignment(job_seniority: str, resume_level: str) -> Dict[str, Any]:
    seniority_ok = not (job_seniority == "senior" and resume_level != "senior")
    if job_seniority == "unknown":
        alignment_points = 5
    elif job_seniority == resume_level:
        alignment_points = 10
    elif job_seniority == "mid" and resume_level == "senior":
        alignment_points = 7
    elif job_seniority == "mid" and resume_level == "junior":
        alignment_points = 3
    elif job_seniority == "junior" and resume_level == "mid":
        alignment_points = 7
    elif job_seniority == "junior" and resume_level == "senior":
        alignment_points = 6
    elif job_seniority == "senior" and resume_level != "senior":
        alignment_points = 0
    else:
        alignment_points = 0
    return {
        "seniority_ok": seniority_ok,
        "alignment_points": int(alignment_points),
        "job_seniority": job_seniority,
        "resume_level": resume_level,
    }
