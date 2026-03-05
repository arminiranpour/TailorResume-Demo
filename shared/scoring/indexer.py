from typing import Any, Dict

from .normalize import extract_signals


def build_resume_index(resume_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")

    if "summary" not in resume_json or resume_json["summary"] is None:
        raise ValueError("resume_json missing required field: summary")
    summary = resume_json["summary"]
    if "text" not in summary:
        raise ValueError("summary missing required field: text")

    if "skills" not in resume_json or resume_json["skills"] is None:
        raise ValueError("resume_json missing required field: skills")
    skills = resume_json["skills"]
    if "lines" not in skills or skills["lines"] is None:
        raise ValueError("skills missing required field: lines")

    if "experience" not in resume_json or resume_json["experience"] is None:
        raise ValueError("resume_json missing required field: experience")

    index: Dict[str, Any] = {
        "summary": {},
        "skills": {},
        "experience": {},
        "all_tokens": set(),
        "all_phrases": set(),
    }

    summary_text = summary.get("text", "")
    summary_signals = extract_signals(summary_text)
    index["summary"] = {
        "source_type": "summary",
        "source_id": "summary",
        "original": summary_text,
        "signals": summary_signals,
    }
    index["all_tokens"].update(summary_signals["tokens"])
    index["all_phrases"].update(summary_signals["phrases"])

    for line in skills["lines"]:
        if "line_id" not in line:
            raise ValueError("skills line missing required field: line_id")
        if "text" not in line:
            raise ValueError("skills line missing required field: text")
        line_id = line["line_id"]
        text = line.get("text", "")
        signals = extract_signals(text)
        index["skills"][line_id] = {
            "source_type": "skill_line",
            "source_id": line_id,
            "original": text,
            "signals": signals,
        }
        index["all_tokens"].update(signals["tokens"])
        index["all_phrases"].update(signals["phrases"])

    for exp in resume_json["experience"]:
        if "exp_id" not in exp:
            raise ValueError("experience entry missing required field: exp_id")
        if "bullets" not in exp or exp["bullets"] is None:
            raise ValueError("experience entry missing required field: bullets")
        exp_id = exp["exp_id"]
        for bullet in exp["bullets"]:
            if "bullet_id" not in bullet:
                raise ValueError("experience bullet missing required field: bullet_id")
            if "bullet_index" not in bullet:
                raise ValueError("experience bullet missing required field: bullet_index")
            if "text" not in bullet:
                raise ValueError("experience bullet missing required field: text")
            bullet_id = bullet["bullet_id"]
            bullet_index = bullet["bullet_index"]
            text = bullet.get("text", "")
            signals = extract_signals(text)
            index["experience"][bullet_id] = {
                "source_type": "experience_bullet",
                "source_id": bullet_id,
                "original": text,
                "signals": signals,
                "exp_id": exp_id,
                "bullet_index": bullet_index,
            }
            index["all_tokens"].update(signals["tokens"])
            index["all_phrases"].update(signals["phrases"])

    projects = resume_json.get("projects")
    if projects is not None:
        index["projects"] = {}
        for project in projects:
            if "project_id" not in project:
                raise ValueError("project entry missing required field: project_id")
            if "bullets" not in project or project["bullets"] is None:
                raise ValueError("project entry missing required field: bullets")
            for bullet in project["bullets"]:
                if "bullet_id" not in bullet:
                    raise ValueError("project bullet missing required field: bullet_id")
                if "bullet_index" not in bullet:
                    raise ValueError("project bullet missing required field: bullet_index")
                if "text" not in bullet:
                    raise ValueError("project bullet missing required field: text")
                bullet_id = bullet["bullet_id"]
                bullet_index = bullet["bullet_index"]
                text = bullet.get("text", "")
                signals = extract_signals(text)
                index["projects"][bullet_id] = {
                    "source_type": "project_bullet",
                    "source_id": bullet_id,
                    "original": text,
                    "signals": signals,
                    "bullet_index": bullet_index,
                }
                index["all_tokens"].update(signals["tokens"])
                index["all_phrases"].update(signals["phrases"])

    return index
