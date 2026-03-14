"""Deterministic ATS signal extraction for ResumeJSON."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Any

from app.ats.canonicalize import extract_canonical_term_pairs, normalize_phrase
from app.ats.types import ResumeSignals, SourceEntry, TermEvidence

_MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_PRESENT_MARKERS = {"current", "now", "present", "today"}


def extract_resume_signals(resume_json: dict[str, Any]) -> ResumeSignals:
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    if "summary" not in resume_json or resume_json["summary"] is None:
        raise ValueError("resume_json missing required field: summary")
    if "skills" not in resume_json or resume_json["skills"] is None:
        raise ValueError("resume_json missing required field: skills")
    if "experience" not in resume_json or resume_json["experience"] is None:
        raise ValueError("resume_json missing required field: experience")

    experience_order = _recent_experience_order(resume_json["experience"])
    experience_rank = {exp_id: index for index, exp_id in enumerate(experience_order)}

    source_entries: list[SourceEntry] = []
    order = 0

    def add_entry(
        source_id: str,
        section: str,
        text: str,
        *,
        parent_id: str | None = None,
        line_id: str | None = None,
        exp_id: str | None = None,
        project_id: str | None = None,
        edu_id: str | None = None,
        bullet_id: str | None = None,
        bullet_index: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        nonlocal order
        if not text:
            return
        source_entries.append(
            SourceEntry(
                source_id=source_id,
                section=section,
                text=text,
                order=order,
                parent_id=parent_id,
                line_id=line_id,
                exp_id=exp_id,
                project_id=project_id,
                edu_id=edu_id,
                bullet_id=bullet_id,
                bullet_index=bullet_index,
                start_date=start_date,
                end_date=end_date,
                experience_order=experience_rank.get(exp_id) if exp_id else None,
            )
        )
        order += 1

    summary = resume_json["summary"]
    if "id" not in summary or "text" not in summary:
        raise ValueError("summary missing required fields: id/text")
    add_entry(summary["id"], "summary", str(summary["text"]))

    skills = resume_json["skills"]
    if "lines" not in skills or skills["lines"] is None:
        raise ValueError("skills missing required field: lines")
    for line in skills["lines"]:
        if "line_id" not in line or "text" not in line:
            raise ValueError("skills line missing required fields: line_id/text")
        add_entry(line["line_id"], "skills", str(line["text"]), line_id=line["line_id"])

    for experience in resume_json["experience"]:
        _validate_experience(experience)
        exp_id = experience["exp_id"]
        start_date = str(experience.get("start_date", ""))
        end_date = str(experience.get("end_date", ""))
        title_source_id = f"{exp_id}__title"
        add_entry(
            title_source_id,
            "experience_title",
            str(experience.get("title", "")),
            parent_id=exp_id,
            exp_id=exp_id,
            start_date=start_date,
            end_date=end_date,
        )
        for bullet in experience["bullets"]:
            if "bullet_id" not in bullet or "text" not in bullet or "bullet_index" not in bullet:
                raise ValueError("experience bullet missing required fields")
            add_entry(
                bullet["bullet_id"],
                "experience_bullet",
                str(bullet["text"]),
                parent_id=exp_id,
                exp_id=exp_id,
                bullet_id=bullet["bullet_id"],
                bullet_index=int(bullet["bullet_index"]),
                start_date=start_date,
                end_date=end_date,
            )

    for project in resume_json.get("projects", []):
        if "project_id" not in project or "name" not in project or "bullets" not in project:
            raise ValueError("project entry missing required fields")
        add_entry(
            f"{project['project_id']}__name",
            "project_name",
            str(project["name"]),
            parent_id=project["project_id"],
            project_id=project["project_id"],
        )
        for bullet in project["bullets"]:
            if "bullet_id" not in bullet or "text" not in bullet or "bullet_index" not in bullet:
                raise ValueError("project bullet missing required fields")
            add_entry(
                bullet["bullet_id"],
                "project_bullet",
                str(bullet["text"]),
                parent_id=project["project_id"],
                project_id=project["project_id"],
                bullet_id=bullet["bullet_id"],
                bullet_index=int(bullet["bullet_index"]),
            )

    for education in resume_json.get("education", []):
        if "edu_id" not in education or "degree" not in education:
            raise ValueError("education entry missing required fields")
        add_entry(
            education["edu_id"],
            "education",
            str(education["degree"]),
            parent_id=education["edu_id"],
            edu_id=education["edu_id"],
            start_date=str(education.get("start_date", "")),
            end_date=str(education.get("end_date", "")),
        )

    canonical_terms: list[str] = []
    skill_terms: list[str] = []
    title_like_terms: list[str] = []
    term_frequencies: dict[str, int] = defaultdict(int)
    section_terms: dict[str, list[str]] = defaultdict(list)
    term_sources: dict[str, list[str]] = defaultdict(list)
    term_source_ids: dict[str, list[str]] = defaultdict(list)
    term_variants: dict[str, list[str]] = defaultdict(list)
    evidence_lookup: dict[tuple[str, str], TermEvidence] = {}
    evidence_map: dict[str, list[TermEvidence]] = defaultdict(list)
    seen_terms: set[str] = set()
    skill_seen: set[str] = set()
    title_seen: set[str] = set()

    for entry in source_entries:
        for canonical, raw in extract_canonical_term_pairs(entry.text):
            if entry.section == "education" and not _keep_education_term(canonical):
                continue
            normalized_raw = normalize_phrase(raw)
            if canonical not in seen_terms:
                canonical_terms.append(canonical)
                seen_terms.add(canonical)
            term_frequencies[canonical] += 1
            _append_unique(section_terms[_section_bucket(entry.section)], canonical)
            _append_unique(term_sources[canonical], entry.section)
            _append_unique(term_source_ids[canonical], entry.source_id)
            if normalized_raw:
                _append_unique(term_variants[canonical], normalized_raw)
            _upsert_evidence(evidence_lookup, evidence_map, entry, canonical, normalized_raw)
            if entry.section == "skills" and canonical not in skill_seen:
                skill_terms.append(canonical)
                skill_seen.add(canonical)
            if entry.section == "experience_title" and canonical not in title_seen:
                title_like_terms.append(canonical)
                title_seen.add(canonical)

    return ResumeSignals(
        all_terms=tuple(canonical_terms),
        section_terms={section: tuple(values) for section, values in section_terms.items()},
        skill_terms=tuple(skill_terms),
        title_like_terms=tuple(title_like_terms),
        evidence_map={term: tuple(values) for term, values in evidence_map.items()},
        term_frequencies=dict(term_frequencies),
        term_sources={term: tuple(values) for term, values in term_sources.items()},
        term_source_ids={term: tuple(values) for term, values in term_source_ids.items()},
        term_variants={term: tuple(values) for term, values in term_variants.items()},
        recent_experience_order=experience_order,
        source_entries=tuple(source_entries),
    )


build_resume_signals = extract_resume_signals


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _validate_experience(experience: dict[str, Any]) -> None:
    required_fields = {"exp_id", "title", "start_date", "end_date", "bullets"}
    missing = [field for field in required_fields if field not in experience]
    if missing:
        raise ValueError(f"experience entry missing required fields: {', '.join(missing)}")


def _section_bucket(section: str) -> str:
    if section == "experience_title" or section == "experience_bullet":
        return "experience"
    if section == "project_name" or section == "project_bullet":
        return "projects"
    return section


def _keep_education_term(term: str) -> bool:
    return " " in term or any(char in term for char in ".#/+")


def _recent_experience_order(experience_entries: list[dict[str, Any]]) -> tuple[str, ...]:
    sort_keys: list[tuple[tuple[int, date, date], str]] = []
    for entry in experience_entries:
        exp_id = entry.get("exp_id")
        if not exp_id:
            raise ValueError("experience entry missing required field: exp_id")
        key = _experience_sort_key(entry)
        if key is None:
            return tuple(str(item["exp_id"]) for item in experience_entries)
        sort_keys.append((key, str(exp_id)))
    sort_keys.sort(key=lambda item: item[0], reverse=True)
    return tuple(exp_id for _, exp_id in sort_keys)


def _experience_sort_key(entry: dict[str, Any]) -> tuple[int, date, date] | None:
    end_date = _parse_date_fragment(entry.get("end_date"))
    start_date = _parse_date_fragment(entry.get("start_date"))
    if end_date is None or start_date is None:
        return None
    present_flag = 1 if _is_present(entry.get("end_date")) else 0
    return (present_flag, end_date, start_date)


def _parse_date_fragment(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if _is_present(text):
        return date(9999, 12, 31)

    direct_match = re.match(r"^(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?$", text)
    if direct_match:
        year = int(direct_match.group(1))
        month = int(direct_match.group(2) or 12)
        day = int(direct_match.group(3) or 28)
        return _safe_date(year, month, day)

    month_match = re.match(r"^([a-z]+)\s+(\d{4})$", text)
    if month_match:
        month = _MONTH_LOOKUP.get(month_match.group(1))
        if month is None:
            return None
        return _safe_date(int(month_match.group(2)), month, 28)

    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _is_present(value: Any) -> bool:
    return str(value).strip().lower() in _PRESENT_MARKERS


def _upsert_evidence(
    evidence_lookup: dict[tuple[str, str], TermEvidence],
    evidence_map: dict[str, list[TermEvidence]],
    entry: SourceEntry,
    canonical_term: str,
    raw_term: str,
) -> None:
    key = (canonical_term, entry.source_id)
    current = evidence_lookup.get(key)
    if current is None:
        evidence = TermEvidence(
            canonical_term=canonical_term,
            raw_term=raw_term or canonical_term,
            section=entry.section,
            source_id=entry.source_id,
            source_text=entry.text,
            occurrence_count=1,
            order=entry.order,
            parent_id=entry.parent_id,
            requirement_id=entry.requirement_id,
            line_id=entry.line_id,
            exp_id=entry.exp_id,
            project_id=entry.project_id,
            edu_id=entry.edu_id,
            bullet_id=entry.bullet_id,
            bullet_index=entry.bullet_index,
            start_date=entry.start_date,
            end_date=entry.end_date,
            experience_order=entry.experience_order,
        )
        evidence_lookup[key] = evidence
        evidence_map[canonical_term].append(evidence)
        return

    updated = TermEvidence(
        canonical_term=current.canonical_term,
        raw_term=current.raw_term,
        section=current.section,
        source_id=current.source_id,
        source_text=current.source_text,
        occurrence_count=current.occurrence_count + 1,
        order=current.order,
        parent_id=current.parent_id,
        requirement_id=current.requirement_id,
        line_id=current.line_id,
        exp_id=current.exp_id,
        project_id=current.project_id,
        edu_id=current.edu_id,
        bullet_id=current.bullet_id,
        bullet_index=current.bullet_index,
        start_date=current.start_date,
        end_date=current.end_date,
        experience_order=current.experience_order,
    )
    evidence_lookup[key] = updated
    evidence_list = evidence_map[canonical_term]
    for index, existing in enumerate(evidence_list):
        if existing.source_id == entry.source_id:
            evidence_list[index] = updated
            break
