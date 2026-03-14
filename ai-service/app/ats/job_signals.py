"""Deterministic ATS signal extraction for JobJSON."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from app.ats.canonicalize import extract_canonical_term_pairs, normalize_phrase
from app.ats.stopwords import GENERIC_JOB_WORDS
from app.ats.types import JobSignals, SourceEntry, TermEvidence

_NOISY_JOB_TERMS = GENERIC_JOB_WORDS | {
    "build",
    "built",
    "building",
    "expert",
    "expertise",
    "lead",
    "led",
    "maintain",
    "maintaining",
    "own",
    "owns",
    "owning",
    "rest",
    "restful",
}

_NOISY_PHRASE_PREFIXES = frozenset({"build", "built", "building", "expert", "expertise", "lead", "led", "maintain", "own"})


def extract_job_signals(job_json: dict[str, Any]) -> JobSignals:
    if not isinstance(job_json, dict):
        raise ValueError("job_json must be a dict")
    if "must_have" not in job_json or job_json["must_have"] is None:
        raise ValueError("job_json missing required field: must_have")
    if "nice_to_have" not in job_json or job_json["nice_to_have"] is None:
        raise ValueError("job_json missing required field: nice_to_have")
    if "responsibilities" not in job_json or job_json["responsibilities"] is None:
        raise ValueError("job_json missing required field: responsibilities")

    source_entries: list[SourceEntry] = []
    order = 0

    def add_entry(
        source_id: str,
        section: str,
        text: str,
        *,
        requirement_id: str | None = None,
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
                requirement_id=requirement_id,
            )
        )
        order += 1

    title = _clean_text(job_json.get("title"))
    if title:
        add_entry("title", "title", title)

    seniority = _clean_text(job_json.get("seniority"))
    if seniority and seniority != "unknown":
        add_entry("seniority", "seniority", seniority)

    for requirement in job_json["must_have"]:
        if "requirement_id" not in requirement:
            raise ValueError("must_have item missing required field: requirement_id")
        if "text" not in requirement:
            raise ValueError("must_have item missing required field: text")
        add_entry(
            requirement["requirement_id"],
            "must_have",
            str(requirement["text"]),
            requirement_id=requirement["requirement_id"],
        )

    for requirement in job_json["nice_to_have"]:
        if "requirement_id" not in requirement:
            raise ValueError("nice_to_have item missing required field: requirement_id")
        if "text" not in requirement:
            raise ValueError("nice_to_have item missing required field: text")
        add_entry(
            requirement["requirement_id"],
            "nice_to_have",
            str(requirement["text"]),
            requirement_id=requirement["requirement_id"],
        )

    for index, responsibility in enumerate(job_json["responsibilities"]):
        add_entry(f"responsibility_{index}", "responsibilities", str(responsibility))

    for index, keyword in enumerate(job_json.get("keywords") or []):
        add_entry(f"keyword_{index}", "keywords", str(keyword))

    raw_terms: list[str] = []
    canonical_terms: list[str] = []
    title_terms: list[str] = []
    required_terms: list[str] = []
    preferred_terms: list[str] = []
    keyword_counts: dict[str, int] = defaultdict(int)
    term_sources: dict[str, list[str]] = defaultdict(list)
    term_source_ids: dict[str, list[str]] = defaultdict(list)
    term_variants: dict[str, list[str]] = defaultdict(list)
    evidence_lookup: dict[tuple[str, str], TermEvidence] = {}
    term_evidence: dict[str, list[TermEvidence]] = defaultdict(list)
    seen_raw: set[str] = set()
    seen_canonical: set[str] = set()
    title_seen: set[str] = set()
    required_seen: set[str] = set()
    preferred_seen: set[str] = set()

    for entry in source_entries:
        for canonical, raw in extract_canonical_term_pairs(entry.text):
            normalized_raw = normalize_phrase(raw)
            if normalized_raw and normalized_raw not in seen_raw:
                raw_terms.append(normalized_raw)
                seen_raw.add(normalized_raw)
            if canonical not in seen_canonical:
                canonical_terms.append(canonical)
                seen_canonical.add(canonical)
            keyword_counts[canonical] += 1
            _append_unique(term_sources[canonical], entry.section)
            _append_unique(term_source_ids[canonical], entry.source_id)
            if normalized_raw:
                _append_unique(term_variants[canonical], normalized_raw)
            _upsert_evidence(evidence_lookup, term_evidence, entry, canonical, normalized_raw)
            if entry.section == "title" and canonical not in title_seen:
                title_terms.append(canonical)
                title_seen.add(canonical)
            if (
                entry.section == "must_have"
                and _keep_inventory_term(canonical)
                and canonical not in required_seen
            ):
                required_terms.append(canonical)
                required_seen.add(canonical)
            if (
                entry.section == "nice_to_have"
                and _keep_inventory_term(canonical)
                and canonical not in preferred_seen
            ):
                preferred_terms.append(canonical)
                preferred_seen.add(canonical)

    repeated_terms = tuple(
        term
        for term in canonical_terms
        if keyword_counts[term] > 1
        and _keep_inventory_term(term)
        and not _source_sections_are_only_title(term_sources.get(term, []))
    )
    domain_terms = tuple(
        term
        for term in canonical_terms
        if _is_domain_term(
            term,
            count=keyword_counts[term],
            source_sections=term_sources.get(term, []),
        )
    )

    return JobSignals(
        all_terms=tuple(raw_terms),
        canonical_terms=tuple(canonical_terms),
        title_terms=tuple(title_terms),
        required_terms=tuple(required_terms),
        preferred_terms=tuple(preferred_terms),
        repeated_terms=repeated_terms,
        domain_terms=domain_terms,
        keyword_counts=dict(keyword_counts),
        term_sources={term: tuple(values) for term, values in term_sources.items()},
        term_source_ids={term: tuple(values) for term, values in term_source_ids.items()},
        term_variants={term: tuple(values) for term, values in term_variants.items()},
        term_evidence={term: tuple(values) for term, values in term_evidence.items()},
        source_entries=tuple(source_entries),
    )


build_job_signals = extract_job_signals


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_domain_term(term: str, *, count: int, source_sections: Iterable[str]) -> bool:
    if not _keep_inventory_term(term):
        return False
    sections = set(source_sections)
    if not sections.intersection({"title", "responsibilities", "keywords"}):
        return False
    if sections.issubset({"title", "seniority"}):
        return False
    if count > 1:
        return True
    if " " in term:
        return True
    return any(char in term for char in ".#/+")


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


def _keep_inventory_term(term: str) -> bool:
    if not term or term in _NOISY_JOB_TERMS:
        return False
    tokens = term.split()
    if tokens[0] in _NOISY_PHRASE_PREFIXES:
        return False
    return True


def _source_sections_are_only_title(source_sections: Iterable[str]) -> bool:
    sections = set(source_sections)
    return bool(sections) and sections.issubset({"title", "seniority"})

build_job_signals = extract_job_signals
