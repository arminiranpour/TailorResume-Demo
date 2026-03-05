from typing import Any, Dict, List, Tuple

from .normalize import extract_signals

PRIORITY_ORDER = ["skills", "experience", "projects", "summary"]


def _iter_sections(resume_index: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    sections: List[Tuple[str, Dict[str, Any]]] = []
    if "skills" in resume_index:
        for item in resume_index["skills"].values():
            sections.append(("skills", item))
    if "experience" in resume_index:
        for item in resume_index["experience"].values():
            sections.append(("experience", item))
    if "projects" in resume_index:
        for item in resume_index["projects"].values():
            sections.append(("projects", item))
    if "summary" in resume_index and resume_index["summary"]:
        sections.append(("summary", resume_index["summary"]))
    return sections


def _priority_rank(section_name: str) -> int:
    try:
        return PRIORITY_ORDER.index(section_name)
    except ValueError:
        return len(PRIORITY_ORDER)


def _best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not candidates:
        return None
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -c["overlap_score"],
            _priority_rank(c["section_name"]),
            c["item"]["source_id"],
        ),
    )
    return sorted_candidates[0]


def _build_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    evidence = {
        "source_type": item["source_type"],
        "source_id": item["source_id"],
        "snippet": item["original"],
    }
    if "exp_id" in item:
        evidence["exp_id"] = item["exp_id"]
    if "bullet_index" in item:
        evidence["bullet_index"] = item["bullet_index"]
    return evidence


def _match_threshold(token_count: int) -> float:
    if token_count <= 0:
        return 1.1
    if token_count == 1:
        return 1.0
    if token_count <= 3:
        return 0.5
    return 0.4


def match_requirement(
    requirement_text: str, requirement_id: str, resume_index: Dict[str, Any]
) -> Dict[str, Any]:
    signals = extract_signals(requirement_text)
    req_tokens = signals["tokens"]
    req_phrases = signals["phrases"]
    req_token_set = set(req_tokens)

    candidates = []
    for section_name, item in _iter_sections(resume_index):
        section_tokens = set(item["signals"]["tokens"])
        overlap_tokens = req_token_set.intersection(section_tokens)
        overlap_score = (
            len(overlap_tokens) / max(1, len(req_token_set))
            if req_token_set
            else 0.0
        )
        section_phrases = set(item["signals"]["phrases"])
        overlap_phrases = req_phrases.intersection(section_phrases)
        candidates.append(
            {
                "section_name": section_name,
                "item": item,
                "overlap_score": overlap_score,
                "overlap_tokens": overlap_tokens,
                "overlap_phrases": overlap_phrases,
            }
        )

    phrase_overlap = req_phrases.intersection(resume_index.get("all_phrases", set()))
    if phrase_overlap:
        phrase_candidates = [c for c in candidates if c["overlap_phrases"]]
        best = _best_candidate(phrase_candidates) or _best_candidate(candidates)
        matched_terms = sorted(best["overlap_phrases"]) if best else sorted(phrase_overlap)
        evidence = _build_evidence(best["item"]) if best else None
        overlap_score = best["overlap_score"] if best else 0.0
        return {
            "requirement_id": requirement_id,
            "text": requirement_text,
            "matched": True,
            "match_method": "phrase",
            "overlap_score": overlap_score,
            "matched_terms": matched_terms,
            "evidence": evidence,
        }

    best = _best_candidate(candidates)
    if not best or not req_token_set:
        return {
            "requirement_id": requirement_id,
            "text": requirement_text,
            "matched": False,
            "match_method": "none",
            "overlap_score": 0.0,
            "matched_terms": [],
            "evidence": None,
        }

    threshold = _match_threshold(len(req_token_set))
    matched = best["overlap_score"] >= threshold and len(best["overlap_tokens"]) > 0
    if matched:
        matched_terms = sorted(best["overlap_tokens"])
        evidence = _build_evidence(best["item"])
        return {
            "requirement_id": requirement_id,
            "text": requirement_text,
            "matched": True,
            "match_method": "token_overlap",
            "overlap_score": best["overlap_score"],
            "matched_terms": matched_terms,
            "evidence": evidence,
        }

    return {
        "requirement_id": requirement_id,
        "text": requirement_text,
        "matched": False,
        "match_method": "none",
        "overlap_score": best["overlap_score"],
        "matched_terms": [],
        "evidence": None,
    }


def match_requirements(
    requirements: List[Dict[str, Any]], resume_index: Dict[str, Any]
) -> List[Dict[str, Any]]:
    matches = []
    for requirement in requirements:
        if "requirement_id" not in requirement:
            raise ValueError("requirement missing required field: requirement_id")
        if "text" not in requirement:
            raise ValueError("requirement missing required field: text")
        matches.append(
            match_requirement(requirement["text"], requirement["requirement_id"], resume_index)
        )
    return matches


def build_job_match(job_json: Dict[str, Any], resume_index: Dict[str, Any]) -> Dict[str, Any]:
    if "must_have" not in job_json or job_json["must_have"] is None:
        raise ValueError("job_json missing required field: must_have")
    if "nice_to_have" not in job_json or job_json["nice_to_have"] is None:
        raise ValueError("job_json missing required field: nice_to_have")

    must_have_matches = match_requirements(job_json["must_have"], resume_index)
    nice_to_have_matches = match_requirements(job_json["nice_to_have"], resume_index)

    return {"must_have": must_have_matches, "nice_to_have": nice_to_have_matches}
