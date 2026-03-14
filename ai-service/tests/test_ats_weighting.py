import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.job_signals import extract_job_signals  # noqa: E402
from app.ats.weighting import build_job_weights  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_build_job_weights_prioritizes_required_title_and_repeated_terms():
    signals = extract_job_signals(_load_fixture("ats_job.json"))

    weights = build_job_weights(signals)

    assert weights.weights_by_term["javascript"].total_weight > weights.weights_by_term["react"].total_weight
    assert weights.weights_by_term["software engineer"].total_weight > weights.weights_by_term["automation"].total_weight
    assert weights.weights_by_term["javascript"].components["repetition"] == 3
    assert weights.title_priority_terms[0] == "javascript"
    assert "software engineer" in weights.high_priority_terms


def test_build_job_weights_downweights_low_signal_terms_and_favors_technical_phrases():
    job_json = {
        "title": "Platform Engineer",
        "must_have": [
            {"requirement_id": "req_1", "text": "Strong communication skills"},
            {"requirement_id": "req_2", "text": "Build REST APIs"},
        ],
        "nice_to_have": [
            {"requirement_id": "nice_1", "text": "Motivated team player"},
        ],
        "responsibilities": [
            "Maintain REST API integrations",
            "Partner with engineering and communication stakeholders",
        ],
        "keywords": ["REST API", "communication"],
    }

    weights = build_job_weights(extract_job_signals(job_json))

    assert weights.weights_by_term["rest api"].total_weight > weights.weights_by_term["communication"].total_weight
    assert weights.weights_by_term["communication"].is_low_signal is True
    assert weights.weights_by_term["communication"].components["low_signal_penalty"] < 0
    assert "technical_term" in weights.weights_by_term["rest api"].reasons
    assert "low_signal" in weights.weights_by_term["communication"].reasons


def test_build_job_weights_uses_stable_lexical_tiebreaking_for_equal_weights():
    job_json = {
        "title": "Engineer",
        "must_have": [],
        "nice_to_have": [
            {"requirement_id": "nice_1", "text": "React"},
            {"requirement_id": "nice_2", "text": "Postgres"},
        ],
        "responsibilities": [],
        "keywords": [],
    }

    weights = build_job_weights(extract_job_signals(job_json))

    assert weights.weights_by_term["postgresql"].total_weight == weights.weights_by_term["react"].total_weight
    assert weights.ordered_terms.index("postgresql") < weights.ordered_terms.index("react")
    assert weights.preferred_priority_terms == ("postgresql", "react")


def test_build_job_weights_preserves_reason_breakdown_and_signal_sources():
    signals = extract_job_signals(_load_fixture("ats_job.json"))

    weights = build_job_weights(signals)
    rest_api = weights.weights_by_term["rest api"]

    assert rest_api.components == {
        "base": 1,
        "must_have": 6,
        "repetition": 2,
        "domain": 2,
        "technical": 4,
    }
    assert rest_api.reasons == (
        "term_present",
        "must_have",
        "repeated",
        "domain_term",
        "technical_term",
    )
    assert rest_api.source_sections == ("must_have", "responsibilities", "keywords")
    assert rest_api.source_ids == ("req_2", "responsibility_0", "keyword_2")
    assert rest_api.source_signals == (
        "canonical_terms",
        "required_terms",
        "repeated_terms",
        "keyword_counts",
        "domain_terms",
        "technical_phrase_rules",
    )


def test_build_job_weights_is_deterministic_across_repeated_runs():
    signals = extract_job_signals(_load_fixture("ats_job.json"))

    first = build_job_weights(signals)
    second = build_job_weights(signals)

    assert first == second
