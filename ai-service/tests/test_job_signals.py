import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.job_signals import extract_job_signals  # noqa: E402


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_extract_job_signals_builds_term_inventories_and_sources():
    signals = extract_job_signals(_load_fixture("ats_job.json"))

    assert "software engineer" in signals.title_terms
    assert "javascript" in signals.required_terms
    assert "rest api" in signals.required_terms
    assert "c#" in signals.required_terms
    assert "postgresql" in signals.preferred_terms
    assert "react" in signals.preferred_terms

    assert signals.keyword_counts["javascript"] == 4
    assert signals.term_sources["javascript"] == ("title", "must_have", "responsibilities", "keywords")
    assert signals.term_source_ids["rest api"] == ("req_2", "responsibility_0", "keyword_2")
    assert "node.js" in signals.domain_terms
    assert "rest api" in signals.repeated_terms


def test_job_signal_evidence_preserves_provenance():
    signals = extract_job_signals(_load_fixture("ats_job.json"))

    rest_api_sources = {entry.source_id: entry for entry in signals.term_evidence["rest api"]}
    assert rest_api_sources["req_2"].section == "must_have"
    assert rest_api_sources["req_2"].requirement_id == "req_2"
    assert rest_api_sources["responsibility_0"].section == "responsibilities"
    assert rest_api_sources["keyword_2"].raw_term == "rest api"
