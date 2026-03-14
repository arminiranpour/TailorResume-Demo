import json
from pprint import pprint

from app.ats.job_signals import build_job_signals
from app.ats.resume_signals import build_resume_signals
from app.ats.weighting import build_job_weights
from app.ats.coverage import build_coverage_model

with open("tests/fixtures/ats_job.json", "r", encoding="utf-8") as f:
    job_json = json.load(f)

with open("tests/fixtures/ats_resume.json", "r", encoding="utf-8") as f:
    resume_json = json.load(f)

job_signals = build_job_signals(job_json)
resume_signals = build_resume_signals(resume_json)
job_weights = build_job_weights(job_signals)
coverage = build_coverage_model(job_signals, resume_signals, job_weights)

print("\n=== COVERED TERMS ===")
pprint(coverage.covered_terms)

print("\n=== MISSING TERMS ===")
pprint(coverage.missing_terms)

print("\n=== UNDER-SUPPORTED TERMS ===")
pprint(coverage.under_supported_terms)

print("\n=== HIGH PRIORITY MISSING ===")
pprint(coverage.high_priority_missing_terms)

print("\n=== REQUIRED MISSING ===")
pprint(coverage.required_missing_terms)

print("\n=== TITLE TERMS MISSING ===")
pprint(coverage.title_terms_missing)

print("\n=== AGGREGATE METRICS ===")
print("overall_distinct_coverage =", coverage.overall_distinct_coverage)
print("high_priority_coverage    =", coverage.high_priority_coverage)
print("required_coverage         =", coverage.required_coverage)

print("\n=== SAMPLE TERM COVERAGE ===")
for term in coverage.coverage_ordered_terms[:10]:
    pprint(coverage.coverage_by_term[term])