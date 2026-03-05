import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.scoring import score_job


def build_resume(summary_text, skills_texts, title):
    return {
        "meta": {"total_pages": 1, "structure_hash": "abc"},
        "summary": {"id": "summary", "text": summary_text},
        "skills": {
            "id": "skills",
            "lines": [
                {"line_id": f"skills_{i}", "text": text}
                for i, text in enumerate(skills_texts, start=1)
            ],
        },
        "experience": [
            {
                "exp_id": "exp_1",
                "company": "Acme",
                "title": title,
                "start_date": "2020",
                "end_date": "2021",
                "bullets": [
                    {
                        "bullet_id": "exp_1_b1",
                        "bullet_index": 0,
                        "text": "Built APIs.",
                        "char_count": 11,
                    }
                ],
            }
        ],
        "projects": [],
        "education": [
            {
                "edu_id": "edu_1",
                "school": "State University",
                "degree": "BS",
                "start_date": "2016",
                "end_date": "2020",
            }
        ],
    }


def build_job(must, nice, seniority=None):
    job = {
        "must_have": [
            {"requirement_id": f"must_{i}", "text": text}
            for i, text in enumerate(must, start=1)
        ],
        "nice_to_have": [
            {"requirement_id": f"nice_{i}", "text": text}
            for i, text in enumerate(nice, start=1)
        ],
        "responsibilities": ["Build"],
        "keywords": ["Backend"],
    }
    if seniority is not None:
        job["seniority"] = seniority
    return job


def test_all_must_have_matched_scores_70():
    resume = build_resume("Backend engineer with Python and FastAPI.", ["Python", "FastAPI"], "Developer")
    job = build_job(["Python", "FastAPI"], [], "mid")
    result = score_job(resume, job)
    assert result["score_breakdown"]["must"] == 70
    assert result["must_have"]["coverage_percent"] == 100


def test_half_must_have_matched_scores_35():
    resume = build_resume("Backend engineer with Python.", ["Python"], "Developer")
    job = build_job(["Python", "FastAPI"], [], "mid")
    result = score_job(resume, job)
    assert result["score_breakdown"]["must"] == 35
    assert result["must_have"]["coverage_percent"] == 50


def test_nice_to_have_scales_to_20():
    resume = build_resume("Frontend engineer.", ["React", "Kubernetes"], "Developer")
    job = build_job([], ["React", "Kubernetes"], "mid")
    result = score_job(resume, job)
    assert result["score_breakdown"]["nice"] == 20


def test_seniority_gate_blocks_junior_for_senior_role():
    resume = build_resume("Intern developer.", ["Python"], "Intern")
    job = build_job([], [], "senior")
    result = score_job(resume, job)
    assert result["seniority"]["seniority_ok"] is False
    assert result["score_breakdown"]["alignment"] == 0


def test_determinism_repeated_scoring_matches():
    resume = build_resume("Backend engineer with Python.", ["Python", "SQL"], "Developer")
    job = build_job(["Python"], ["SQL"], "mid")
    result_a = score_job(resume, job)
    result_b = score_job(resume, job)
    assert result_a == result_b
