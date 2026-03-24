import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.scoring import build_job_match, build_resume_index


resume = {
    "meta": {"total_pages": 1, "structure_hash": "abc"},
    "summary": {"id": "summary", "text": "Backend engineer with Python and FastAPI."},
    "skills": {
        "id": "skills",
        "lines": [
            {"line_id": "skills_1", "text": "Python, FastAPI, SQL"},
            {"line_id": "skills_2", "text": "Docker"},
        ],
    },
    "experience": [
        {
            "exp_id": "exp_1",
            "company": "Acme",
            "title": "Developer",
            "start_date": "2020",
            "end_date": "2021",
            "bullets": [
                {
                    "bullet_id": "exp_1_b1",
                    "bullet_index": 0,
                    "text": "Built APIs using FastAPI and PostgreSQL.",
                    "char_count": 43,
                },
                {
                    "bullet_id": "exp_1_b2",
                    "bullet_index": 1,
                    "text": "Wrote scripts in Python.",
                    "char_count": 27,
                },
            ],
        }
    ],
    "projects": [
        {
            "project_id": "proj_1",
            "name": "Tool",
            "bullets": [
                {
                    "bullet_id": "proj_1_b1",
                    "bullet_index": 0,
                    "text": "Created React app.",
                    "char_count": 20,
                }
            ],
        }
    ],
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

job = {
    "must_have": [
        {"requirement_id": "req_python", "text": "Python"},
        {"requirement_id": "req_fastapi", "text": "FastAPI"},
        {"requirement_id": "req_injection", "text": "SQL injection; DROP TABLE users;"},
    ],
    "nice_to_have": [
        {"requirement_id": "req_react", "text": "React"},
        {"requirement_id": "req_k8s", "text": "Kubernetes"},
    ],
    "responsibilities": ["Build APIs"],
    "keywords": ["Backend"],
}

resume_index = build_resume_index(resume)
match = build_job_match(job, resume_index)

python_match = next(m for m in match["must_have"] if m["requirement_id"] == "req_python")
assert python_match["matched"] is True
assert python_match["evidence"] is not None
assert python_match["evidence"]["source_id"] == "skills_1"

fastapi_match = next(m for m in match["must_have"] if m["requirement_id"] == "req_fastapi")
assert fastapi_match["matched"] is True
assert fastapi_match["evidence"] is not None

inj_match = next(m for m in match["must_have"] if m["requirement_id"] == "req_injection")
assert inj_match["matched"] is False
assert inj_match["evidence"] is None

k8s_match = next(m for m in match["nice_to_have"] if m["requirement_id"] == "req_k8s")
assert k8s_match["matched"] is False
assert k8s_match["evidence"] is None
