import json
import sys
from pathlib import Path

import requests


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    fixtures_dir = base_dir / "fixtures"
    resume_path = Path(sys.argv[1]) if len(sys.argv) > 1 else fixtures_dir / "resume.json"
    job_path = Path(sys.argv[2]) if len(sys.argv) > 2 else fixtures_dir / "job.json"
    score_path = Path(sys.argv[3]) if len(sys.argv) > 3 else fixtures_dir / "score_result.json"

    if not resume_path.exists():
        print(f"Resume file not found: {resume_path}")
        return 1
    if not job_path.exists():
        print(f"Job file not found: {job_path}")
        return 1
    if not score_path.exists():
        print(f"ScoreResult file not found: {score_path}")
        return 1

    resume_json = json.loads(resume_path.read_text(encoding="utf-8"))
    job_json = json.loads(job_path.read_text(encoding="utf-8"))
    score_result = json.loads(score_path.read_text(encoding="utf-8"))

    response = requests.post(
        "http://localhost:8000/tailor-plan",
        json={"resume_json": resume_json, "job_json": job_json, "score_result": score_result},
        timeout=120,
    )
    print(f"status={response.status_code}")
    try:
        payload = response.json()
    except json.JSONDecodeError:
        print("Non-JSON response")
        return 1
    if response.status_code != 200:
        print(json.dumps(payload, indent=2))
        return 1
    plan = payload.get("tailoring_plan", {})
    actions = plan.get("bullet_actions", [])
    print("ok=True")
    for action in actions[:2]:
        print(json.dumps(action, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
