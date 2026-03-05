import json
import sys
from pathlib import Path

import requests


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_resume_extractor.py /path/to/resume.txt")
        return 1
    resume_path = Path(sys.argv[1])
    if not resume_path.exists():
        print(f"Resume file not found: {resume_path}")
        return 1
    resume_text = resume_path.read_text(encoding="utf-8")
    response = requests.post(
        "http://localhost:8000/parse-resume",
        json={"resume_text": resume_text},
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
    resume_json = payload.get("resume_json", {})
    exp_count, bullet_count = _count_experiences(resume_json)
    print("schema_valid=True")
    print(f"experiences={exp_count}")
    print(f"bullets={bullet_count}")
    return 0


def _count_experiences(resume_json: dict) -> tuple[int, int]:
    total_bullets = 0
    total_exps = 0
    for key in ("experience", "experiences", "work_experience"):
        experiences = resume_json.get(key)
        if not isinstance(experiences, list):
            continue
        for exp in experiences:
            total_exps += 1
            if isinstance(exp, dict):
                bullets = exp.get("bullets")
                if isinstance(bullets, list):
                    total_bullets += len(bullets)
    return total_exps, total_bullets


if __name__ == "__main__":
    raise SystemExit(main())
