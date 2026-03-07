import json
import sys
from pathlib import Path

import requests


def _count_bullets(resume_json: dict) -> int:
    count = 0
    for exp in resume_json.get("experience", []) or []:
        if isinstance(exp, dict):
            count += len(exp.get("bullets", []) or [])
    for proj in resume_json.get("projects", []) or []:
        if isinstance(proj, dict):
            count += len(proj.get("bullets", []) or [])
    return count


def _first_bullets(resume_json: dict, limit: int = 2) -> list[str]:
    texts = []
    for exp in resume_json.get("experience", []) or []:
        if not isinstance(exp, dict):
            continue
        for bullet in exp.get("bullets", []) or []:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                texts.append(bullet["text"])
                if len(texts) >= limit:
                    return texts
    for proj in resume_json.get("projects", []) or []:
        if not isinstance(proj, dict):
            continue
        for bullet in proj.get("bullets", []) or []:
            if isinstance(bullet, dict) and isinstance(bullet.get("text"), str):
                texts.append(bullet["text"])
                if len(texts) >= limit:
                    return texts
    return texts


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    fixtures_dir = base_dir / "fixtures"
    request_path = Path(sys.argv[1]) if len(sys.argv) > 1 else fixtures_dir / "rewrite_request.json"
    if not request_path.exists():
        print(f"Rewrite request file not found: {request_path}")
        return 1

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    original_resume = request_payload.get("resume_json", {})
    original_count = _count_bullets(original_resume)

    response = requests.post(
        "http://localhost:8000/rewrite-bullets",
        json=request_payload,
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

    tailored = payload.get("tailored_resume_json", {})
    new_count = _count_bullets(tailored)
    if new_count != original_count:
        raise AssertionError("Bullet counts changed")

    print("ok=True")
    for text in _first_bullets(tailored, 2):
        print(f"len={len(text)} text={text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
