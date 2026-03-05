import json
import sys
from pathlib import Path

import requests


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_job_extractor.py /path/to/job.txt [url]")
        return 1
    job_path = Path(sys.argv[1])
    if not job_path.exists():
        print(f"Job file not found: {job_path}")
        return 1
    job_text = job_path.read_text(encoding="utf-8")
    url = sys.argv[2] if len(sys.argv) > 2 else None
    response = requests.post(
        "http://localhost:8000/parse-job",
        json={"job_text": job_text, "url": url},
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
    job_json = payload.get("job_json", {})
    title = _safe_str(job_json.get("title"))
    company = _safe_str(job_json.get("company"))
    seniority = _safe_str(job_json.get("seniority"))
    remote = job_json.get("remote")
    print(f"title={title}")
    print(f"company={company}")
    print(f"seniority={seniority}")
    print(f"remote={remote}")
    print(f"must_have={_count_list(job_json.get('must_have'))}")
    print(f"nice_to_have={_count_list(job_json.get('nice_to_have'))}")
    print(f"responsibilities={_count_list(job_json.get('responsibilities'))}")
    print(f"keywords={_count_list(job_json.get('keywords'))}")
    return 0


def _count_list(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
