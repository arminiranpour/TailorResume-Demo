import json
import sys
from pathlib import Path

import requests


def _iter_bullet_texts(resume_json):
    for exp in resume_json.get("experience", []) or []:
        if not isinstance(exp, dict):
            continue
        for bullet in exp.get("bullets", []) or []:
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str) and isinstance(
                bullet.get("text"), str
            ):
                yield bullet["bullet_id"], bullet["text"]
    for proj in resume_json.get("projects", []) or []:
        if not isinstance(proj, dict):
            continue
        for bullet in proj.get("bullets", []) or []:
            if isinstance(bullet, dict) and isinstance(bullet.get("bullet_id"), str) and isinstance(
                bullet.get("text"), str
            ):
                yield bullet["bullet_id"], bullet["text"]


def _iter_skill_lines(resume_json):
    skills = resume_json.get("skills") if isinstance(resume_json.get("skills"), dict) else {}
    for line in skills.get("lines", []) or []:
        if isinstance(line, dict) and isinstance(line.get("line_id"), str) and isinstance(line.get("text"), str):
            yield line["line_id"], line["text"]


def _summary_text(resume_json):
    summary = resume_json.get("summary") if isinstance(resume_json.get("summary"), dict) else {}
    text = summary.get("text")
    return text if isinstance(text, str) else ""


def _compute_baseline_budgets(resume_json):
    summary_len = len(_summary_text(resume_json))
    bullet_budgets = {}
    for bullet_id, text in _iter_bullet_texts(resume_json):
        if bullet_id not in bullet_budgets:
            bullet_budgets[bullet_id] = len(text)
    skills_budgets = {}
    for line_id, text in _iter_skill_lines(resume_json):
        if line_id not in skills_budgets:
            skills_budgets[line_id] = len(text)
    total = summary_len + sum(bullet_budgets.values()) + sum(skills_budgets.values())
    return summary_len, bullet_budgets, skills_budgets, total


def _compute_total_len(resume_json):
    total = len(_summary_text(resume_json))
    total += sum(len(text) for _, text in _iter_bullet_texts(resume_json))
    total += sum(len(text) for _, text in _iter_skill_lines(resume_json))
    return total


def _check_invariants(original, final):
    original_exps = original.get("experience", []) or []
    final_exps = final.get("experience", []) or []
    if len(original_exps) != len(final_exps):
        raise AssertionError("experience count changed")
    if [exp.get("exp_id") for exp in original_exps] != [exp.get("exp_id") for exp in final_exps]:
        raise AssertionError("exp_id sequence changed")
    for orig, new in zip(original_exps, final_exps):
        orig_bullets = orig.get("bullets", []) or []
        new_bullets = new.get("bullets", []) or []
        if len(orig_bullets) != len(new_bullets):
            raise AssertionError("bullet count changed")
        orig_ids = [b.get("bullet_id") for b in orig_bullets]
        new_ids = [b.get("bullet_id") for b in new_bullets]
        if sorted(orig_ids) != sorted(new_ids):
            raise AssertionError("bullet ids changed")
        orig_index = {b.get("bullet_id"): b.get("bullet_index") for b in orig_bullets}
        new_index = {b.get("bullet_id"): b.get("bullet_index") for b in new_bullets}
        if orig_index != new_index:
            raise AssertionError("bullet_index changed")

    orig_lines = [line.get("line_id") for line in (original.get("skills", {}) or {}).get("lines", [])]
    new_lines = [line.get("line_id") for line in (final.get("skills", {}) or {}).get("lines", [])]
    if len(orig_lines) != len(new_lines) or sorted(orig_lines) != sorted(new_lines):
        raise AssertionError("skills line ids changed")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    fixtures_dir = base_dir / "fixtures"
    request_path = Path(sys.argv[1]) if len(sys.argv) > 1 else fixtures_dir / "enforce_budget_request.json"
    if not request_path.exists():
        print(f"Request file not found: {request_path}")
        return 1

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    original_resume = request_payload.get("original_resume_json", {})
    summary_budget, bullet_budgets, skills_budgets, total_budget = _compute_baseline_budgets(original_resume)

    response = requests.post(
        "http://localhost:8000/enforce-budgets",
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

    final_resume = payload.get("final_resume_json", {})
    for bullet_id, text in _iter_bullet_texts(final_resume):
        max_chars = bullet_budgets.get(bullet_id)
        if max_chars is None:
            raise AssertionError(f"Missing budget for {bullet_id}")
        if len(text) > max_chars:
            raise AssertionError(f"Bullet {bullet_id} exceeds budget")

    total_len = _compute_total_len(final_resume)
    if total_len > total_budget:
        raise AssertionError("Total length exceeds budget")

    if len(_summary_text(final_resume)) > summary_budget:
        raise AssertionError("Summary exceeds budget")

    for line_id, text in _iter_skill_lines(final_resume):
        max_chars = skills_budgets.get(line_id)
        if max_chars is None:
            raise AssertionError(f"Missing skills budget for {line_id}")
        if len(text) > max_chars:
            raise AssertionError("Skills line exceeds budget")

    _check_invariants(original_resume, final_resume)
    print("ok=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
