from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.schemas.validator import validate_json  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ats_scenarios"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_fixture_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [("job", FIXTURES_DIR / "shared_job.json")]
    for path in sorted(FIXTURES_DIR.glob("resume_*.json")):
        paths.append(("resume", path))
    return paths


def main() -> int:
    failures = 0
    for schema_name, path in _iter_fixture_paths():
        try:
            data = _load_json(path)
            ok, errors = validate_json(schema_name, data)
        except Exception as exc:
            ok = False
            errors = [str(exc)]

        if ok:
            print(f"PASS {schema_name:6} {path.name}")
            continue

        failures += 1
        print(f"FAIL {schema_name:6} {path.name}")
        for error in errors:
            print(f"  - {error}")

    if failures:
        print(f"\nFixture validation failed: {failures} file(s)")
        return 1

    print("\nAll ATS scenario fixtures are schema-valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
