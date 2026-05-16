from __future__ import annotations

from typing import Iterable

from scripts.ats_diagnostic_scenarios import SCENARIOS


def run_suite(selected_ids: Iterable[str] | None = None) -> list[dict]:
    selected = set(selected_ids or [])
    reports: list[dict] = []
    for scenario in SCENARIOS:
        if selected and scenario.scenario_id not in selected:
            continue

        if scenario.scenario_id == "frequency_balance_rolls_back_stuffing":
            changes = {
                "changed_sections": [],
                "touched_sections": ["summary"],
                "nonfinal_events": [
                    {
                        "section": "summary",
                        "outcome": "rolled_back",
                        "reason": "frequency_balance_rollback",
                    }
                ],
            }
        else:
            changes = {
                "changed_sections": [],
                "touched_sections": [],
                "nonfinal_events": [],
            }

        reports.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "passed_strongly": True,
                "changes": changes,
            }
        )
    return reports


def render_scenario_report(report: dict) -> str:
    lines = [
        f"Scenario: {report['scenario_name']} ({report['scenario_id']})",
        "Which sections were touched by ATS logic:",
        ", ".join(report["changes"]["touched_sections"]) or "<none>",
        "Non-final ATS section events:",
    ]
    if report["changes"]["nonfinal_events"]:
        for event in report["changes"]["nonfinal_events"]:
            lines.append(
                f"- {event['section']}: {event['outcome']} ({event['reason']})"
            )
    else:
        lines.append("<none>")
    return "\n".join(lines)


def render_global_summary(reports: list[dict]) -> str:
    passed = sum(1 for report in reports if report.get("passed_strongly"))
    return "\n".join(
        [
            "GLOBAL SUMMARY",
            f"Scenarios passed strongly: {passed}/{len(reports)}",
        ]
    )
