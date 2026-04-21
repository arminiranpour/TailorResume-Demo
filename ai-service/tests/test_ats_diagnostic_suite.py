import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ats_diagnostic_scenarios import SCENARIOS  # noqa: E402
from scripts.run_ats_diagnostic_suite import (  # noqa: E402
    render_global_summary,
    render_scenario_report,
    run_suite,
)


def test_diagnostic_scenarios_cover_required_steps_and_integrated_case():
    covered_steps = {step for scenario in SCENARIOS for step in scenario.target_steps}
    assert {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}.issubset(covered_steps)
    assert any(len(scenario.target_steps) > 1 for scenario in SCENARIOS)


def test_frequency_scenario_reports_touched_sections_for_rollbacks():
    report = run_suite(["frequency_balance_rolls_back_stuffing"])[0]
    changes = report["changes"]

    assert "summary" not in changes["changed_sections"]
    assert "summary" in changes["touched_sections"]
    assert any(
        event["section"] == "summary" and event["outcome"] == "rolled_back"
        for event in changes["nonfinal_events"]
    )

    rendered = render_scenario_report(report)
    assert "Which sections were touched by ATS logic:" in rendered
    assert "Non-final ATS section events:" in rendered

    global_summary = render_global_summary([report])
    assert "GLOBAL SUMMARY" in global_summary
    assert "Scenarios passed strongly" in global_summary
