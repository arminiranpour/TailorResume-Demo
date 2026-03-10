from __future__ import annotations

import io
import logging
from typing import Any, Dict, Iterable, List, Tuple

from docx import Document

from app.docx_engine.editor import get_resume_replacement_targets
from app.docx_engine.metrics import compute_docx_metrics
from app.docx_engine.types import DocxOverflowError, DocxReplacementError

logger = logging.getLogger(__name__)

_DEFAULT_LARGEST_DELTA_COUNT = 5


def default_overflow_thresholds() -> Dict[str, float]:
    """Return default heuristic overflow thresholds."""
    return {
        "max_total_growth_chars": 10,
        "max_total_growth_percent": 0.03,
        "max_field_growth_chars": 20,
        "max_field_growth_percent": 0.10,
        "max_longest_paragraph_growth_chars": 20,
        "max_longest_paragraph_growth_percent": 0.10,
        "max_increased_paragraphs": 2,
    }


def compare_metrics(
    original_metrics: Dict[str, Any],
    edited_metrics: Dict[str, Any],
    thresholds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compare original vs edited metrics and return an overflow report."""
    effective_thresholds = default_overflow_thresholds()
    if thresholds:
        effective_thresholds.update(thresholds)

    original_records = _require_field_records(original_metrics, "original")
    edited_records = _require_field_records(edited_metrics, "edited")

    original_map = _index_records(original_records)
    edited_map = _index_records(edited_records)

    if set(original_map.keys()) != set(edited_map.keys()):
        missing = sorted(set(original_map.keys()) - set(edited_map.keys()))
        extra = sorted(set(edited_map.keys()) - set(original_map.keys()))
        raise DocxOverflowError(
            "Field records mismatch between original and edited metrics",
            details={"missing": missing, "extra": extra},
        )

    field_reports: List[Dict[str, Any]] = []
    increased_paragraphs = 0
    replaced_field_count = 0
    replaced_style_names: List[str] = []
    field_growth_violation = False
    field_growth_violations: List[Tuple[str, str, int, float]] = []

    for original_record in original_records:
        key = (original_record["field_type"], original_record["field_id"])
        edited_record = edited_map[key]

        if edited_record["paragraph_index"] != original_record["paragraph_index"]:
            raise DocxOverflowError(
                "Paragraph index mismatch between original and edited metrics",
                details={
                    "field_type": original_record["field_type"],
                    "field_id": original_record["field_id"],
                    "original_paragraph_index": original_record["paragraph_index"],
                    "edited_paragraph_index": edited_record["paragraph_index"],
                },
            )

        original_length = original_record["visible_length"]
        edited_length = edited_record["visible_length"]
        delta_chars = edited_length - original_length
        delta_percent = _safe_delta_percent(delta_chars, original_length)

        if delta_chars > 0:
            increased_paragraphs += 1

        if original_record["expected_text"] != edited_record["expected_text"]:
            replaced_field_count += 1
            style_name = edited_record.get("paragraph_style_name")
            if isinstance(style_name, str) and style_name:
                replaced_style_names.append(style_name)

        if (
            delta_chars > effective_thresholds["max_field_growth_chars"]
            or delta_percent > effective_thresholds["max_field_growth_percent"]
        ):
            field_growth_violation = True
            field_growth_violations.append(
                (
                    original_record["field_type"],
                    original_record["field_id"],
                    delta_chars,
                    delta_percent,
                )
            )

        field_reports.append(
            {
                "field_type": original_record["field_type"],
                "field_id": original_record["field_id"],
                "paragraph_index": original_record["paragraph_index"],
                "original_length": original_length,
                "edited_length": edited_length,
                "delta_chars": delta_chars,
                "delta_percent": delta_percent,
                "run_count_before": original_record["run_count"],
                "run_count_after": edited_record["run_count"],
            }
        )

    replaced_style_names = sorted(set(replaced_style_names))

    original_totals = _require_totals(original_metrics, "original")
    edited_totals = _require_totals(edited_metrics, "edited")

    total_original = original_totals["total_editable_chars"]
    total_edited = edited_totals["total_editable_chars"]
    total_delta = total_edited - total_original
    total_delta_percent = _safe_delta_percent(total_delta, total_original)

    longest_original = original_totals["max_paragraph_visible_length"]
    longest_edited = edited_totals["max_paragraph_visible_length"]
    longest_delta = longest_edited - longest_original
    longest_delta_percent = _safe_delta_percent(longest_delta, longest_original)

    rules_failed: List[str] = []

    if total_delta > effective_thresholds["max_total_growth_chars"]:
        rules_failed.append("total_editable_chars_growth_over_max")
    if total_delta_percent > effective_thresholds["max_total_growth_percent"]:
        rules_failed.append("total_editable_chars_growth_percent_over_max")
    if field_growth_violation:
        rules_failed.append("field_growth_over_max")
    if longest_delta > effective_thresholds["max_longest_paragraph_growth_chars"]:
        rules_failed.append("longest_paragraph_growth_over_max")
    if (
        longest_delta_percent
        > effective_thresholds["max_longest_paragraph_growth_percent"]
    ):
        rules_failed.append("longest_paragraph_growth_percent_over_max")
    if increased_paragraphs > effective_thresholds["max_increased_paragraphs"]:
        rules_failed.append("increased_paragraph_count_over_max")

    for field_type, field_id, delta_chars, delta_percent in field_growth_violations:
        logger.warning(
            "Overflow risk: %s %s grew by %s chars (%.2f%%)",
            field_type,
            field_id,
            delta_chars,
            delta_percent * 100,
        )

    logger.info("Computed original editable char total: %s", total_original)
    logger.info("Computed edited editable char total: %s", total_edited)

    largest_deltas = _largest_growth_fields(field_reports, _DEFAULT_LARGEST_DELTA_COUNT)

    report = {
        "valid": True,
        "overflow_risk": bool(rules_failed),
        "rules_failed": rules_failed,
        "summary": {
            "total_editable_chars_original": total_original,
            "total_editable_chars_edited": total_edited,
            "total_editable_normalized_chars_original": original_totals[
                "total_editable_normalized_chars"
            ],
            "total_editable_normalized_chars_edited": edited_totals[
                "total_editable_normalized_chars"
            ],
            "delta_chars": total_delta,
            "delta_percent": total_delta_percent,
            "editable_field_count": original_totals["editable_field_count"],
            "replaced_field_count": replaced_field_count,
            "longest_editable_paragraph_original": longest_original,
            "longest_editable_paragraph_edited": longest_edited,
            "longest_editable_paragraph_delta_chars": longest_delta,
            "longest_editable_paragraph_delta_percent": longest_delta_percent,
            "sum_paragraph_visible_length_original": original_totals[
                "sum_paragraph_visible_length"
            ],
            "sum_paragraph_visible_length_edited": edited_totals[
                "sum_paragraph_visible_length"
            ],
            "increased_paragraph_count": increased_paragraphs,
            "paragraph_count_original": original_metrics["paragraph_count"],
            "paragraph_count_edited": edited_metrics["paragraph_count"],
            "replaced_paragraph_style_names": replaced_style_names,
        },
        "largest_deltas": largest_deltas,
        "field_reports": field_reports,
        "thresholds": dict(effective_thresholds),
    }
    return report


def evaluate_docx_overflow_risk(
    original_docx_path: str,
    edited_docx_bytes: bytes,
    original_resume_json: Dict[str, Any],
    tailored_resume_json: Dict[str, Any],
    docx_mapping: Dict[str, Any],
    thresholds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate heuristic overflow risk for edited DOCX content."""
    if not isinstance(edited_docx_bytes, (bytes, bytearray)):
        raise DocxOverflowError(
            "edited_docx_bytes must be bytes",
            details={"type": type(edited_docx_bytes).__name__},
        )

    try:
        _ = get_resume_replacement_targets(original_resume_json, tailored_resume_json)
    except DocxReplacementError as exc:
        raise DocxOverflowError(
            "Resume structure mismatch between original and tailored JSON",
            details=getattr(exc, "details", None),
        ) from exc

    try:
        original_doc = Document(original_docx_path)
    except Exception as exc:  # pragma: no cover - depends on python-docx errors
        raise DocxOverflowError(
            "Failed to load original DOCX",
            details={"error": str(exc)},
        ) from exc

    try:
        edited_doc = Document(io.BytesIO(edited_docx_bytes))
    except Exception as exc:  # pragma: no cover - depends on python-docx errors
        raise DocxOverflowError(
            "Failed to load edited DOCX bytes",
            details={"error": str(exc)},
        ) from exc

    original_metrics = compute_docx_metrics(
        original_doc, original_resume_json, docx_mapping
    )
    edited_metrics = compute_docx_metrics(
        edited_doc, tailored_resume_json, docx_mapping
    )

    return compare_metrics(original_metrics, edited_metrics, thresholds=thresholds)


def _require_field_records(metrics: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    records = metrics.get("field_records")
    if not isinstance(records, list):
        raise DocxOverflowError(
            f"{label} metrics missing field_records",
            details={"type": type(records).__name__},
        )
    return records


def _require_totals(metrics: Dict[str, Any], label: str) -> Dict[str, Any]:
    totals = metrics.get("totals")
    if not isinstance(totals, dict):
        raise DocxOverflowError(
            f"{label} metrics missing totals",
            details={"type": type(totals).__name__},
        )
    return totals


def _index_records(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record["field_type"], record["field_id"])
        if key in indexed:
            raise DocxOverflowError(
                "Duplicate field record key detected",
                details={"field_type": record["field_type"], "field_id": record["field_id"]},
            )
        indexed[key] = record
    return indexed


def _safe_delta_percent(delta: int, original: int) -> float:
    if original <= 0:
        return 1.0 if delta > 0 else 0.0
    return delta / original


def _largest_growth_fields(
    field_reports: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    growing = [
        report for report in field_reports if report.get("delta_chars", 0) > 0
    ]
    growing.sort(
        key=lambda item: (
            -item["delta_chars"],
            -item["delta_percent"],
            item["field_type"],
            item["field_id"],
        )
    )
    trimmed = growing[:limit]
    return [
        {
            "field_type": report["field_type"],
            "field_id": report["field_id"],
            "paragraph_index": report["paragraph_index"],
            "delta_chars": report["delta_chars"],
            "delta_percent": report["delta_percent"],
        }
        for report in trimmed
    ]
