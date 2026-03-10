from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReplacementAuditRecord:
    field_type: str
    field_id: str
    paragraph_index: int
    old_text: str
    new_text: str
    verification_score: float
    old_run_count: int
    replacement_status: str
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCollector:
    total_targets: int
    replacements: List[ReplacementAuditRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_replacement(self, record: ReplacementAuditRecord) -> None:
        self.replacements.append(record)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        replaced = sum(1 for r in self.replacements if r.replacement_status == "replaced")
        skipped = sum(1 for r in self.replacements if r.replacement_status == "skipped")
        failed = sum(1 for r in self.replacements if r.replacement_status == "failed")
        return {
            "replacements": [record.to_dict() for record in self.replacements],
            "warnings": list(self.warnings),
            "summary": {
                "total_targets": self.total_targets,
                "replaced": replaced,
                "skipped": skipped,
                "failed": failed,
            },
        }
