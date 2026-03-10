from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ReplacementTarget:
    field_type: str
    field_id: str
    old_text: str
    new_text: str


@dataclass
class DocxReplacementError(Exception):
    message: str
    details: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        return self.message


@dataclass
class DocxAlignmentError(DocxReplacementError):
    """Raised when paragraph alignment verification fails."""


@dataclass
class DocxInvariantError(DocxReplacementError):
    """Raised when document invariants are violated after replacement."""


@dataclass
class DocxOverflowError(DocxReplacementError):
    """Raised when overflow evaluation inputs or invariants are invalid."""
