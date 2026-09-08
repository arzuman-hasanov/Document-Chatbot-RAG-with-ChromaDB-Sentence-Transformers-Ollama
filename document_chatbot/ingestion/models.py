"""Internal document model used across ingestion and future pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Normalized internal representation for extracted document content."""

    source_filename: str
    file_type: str
    extracted_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extracted_text = self.extracted_text.strip()
