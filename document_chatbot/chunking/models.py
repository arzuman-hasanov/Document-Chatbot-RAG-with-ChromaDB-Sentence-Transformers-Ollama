"""Normalized chunk models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    """A chunk extracted from a source document."""

    chunk_id: str
    source_document_filename: str
    chunk_text: str
    chunk_index: int
    source_metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_distance: float | None = None

    def __post_init__(self) -> None:
        self.chunk_text = self.chunk_text.strip()
