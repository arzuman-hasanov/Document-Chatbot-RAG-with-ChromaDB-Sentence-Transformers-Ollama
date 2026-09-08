"""Chat response models for the document chatbot MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatResponse:
    """Structured result returned by the chat service."""

    answer: str
    source_documents: list[str] = field(default_factory=list)
    source_chunk_indexes: list[int] = field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
