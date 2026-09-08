"""Base abstraction for document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import Document


class DocumentParser(ABC):
    """All document parsers must implement a common interface."""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """The supported file type identifier, e.g. 'txt', 'pdf', 'docx'."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower() == self.file_type.lower()

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Extract text and metadata from a document file."""
