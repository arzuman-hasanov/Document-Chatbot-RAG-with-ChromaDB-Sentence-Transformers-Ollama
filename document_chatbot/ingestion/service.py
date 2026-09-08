"""Service entry point for document ingestion."""

from __future__ import annotations

from pathlib import Path

from .exceptions import UnsupportedDocumentFormatError
from .models import Document
from .parsers import DocxParser, PdfParser, TxtParser


class DocumentIngestionService:
    """Detect a document type and parse it using the correct parser."""

    _parsers = {
        "txt": TxtParser(),
        "pdf": PdfParser(),
        "docx": DocxParser(),
    }

    def parse_document(self, file_path: str | Path, source_filename: str | None = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        file_type = path.suffix.lower().lstrip(".")
        parser = self._parsers.get(file_type)
        if parser is None:
            raise UnsupportedDocumentFormatError(
                f"Unsupported document format: {file_type}. Supported formats are: txt, pdf, docx."
            )

        document = parser.parse(path)
        if source_filename:
            document.source_filename = source_filename
        return document
