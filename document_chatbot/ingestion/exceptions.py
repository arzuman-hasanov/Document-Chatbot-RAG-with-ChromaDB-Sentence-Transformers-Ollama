"""Custom exceptions for document parsing failures."""


class DocumentParsingError(ValueError):
    """Base exception for document parsing issues."""


class UnsupportedDocumentFormatError(DocumentParsingError):
    """Raised when a document type is not supported."""


class EmptyDocumentError(DocumentParsingError):
    """Raised when a document exists but contains no usable text."""
