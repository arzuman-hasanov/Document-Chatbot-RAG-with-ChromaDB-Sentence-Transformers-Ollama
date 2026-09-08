"""Chunking package for splitting normalized documents into RAG-friendly chunks."""

from .models import DocumentChunk
from .service import DocumentChunker

__all__ = ["DocumentChunk", "DocumentChunker"]
