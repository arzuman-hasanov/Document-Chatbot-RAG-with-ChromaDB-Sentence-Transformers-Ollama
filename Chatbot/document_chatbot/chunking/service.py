"""Service for splitting source documents into manageable text chunks."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from document_chatbot.config import CHUNK_OVERLAP, CHUNK_SIZE
from document_chatbot.ingestion.models import Document

from .models import DocumentChunk


class DocumentChunker:
    """Split normalized documents into overlapping chunks for later embedding."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        if document is None:
            return []

        text = (document.extracted_text or "").strip()
        if not text:
            return []

        segments = self._split_text_into_segments(text)
        if not segments:
            return []

        chunks: list[DocumentChunk] = []
        current = ""

        for segment in segments:
            if not current:
                current = segment
                continue

            combined = f"{current} {segment}".strip()
            if len(combined) <= self.chunk_size:
                current = combined
                continue

            chunks.append(self._build_chunk(document, len(chunks), current))
            overlap_text = self._get_overlap(current, self.chunk_overlap)
            current = f"{overlap_text} {segment}".strip() if overlap_text else segment
            if len(current) > self.chunk_size:
                current = current[: self.chunk_size].rstrip()

        if current.strip():
            chunks.append(self._build_chunk(document, len(chunks), current))

        return chunks

    def _split_text_into_segments(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]

        segments: list[str] = []
        for paragraph in paragraphs:
            segments.extend(self._split_paragraph(paragraph))

        return [segment.strip() for segment in segments if segment and segment.strip()]

    def _split_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.chunk_size:
            return [paragraph]

        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", paragraph) if piece.strip()]
        if not sentences:
            return [paragraph]

        segments: list[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) <= self.chunk_size:
                if not current:
                    current = sentence
                elif len(f"{current} {sentence}") <= self.chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    segments.append(current.strip())
                    current = sentence
            else:
                if current:
                    segments.append(current.strip())
                    current = ""
                segments.extend(self._split_long_sentence(sentence))

        if current.strip():
            segments.append(current.strip())

        return segments

    def _split_long_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        if not words:
            return []

        segments: list[str] = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    segments.append(current)
                current = word

        if current:
            segments.append(current)

        return segments

    def _get_overlap(self, text: str, overlap_size: int) -> str:
        if overlap_size <= 0 or len(text) <= overlap_size:
            return ""

        tail = text[-overlap_size:]
        stripped = tail.strip()
        if not stripped:
            return ""
        return stripped

    def _build_chunk(self, document: Document, chunk_index: int, chunk_text: str) -> DocumentChunk:
        metadata: dict[str, Any] = dict(document.metadata or {})
        metadata["source_document_filename"] = document.source_filename
        metadata["source_file_type"] = document.file_type

        return DocumentChunk(
            chunk_id=uuid4().hex,
            source_document_filename=document.source_filename,
            chunk_text=chunk_text.strip(),
            chunk_index=chunk_index,
            source_metadata=metadata,
        )
