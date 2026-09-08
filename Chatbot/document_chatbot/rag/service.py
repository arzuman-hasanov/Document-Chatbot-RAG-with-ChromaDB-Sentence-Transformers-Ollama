"""RAG orchestration service for document indexing and retrieval."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from document_chatbot.chunking import DocumentChunker
from document_chatbot.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SEARCH_RESULTS,
    RETRIEVAL_DISTANCE_THRESHOLD,
)
from document_chatbot.embeddings import EmbeddingService
from document_chatbot.ingestion import DocumentIngestionService
from document_chatbot.llm.service import build_grounded_prompt
from document_chatbot.vectorstore import VectorStoreService


class IndexingResult(list[dict[str, Any]]):
    """Indexed chunks plus a status that preserves the historical list API."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        status: str,
        source_filename: str,
        document_hash: str,
    ) -> None:
        super().__init__(chunks)
        self.status = status
        self.source_filename = source_filename
        self.document_hash = document_hash


class RAGService:
    """Connects ingestion, chunking, embedding, and vector search."""

    def __init__(
        self,
        ingestion_service: DocumentIngestionService | None = None,
        chunker: DocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store_service: VectorStoreService | None = None,
        retrieval_result_count: int = DEFAULT_SEARCH_RESULTS,
        retrieval_distance_threshold: float = RETRIEVAL_DISTANCE_THRESHOLD,
    ) -> None:
        self.ingestion_service = ingestion_service or DocumentIngestionService()
        self.chunker = chunker or DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.retrieval_result_count = retrieval_result_count
        self.retrieval_distance_threshold = retrieval_distance_threshold

    def index_document(
        self,
        file_path: str | Path,
        source_filename: str | None = None,
    ) -> IndexingResult:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document does not exist: {path}")

        document_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        document = self.ingestion_service.parse_document(path, source_filename=source_filename)
        document.metadata["document_hash"] = document_hash
        already_indexed = self.vector_store_service.has_document_hash(document_hash)
        if already_indexed:
            return IndexingResult(
                [],
                status="already_indexed",
                source_filename=document.source_filename,
                document_hash=document_hash,
            )

        had_existing_filename = self.vector_store_service.has_source_document(document.source_filename)
        chunks = self.chunker.chunk_document(document)
        if not chunks:
            raise ValueError(f"No chunks were produced from document: {document.source_filename}")

        self.vector_store_service.delete_source_document(document.source_filename)

        embeddings = self.embedding_service.embed_chunks(chunks)
        self.vector_store_service.add_chunks(chunks, embeddings)

        result = [
            {
                "chunk_id": chunk.chunk_id,
                "source_document_filename": chunk.source_document_filename,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
            }
            for chunk in chunks
        ]
        return IndexingResult(
            result,
            status="reindexed" if had_existing_filename else "indexed",
            source_filename=document.source_filename,
            document_hash=document_hash,
        )

    def retrieve(self, question: str, n_results: int | None = None) -> list[Any]:
        if question is None or not str(question).strip():
            raise ValueError("Question cannot be empty.")

        limit = n_results if n_results is not None else self.retrieval_result_count
        if limit <= 0:
            raise ValueError("n_results must be greater than zero.")

        collection_count = self.vector_store_service.collection.count()
        retrieved = self.vector_store_service.similarity_search(
            query=str(question).strip(),
            n_results=max(limit, collection_count),
            embedding_service=self.embedding_service,
            distance_threshold=self.retrieval_distance_threshold,
        )
        return self.deduplicate_chunks(retrieved)[:limit]

    @staticmethod
    def deduplicate_chunks(chunks: list[Any]) -> list[Any]:
        """Remove duplicate evidence, preferring canonical names over temporary names."""
        canonical_content_keys = {
            RAGService._content_identity(chunk)
            for chunk in chunks
            if chunk.source_metadata.get("document_hash")
            and not RAGService._is_temporary_filename(chunk.source_document_filename)
        }
        selected: dict[str, Any] = {}
        for chunk in chunks:
            if (
                not chunk.source_metadata.get("document_hash")
                and RAGService._is_temporary_filename(chunk.source_document_filename)
                and RAGService._content_identity(chunk) in canonical_content_keys
            ):
                continue
            identity = RAGService._chunk_identity(chunk)
            existing = selected.get(identity)
            if existing is None or RAGService._is_preferred_chunk(chunk, existing):
                selected[identity] = chunk

        return [chunk for chunk in chunks if selected.get(RAGService._chunk_identity(chunk)) is chunk]

    @staticmethod
    def _normalized_chunk_text(chunk: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(chunk.chunk_text))
        return " ".join(normalized.split())

    @staticmethod
    def _content_identity(chunk: Any) -> str:
        normalized = RAGService._normalized_chunk_text(chunk)
        return f"content:{chunk.chunk_index}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _chunk_identity(chunk: Any) -> str:
        document_hash = chunk.source_metadata.get("document_hash")
        if document_hash:
            return f"hash:{document_hash}:{chunk.chunk_index}"
        return RAGService._content_identity(chunk)

    @staticmethod
    def _is_temporary_filename(filename: str) -> bool:
        return bool(re.match(r"^tmp[a-z0-9]+(?:\.[a-z0-9]+)?$", filename, re.IGNORECASE))

    @staticmethod
    def _is_preferred_chunk(candidate: Any, current: Any) -> bool:
        candidate_temporary = RAGService._is_temporary_filename(candidate.source_document_filename)
        current_temporary = RAGService._is_temporary_filename(current.source_document_filename)
        if candidate_temporary != current_temporary:
            return not candidate_temporary
        return bool(candidate.source_metadata.get("document_hash")) and not bool(
            current.source_metadata.get("document_hash")
        )

    def cleanup_legacy_duplicates(self) -> list[dict[str, str | int]]:
        return self.vector_store_service.cleanup_legacy_duplicates()

    def build_context(self, chunks: list[Any], max_chars: int = 4000) -> str:
        if not chunks:
            return ""

        references: list[str] = []
        for chunk in chunks:
            references.append(
                f"[Source: {chunk.source_document_filename} | Chunk {chunk.chunk_index}]\n{chunk.chunk_text}"
            )

        context = "\n\n".join(references)
        if len(context) > max_chars:
            return context[:max_chars].rstrip() + "..."
        return context

    def build_prompt(self, question: str, chunks: list[Any]) -> str:
        if question is None or not str(question).strip():
            raise ValueError("Question cannot be empty.")

        context = self.build_context(chunks)
        return build_grounded_prompt(question, context)

    def list_indexed_documents(self) -> list[str]:
        return self.vector_store_service.list_indexed_documents()
