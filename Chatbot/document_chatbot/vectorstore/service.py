"""Local vector storage built on ChromaDB."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path

import chromadb

from document_chatbot.chunking.models import DocumentChunk
from document_chatbot.config import (
    DEFAULT_SEARCH_RESULTS,
    RETRIEVAL_DISTANCE_THRESHOLD,
    VECTOR_DB_COLLECTION_NAME,
    VECTOR_DB_PATH,
)
from document_chatbot.embeddings.service import EmbeddingService


logger = logging.getLogger(__name__)
TEMPORARY_SOURCE_PATTERN = re.compile(r"^tmp[a-z0-9]+(?:\.[a-z0-9]+)?$", re.IGNORECASE)


class VectorStoreService:
    """Persist document chunks and perform local similarity search with ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = VECTOR_DB_PATH,
        collection_name: str = VECTOR_DB_COLLECTION_NAME,
    ) -> None:
        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self) -> bool:
        return any(collection.name == self.collection_name for collection in self.client.list_collections())

    def is_usable(self) -> bool:
        return self.collection_exists() and self.collection is not None

    def _flatten_metadata(self, chunk: DocumentChunk) -> dict[str, str | int | float | bool | None | list[str | int | float | bool | None]]:
        flattened: dict[str, str | int | float | bool | None | list[str | int | float | bool | None]] = {
            "chunk_id": chunk.chunk_id,
            "source_document_filename": chunk.source_document_filename,
            "chunk_index": chunk.chunk_index,
            "source_metadata_json": json.dumps(chunk.source_metadata, ensure_ascii=False),
        }

        for key, value in chunk.source_metadata.items():
            if isinstance(value, (dict, list)):
                flattened[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flattened[key] = value
            else:
                flattened[key] = str(value)

        return flattened

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        if chunks is None or not chunks:
            raise ValueError("At least one chunk is required.")

        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("Embeddings count must match the number of chunks.")

        if embeddings is None:
            raise ValueError("Embeddings are required when indexing chunks.")

        documents = [chunk.chunk_text for chunk in chunks]
        metadatas = [self._flatten_metadata(chunk) for chunk in chunks]

        ids = [chunk.chunk_id for chunk in chunks]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=list(embeddings),
        )
        return ids

    def similarity_search(
        self,
        query: str,
        n_results: int = DEFAULT_SEARCH_RESULTS,
        embedding_service: EmbeddingService | None = None,
        distance_threshold: float = RETRIEVAL_DISTANCE_THRESHOLD,
    ) -> list[DocumentChunk]:
        if query is None or not str(query).strip():
            raise ValueError("Query text cannot be empty.")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be zero or greater.")

        if not self.collection_exists():
            return []

        if embedding_service is None:
            embedding_service = EmbeddingService()

        embedding_started = time.perf_counter()
        embedding = embedding_service.embed_text(query)
        embedding_elapsed = time.perf_counter() - embedding_started
        logger.info(
            "RAG timing: query_embedding_seconds=%.4f embedding_model=%s",
            embedding_elapsed,
            getattr(embedding_service, "model_name", "unknown"),
        )

        retrieval_started = time.perf_counter()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_elapsed = time.perf_counter() - retrieval_started

        document_texts = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: list[DocumentChunk] = []
        for index, document_text in enumerate(document_texts):
            distance = float(distances[index]) if index < len(distances) else None
            if distance is not None and distance > distance_threshold:
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            source_metadata_json = metadata.get("source_metadata_json")
            source_metadata: dict[str, object] = {}
            if source_metadata_json:
                try:
                    source_metadata = json.loads(source_metadata_json)
                except json.JSONDecodeError:
                    source_metadata = {}

            for key, value in metadata.items():
                if key in {"chunk_id", "source_document_filename", "chunk_index", "source_metadata_json"}:
                    continue
                source_metadata[key] = value

            chunks.append(
                DocumentChunk(
                    chunk_id=str(ids[index]) if index < len(ids) else str(index),
                    source_document_filename=str(metadata.get("source_document_filename", "unknown")),
                    chunk_text=str(document_text),
                    chunk_index=int(metadata.get("chunk_index", index)),
                    source_metadata=source_metadata,
                    retrieval_distance=distance,
                )
            )
        logger.info(
            "RAG timing: chroma_retrieval_seconds=%.4f retrieved_chunks=%d",
            retrieval_elapsed,
            len(chunks),
        )
        return chunks

    def delete_source_document(self, source_document_filename: str) -> int:
        if not source_document_filename:
            raise ValueError("source_document_filename cannot be empty.")
        if not self.collection_exists():
            return 0

        matches = self.collection.get(
            where={"source_document_filename": source_document_filename},
        )
        ids = matches.get("ids", [])
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)

    def has_source_document(self, source_document_filename: str) -> bool:
        if not source_document_filename or not self.collection_exists():
            return False
        matches = self.collection.get(where={"source_document_filename": source_document_filename}, include=["metadatas"])
        return bool(matches.get("ids", []))

    def has_document_hash(self, document_hash: str) -> bool:
        if not document_hash or not self.collection_exists():
            return False
        matches = self.collection.get(where={"document_hash": document_hash}, include=["metadatas"])
        return bool(matches.get("ids", []))

    def cleanup_legacy_duplicates(self) -> list[dict[str, str | int]]:
        """Remove temporary-name chunks only when canonical chunks contain identical evidence."""
        if not self.collection_exists():
            return []

        records = self.collection.get(include=["documents", "metadatas"])
        documents = records.get("documents", [])
        metadatas = records.get("metadatas", [])
        ids = records.get("ids", [])
        canonical_keys: set[tuple[int, str]] = set()
        temporary_records: list[tuple[str, str, int, str]] = []

        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            filename = str(metadata.get("source_document_filename", ""))
            chunk_index = int(metadata.get("chunk_index", index))
            document_text = str(documents[index]) if index < len(documents) else ""
            key = (chunk_index, " ".join(document_text.split()))
            if TEMPORARY_SOURCE_PATTERN.match(filename):
                if index < len(ids):
                    temporary_records.append((str(ids[index]), filename, chunk_index, key[1]))
            else:
                canonical_keys.add(key)

        removed: list[dict[str, str | int]] = []
        ids_to_delete: list[str] = []
        for record_id, filename, chunk_index, normalized_text in temporary_records:
            if (chunk_index, normalized_text) not in canonical_keys:
                continue
            ids_to_delete.append(record_id)
            removed.append(
                {
                    "chunk_id": record_id,
                    "source_document_filename": filename,
                    "chunk_index": chunk_index,
                }
            )

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return removed

    def list_indexed_documents(self) -> list[str]:
        if not self.collection_exists():
            return []

        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        filenames: set[str] = set()
        for metadata in metadatas:
            if isinstance(metadata, dict) and "source_document_filename" in metadata:
                filename = metadata.get("source_document_filename")
                if filename:
                    filenames.add(str(filename))
        return sorted(filenames)

    def delete_all(self) -> None:
        self.collection.delete(where={})
