"""Service for generating local text embeddings using sentence-transformers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from document_chatbot.chunking.models import DocumentChunk
from document_chatbot.config import EMBEDDING_MODEL_CACHE_DIR, EMBEDDING_MODEL_NAME


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for text and chunk objects using a local multilingual model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str | None = EMBEDDING_MODEL_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
        logger.info(
            "Embedding model configured: model=%s device=%s",
            self.model_name,
            getattr(self.model, "device", "unknown"),
        )

    def embed_text(self, text: str) -> list[float]:
        if text is None or not str(text).strip():
            raise ValueError("Text input cannot be empty.")

        embedding = self.model.encode(str(text), convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(float).tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("At least one text value is required.")

        cleaned = [str(text).strip() for text in texts]
        if any(not value for value in cleaned):
            raise ValueError("Text inputs cannot contain empty values.")

        embeddings = self.model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(float).tolist()

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[list[float]]:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        return self.embed_texts([chunk.chunk_text for chunk in chunks])
