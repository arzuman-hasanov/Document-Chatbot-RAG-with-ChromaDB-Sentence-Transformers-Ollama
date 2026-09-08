"""Chat service orchestrating retrieval and answer generation."""

from __future__ import annotations

import logging
import time

from document_chatbot.config import DEFAULT_SEARCH_RESULTS, OLLAMA_MODEL
from document_chatbot.llm import LLMService, OllamaChatService
from document_chatbot.rag import RAGService

from .models import ChatResponse


logger = logging.getLogger(__name__)


class ChatService:
    """Expose a simple ask(question) interface for the MVP."""

    def __init__(
        self,
        rag_service: RAGService | None = None,
        llm_service: LLMService | None = None,
        retrieval_result_count: int = DEFAULT_SEARCH_RESULTS,
    ) -> None:
        self.rag_service = rag_service or RAGService(retrieval_result_count=retrieval_result_count)
        self.llm_service = llm_service

    def _get_llm_service(self) -> LLMService:
        if self.llm_service is None:
            self.llm_service = OllamaChatService()
        return self.llm_service

    def ask(self, question: str) -> ChatResponse:
        total_started = time.perf_counter()
        if question is None or not str(question).strip():
            raise ValueError("Question cannot be empty.")

        retrieved_chunks = self.rag_service.retrieve(str(question).strip(), self.rag_service.retrieval_result_count)
        logger.info(
            "RAG request: embedding_model=%s ollama_model=%s retrieved_chunks=%d",
            getattr(self.rag_service.embedding_service, "model_name", "unknown"),
            getattr(self.llm_service, "model", OLLAMA_MODEL),
            len(retrieved_chunks),
        )
        if not retrieved_chunks:
            logger.info(
                "RAG timing: total_response_seconds=%.4f",
                time.perf_counter() - total_started,
            )
            return ChatResponse(
                answer="I could not find relevant information in the provided documents.",
                source_documents=[],
                source_chunk_indexes=[],
                retrieved_context=[],
            )

        context = self.rag_service.build_context(retrieved_chunks)
        logger.info(
            "RAG context: context_chars=%d context_approx_tokens=%d",
            len(context),
            max(1, len(context) // 4),
        )
        try:
            answer = self._get_llm_service().generate(str(question).strip(), context)
        except ValueError as exc:
            response = ChatResponse(
                answer=str(exc),
                source_documents=list(dict.fromkeys(chunk.source_document_filename for chunk in retrieved_chunks)),
                source_chunk_indexes=[chunk.chunk_index for chunk in retrieved_chunks],
                retrieved_context=[
                    {
                        "source_document": chunk.source_document_filename,
                        "chunk_index": chunk.chunk_index,
                        "chunk_text": chunk.chunk_text,
                    }
                    for chunk in retrieved_chunks
                ],
            )
            logger.info(
                "RAG timing: total_response_seconds=%.4f",
                time.perf_counter() - total_started,
            )
            return response

        response = ChatResponse(
            answer=answer,
            source_documents=list(dict.fromkeys(chunk.source_document_filename for chunk in retrieved_chunks)),
            source_chunk_indexes=[chunk.chunk_index for chunk in retrieved_chunks],
            retrieved_context=[
                {
                    "source_document": chunk.source_document_filename,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.chunk_text,
                }
                for chunk in retrieved_chunks
            ],
        )

        logger.info(
            "RAG timing: total_response_seconds=%.4f",
            time.perf_counter() - total_started,
        )
        return response

    def list_indexed_documents(self) -> list[str]:
        return self.rag_service.list_indexed_documents()
