"""Document chatbot application package."""

from .chat import ChatResponse, ChatService
from .chunking import DocumentChunk, DocumentChunker
from .embeddings import EmbeddingService
from .ingestion import Document, DocumentIngestionService
from .llm import LLMService, OllamaChatService
from .rag import RAGService
from .vectorstore import VectorStoreService

__all__ = [
    "__version__",
    "ChatResponse",
    "ChatService",
    "Document",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentIngestionService",
    "EmbeddingService",
    "LLMService",
    "OllamaChatService",
    "RAGService",
    "VectorStoreService",
]

__version__ = "0.1.0"
