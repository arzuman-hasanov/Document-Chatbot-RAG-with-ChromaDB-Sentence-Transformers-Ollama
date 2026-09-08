"""Application configuration settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "document_chatbot"
APP_VERSION = "0.1.0"

# Default chunking settings used for future embedding and retrieval workflows.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Default embedding configuration.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_MODEL_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface")

# Default local vector database configuration.
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "chroma_db"))
VECTOR_DB_COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "document_chunks")
DEFAULT_SEARCH_RESULTS = int(os.getenv("RETRIEVAL_RESULT_COUNT", "5"))
# Chroma cosine distance is lower for more relevant matches; 1.0 keeps close
# matches while filtering weak cross-document results. Set to 2.0 to disable.
RETRIEVAL_DISTANCE_THRESHOLD = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "1.0"))

# Local LLM configuration.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
