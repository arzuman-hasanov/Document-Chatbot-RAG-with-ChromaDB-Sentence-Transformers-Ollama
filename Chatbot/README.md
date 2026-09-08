# Document Chatbot MVP

This project is a lightweight document Q&A application for English documents and English questions. It includes ingestion, chunking, local embeddings, ChromaDB vector storage, retrieval, a local Ollama LLM layer, and a Streamlit UI.

## Selected Ollama model

I selected `llama3.2:3b` for the local LLM. It is a practical, small model for a normal Windows PC and works well for local document-grounded Q&A without requiring any paid API.

## Overview

The MVP follows this flow:

1. Upload a PDF, DOCX, or TXT file.
2. Parse and normalize the document text.
3. Split the text into chunks with overlap.
4. Generate local embeddings with `sentence-transformers`.
5. Store chunks and metadata in a persistent ChromaDB collection.
6. Embed the user question and retrieve the most relevant chunks.
7. Send those chunks to Ollama as context.
8. Return an answer with source documents and chunk references.

## Project structure

```text
.
├── .env.example
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
├── app.py
├── main.py
├── document_chatbot/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── config.py
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── service.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── service.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── service.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── parsers.py
│   │   └── service.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── service.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── service.py
│   ├── services/
│   │   └── __init__.py
│   └── vectorstore/
│       ├── __init__.py
│       └── service.py
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings_vectorstore.py
│   ├── test_ingestion.py
│   ├── test_rag_mvp.py
│   └── __init__.py
├── data/
│   └── chroma_db/
├── .venv/
└── .cache/
```

## Python version

This project targets Python 3.13.

## Setup

Create and activate a virtual environment on Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ollama setup on Windows

1. Download and install Ollama from:
   https://ollama.com/download/windows
2. After installation, restart your terminal or PowerShell session.
3. Pull the selected model:

```powershell
ollama pull llama3.2:3b
```

4. Start the local Ollama server:

```powershell
ollama serve
```

5. Optionally verify the model is available:

```powershell
ollama list
```

If you prefer to run the model directly for testing:

```powershell
ollama run llama3.2:3b
```

## Environment configuration

Copy the example environment file and set your values:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and configure:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_RESULT_COUNT=5
RETRIEVAL_DISTANCE_THRESHOLD=1.0
VECTOR_DB_PATH=./data/chroma_db
VECTOR_DB_COLLECTION_NAME=document_chunks
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

`RETRIEVAL_DISTANCE_THRESHOLD` is a maximum Chroma cosine distance: lower values
are more relevant. The default `1.0` filters weak cross-document matches while
retaining close matches. Set it to `2.0` to disable distance filtering.

Important notes:
- There is no API key required for Ollama.
- The app uses local ChromaDB storage under `data/chroma_db` by default.
- The first run downloads the local embedding model to the Hugging Face cache if it is not already available.

## Start the Streamlit app

```powershell
streamlit run app.py
```

The app will allow you to:
- upload PDF, DOCX, and TXT files
- index documents
- view indexed source documents
- ask English questions
- view the answer and source references

## How it works

The document chatbot uses:

- `DocumentIngestionService` to parse uploaded files.
- `DocumentChunker` to split text into useful chunks.
- `EmbeddingService` to create local embeddings with a multilingual sentence-transformers model.
- `VectorStoreService` to store chunks and metadata in ChromaDB.
- `RAGService` to orchestrate retrieval.
- `OllamaChatService` to generate final answers using the retrieved document context.

The LLM is instructed to answer only from the provided context and to say clearly when the information is not found in the uploaded documents.

## Development timing logs

The application emits INFO-level timing logs while answering a question. The logs include:

- query embedding time
- ChromaDB retrieval time
- Ollama generation time
- total response time
- configured embedding and Ollama models
- embedding device (CPU/GPU when reported by sentence-transformers)
- retrieved chunk count
- approximate context/prompt size

Run Streamlit from a terminal to see these logs:

```powershell
streamlit run app.py
```

The Ollama HTTP API does not expose CPU/GPU usage for an individual chat request, so Ollama acceleration is logged as `unknown` rather than guessed.

## Example usage

```python
from pathlib import Path

from document_chatbot.chat import ChatService
from document_chatbot.rag import RAGService

rag_service = RAGService()
rag_service.index_document(Path("sample.pdf"))

chat_service = ChatService(rag_service=rag_service)
response = chat_service.ask("What is the main topic of the document?")

print(response.answer)
print(response.source_documents)
print(response.source_chunk_indexes)
```

## Testing

Run the full suite:

```powershell
python -m pytest
```

## Current limitations

- This MVP is intentionally limited to English documents and English questions.
- It uses a local model for embeddings and a local ChromaDB instance; it is not a production-scale distributed deployment.
- The app does not include authentication, multi-user support, or advanced UI features.
- Ollama must be installed and running locally on the machine.
- The first embedding model download may take time on the first run.
