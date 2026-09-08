# Document Chatbot MVP

A local document question-answering chatbot built with **Streamlit, Retrieval-Augmented Generation (RAG), ChromaDB, Sentence Transformers, and Ollama**.

Upload PDF, DOCX, or TXT documents, index them locally, and ask questions about their contents. The system retrieves relevant document chunks and provides grounded answers using a locally running LLM.

> **MVP status:** This project is designed as a local, single-user prototype for document-based question answering.

## Features

* 📄 Upload **PDF, DOCX, and TXT** documents
* 🔎 Semantic search using local embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 💾 Persistent local vector storage with ChromaDB
* 🤖 Local LLM inference with Ollama
* 🖥️ Streamlit web interface
* 📚 Source document and chunk references
* 🎯 Relevance filtering for retrieved chunks
* 🔒 No external LLM API required
* 🧪 Automated tests for RAG and vector-store behavior

## Architecture

```text
                     ┌─────────────────────┐
                     │   Streamlit UI      │
                     │      app.py         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     ChatService     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     RAGService      │
                     └───────┬───────┬─────┘
                             │       │
                ┌────────────┘       └──────────────┐
                ▼                                   ▼
      ┌───────────────────┐               ┌──────────────────┐
      │ Document Ingestion│               │ Vector Store     │
      │      & Chunking   │               │    ChromaDB      │
      └─────────┬─────────┘               └────────┬─────────┘
                │                                  │
                ▼                                  ▼
      ┌───────────────────┐               ┌──────────────────┐
      │ Sentence          │               │ Semantic         │
      │ Transformers      │◄──────────────┤ Retrieval        │
      └───────────────────┘               └────────┬─────────┘
                                                   │
                                                   ▼
                                        ┌────────────────────┐
                                        │   Ollama LLM       │
                                        │  Local Generation  │
                                        └────────────────────┘
```

## How It Works

The application follows a standard local RAG pipeline:

```text
Document
   ↓
Text extraction
   ↓
Text normalization
   ↓
Chunking
   ↓
Embedding generation
   ↓
ChromaDB
   ↓
User question
   ↓
Question embedding
   ↓
Semantic retrieval
   ↓
Relevance filtering
   ↓
Relevant context
   ↓
Ollama
   ↓
Grounded answer
```

### 1. Document ingestion

Uploaded PDF, DOCX, and TXT files are parsed and converted into normalized text.

### 2. Chunking

Documents are divided into smaller chunks so that relevant sections can be retrieved independently.

Default configuration:

* Chunk size: `800` characters
* Chunk overlap: `120` characters

### 3. Embeddings

The project uses a local Sentence Transformer model to convert document chunks and user questions into vector embeddings.

Default model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 4. Vector search

Embeddings are stored in a persistent **ChromaDB** collection using cosine distance.

Relevant chunks are retrieved based on semantic similarity.

The application also applies a configurable retrieval-distance threshold so that weak or unrelated results are not unnecessarily passed to the LLM.

### 5. Answer generation

Relevant document context is sent to a locally running **Ollama** model.

The LLM is instructed to:

* answer using only the supplied document context
* answer the question directly
* avoid unnecessary explanations
* avoid repeating retrieved context
* avoid inventing information
* indicate when the answer cannot be found

## Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| UI              | Streamlit             |
| Language        | Python                |
| RAG             | Custom RAG pipeline   |
| Vector database | ChromaDB              |
| Embeddings      | Sentence Transformers |
| LLM             | Ollama                |
| Default LLM     | `llama3.2:3b`         |
| PDF extraction  | PyMuPDF               |
| DOCX extraction | python-docx           |
| Testing         | pytest                |

## Project Structure

```text
.
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
│
├── document_chatbot/
│   ├── chat/
│   ├── chunking/
│   ├── embeddings/
│   ├── ingestion/
│   ├── llm/
│   ├── rag/
│   └── vectorstore/
│
├── tests/
│   ├── test_rag_mvp.py
│   └── test_embeddings_vectorstore.py
│
└── data/
    └── chroma_db/
```

## Requirements

* Python 3.10+
* Ollama
* A local Ollama model
* Enough RAM/storage to run the embedding model and LLM locally

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/document-chatbot.git
cd document-chatbot
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and make sure the Ollama server is running.

Pull the default model:

```bash
ollama pull llama3.2:3b
```

Start the Ollama server if necessary:

```bash
ollama serve
```

The application expects Ollama at:

```text
http://localhost:11434
```

You can change this through environment variables.

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Example configuration:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

CHUNK_SIZE=800
CHUNK_OVERLAP=120

RETRIEVAL_RESULT_COUNT=5
RETRIEVAL_DISTANCE_THRESHOLD=1.0

CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=document_chunks
```

On Windows, you can simply create `.env` manually if `cp` is unavailable.

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

## Example

Upload a document such as a CV or technical document.

Then ask:

```text
What is his name?
```

The system retrieves the relevant document chunk and generates a concise grounded response such as:

```text
Arzuman Hasanov
```

The UI also displays the source document and chunk used for the retrieval.

## Testing

Run the complete test suite:

```bash
python -m pytest
```

Compile-check the project:

```bash
python -m compileall -q app.py document_chatbot tests
```

The test suite covers areas including:

* document ingestion
* chunking
* embeddings
* ChromaDB retrieval
* retrieval-distance filtering
* RAG behavior
* source metadata
* no-result behavior
* stale document re-indexing
* cross-document retrieval isolation

## Privacy

The application is designed to run locally.

Documents are processed and stored locally, embeddings are generated locally, and the LLM is accessed through a local Ollama server.

No external LLM API is required for the MVP.

## Current Limitations

This is an MVP and has several known limitations:

* No OCR for scanned/image-only PDFs
* English-focused question answering
* Character-based chunking
* No reranking model
* No exact citation offsets
* No answer confidence score
* Limited document-format support
* DOCX extraction focuses primarily on document paragraphs
* Local/single-user architecture
* Performance depends on local hardware
* Ollama must be installed and running for answer generation

## Future Improvements

Potential future improvements include:

* OCR support for scanned PDFs
* Better semantic/structure-aware chunking
* Retrieval reranking
* Hybrid keyword + semantic search
* Improved citation and source highlighting
* Streaming LLM responses
* Conversation memory
* Multiple embedding-model support
* Additional document formats
* Authentication and multi-user support
* Production deployment
* Evaluation datasets and retrieval/answer quality metrics

## License

Add your preferred license here.

For example:

```text
MIT License
```

## Author

**Arzuman Hasanov**

Built as a local RAG/document-Q&A MVP using Python, Streamlit, ChromaDB, Sentence Transformers, and Ollama.
