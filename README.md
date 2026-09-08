# Document Chatbot MVP

A local document question-answering chatbot built with **Streamlit, Retrieval-Augmented Generation (RAG), ChromaDB, Sentence Transformers, and Ollama**.

Upload PDF, DOCX, or TXT documents, index them locally, and ask questions about their contents. The system retrieves relevant document chunks and provides grounded answers using a locally running LLM.



## Features

- 📄 Upload **PDF, DOCX, and TXT** documents
- 🔎 Semantic search using local embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 💾 Persistent local vector storage with ChromaDB
- 🤖 Local LLM inference with Ollama
- 🖥️ Streamlit web interface
- 📚 Source document and chunk references
- 🎯 Relevance filtering for retrieved chunks
- 🔒 No external LLM API required
- 🧪 Automated tests for RAG and vector-store behavior

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



### 1. Document ingestion

Uploaded PDF, DOCX, and TXT files are parsed and converted into normalized text.

### 2. Chunking

Documents are divided into smaller chunks so that relevant sections can be retrieved independently.

Default configuration:

- Chunk size: `800` characters
- Chunk overlap: `120` characters

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

- Python 3.10+
- Ollama
- A local Ollama model
- Enough RAM/storage to run the embedding model and LLM locally

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


