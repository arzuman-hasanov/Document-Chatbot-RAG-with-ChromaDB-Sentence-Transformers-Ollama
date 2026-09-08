# Project Context: Document Chatbot MVP

This document is a technical briefing for ChatGPT or another coding assistant working on this repository. It describes the current implementation, contracts, runtime flow, configuration, tests, and limitations. Treat the code as the source of truth if this document and the implementation ever disagree.

## 1. Purpose and scope

This is a local document question-answering application for English documents and English questions. It implements a retrieval-augmented generation (RAG) pipeline:

1. Accept PDF, DOCX, or TXT files.
2. Extract and normalize their text.
3. Split text into overlapping chunks.
4. Generate local sentence-transformer embeddings.
5. Persist chunks, embeddings, and metadata in ChromaDB.
6. Embed a user question and retrieve similar chunks.
7. Send the retrieved context to a local Ollama model.
8. Return an answer plus source document and chunk references.

The project is an MVP. It is local-only, single-user, English-focused, and not a production distributed service.

## 2. Current entry points

### Streamlit application: the real user-facing product

Run:

```powershell
streamlit run app.py
```

The root `app.py` creates a cached `ChatService`, accepts multiple uploads, indexes supported files, lists indexed filenames, accepts a question, and displays the answer, source references, and retrieved context.

Uploaded files are copied to temporary files because the ingestion layer expects a filesystem path. The temporary file is removed after indexing, whether indexing succeeds or fails.

### CLI/package entry point: scaffold only

`main.py` delegates to `document_chatbot.__main__`, which delegates to `document_chatbot.app.main`. That function only prints the package name/version and scaffold text and exits successfully. It does not start Streamlit or run the RAG pipeline.

Therefore, do not treat `python main.py` or `python -m document_chatbot` as the chatbot UI. The working UI is the root Streamlit script.

## 3. Architecture

The main service chain is:

```text
Streamlit app
    -> ChatService
        -> RAGService
            -> DocumentIngestionService
            -> DocumentChunker
            -> EmbeddingService
            -> VectorStoreService / ChromaDB
        -> OllamaChatService
```

Indexing uses ingestion, chunking, embeddings, and vector storage. Asking a question uses retrieval, context construction, and the LLM. The LLM is lazy: `ChatService` does not create `OllamaChatService` until the first question that has retrieved chunks.

All package-level services are importable from `document_chatbot.__init__`, including `ChatService`, `RAGService`, `DocumentIngestionService`, `DocumentChunker`, `EmbeddingService`, `VectorStoreService`, and the LLM abstractions.

## 4. Configuration

Configuration is loaded in `document_chatbot/config.py`. `python-dotenv` loads `.env` automatically. Environment values are read at import time, so restart the Python process after changing them.

Defaults:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Local generation model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP server |
| `CHUNK_SIZE` | `800` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `120` | Character tail copied into the next chunk |
| `RETRIEVAL_RESULT_COUNT` | `5` | Number of Chroma results requested |
| `VECTOR_DB_PATH` | `./data/chroma_db` | Persistent Chroma directory |
| `VECTOR_DB_COLLECTION_NAME` | `document_chunks` | Chroma collection name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model |

The embedding model is cached under `.cache/huggingface` relative to the repository. The first use may download model files. The model is multilingual even though the MVP UI and prompt are designed for English.

Python version target is 3.13, recorded in `.python-version` and the README.

## 5. Ingestion layer

Files in `document_chatbot/ingestion/` define a common parser abstraction and three concrete parsers.

### `Document`

`Document` is a slotted dataclass with:

- `source_filename: str`: basename used as the source identity.
- `file_type: str`: `txt`, `pdf`, or `docx`.
- `extracted_text: str`: normalized text; `__post_init__` strips outer whitespace.
- `metadata: dict[str, Any]`: parser-specific metadata.

### `DocumentParser`

The abstract parser contract has a `file_type` property, a case-insensitive `supports()` helper, and `parse(Path) -> Document`.

### `TxtParser`

- Rejects a missing path with `DocumentParsingError`.
- Rejects empty bytes and whitespace-only text with `EmptyDocumentError`.
- Tries `utf-8-sig`, then `utf-16`, then `latin-1`.
- Strips the resulting text.
- Records byte size, line count, and the selected encoding.

### `PdfParser`

- Rejects a missing path.
- Wraps PDF open failures as `DocumentParsingError`.
- Attempts to decrypt encrypted PDFs with an empty password.
- Extracts text from each page using `pypdf`; blank pages are skipped.
- Joins non-empty page text with two newlines.
- Rejects PDFs with no extractable text.
- Records page count, encryption status, and normalized PDF metadata.

Scanned/image-only PDFs will generally fail as empty because OCR is not implemented.

### `DocxParser`

- Rejects a missing path.
- Opens the file with `python-docx` and wraps invalid-file failures.
- Keeps non-empty paragraphs, strips each paragraph, and joins them with newlines.
- Rejects documents containing no non-empty paragraphs.
- Records paragraph count, section count, and selected core properties: author, title, subject, keywords, created, and modified.

Tables, headers, footers, images, and OCR content are not explicitly extracted.

### `DocumentIngestionService`

`parse_document(path)` validates that the path exists and is a file, determines the format from the lowercase suffix, and dispatches to a singleton parser from its internal map. Supported suffixes are exactly `txt`, `pdf`, and `docx`.

Exceptions:

- `DocumentParsingError`: base value error for parsing problems.
- `UnsupportedDocumentFormatError`: unsupported suffix.
- `EmptyDocumentError`: file exists but has no usable text.

## 6. Chunking layer

`DocumentChunker` converts a `Document` into `DocumentChunk` objects.

`DocumentChunk` contains:

- `chunk_id`: random UUID hex string.
- `source_document_filename`: source basename.
- `chunk_text`: stripped text.
- `chunk_index`: zero-based sequential index.
- `source_metadata`: copied document metadata plus `source_document_filename` and `source_file_type`.

Constructor validation requires:

- `chunk_size > 0`.
- `chunk_overlap >= 0`.
- `chunk_overlap < chunk_size`.

The algorithm is character-based, not token-based:

1. Normalize line endings.
2. Split on blank-line paragraph boundaries.
3. Keep short paragraphs intact.
4. Split long paragraphs into sentences using punctuation `.`, `!`, or `?` followed by whitespace.
5. Group sentences up to `chunk_size`.
6. Split a sentence longer than the limit by whitespace-separated words.
7. Combine segments into chunks up to the limit.
8. When starting a new chunk, prepend the last `chunk_overlap` characters from the previous chunk.

A `None` document, empty text, or no segments produces an empty list. Chunk indexes are regenerated from zero for every document. The overlap is a raw character tail, so it can begin or end mid-word; this is intentional in the current simple implementation.

## 7. Embeddings

`EmbeddingService` wraps `sentence_transformers.SentenceTransformer`.

- It loads the configured model at construction time.
- It passes the configured cache directory to SentenceTransformer.
- It uses the model's selected device and logs it when available.
- `embed_text()` rejects empty input and returns one normalized embedding as `list[float]`.
- `embed_texts()` rejects an empty sequence or any empty item and returns one embedding per input.
- `embed_chunks()` embeds each `DocumentChunk.chunk_text` and rejects an empty chunk list.

Embeddings are normalized (`normalize_embeddings=True`) and converted from NumPy arrays to Python float lists. The same embedding service is used for indexing and query embedding, which keeps vector dimensions and model semantics consistent.

## 8. Vector storage

`VectorStoreService` wraps a persistent ChromaDB client.

At construction it creates the configured directory, opens a `PersistentClient`, and gets or creates the configured collection with cosine distance (`hnsw:space: cosine`).

### Indexing

`add_chunks(chunks, embeddings)` requires non-empty chunks, requires embeddings, and requires equal counts. It stores:

- Chroma IDs from `chunk.chunk_id`.
- Documents from `chunk.chunk_text`.
- Explicit embeddings.
- Flattened metadata.

Metadata includes the chunk ID, source filename, chunk index, a JSON copy of all source metadata, and scalar values directly. Dict/list values are JSON-encoded because Chroma metadata values must be primitive-compatible.

### Similarity search

`similarity_search(query, n_results, embedding_service)` rejects blank queries and non-positive result counts. It embeds the query, runs Chroma `query()` with documents, metadata, and distances, then reconstructs `DocumentChunk` objects. It logs query embedding and Chroma retrieval timings.

The returned chunk metadata is reconstructed from `source_metadata_json` and the flattened fields. Retrieval returns Chroma's relevance ordering; distances are requested but not exposed on `DocumentChunk`.

If the collection does not exist, search returns an empty list. A normal constructed store creates the collection, so an empty existing collection is a separate runtime condition.

### Maintenance

- `delete_source_document(filename)` finds all matching metadata rows, deletes their IDs, and returns the number removed.
- `list_indexed_documents()` reads metadata, deduplicates filenames, and returns them sorted.
- `delete_all()` deletes all rows in the collection.
- `collection_exists()` and `is_usable()` expose basic collection status.

## 9. RAG orchestration

`RAGService` wires together ingestion, chunking, embedding, and vector storage. Dependencies can be injected, which is how tests isolate the persistent store and avoid hard-coding paths.

### `index_document(path)`

1. Parse the path.
2. Chunk the normalized document.
3. Reject a document that produced no chunks.
4. Delete all old chunks with the same source filename.
5. Embed the new chunks.
6. Add chunks and embeddings to Chroma.
7. Return serializable dictionaries containing chunk ID, filename, index, and text.

The delete-before-add behavior makes re-indexing a file by the same basename replace its old content. Source identity is the filename only, not the full path, so two different directories with the same basename share one logical document identity.

### `retrieve(question, n_results)`

Validates the question, chooses the explicit result count or configured default, and delegates to vector search while reusing the service's embedding model.

### Context and prompt helpers

`build_context(chunks, max_chars=4000)` formats each chunk as:

```text
[Source: filename | Chunk index]
chunk text
```

It joins chunks with blank lines and truncates the final context at 4000 characters by default, appending `...` when truncated.

`build_prompt()` creates a grounded prompt that tells the model to use only supplied context, state when information is absent, and never guess. The current `ChatService` does not call this helper; `OllamaChatService.generate()` builds an equivalent prompt itself. This duplication is a maintainability detail to preserve or consolidate carefully.

## 10. LLM layer

`LLMService` is an abstract provider interface with `generate(question, context) -> str`.

`OllamaChatService` implements it using Python's standard-library `urllib`, not the Ollama Python SDK.

### Startup validation

Construction calls `GET {base_url}/api/tags`. It fails with a user-readable `ValueError` if:

- Ollama cannot be reached.
- The server returns an HTTP error.
- No models are installed.
- The configured model is not present. A model name with a matching tag prefix is accepted.

This means Ollama must be running and the model must already be pulled before the first generation.

### Generation

`generate()` rejects blank questions and blank context. It sends `POST /api/chat` with:

- the configured model,
- a system message requiring document-grounded answers,
- a user prompt containing context and question,
- `stream: false`,
- temperature `0.1`.

It expects `message.content` in the response and strips the answer. It logs prompt/context size and generation duration. Ollama acceleration is logged as unknown because the response does not expose per-request CPU/GPU usage.

## 11. Chat service

`ChatService` exposes the top-level `ask(question)` API.

`ChatResponse` contains:

- `answer: str`.
- `source_documents: list[str]`, deduplicated while preserving retrieval order.
- `source_chunk_indexes: list[int]`, one entry per retrieved chunk.
- `retrieved_context: list[dict]`, each with `source_document`, `chunk_index`, and `chunk_text`.

Ask flow:

1. Reject an empty question.
2. Retrieve configured top-k chunks.
3. If none are returned, return the fixed answer `I could not find relevant information in the provided documents.` without initializing Ollama.
4. Build bounded context through `RAGService.build_context()`.
5. Lazily create or reuse `OllamaChatService`.
6. Generate an answer.
7. Return answer and evidence fields.
8. Log total response time.

Ollama `ValueError`s are converted into a `ChatResponse` whose `answer` is the error text. Retrieved sources and context are retained in that response. Other exceptions propagate to the Streamlit catch-all handler.

The service also exposes `list_indexed_documents()` through RAG.

## 12. Streamlit UI behavior

The UI is intentionally small:

- Page title: `Document Chatbot MVP`.
- Multiple file uploader for PDF, DOCX, and TXT.
- Index button.
- Sorted indexed filename list from Chroma.
- Text input for an English question.
- Ask button.
- Answer output.
- Source references with filename and chunk number.
- Full retrieved chunk text for transparency.

Unsupported suffixes are skipped with a warning. Per-file indexing errors are shown without stopping later files. Empty uploads warn the user. The UI catches validation errors separately and displays unexpected exceptions as generic errors.

`st.cache_resource` caches the `ChatService` for the Streamlit process. The persistent Chroma database survives app restarts; uploaded temporary files do not.

## 13. Tests and expected guarantees

Tests are in `tests/`:

- `test_ingestion.py`: TXT, PDF, DOCX parsing; empty documents; unsupported formats.
- `test_chunking.py`: normal splitting, one-chunk short files, overlap, empty input, metadata preservation, and long-sentence splitting.
- `test_embeddings_vectorstore.py`: real embedding generation, metadata round trip, semantic retrieval, deletion, and validation errors. These tests may download/load the embedding model and can be slow or environment-sensitive.
- `test_rag_mvp.py`: indexing/retrieval, same-filename re-indexing, grounded prompt construction, structured chat responses, missing Ollama behavior, and no-result chat behavior.

Recommended command:

```powershell
python -m pytest
```

The repository's tests use temporary Chroma directories for isolation. The default application database remains under `data/chroma_db`.

## 14. Dependencies and external services

`requirements.txt` includes:

- `pypdf` for PDF extraction.
- `python-docx` for DOCX extraction.
- `sentence-transformers` for local embeddings.
- `chromadb` for persistent vector storage.
- `python-dotenv` for `.env` loading.
- `streamlit` for the UI.
- `pytest` for tests.

Ollama is an external local runtime, not a Python requirement. Install Ollama on Windows, run `ollama pull llama3.2:3b`, and start `ollama serve`.

## 15. Known limitations and design risks

- No OCR for scanned PDFs.
- DOCX extraction is paragraph-focused and does not explicitly process tables or embedded content.
- No authentication, authorization, multi-user isolation, or tenant separation.
- Filename-only document identity can cause collisions across directories.
- No document deletion control in the UI, although the service supports deletion.
- No reranking, citations with exact character offsets, answer confidence, or distance display.
- Chunking uses character counts rather than model token counts.
- Overlap can cut through words.
- Prompt grounding reduces hallucination risk but cannot guarantee factual answers.
- Ollama availability is checked only when generation is first needed.
- Model loading is eager for embeddings, so app startup can be expensive.
- `RAGService.build_prompt()` and the prompt construction in `OllamaChatService` duplicate policy text.
- The current ChatResponse source chunk indexes are not deduplicated, while source filenames are.
- The application is documented as English-focused even though the embedding model supports multiple languages.

## 16. Safe extension points

Prefer these existing abstractions when changing the project:

- Add another file format by implementing `DocumentParser` and registering it in `DocumentIngestionService._parsers`.
- Change chunk behavior inside `DocumentChunker`, preserving `DocumentChunk` fields unless the vector store contract is updated too.
- Swap embedding providers behind `EmbeddingService`-compatible methods.
- Swap vector databases behind the operations used by `RAGService`.
- Add an LLM provider by implementing `LLMService`.
- Test orchestration with injected ingestion, chunker, embedding, vector-store, and LLM services rather than relying on the default persistent database or live Ollama.

Keep source metadata and filename/chunk-index references intact when modifying retrieval, because the UI and ChatResponse use them for traceability.

## 17. Typical end-to-end example

```python
from pathlib import Path

from document_chatbot.chat import ChatService
from document_chatbot.rag import RAGService

rag = RAGService()
rag.index_document(Path("sample.pdf"))

chat = ChatService(rag_service=rag)
response = chat.ask("What is the main topic of the document?")

print(response.answer)
print(response.source_documents)
print(response.source_chunk_indexes)
```

This example assumes the embedding model is available locally and, for a question with retrieved context, Ollama is running with the configured model.
