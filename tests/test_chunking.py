from __future__ import annotations

from document_chatbot.chunking import DocumentChunker
from document_chatbot.ingestion.models import Document


def _build_document(text: str, *, filename: str = "sample.txt", metadata: dict | None = None) -> Document:
    return Document(
        source_filename=filename,
        file_type="txt",
        extracted_text=text,
        metadata=metadata or {"author": "Test User", "pages": 1},
    )


def test_chunk_document_normal_text() -> None:
    text = (
        "Paragraph one with enough text to be chunked into manageable parts. "
        "This sentence keeps going and makes the chunk useful for RAG.\n\n"
        "Paragraph two introduces a second topic and gives more context for retrieval. "
        "The chunking logic should prefer paragraph boundaries when possible.\n\n"
        "Paragraph three is still relevant and goes on to ensure the document is split into multiple chunks."
    )

    chunks = DocumentChunker(chunk_size=180, chunk_overlap=30).chunk_document(_build_document(text))

    assert len(chunks) > 1
    assert all(chunk.chunk_text for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.source_document_filename == "sample.txt" for chunk in chunks)


def test_chunk_document_short_document_returns_single_chunk() -> None:
    document = _build_document("Short document with only a few words.")

    chunks = DocumentChunker(chunk_size=200, chunk_overlap=40).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_text == document.extracted_text
    assert chunks[0].chunk_index == 0


def test_chunk_document_long_document_has_overlap() -> None:
    text = " ".join(["sentence about retrieval in a document" for _ in range(60)])
    document = _build_document(text, metadata={"author": "Alice"})

    chunker = DocumentChunker(chunk_size=220, chunk_overlap=40)
    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert chunks[1].chunk_text.startswith(chunks[0].chunk_text[-40:])
    assert chunks[0].source_metadata["author"] == "Alice"


def test_chunk_document_empty_text_returns_no_chunks() -> None:
    chunks = DocumentChunker().chunk_document(_build_document("   \n\t  "))

    assert chunks == []


def test_chunk_document_preserves_source_metadata() -> None:
    metadata = {"author": "Alice", "pages": 7, "source_type": "research"}
    document = _build_document("This is a sample document for metadata preservation.", metadata=metadata)

    chunks = DocumentChunker(chunk_size=80, chunk_overlap=10).chunk_document(document)

    assert chunks
    for chunk in chunks:
        assert chunk.source_metadata["author"] == "Alice"
        assert chunk.source_metadata["pages"] == 7
        assert chunk.source_metadata["source_file_type"] == "txt"


def test_chunk_document_very_long_sentence_is_split() -> None:
    sentence = " ".join(["word" for _ in range(200)])
    chunks = DocumentChunker(chunk_size=80, chunk_overlap=20).chunk_document(_build_document(sentence))

    assert len(chunks) > 1
    assert all(len(chunk.chunk_text) <= 80 for chunk in chunks)
