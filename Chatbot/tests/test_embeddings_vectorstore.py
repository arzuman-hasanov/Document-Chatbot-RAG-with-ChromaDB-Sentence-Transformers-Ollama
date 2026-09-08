from __future__ import annotations

import pytest

from document_chatbot.chunking.models import DocumentChunk
from document_chatbot.embeddings.service import EmbeddingService
from document_chatbot.vectorstore.service import VectorStoreService


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


@pytest.fixture
def vector_store(tmp_path) -> VectorStoreService:
    return VectorStoreService(persist_directory=tmp_path / "chroma_db", collection_name="test_collection")


def _make_chunk(source_filename: str, chunk_text: str, chunk_index: int, metadata: dict | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{source_filename}-{chunk_index}",
        source_document_filename=source_filename,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        source_metadata=metadata or {"author": "Test User", "language": "tr"},
    )


def test_embedding_service_generates_embeddings(embedding_service: EmbeddingService) -> None:
    embedding = embedding_service.embed_text("Bu metin Türkçe ve İngilizce olarak benzer anlam taşır.")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)


def test_vector_store_indexes_chunks_and_preserves_metadata(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunks = [
        _make_chunk(
            "research.txt",
            "Türkiye'nin eğitim politikası kapsayıcı ve erişilebilir olmalıdır.",
            0,
            {"author": "Ali", "department": "education"},
        )
    ]
    embeddings = embedding_service.embed_chunks(chunks)

    ids = vector_store.add_chunks(chunks, embeddings)
    assert ids == ["research.txt-0"]
    assert vector_store.collection.count() == 1

    result = vector_store.similarity_search("eğitim politikası ve erişilebilirlik", n_results=1, embedding_service=embedding_service)
    assert result
    assert result[0].source_document_filename == "research.txt"
    assert result[0].source_metadata["department"] == "education"


def test_vector_store_similarity_search_returns_relevant_results(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunks = [
        _make_chunk("doc_a.txt", "Bu belge eğitim ve öğretim alanında yenilikçi uygulamalar anlatır.", 0),
        _make_chunk("doc_b.txt", "Müşteri hizmetleri ve çağrı merkezleri günlük operasyonları yönetir.", 0),
    ]
    embeddings = embedding_service.embed_chunks(chunks)
    vector_store.add_chunks(chunks, embeddings)

    results = vector_store.similarity_search("öğretim ve eğitim yöntemleri", n_results=1, embedding_service=embedding_service)

    assert results
    assert results[0].source_document_filename == "doc_a.txt"
    assert results[0].retrieval_distance is not None


def test_vector_store_filters_results_above_cosine_distance_threshold(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunks = [
        _make_chunk("relevant.txt", "The warranty policy covers repairs and returns.", 0),
        _make_chunk("unrelated.txt", "A recipe for tomato soup with basil and garlic.", 0),
    ]
    vector_store.add_chunks(chunks, embedding_service.embed_chunks(chunks))

    results = vector_store.similarity_search(
        "What does the warranty policy cover?",
        n_results=2,
        embedding_service=embedding_service,
        distance_threshold=0.8,
    )

    assert [result.source_document_filename for result in results] == ["relevant.txt"]


def test_vector_store_persists_document_hash_metadata(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunk = _make_chunk("hashed.txt", "Hashable content.", 0, {"document_hash": "sha256-value"})
    vector_store.add_chunks([chunk], embedding_service.embed_chunks([chunk]))

    metadata = vector_store.collection.get(include=["metadatas"])["metadatas"][0]

    assert metadata["document_hash"] == "sha256-value"


def test_vector_store_cleans_only_temporary_duplicates(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunks = [
        _make_chunk("tmp8xi9t88w.pdf", "CV chunk zero", 0),
        _make_chunk("Arzuman_Hasanov_CV.pdf", "CV chunk zero", 0, {"document_hash": "cv-hash"}),
        _make_chunk("policy.txt", "CV chunk zero!", 0, {"document_hash": "policy-hash"}),
    ]
    vector_store.add_chunks(chunks, embedding_service.embed_chunks(chunks))

    removed = vector_store.cleanup_legacy_duplicates()
    remaining = vector_store.list_indexed_documents()

    assert len(removed) == 1
    assert removed[0]["source_document_filename"] == "tmp8xi9t88w.pdf"
    assert remaining == ["Arzuman_Hasanov_CV.pdf", "policy.txt"]


def test_vector_store_delete_source_document_removes_matching_chunks(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
) -> None:
    chunks = [
        _make_chunk("delete_me.txt", "Bu belge silinecek içerik içeriyor.", 0),
        _make_chunk("keep_me.txt", "Bu belge kalacak içerik içeriyor.", 0),
    ]
    vector_store.add_chunks(chunks, embedding_service.embed_chunks(chunks))

    deleted_count = vector_store.delete_source_document("delete_me.txt")

    assert deleted_count == 1
    assert vector_store.collection.count() == 1

    remaining = vector_store.collection.get(include=["metadatas"])
    assert remaining["metadatas"][0]["source_document_filename"] == "keep_me.txt"


def test_vector_store_rejects_empty_or_invalid_input(vector_store: VectorStoreService) -> None:
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        vector_store.similarity_search("  ")

    with pytest.raises(ValueError, match="At least one chunk is required"):
        vector_store.add_chunks([], [])

    with pytest.raises(ValueError, match="Embeddings are required"):
        vector_store.add_chunks([_make_chunk("empty.txt", "example text", 0)], None)


def test_embedding_service_rejects_empty_input(embedding_service: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="Text input cannot be empty"):
        embedding_service.embed_text("   ")

    with pytest.raises(ValueError, match="At least one chunk is required"):
        embedding_service.embed_chunks([])
