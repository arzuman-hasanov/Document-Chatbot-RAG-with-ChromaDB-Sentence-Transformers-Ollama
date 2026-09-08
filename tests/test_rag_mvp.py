from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from document_chatbot.chunking.models import DocumentChunk
from document_chatbot.chat import ChatService
from document_chatbot.llm import OllamaChatService
from document_chatbot.llm.service import build_grounded_prompt
from document_chatbot.rag import RAGService
from document_chatbot.vectorstore import VectorStoreService


class CapturingLLM:
    model = "test"

    def __init__(self) -> None:
        self.context = ""

    def generate(self, question: str, context: str) -> str:
        self.context = context
        return "Arzuman Hasanov"


@pytest.fixture
def rag_service(tmp_path: Path) -> RAGService:
    return RAGService(
        vector_store_service=VectorStoreService(
            persist_directory=tmp_path / "chroma_test_db",
            collection_name="rag_tests",
        ),
        retrieval_result_count=2,
    )


def test_rag_service_indexes_and_retrieves_document(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "policy.txt"
    file_path.write_text(
        "The company emphasizes training and quality assurance in every team. "
        "Employees are encouraged to document lessons learned and improve processes.",
        encoding="utf-8",
    )

    chunks = rag_service.index_document(file_path)
    results = rag_service.retrieve("What does the company emphasize for teams?", n_results=2)

    assert chunks
    assert results
    assert results[0].source_document_filename == "policy.txt"
    assert any("training" in result.chunk_text.lower() for result in results)


def test_rag_service_reindexes_document_by_filename(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "revision.txt"
    file_path.write_text("First version says to follow the legacy process and review manually.", encoding="utf-8")
    rag_service.index_document(file_path)

    file_path.write_text("Second version says to use automated quality checks and testing pipelines.", encoding="utf-8")
    rag_service.index_document(file_path)

    results = rag_service.retrieve("What is the new process recommendation?", n_results=2)

    assert results
    assert "automated quality checks" in results[0].chunk_text.lower()
    assert len(rag_service.list_indexed_documents()) == 1


def test_rag_service_preserves_explicit_source_filename_and_hash(
    rag_service: RAGService,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "tmp-upload.txt"
    content = "ARZUMAN HASANOV is a software engineer."
    file_path.write_text(content, encoding="utf-8")

    result = rag_service.index_document(file_path, source_filename="Arzuman_Hasanov_CV.pdf")

    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result.status == "indexed"
    assert result.source_filename == "Arzuman_Hasanov_CV.pdf"
    assert result[0]["source_document_filename"] == "Arzuman_Hasanov_CV.pdf"
    stored = rag_service.vector_store_service.collection.get(include=["metadatas"])
    assert stored["metadatas"][0]["document_hash"] == expected_hash
    assert stored["metadatas"][0]["source_document_filename"] == "Arzuman_Hasanov_CV.pdf"
    assert "tmp-upload.txt" not in rag_service.list_indexed_documents()


def test_rag_service_skips_same_content_under_same_or_different_filename(
    rag_service: RAGService,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("The same document content.", encoding="utf-8")
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    initial = rag_service.index_document(first, source_filename="Original.txt")
    duplicate_same_name = rag_service.index_document(first, source_filename="Original.txt")
    duplicate_other_name = rag_service.index_document(second, source_filename="Renamed.txt")

    assert initial.status == "indexed"
    assert duplicate_same_name.status == "already_indexed"
    assert duplicate_other_name.status == "already_indexed"
    assert rag_service.vector_store_service.collection.count() == len(initial)
    assert rag_service.list_indexed_documents() == ["Original.txt"]


def test_rag_service_reindexes_same_filename_when_content_changes(
    rag_service: RAGService,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "upload.txt"
    file_path.write_text("The original version.", encoding="utf-8")
    rag_service.index_document(file_path, source_filename="Document.txt")

    file_path.write_text("The revised version.", encoding="utf-8")
    result = rag_service.index_document(file_path, source_filename="Document.txt")

    assert result.status == "reindexed"
    assert rag_service.list_indexed_documents() == ["Document.txt"]
    retrieved = rag_service.retrieve("What is the revised version?", n_results=2)
    assert retrieved
    assert "revised version" in retrieved[0].chunk_text


def test_rag_service_deduplicates_hash_and_legacy_evidence(rag_service: RAGService) -> None:
    duplicate_hash_chunks = [
        DocumentChunk("one", "tmp8xi9t88w.pdf", "ARZUMAN HASANOV", 0, {}),
        DocumentChunk("two", "Arzuman_Hasanov_CV.pdf", "ARZUMAN HASANOV", 0, {"document_hash": "abc"}),
    ]
    legacy_chunks = [
        DocumentChunk("three", "tmp-one.pdf", "Same legacy evidence", 0, {}),
        DocumentChunk("four", "tmp-two.pdf", "Same legacy evidence", 0, {}),
    ]

    deduplicated = rag_service.deduplicate_chunks(duplicate_hash_chunks)
    assert len(deduplicated) == 1
    assert deduplicated[0].source_document_filename == "Arzuman_Hasanov_CV.pdf"
    assert len(rag_service.deduplicate_chunks(legacy_chunks)) == 1


def test_rag_service_deduplicates_legacy_cv_chunks_0_2_and_3_without_merging_unrelated_content(
    rag_service: RAGService,
) -> None:
    chunks = []
    for chunk_index in (0, 2, 3):
        text = f"CV evidence for chunk {chunk_index}"
        chunks.extend(
            [
                DocumentChunk(f"tmp-{chunk_index}", "tmp8xi9t88w.pdf", text, chunk_index, {}),
                DocumentChunk(
                    f"canonical-{chunk_index}",
                    "Arzuman_Hasanov_CV.pdf",
                    text,
                    chunk_index,
                    {"document_hash": "cv-hash"},
                ),
            ]
        )
    chunks.extend(
        [
            DocumentChunk("other-a", "other-a.pdf", "Nearly identical CV evidence", 0, {"document_hash": "other-a"}),
            DocumentChunk("other-b", "other-b.pdf", "Nearly identical CV evidence!", 0, {"document_hash": "other-b"}),
        ]
    )

    deduplicated = rag_service.deduplicate_chunks(chunks)

    assert [chunk.source_document_filename for chunk in deduplicated[:3]] == [
        "Arzuman_Hasanov_CV.pdf",
        "Arzuman_Hasanov_CV.pdf",
        "Arzuman_Hasanov_CV.pdf",
    ]
    assert {chunk.source_document_filename for chunk in deduplicated[3:]} == {"other-a.pdf", "other-b.pdf"}


def test_rag_service_build_prompt_uses_context_only(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "context.txt"
    file_path.write_text("The user manual says the product ships with a warranty policy.", encoding="utf-8")
    rag_service.index_document(file_path)

    chunks = rag_service.retrieve("What does the warranty policy say?", n_results=1)
    prompt = rag_service.build_prompt("What does the warranty policy say?", chunks)

    assert "Answer only using the information in the supplied context" in prompt
    assert "warranty policy" in prompt.lower()
    assert "source:" in prompt.lower()


def test_grounded_prompt_requires_concise_direct_answers() -> None:
    prompt = build_grounded_prompt("What is his name?", "[Source: cv.pdf | Chunk 0]\nARZUMAN HASANOV")

    assert "directly and concisely" in prompt
    assert "do not repeat unrelated context" in prompt.lower()
    assert "shortest correct answer" in prompt.lower()
    assert "never guess" in prompt.lower()


def test_chat_service_preserves_sources_with_concise_answer(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "cv.txt"
    file_path.write_text(
        "ARZUMAN HASANOV\nEmail: hasanovarzuman7@gmail.com\nEducation: UFAZ and ASOIU.",
        encoding="utf-8",
    )
    rag_service.index_document(file_path)

    with patch.object(OllamaChatService, "_ensure_ollama_available"), patch.object(
        OllamaChatService,
        "generate",
        return_value="Arzuman Hasanov",
    ):
        response = ChatService(
            rag_service=rag_service,
            llm_service=OllamaChatService(model="llama3.2:3b"),
        ).ask("What is his name?")

    assert response.answer == "Arzuman Hasanov"
    assert response.source_documents == ["cv.txt"]
    assert response.source_chunk_indexes
    assert response.retrieved_context


def test_chat_service_filters_unrelated_document_context(rag_service: RAGService, tmp_path: Path) -> None:
    person_path = tmp_path / "person.txt"
    context_path = tmp_path / "context.txt"
    person_path.write_text("Arzuman Hasanov is a Data Science and AI researcher.", encoding="utf-8")
    context_path.write_text("The user manual says the product ships with a warranty policy.", encoding="utf-8")
    rag_service.index_document(person_path)
    rag_service.index_document(context_path)
    llm = CapturingLLM()

    response = ChatService(rag_service=rag_service, llm_service=llm).ask("What is his name?")

    assert response.answer == "Arzuman Hasanov"
    assert "person.txt" in llm.context
    assert "context.txt" not in llm.context
    assert response.source_documents == ["person.txt"]


def test_chat_service_returns_no_result_without_initializing_ollama(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "person.txt"
    file_path.write_text("Arzuman Hasanov is a Data Science and AI researcher.", encoding="utf-8")
    rag_service.index_document(file_path)
    rag_service.retrieval_distance_threshold = 0.0

    with patch("document_chatbot.chat.service.OllamaChatService") as ollama:
        response = ChatService(rag_service=rag_service).ask("What is his favorite color?")

    assert response.answer == "I could not find relevant information in the provided documents."
    ollama.assert_not_called()


def test_chat_service_returns_structured_response(rag_service: RAGService, tmp_path: Path) -> None:
    file_path = tmp_path / "team.txt"
    file_path.write_text("The onboarding process includes training and documentation review.", encoding="utf-8")
    rag_service.index_document(file_path)

    with patch.object(OllamaChatService, "_ensure_ollama_available"), patch.object(
        OllamaChatService,
        "generate",
        return_value="The onboarding process includes training and documentation review.",
    ):
        chat_service = ChatService(rag_service=rag_service, llm_service=OllamaChatService(model="llama3.2:3b"))
        response = chat_service.ask("What is included in the onboarding process?")

    assert response.answer == "The onboarding process includes training and documentation review."
    assert response.source_documents == ["team.txt"]
    assert response.source_chunk_indexes
    assert response.retrieved_context


def test_chat_service_handles_missing_ollama_runtime() -> None:
    with patch("document_chatbot.llm.service.urllib.request.urlopen", side_effect=Exception("down")):
        with pytest.raises(ValueError, match="Ollama is not running or unavailable"):
            OllamaChatService(model="llama3.2:3b", base_url="http://localhost:11434")


def test_chat_service_returns_user_friendly_message_when_no_relevant_documents(rag_service: RAGService) -> None:
    chat_service = ChatService(rag_service=rag_service)
    response = chat_service.ask("What is the company policy on retirement?")

    assert "I could not find relevant information" in response.answer
    assert response.source_documents == []
