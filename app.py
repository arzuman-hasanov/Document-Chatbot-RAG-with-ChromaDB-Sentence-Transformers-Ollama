from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from document_chatbot.chat import ChatService


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SUPPORTED_TYPES = {"pdf", "docx", "txt"}


@st.cache_resource
def get_chat_service() -> ChatService:
    chat_service = ChatService()
    chat_service.rag_service.cleanup_legacy_duplicates()
    return chat_service


def index_uploaded_documents(chat_service: ChatService, uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Please upload at least one PDF, DOCX, or TXT file.")
        return

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_TYPES:
            st.warning(f"Skipping unsupported file type: {filename}")
            continue

        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        try:
            result = chat_service.rag_service.index_document(temp_path, source_filename=filename)
            if result.status == "already_indexed":
                st.info(f"Document already indexed: {filename}")
            elif result.status == "reindexed":
                st.success(f"Document updated and re-indexed: {filename}")
            else:
                st.success(f"Indexed document: {filename}")
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Could not index {filename}: {exc}")
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    st.session_state["indexed_documents"] = chat_service.list_indexed_documents()


st.set_page_config(page_title="Document Chatbot MVP", page_icon="📄")
st.title("Document Chatbot MVP")

chat_service = get_chat_service()
st.session_state["indexed_documents"] = chat_service.list_indexed_documents()

st.subheader("1. Upload and index documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Index uploaded documents"):
    index_uploaded_documents(chat_service, uploaded_files)

if st.session_state["indexed_documents"]:
    st.write("Indexed documents:")
    for doc in st.session_state["indexed_documents"]:
        st.write(f"- {doc}")
else:
    st.info("No documents have been indexed yet.")

st.subheader("2. Ask a question")
question = st.text_input("Type your question about the documents", placeholder="Ask a question in English")

if st.button("Ask") and question:
    try:
        response = chat_service.ask(question)
        st.subheader("Answer")
        st.write(response.answer)

        if response.source_documents:
            st.subheader("Source references")
            for idx, document_name in enumerate(response.source_documents):
                chunk_index = response.source_chunk_indexes[idx] if idx < len(response.source_chunk_indexes) else "n/a"
                st.write(f"- {document_name} (chunk {chunk_index})")

        if response.retrieved_context:
            st.subheader("Retrieved context")
            for item in response.retrieved_context:
                st.markdown(f"**{item['source_document']}** (chunk {item['chunk_index']})\n{item['chunk_text']}")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"An unexpected error occurred: {exc}")
