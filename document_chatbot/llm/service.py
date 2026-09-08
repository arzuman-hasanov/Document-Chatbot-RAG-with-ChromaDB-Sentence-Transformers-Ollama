"""LLM service abstraction and local Ollama-backed implementation."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from document_chatbot.config import OLLAMA_BASE_URL, OLLAMA_MODEL


logger = logging.getLogger(__name__)


def build_grounded_prompt(question: str, context: str) -> str:
    """Build the single grounded, concise-answer policy used by all callers."""
    if question is None or not str(question).strip():
        raise ValueError("Question cannot be empty.")
    if context is None or not str(context).strip():
        raise ValueError("Document context is required to answer the question.")

    return (
        "Answer the user's question directly and concisely using only the supplied document context. "
        "Answer only using the information in the supplied context. "
        "Treat the context as evidence, not as text to repeat. Do not repeat unrelated context. "
        "Do not provide unrelated information or summarize the document unless the user explicitly asks for a summary. "
        "For simple factual questions, give the shortest correct answer possible. "
        "Return only the requested fact when the question asks for a name, email, phone number, date, location, or other specific value. "
        "If the requested information is not present in the context, say that it was not found in the provided documents. "
        "Never guess, infer, or use outside knowledge.\n\n"
        f"Document context:\n{context}\n\nQuestion:\n{question}"
    )


class LLMService(ABC):
    """Small service interface for future LLM providers."""

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Return the final answer using the provided context."""


class OllamaChatService(LLMService):
    """Local Ollama-backed chat implementation for English document Q&A."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)).rstrip("/")
        self._ensure_ollama_available()
        logger.info(
            "Ollama runtime configured: model=%s acceleration=unknown "
            "(the Ollama chat response does not expose CPU/GPU per-request)",
            self.model,
        )

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Ollama request failed for {url}: {exc.reason}. {body}") from exc
        except (urllib.error.URLError, OSError, TimeoutError, ConnectionError) as exc:
            raise ValueError(
                f"Ollama is not running or unavailable at {self.base_url}. Start it with 'ollama serve' and confirm the server is reachable."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Ollama is not running or unavailable at {self.base_url}. Start it with 'ollama serve' and confirm the server is reachable."
            ) from exc

    def _ensure_ollama_available(self) -> None:
        try:
            info = self._request_json("GET", "/api/tags")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        models = info.get("models", [])
        if not models:
            raise ValueError(
                f"Ollama model '{self.model}' is not available. Run 'ollama pull {self.model}' to download it locally."
            )

        available = {
            model.get("name", "")
            for model in models
            if isinstance(model, dict)
        }
        if self.model not in available and not any(name.startswith(f"{self.model}:") for name in available):
            raise ValueError(
                f"Ollama model '{self.model}' is not available. Run 'ollama pull {self.model}' to download it locally."
            )

    def generate(self, question: str, context: str) -> str:
        prompt = build_grounded_prompt(question, context)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied document context. "
                        "Be concise, direct, and do not repeat unrelated context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        logger.info(
            "Ollama request: model=%s context_chars=%d prompt_chars=%d",
            self.model,
            len(context),
            len(prompt),
        )
        generation_started = time.perf_counter()
        try:
            response = self._request_json("POST", "/api/chat", payload)
        except Exception:
            logger.info(
                "RAG timing: ollama_generation_seconds=%.4f model=%s status=error",
                time.perf_counter() - generation_started,
                self.model,
            )
            raise
        generation_elapsed = time.perf_counter() - generation_started
        message = response.get("message", {}).get("content")
        if message is None:
            raise ValueError("The Ollama model returned no answer.")
        logger.info(
            "RAG timing: ollama_generation_seconds=%.4f model=%s",
            generation_elapsed,
            self.model,
        )
        return str(message).strip()
