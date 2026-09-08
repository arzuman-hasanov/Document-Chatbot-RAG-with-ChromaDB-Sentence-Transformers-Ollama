"""Minimal application entry logic."""

from __future__ import annotations

from .config import APP_NAME, APP_VERSION


def main() -> int:
    print(f"{APP_NAME} v{APP_VERSION}")
    print("Application scaffold is ready.")
    print("Add document parsing, chunking, embeddings, and RAG features here later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
