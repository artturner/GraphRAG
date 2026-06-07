"""JSON pages connector for loading pre-extracted textbook page datasets.

Reads a JSON file produced by scripts/pdf_extract.py — an array of objects
with chapter, page_number, source, and text fields — and returns one Document
per page with chapter and page_number preserved in metadata so they survive
chunking and surface in citations.
"""

import json
import logging
from pathlib import Path
from typing import Union

from src.connectors.base import BaseConnector
from src.connectors.document import create_document_id
from src.exceptions import ConnectorError
from src.types import Document, DocumentType

logger = logging.getLogger(__name__)


class JsonPagesConnector(BaseConnector):
    """Connector for JSON datasets of pre-extracted textbook pages.

    Each entry in the JSON array becomes one Document. The chapter and
    page_number fields are stored in Document.metadata so the ingestion
    pipeline propagates them into every Chunk produced from that page,
    enabling precise chapter/page citations at query time.

    Expected JSON format::

        [
          {
            "chapter": 3,
            "page_number": 47,
            "source": "chapter_03.pdf",
            "text": "..."
          },
          ...
        ]

    Example::

        connector = JsonPagesConnector("data/full_book.json")
        documents = connector.load()
        # Each document has metadata["chapter"] and metadata["page_number"]
    """

    def __init__(self, source_path: Union[str, Path]) -> None:
        self._source_path = Path(source_path)
        super().__init__(str(self._source_path))

    def validate_source(self) -> bool:
        return (
            self._source_path.exists()
            and self._source_path.is_file()
            and self._source_path.suffix.lower() == ".json"
        )

    def list_documents(self) -> list[str]:
        """Return human-readable identifiers for every page in the dataset."""
        entries = self._read_entries()
        return [_page_label(e) for e in entries]

    def load(self) -> list[Document]:
        """Load all pages as Documents with chapter/page metadata."""
        if not self.validate_source():
            raise ConnectorError(
                f"JSON pages file not found or not a .json file: {self._source_path}"
            )

        entries = self._read_entries()
        documents: list[Document] = []

        for entry in entries:
            try:
                doc = self._entry_to_document(entry)
                if doc is not None:
                    documents.append(doc)
            except Exception as exc:
                logger.warning("Skipping malformed entry %s: %s", entry, exc)

        logger.info(
            "JsonPagesConnector loaded %d pages from %s",
            len(documents),
            self._source_path.name,
        )
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_entries(self) -> list[dict]:
        try:
            raw = self._source_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConnectorError(
                f"Failed to read JSON pages file: {self._source_path}",
                details=str(exc),
            )

        if not isinstance(data, list):
            raise ConnectorError(
                f"JSON pages file must contain an array at the top level: {self._source_path}"
            )

        return data

    def _entry_to_document(self, entry: dict) -> Document | None:
        text = (entry.get("text") or "").strip()
        if not text:
            return None

        chapter = entry.get("chapter")
        page_number = entry.get("page_number")
        source_file = entry.get("source", self._source_path.name)

        # Human-readable source string used in citations
        source = _page_label(entry)

        doc_id = create_document_id(f"{source_file}::page::{page_number}")

        return Document(
            id=doc_id,
            content=text,
            source=source,
            document_type=DocumentType.JSON,
            metadata={
                "chapter": chapter,
                "page_number": page_number,
                "source_file": source_file,
            },
        )


def _page_label(entry: dict) -> str:
    """Build a citation-friendly label from a page entry."""
    chapter = entry.get("chapter")
    page = entry.get("page_number")
    if chapter is not None and page is not None:
        return f"Chapter {chapter}, p. {page}"
    if page is not None:
        return f"p. {page}"
    return entry.get("source", "unknown")
