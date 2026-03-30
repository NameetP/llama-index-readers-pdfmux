"""LlamaIndex reader for pdfmux.

Usage:
    from llama_index_readers_pdfmux import PDFMuxReader

    reader = PDFMuxReader()
    docs = reader.load_data("report.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_VALID_QUALITY = {"fast", "standard", "high"}


class PDFMuxReader(BaseReader):
    """LlamaIndex reader powered by pdfmux.

    Extracts text from PDFs using pdfmux's self-healing multi-pass pipeline
    and returns LlamaIndex Document objects with rich metadata.

    Supports single files and directories (all .pdf files in the directory).

    Args:
        quality: Extraction quality -- "fast", "standard" (default), or "high".
        glob: Glob pattern for matching PDF files in a directory.
              Defaults to "*.pdf".

    Example::

        from llama_index_readers_pdfmux import PDFMuxReader

        reader = PDFMuxReader(quality="high")
        docs = reader.load_data("report.pdf")
        for doc in docs:
            print(doc.metadata["confidence"])
    """

    def __init__(
        self,
        quality: str = "standard",
        glob: str = "*.pdf",
    ) -> None:
        if quality not in _VALID_QUALITY:
            raise ValueError(
                f"Invalid quality {quality!r}. Must be one of: {', '.join(sorted(_VALID_QUALITY))}"
            )
        self.quality = quality
        self.glob = glob

    def load_data(
        self,
        file: str | Path,
        extra_info: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Load documents from a PDF file or directory.

        Args:
            file: Path to a PDF file or directory containing PDFs.
            extra_info: Optional dict of extra metadata to attach to each Document.

        Returns:
            List of Document objects with page_content and metadata.

        Raises:
            ImportError: If pdfmux is not installed.
            ValueError: If the path does not exist.
            RuntimeError: If pdfmux fails to process a file.
        """
        try:
            import pdfmux
        except ImportError:
            raise ImportError(
                "pdfmux is required for PDFMuxReader. Install it with: pip install pdfmux"
            )

        file_path = Path(file)
        if not file_path.exists():
            raise ValueError(
                f"Path does not exist: {file_path}. "
                "Provide a valid path to a PDF file or directory."
            )

        if file_path.is_dir():
            paths = sorted(file_path.glob(self.glob))
            if not paths:
                logger.warning("No files matching %r found in %s", self.glob, file_path)
        else:
            paths = [file_path]

        documents: list[Document] = []

        for path in paths:
            try:
                chunks = pdfmux.load_llm_context(path, quality=self.quality)
            except Exception as exc:
                raise RuntimeError(f"pdfmux failed to process {path}: {exc}") from exc

            for chunk in chunks:
                metadata = {
                    "source": str(path),
                    "title": chunk["title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "tokens": chunk["tokens"],
                    "confidence": chunk["confidence"],
                }
                if extra_info:
                    metadata.update(extra_info)

                documents.append(
                    Document(
                        text=chunk["text"],
                        metadata=metadata,
                    )
                )

        return documents
