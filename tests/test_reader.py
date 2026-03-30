"""Tests for PDFMuxReader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import Document

from llama_index_readers_pdfmux.base import PDFMuxReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "text": "Introduction to the report.",
        "title": "Introduction",
        "page_start": 1,
        "page_end": 2,
        "tokens": 120,
        "confidence": 0.95,
    },
    {
        "text": "Detailed findings section.",
        "title": "Findings",
        "page_start": 3,
        "page_end": 5,
        "tokens": 340,
        "confidence": 0.88,
    },
]


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a dummy PDF file (content doesn't matter since pdfmux is mocked)."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple dummy PDF files."""
    for name in ("a.pdf", "b.pdf", "notes.txt"):
        (tmp_path / name).write_bytes(b"%PDF-1.4 fake")
    return tmp_path


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_quality(self) -> None:
        reader = PDFMuxReader()
        assert reader.quality == "standard"

    def test_custom_quality(self) -> None:
        for q in ("fast", "standard", "high"):
            reader = PDFMuxReader(quality=q)
            assert reader.quality == q

    def test_invalid_quality_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid quality"):
            PDFMuxReader(quality="ultra")

    def test_default_glob(self) -> None:
        reader = PDFMuxReader()
        assert reader.glob == "*.pdf"

    def test_custom_glob(self) -> None:
        reader = PDFMuxReader(glob="**/*.pdf")
        assert reader.glob == "**/*.pdf"


# ---------------------------------------------------------------------------
# load_data tests
# ---------------------------------------------------------------------------


class TestLoadData:
    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_returns_documents(self, mock_pdfmux: MagicMock, sample_pdf: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            reader = PDFMuxReader()
            docs = reader.load_data(sample_pdf)

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_document_content(self, mock_pdfmux: MagicMock, sample_pdf: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(sample_pdf)

        assert docs[0].text == "Introduction to the report."
        assert docs[1].text == "Detailed findings section."

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_metadata_fields(self, mock_pdfmux: MagicMock, sample_pdf: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(sample_pdf)

        meta = docs[0].metadata
        assert meta["source"] == str(sample_pdf)
        assert meta["title"] == "Introduction"
        assert meta["page_start"] == 1
        assert meta["page_end"] == 2
        assert meta["tokens"] == 120
        assert meta["confidence"] == 0.95

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_quality_passed_to_pdfmux(self, mock_pdfmux: MagicMock, sample_pdf: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = []
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            PDFMuxReader(quality="high").load_data(sample_pdf)

        mock_pdfmux.load_llm_context.assert_called_once_with(sample_pdf, quality="high")

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_extra_info_merged_into_metadata(
        self, mock_pdfmux: MagicMock, sample_pdf: Path
    ) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS[:1]
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(sample_pdf, extra_info={"project": "Q4"})

        assert docs[0].metadata["project"] == "Q4"
        assert docs[0].metadata["source"] == str(sample_pdf)

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_string_path_accepted(self, mock_pdfmux: MagicMock, sample_pdf: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS[:1]
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(str(sample_pdf))

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Directory loading tests
# ---------------------------------------------------------------------------


class TestDirectoryLoading:
    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_loads_all_pdfs_in_directory(self, mock_pdfmux: MagicMock, sample_dir: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS[:1]
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(sample_dir)

        # sample_dir has a.pdf and b.pdf (notes.txt excluded by glob)
        assert len(docs) == 2
        sources = {d.metadata["source"] for d in docs}
        assert str(sample_dir / "a.pdf") in sources
        assert str(sample_dir / "b.pdf") in sources

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_custom_glob_pattern(self, mock_pdfmux: MagicMock, sample_dir: Path) -> None:
        mock_pdfmux.load_llm_context.return_value = SAMPLE_CHUNKS[:1]
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader(glob="a.*").load_data(sample_dir)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == str(sample_dir / "a.pdf")

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_empty_directory_yields_nothing(self, mock_pdfmux: MagicMock, tmp_path: Path) -> None:
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            docs = PDFMuxReader().load_data(tmp_path)

        assert docs == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_nonexistent_path_raises(self) -> None:
        reader = PDFMuxReader()
        with pytest.raises(ValueError, match="Path does not exist"):
            reader.load_data("/tmp/does_not_exist_abc123.pdf")

    def test_import_error_message(self, sample_pdf: Path) -> None:
        with patch.dict("sys.modules", {"pdfmux": None}):
            reader = PDFMuxReader()
            with pytest.raises(ImportError, match="pip install pdfmux"):
                reader.load_data(sample_pdf)

    @patch("llama_index_readers_pdfmux.base.pdfmux", create=True)
    def test_conversion_failure_raises_runtime_error(
        self, mock_pdfmux: MagicMock, sample_pdf: Path
    ) -> None:
        mock_pdfmux.load_llm_context.side_effect = Exception("corrupt PDF")
        with patch.dict("sys.modules", {"pdfmux": mock_pdfmux}):
            reader = PDFMuxReader()
            with pytest.raises(RuntimeError, match="pdfmux failed to process"):
                reader.load_data(sample_pdf)
