"""Tests for PDF rendering stage."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docint.models import ProcessingStatus
from docint.pipeline.stage_render import PDFRenderer, compute_file_hash


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_hash_consistency(self, tmp_path):
        """Same file should always produce same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_hash_different_content(self, tmp_path):
        """Different content should produce different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestPDFRenderer:
    """Tests for PDF rendering."""

    @pytest.fixture
    def renderer(self):
        """Create renderer with default settings."""
        return PDFRenderer(dpi=300, max_workers=2)

    @pytest.fixture
    def mock_pdf(self, tmp_path):
        """Create a minimal mock PDF file."""
        # Create a very simple PDF file for testing
        pdf_path = tmp_path / "test.pdf"
        # Write minimal PDF structure
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path

    def test_renderer_initialization(self, renderer):
        """Test renderer initializes with correct settings."""
        assert renderer.dpi == 300
        assert renderer.max_workers == 2

    def test_renderer_default_settings(self):
        """Test renderer uses settings defaults."""
        renderer = PDFRenderer()
        assert renderer.dpi > 0
        assert renderer.max_workers > 0

    @patch("docint.pipeline.stage_render.fitz")
    def test_create_document(self, mock_fitz, renderer, tmp_path):
        """Test document creation from PDF."""
        # Create test PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock PDF document
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.__len__ = MagicMock(return_value=5)
        mock_pdf_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
        }
        mock_pdf_doc.is_pdf = True
        mock_pdf_doc.pdf_version = MagicMock(return_value=17)  # 1.7
        mock_fitz.open.return_value = mock_pdf_doc

        # Create document
        doc = renderer.create_document(pdf_path, output_dir)

        # Verify document properties
        assert doc.source_filename == "test.pdf"
        assert doc.page_count == 5
        assert doc.status == ProcessingStatus.PENDING
        assert doc.metadata.title == "Test Document"
        assert doc.metadata.author == "Test Author"
        assert len(doc.source_hash) == 64

    def test_create_document_file_not_found(self, renderer, tmp_path):
        """Test error when PDF file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            renderer.create_document(
                tmp_path / "nonexistent.pdf",
                tmp_path / "output",
            )

    @patch("docint.pipeline.stage_render.fitz")
    def test_render_page(self, mock_fitz, renderer, tmp_path):
        """Test single page rendering."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")
        output_path = tmp_path / "page_0001.png"

        # Mock PDF and page
        mock_page = MagicMock()
        mock_page.rotation = 0
        mock_page.rect.width = 612
        mock_page.rect.height = 792

        mock_pixmap = MagicMock()
        mock_pixmap.width = 2550  # 612 * 300/72
        mock_pixmap.height = 3300  # 792 * 300/72
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_pdf_doc = MagicMock()
        mock_pdf_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_pdf_doc
        mock_fitz.Matrix.return_value = MagicMock()

        # Create dummy output file
        output_path.write_bytes(b"PNG data")

        # Render page
        render_info = renderer.render_page(pdf_path, 0, output_path)

        # Verify render info
        assert render_info.image_path == str(output_path)
        assert render_info.dpi == 300
        assert render_info.format == "png"


class TestPageRenderInfo:
    """Tests for PageRenderInfo model."""

    def test_render_info_creation(self, tmp_path):
        """Test PageRenderInfo can be created."""
        from docint.models import PageRenderInfo

        info = PageRenderInfo(
            image_path=str(tmp_path / "test.png"),
            width_pixels=2550,
            height_pixels=3300,
            dpi=300,
            format="png",
            file_size_bytes=1024000,
        )

        assert info.width_pixels == 2550
        assert info.height_pixels == 3300
        assert info.dpi == 300

    def test_render_info_path_property(self, tmp_path):
        """Test image_path_obj property returns Path."""
        from docint.models import PageRenderInfo

        path = tmp_path / "test.png"
        info = PageRenderInfo(
            image_path=str(path),
            width_pixels=100,
            height_pixels=100,
            dpi=300,
        )

        assert info.image_path_obj == path
        assert isinstance(info.image_path_obj, Path)
