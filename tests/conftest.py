"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary path for sample PDFs."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return pdf_dir


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir
