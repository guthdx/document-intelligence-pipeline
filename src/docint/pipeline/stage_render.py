"""PDF Rendering Stage - Convert PDF pages to PNG images.

This is the first stage of the deterministic pipeline.
Uses PyMuPDF (fitz) for fast, high-quality rendering.
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Generator
from uuid import UUID

import fitz  # PyMuPDF

from docint.config import settings
from docint.models import (
    Document,
    DocumentMetadata,
    Page,
    PageRenderInfo,
    ProcessingStatus,
)


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_pdf_metadata(pdf_doc: fitz.Document, file_path: Path) -> DocumentMetadata:
    """Extract metadata from PDF document."""
    metadata = pdf_doc.metadata or {}

    return DocumentMetadata(
        title=metadata.get("title"),
        author=metadata.get("author"),
        subject=metadata.get("subject"),
        creator=metadata.get("creator"),
        producer=metadata.get("producer"),
        creation_date=None,  # Would need to parse PDF date format
        modification_date=None,
        keywords=metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
        page_count=len(pdf_doc),
        file_size_bytes=file_path.stat().st_size,
        pdf_version=f"{pdf_doc.pdf_version() / 10:.1f}" if pdf_doc.is_pdf else None,
    )


def _render_page_worker(args: tuple) -> tuple[int, str, int, int, int]:
    """Worker function for parallel page rendering.

    Args:
        args: Tuple of (pdf_path, page_num, output_path, dpi)

    Returns:
        Tuple of (page_num, output_path, width, height, file_size)
    """
    pdf_path, page_num, output_path, dpi = args

    # Open PDF in worker process
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]

    # Calculate zoom factor for target DPI (PDF base is 72 DPI)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    # Render page to pixmap
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Save as PNG
    pixmap.save(output_path)

    # Get dimensions and file size
    width = pixmap.width
    height = pixmap.height
    file_size = os.path.getsize(output_path)

    pdf_doc.close()

    return page_num, output_path, width, height, file_size


class PDFRenderer:
    """Renders PDF pages to PNG images.

    Uses PyMuPDF for high-quality, fast rendering with parallel processing
    for large documents. Memory-efficient by processing pages individually.
    """

    def __init__(
        self,
        dpi: int = None,
        max_workers: int = None,
    ):
        """Initialize renderer.

        Args:
            dpi: Rendering DPI (default from settings, typically 300)
            max_workers: Maximum parallel workers (default from settings)
        """
        self.dpi = dpi or settings.render_dpi
        self.max_workers = max_workers or settings.max_workers

    def create_document(self, pdf_path: Path, output_dir: Path) -> Document:
        """Create Document IR from PDF file.

        Args:
            pdf_path: Path to source PDF
            output_dir: Directory for output files

        Returns:
            Document with metadata populated
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Compute hash for deduplication
        source_hash = compute_file_hash(pdf_path)

        # Create render directory
        render_dir = output_dir / source_hash[:16]
        render_dir.mkdir(parents=True, exist_ok=True)

        # Extract metadata
        pdf_doc = fitz.open(str(pdf_path))
        metadata = extract_pdf_metadata(pdf_doc, pdf_path)
        page_count = len(pdf_doc)
        pdf_doc.close()

        # Create document
        doc = Document(
            source_path=str(pdf_path),
            source_filename=pdf_path.name,
            source_hash=source_hash,
            metadata=metadata,
            page_count=page_count,
            status=ProcessingStatus.PENDING,
            output_dir=str(output_dir),
            render_dir=str(render_dir),
        )

        return doc

    def render_page(
        self,
        pdf_path: Path,
        page_num: int,
        output_path: Path,
    ) -> PageRenderInfo:
        """Render a single page to PNG.

        Args:
            pdf_path: Path to PDF file
            page_num: 0-indexed page number
            output_path: Output PNG path

        Returns:
            PageRenderInfo with image details
        """
        pdf_doc = fitz.open(str(pdf_path))
        page = pdf_doc[page_num]

        # Calculate zoom factor for target DPI
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render to pixmap
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(str(output_path))

        render_info = PageRenderInfo(
            image_path=str(output_path),
            width_pixels=pixmap.width,
            height_pixels=pixmap.height,
            dpi=self.dpi,
            format="png",
            file_size_bytes=output_path.stat().st_size,
        )

        pdf_doc.close()
        return render_info

    def render_document(
        self,
        document: Document,
        parallel: bool = True,
    ) -> Generator[Page, None, None]:
        """Render all pages of a document.

        Uses parallel processing for large documents.
        Yields Page objects as they are rendered for memory efficiency.

        Args:
            document: Document to render
            parallel: Use parallel processing (default True)

        Yields:
            Page objects with render_info populated
        """
        pdf_path = Path(document.source_path)
        render_dir = Path(document.render_dir)

        if parallel and document.page_count > 4:
            # Use parallel rendering for documents with many pages
            yield from self._render_parallel(document, pdf_path, render_dir)
        else:
            # Sequential rendering for small documents
            yield from self._render_sequential(document, pdf_path, render_dir)

    def _render_sequential(
        self,
        document: Document,
        pdf_path: Path,
        render_dir: Path,
    ) -> Generator[Page, None, None]:
        """Render pages sequentially."""
        pdf_doc = fitz.open(str(pdf_path))

        for page_num in range(document.page_count):
            output_path = render_dir / f"page_{page_num + 1:04d}.png"

            # Render page
            pdf_page = pdf_doc[page_num]
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(output_path))

            # Create render info
            render_info = PageRenderInfo(
                image_path=str(output_path),
                width_pixels=pixmap.width,
                height_pixels=pixmap.height,
                dpi=self.dpi,
                format="png",
                file_size_bytes=output_path.stat().st_size,
            )

            # Create page object
            page = Page(
                document_id=document.id,
                page_number=page_num + 1,  # 1-indexed
                render_info=render_info,
                original_width=int(pdf_page.rect.width),
                original_height=int(pdf_page.rect.height),
                rotation=pdf_page.rotation,
                status=ProcessingStatus.COMPLETE,
            )

            yield page

        pdf_doc.close()

    def _render_parallel(
        self,
        document: Document,
        pdf_path: Path,
        render_dir: Path,
    ) -> Generator[Page, None, None]:
        """Render pages in parallel using ProcessPoolExecutor."""
        # Prepare work items
        work_items = []
        for page_num in range(document.page_count):
            output_path = render_dir / f"page_{page_num + 1:04d}.png"
            work_items.append((str(pdf_path), page_num, str(output_path), self.dpi))

        # Get original page dimensions (need to do this before parallel processing)
        pdf_doc = fitz.open(str(pdf_path))
        page_dimensions = [
            (int(pdf_doc[i].rect.width), int(pdf_doc[i].rect.height), pdf_doc[i].rotation)
            for i in range(document.page_count)
        ]
        pdf_doc.close()

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(_render_page_worker, work_items)

            for page_num, output_path, width, height, file_size in results:
                orig_width, orig_height, rotation = page_dimensions[page_num]

                render_info = PageRenderInfo(
                    image_path=output_path,
                    width_pixels=width,
                    height_pixels=height,
                    dpi=self.dpi,
                    format="png",
                    file_size_bytes=file_size,
                )

                page = Page(
                    document_id=document.id,
                    page_number=page_num + 1,  # 1-indexed
                    render_info=render_info,
                    original_width=orig_width,
                    original_height=orig_height,
                    rotation=rotation,
                    status=ProcessingStatus.COMPLETE,
                )

                yield page

    def render_page_range(
        self,
        document: Document,
        start_page: int,
        end_page: int,
    ) -> list[Page]:
        """Render a specific range of pages.

        Useful for bounded-context operations that only need a window.

        Args:
            document: Document to render from
            start_page: Start page (1-indexed, inclusive)
            end_page: End page (1-indexed, inclusive)

        Returns:
            List of rendered Page objects
        """
        if start_page < 1:
            start_page = 1
        if end_page > document.page_count:
            end_page = document.page_count

        pdf_path = Path(document.source_path)
        render_dir = Path(document.render_dir)
        pdf_doc = fitz.open(str(pdf_path))

        pages = []
        for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
            output_path = render_dir / f"page_{page_num + 1:04d}.png"

            # Check if already rendered
            if output_path.exists():
                render_info = PageRenderInfo(
                    image_path=str(output_path),
                    width_pixels=0,  # Would need to read image to get this
                    height_pixels=0,
                    dpi=self.dpi,
                    format="png",
                    file_size_bytes=output_path.stat().st_size,
                )
            else:
                # Render page
                pdf_page = pdf_doc[page_num]
                zoom = self.dpi / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(str(output_path))

                render_info = PageRenderInfo(
                    image_path=str(output_path),
                    width_pixels=pixmap.width,
                    height_pixels=pixmap.height,
                    dpi=self.dpi,
                    format="png",
                    file_size_bytes=output_path.stat().st_size,
                )

            pdf_page = pdf_doc[page_num]
            page = Page(
                document_id=document.id,
                page_number=page_num + 1,
                render_info=render_info,
                original_width=int(pdf_page.rect.width),
                original_height=int(pdf_page.rect.height),
                rotation=pdf_page.rotation,
                status=ProcessingStatus.COMPLETE,
            )
            pages.append(page)

        pdf_doc.close()
        return pages
