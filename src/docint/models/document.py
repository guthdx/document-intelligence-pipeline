"""Document-level IR models."""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import BaseIRModel, ProcessingStatus


class DocumentMetadata(BaseIRModel):
    """Metadata extracted from PDF document."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None  # Software that created the PDF
    producer: Optional[str] = None  # PDF producer
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    keywords: list[str] = Field(default_factory=list)
    page_count: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=0)
    pdf_version: Optional[str] = None


class Document(BaseIRModel):
    """
    Top-level document container.

    Represents a single PDF document being processed through the pipeline.
    Contains references to pages but does NOT hold all page content in memory
    to maintain document-size agnosticism.
    """

    # Source file info
    source_path: str = Field(..., description="Original file path")
    source_filename: str = Field(..., description="Original filename")
    source_hash: str = Field(..., description="SHA-256 hash for deduplication")

    # Metadata
    metadata: Optional[DocumentMetadata] = None
    page_count: int = Field(..., ge=1)

    # Processing state
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    current_page: int = Field(default=0, description="Last processed page")
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None

    # Output paths
    output_dir: Optional[str] = None
    render_dir: Optional[str] = Field(None, description="Directory with rendered page images")

    # Audit flags
    needs_manual_review: bool = Field(default=False)
    review_notes: Optional[str] = None

    @property
    def source_path_obj(self) -> Path:
        """Return source path as Path object."""
        return Path(self.source_path)

    @property
    def is_complete(self) -> bool:
        """Check if document processing is complete."""
        return self.status == ProcessingStatus.COMPLETE

    @property
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == ProcessingStatus.FAILED

    def mark_started(self) -> None:
        """Mark document as started processing."""
        self.status = ProcessingStatus.RENDERING
        self.processing_started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_complete(self) -> None:
        """Mark document as complete."""
        self.status = ProcessingStatus.COMPLETE
        self.processing_completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        """Mark document as failed with error message."""
        self.status = ProcessingStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.utcnow()


class DocumentSummary(BaseIRModel):
    """
    Document-level summary and statistics.

    Generated after all pages are processed. Contains aggregate stats
    but NOT the actual content (document-size agnostic).
    """

    document_id: UUID
    page_count: int
    total_blocks: int = 0
    total_tables: int = 0
    total_handwriting_blocks: int = 0
    total_entities: int = 0
    total_chunks: int = 0

    # Quality metrics
    avg_ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence_block_count: int = 0
    audit_required_block_count: int = 0

    # Content classification
    detected_document_types: list[str] = Field(
        default_factory=list,
        description="e.g., ['lab_report', 'physician_notes', 'medication_list']",
    )
    primary_language: str = Field(default="en")
    contains_handwriting: bool = False
    contains_tables: bool = False
    contains_figures: bool = False
