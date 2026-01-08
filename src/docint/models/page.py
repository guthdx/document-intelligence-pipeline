"""Page-level IR models."""

from pathlib import Path
from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import BaseIRModel, ProcessingStatus


class PageRenderInfo(BaseIRModel):
    """Information about rendered page image."""

    image_path: str
    width_pixels: int = Field(..., gt=0)
    height_pixels: int = Field(..., gt=0)
    dpi: int = Field(default=300, gt=0)
    format: str = Field(default="png")
    file_size_bytes: int = Field(default=0, ge=0)

    @property
    def image_path_obj(self) -> Path:
        """Return image path as Path object."""
        return Path(self.image_path)


class Page(BaseIRModel):
    """
    Single page from a document.

    Contains page-level metadata and references to blocks.
    Actual block data is stored separately for memory efficiency.
    """

    document_id: UUID
    page_number: int = Field(..., ge=1, description="1-indexed page number")

    # Render info
    render_info: Optional[PageRenderInfo] = None
    original_width: Optional[int] = Field(None, description="Original PDF page width in points")
    original_height: Optional[int] = Field(None, description="Original PDF page height in points")
    rotation: int = Field(default=0, description="Page rotation in PDF (0, 90, 180, 270)")

    # Processing state
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: Optional[str] = None

    # Block counts (summary, not actual blocks)
    block_count: int = Field(default=0)
    text_block_count: int = Field(default=0)
    table_count: int = Field(default=0)
    handwriting_count: int = Field(default=0)
    figure_count: int = Field(default=0)

    # Continuation flags
    continues_from_previous: bool = Field(
        default=False, description="Content continues from previous page"
    )
    continues_to_next: bool = Field(
        default=False, description="Content continues to next page"
    )
    continuation_type: Optional[str] = Field(
        None, description="Type of continuation: 'table', 'paragraph', 'section'"
    )

    # Quality metrics
    avg_ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_audit: bool = Field(default=False)
    audit_reason: Optional[str] = None

    @property
    def has_handwriting(self) -> bool:
        """Check if page contains handwriting."""
        return self.handwriting_count > 0

    @property
    def has_tables(self) -> bool:
        """Check if page contains tables."""
        return self.table_count > 0


class PageSynthesis(BaseIRModel):
    """
    LLM-generated synthesis of a single page.

    Single-page scope only - never crosses page boundaries.
    """

    document_id: UUID
    page_id: UUID
    page_number: int = Field(..., ge=1)

    # Synthesis content
    summary: str = Field(..., description="Brief page summary")
    section_type: Optional[str] = Field(
        None, description="e.g., 'lab_results', 'physician_note', 'consent_form'"
    )
    key_facts: list[str] = Field(
        default_factory=list, description="Extracted key facts from this page"
    )
    topics: list[str] = Field(
        default_factory=list, description="Topic labels"
    )

    # Model info
    model_name: str = Field(..., description="LLM used for synthesis")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)

    # Provenance - links to source blocks
    source_block_ids: list[UUID] = Field(
        default_factory=list, description="Blocks used to generate synthesis"
    )
