"""Block-level IR models for detected regions."""

from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import (
    BaseIRModel,
    BlockType,
    BoundingBox,
    ConfidenceLevel,
    Orientation,
    Provenance,
)


class WordBox(BaseIRModel):
    """Individual word with bounding box from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    char_boxes: Optional[list[BoundingBox]] = Field(
        None, description="Character-level boxes if available"
    )


class OCRResult(BaseIRModel):
    """OCR output for a text block."""

    # Raw OCR output
    raw_text: str = Field(..., description="Verbatim OCR output")
    words: list[WordBox] = Field(default_factory=list)

    # Confidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

    # Normalized text (LLM stage)
    normalized_text: Optional[str] = Field(
        None, description="OCR-error-corrected text"
    )
    normalization_edits: list[dict] = Field(
        default_factory=list,
        description="List of edits: [{span: [start, end], original: str, normalized: str, reason: str}]",
    )

    # Model info
    ocr_engine: str = Field(..., description="tesseract, paddleocr, surya, etc.")
    ocr_version: Optional[str] = None


class Block(BaseIRModel):
    """
    Detected region within a page.

    Core unit of the IR. Each block represents a single detected region
    such as a text paragraph, table, figure, or handwriting annotation.
    """

    # Parent references
    document_id: UUID
    page_id: UUID
    page_number: int = Field(..., ge=1)

    # Detection info
    block_type: BlockType
    bbox: BoundingBox
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    detector_model: str = Field(..., description="Model used for detection")

    # Reading order
    reading_order: int = Field(
        ..., ge=0, description="Position in reading order (0-indexed)"
    )

    # Orientation (critical for handwriting per ADR-003)
    detected_orientation: Orientation = Field(default=Orientation.DEG_0)
    applied_orientation: Orientation = Field(
        default=Orientation.DEG_0,
        description="Rotation applied before OCR",
    )

    # OCR result (for text and handwriting blocks)
    ocr_result: Optional[OCRResult] = None

    # Semantic labels (from LayoutLMv3)
    semantic_label: Optional[str] = Field(
        None,
        description="Semantic classification: lab_results, medication_list, etc.",
    )
    semantic_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Cross-references
    table_id: Optional[UUID] = Field(None, description="Reference to Table if block_type=TABLE")
    parent_block_id: Optional[UUID] = Field(
        None, description="Parent block for nested structures"
    )
    linked_block_ids: list[UUID] = Field(
        default_factory=list, description="Related blocks (e.g., caption linked to figure)"
    )

    # Audit flags
    needs_audit: bool = Field(default=False)
    audit_reason: Optional[str] = None
    is_verified: bool = Field(default=False)
    verified_by: Optional[str] = None

    @property
    def has_text(self) -> bool:
        """Check if block has OCR text."""
        return self.ocr_result is not None and bool(self.ocr_result.raw_text)

    @property
    def text(self) -> str:
        """Get normalized text if available, else raw text."""
        if self.ocr_result is None:
            return ""
        return self.ocr_result.normalized_text or self.ocr_result.raw_text

    @property
    def raw_text(self) -> str:
        """Get raw OCR text."""
        if self.ocr_result is None:
            return ""
        return self.ocr_result.raw_text

    @property
    def ocr_confidence(self) -> float:
        """Get OCR confidence score."""
        if self.ocr_result is None:
            return 0.0
        return self.ocr_result.confidence

    def flag_for_audit(self, reason: str) -> None:
        """Flag this block for manual review."""
        self.needs_audit = True
        self.audit_reason = reason
        self.updated_at = self.updated_at  # Trigger update

    def get_provenance(self) -> Provenance:
        """Get provenance object for this block."""
        return Provenance(
            document_id=self.document_id,
            page_number=self.page_number,
            block_id=self.id,
            bbox=self.bbox,
            confidence=self.detection_confidence,
            stage="detection",
            model_name=self.detector_model,
        )


class HandwritingBlock(Block):
    """
    Specialized block for handwritten content.

    Extends Block with handwriting-specific fields per ADR-003.
    """

    block_type: BlockType = Field(default=BlockType.HANDWRITING)

    # Handwriting-specific
    annotation_type: Optional[str] = Field(
        None,
        description="margin_note, inline_annotation, signature, etc.",
    )
    is_legible: bool = Field(default=True)
    language_detected: str = Field(default="en")
    script_style: Optional[str] = Field(
        None, description="cursive, print, mixed"
    )

    # HTR (Handwritten Text Recognition) specific
    htr_model: Optional[str] = Field(None, description="TrOCR variant used")
    char_level_confidence: Optional[list[float]] = Field(
        None, description="Per-character confidence scores"
    )

    # Always flag low-confidence handwriting for audit
    def __init__(self, **data):
        super().__init__(**data)
        if self.ocr_result and self.ocr_result.confidence < 0.7:
            self.flag_for_audit("Low confidence handwriting OCR")
