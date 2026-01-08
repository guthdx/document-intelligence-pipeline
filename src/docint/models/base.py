"""Base models and common types for Document Intelligence Pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Types of detected blocks in a page."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HANDWRITING = "handwriting"
    STAMP = "stamp"
    SIGNATURE = "signature"
    FORM_FIELD = "form_field"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class Orientation(int, Enum):
    """Block orientation in degrees."""

    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270


class ProcessingStatus(str, Enum):
    """Status of document/page processing."""

    PENDING = "pending"
    RENDERING = "rendering"
    LAYOUT_DETECTION = "layout_detection"
    OCR = "ocr"
    EXTRACTION = "extraction"
    COMPLETE = "complete"
    FAILED = "failed"
    NEEDS_AUDIT = "needs_audit"


class ConfidenceLevel(str, Enum):
    """Confidence classification for extracted values."""

    HIGH = "high"  # >0.9 confidence
    MEDIUM = "medium"  # 0.7-0.9 confidence
    LOW = "low"  # 0.5-0.7 confidence
    VERY_LOW = "very_low"  # <0.5 confidence - triggers audit


class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized 0-1 or absolute pixels)."""

    x: float = Field(..., description="Left edge X coordinate")
    y: float = Field(..., description="Top edge Y coordinate")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")
    unit: str = Field(default="normalized", description="'normalized' (0-1) or 'pixels'")

    @property
    def x2(self) -> float:
        """Right edge X coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge Y coordinate."""
        return self.y + self.height

    def to_pixels(self, page_width: int, page_height: int) -> "BoundingBox":
        """Convert normalized coordinates to pixel coordinates."""
        if self.unit == "pixels":
            return self
        return BoundingBox(
            x=self.x * page_width,
            y=self.y * page_height,
            width=self.width * page_width,
            height=self.height * page_height,
            unit="pixels",
        )

    def to_normalized(self, page_width: int, page_height: int) -> "BoundingBox":
        """Convert pixel coordinates to normalized coordinates."""
        if self.unit == "normalized":
            return self
        return BoundingBox(
            x=self.x / page_width,
            y=self.y / page_height,
            width=self.width / page_width,
            height=self.height / page_height,
            unit="normalized",
        )


class Provenance(BaseModel):
    """Tracks the source of extracted data for audit trail."""

    document_id: UUID
    page_number: int = Field(..., ge=1)
    block_id: Optional[UUID] = None
    bbox: Optional[BoundingBox] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stage: str = Field(..., description="Pipeline stage that produced this data")
    model_name: Optional[str] = Field(None, description="ML model used if applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseIRModel(BaseModel):
    """Base class for all IR models with common fields."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True  # For SQLAlchemy compatibility
