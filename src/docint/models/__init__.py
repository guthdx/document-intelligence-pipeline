"""IR (Intermediate Representation) models for Document Intelligence Pipeline.

This module defines the Pydantic models that represent data flowing through
the pipeline stages. All models support JSON serialization and SQLAlchemy
compatibility via `from_attributes = True`.

Key Design Principles:
1. Document-size agnostic: No model requires whole document in memory
2. Provenance preservation: Every value traces to source coordinates
3. Bounded context: Cross-page references limited to 2-10 page windows
4. Conservative normalization: Original values always preserved

Model Hierarchy:
- Document → Pages → Blocks → OCR/Entities
- Document → Tables → TableCells
- Document → Chunks (for retrieval)
"""

from .base import (
    BaseIRModel,
    BlockType,
    BoundingBox,
    ConfidenceLevel,
    Orientation,
    ProcessingStatus,
    Provenance,
)
from .block import (
    Block,
    HandwritingBlock,
    OCRResult,
    WordBox,
)
from .chunk import (
    Chunk,
    ChunkOverflow,
    RetrievalResult,
)
from .document import (
    Document,
    DocumentMetadata,
    DocumentSummary,
)
from .entity import (
    DateEntity,
    DiagnosisEntity,
    Entity,
    EntityType,
    LabValue,
    Medication,
)
from .page import (
    Page,
    PageRenderInfo,
    PageSynthesis,
)
from .table import (
    Table,
    TableCell,
)

__all__ = [
    # Base types
    "BaseIRModel",
    "BlockType",
    "BoundingBox",
    "ConfidenceLevel",
    "Orientation",
    "ProcessingStatus",
    "Provenance",
    # Document
    "Document",
    "DocumentMetadata",
    "DocumentSummary",
    # Page
    "Page",
    "PageRenderInfo",
    "PageSynthesis",
    # Block
    "Block",
    "HandwritingBlock",
    "OCRResult",
    "WordBox",
    # Table
    "Table",
    "TableCell",
    # Entity
    "Entity",
    "EntityType",
    "DateEntity",
    "DiagnosisEntity",
    "LabValue",
    "Medication",
    # Chunk
    "Chunk",
    "ChunkOverflow",
    "RetrievalResult",
]
