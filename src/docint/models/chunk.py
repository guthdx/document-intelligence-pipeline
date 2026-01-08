"""Chunk IR models for embeddings and retrieval."""

from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import BaseIRModel, BlockType, Provenance


class Chunk(BaseIRModel):
    """
    Embedding unit for retrieval.

    Chunks are created from blocks following specific rules:
    - Max 512 tokens, min 64 tokens
    - Respect block boundaries (never split mid-block)
    - Tables stored as single chunks (with overflow handling)
    - Handwriting stored as single chunks
    """

    # Parent references
    document_id: UUID
    page_id: UUID
    page_number: int = Field(..., ge=1)

    # Source block(s) - usually one, but may combine small blocks
    source_block_ids: list[UUID] = Field(
        ..., min_length=1, description="Block IDs that form this chunk"
    )
    source_type: BlockType = Field(
        ..., description="Primary type of source content"
    )

    # Content
    text: str = Field(..., description="Chunk text for embedding")
    token_count: int = Field(..., ge=1, description="Token count")

    # Embedding (stored as list for portability, use pgvector in DB)
    embedding: Optional[list[float]] = Field(
        None, description="Dense vector embedding"
    )
    embedding_model: Optional[str] = Field(
        None, description="Model used: all-MiniLM-L6-v2, etc."
    )
    embedding_dim: Optional[int] = Field(None, description="Embedding dimensions")

    # Metadata for retrieval
    semantic_labels: list[str] = Field(
        default_factory=list,
        description="Semantic labels for filtering",
    )
    contains_table: bool = Field(default=False)
    contains_handwriting: bool = Field(default=False)
    contains_entities: bool = Field(default=False)
    entity_count: int = Field(default=0)

    # Context for retrieval results
    page_context: Optional[str] = Field(
        None, description="Brief page context for display"
    )
    section_header: Optional[str] = Field(
        None, description="Section header if detected"
    )

    # Quality
    avg_ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def get_provenances(self) -> list[Provenance]:
        """Get provenance objects for all source blocks."""
        return [
            Provenance(
                document_id=self.document_id,
                page_number=self.page_number,
                block_id=block_id,
                confidence=self.avg_ocr_confidence,
                stage="chunking",
                model_name=self.embedding_model,
            )
            for block_id in self.source_block_ids
        ]

    @property
    def is_oversized(self) -> bool:
        """Check if chunk exceeds recommended size."""
        return self.token_count > 512

    @property
    def is_undersized(self) -> bool:
        """Check if chunk is below recommended size."""
        return self.token_count < 64


class ChunkOverflow(BaseIRModel):
    """
    Overflow record for chunks that exceed token limits.

    Used for tables and other content that can't be split.
    The main chunk stores the first 512 tokens, overflow stores the rest.
    """

    chunk_id: UUID
    sequence: int = Field(..., ge=1, description="Overflow sequence number")
    text: str
    token_count: int = Field(..., ge=1)
    embedding: Optional[list[float]] = None

    @property
    def total_tokens_with_overflow(self) -> int:
        """Combined token count."""
        return self.token_count


class RetrievalResult(BaseIRModel):
    """
    Result from retrieval layer with provenance.

    Contains the chunk plus metadata about how it was retrieved.
    """

    chunk_id: UUID
    document_id: UUID
    page_number: int

    # Content
    text: str
    source_type: BlockType

    # Retrieval info
    retrieval_method: str = Field(
        ..., description="fts, vector, structured, or fusion"
    )
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., ge=1, description="Position in results")

    # For RRF (Reciprocal Rank Fusion)
    fts_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    structured_rank: Optional[int] = None
    rrf_score: Optional[float] = None

    # Highlights for display
    highlights: list[str] = Field(
        default_factory=list, description="Matched text snippets"
    )

    # Provenance chain
    provenances: list[Provenance] = Field(default_factory=list)

    # Related content
    table_id: Optional[UUID] = Field(None, description="Related table if chunk from table")
    entity_ids: list[UUID] = Field(
        default_factory=list, description="Entities in this chunk"
    )
