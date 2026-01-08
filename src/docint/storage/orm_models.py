"""SQLAlchemy ORM models for Document Intelligence Pipeline.

These models define the database schema for persisting IR data.
Uses PostgreSQL with pgvector for embedding storage.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docint.models.base import BlockType, ConfidenceLevel, Orientation, ProcessingStatus

from .database import Base


class DocumentORM(Base):
    """Document table - top-level container."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Source file info
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Metadata
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Processing state
    status: Mapped[str] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING
    )
    current_page: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Output paths
    output_dir: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    render_dir: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Audit
    needs_manual_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    pages: Mapped[list["PageORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    tables: Mapped[list["TableORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["ChunkORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_source_hash", "source_hash"),
        Index("ix_documents_status", "status"),
    )


class PageORM(Base):
    """Page table - individual pages from documents."""

    __tablename__ = "pages"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Render info
    render_info_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    original_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    original_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rotation: Mapped[int] = mapped_column(Integer, default=0)

    # Processing state
    status: Mapped[str] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Block counts
    block_count: Mapped[int] = mapped_column(Integer, default=0)
    text_block_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    handwriting_count: Mapped[int] = mapped_column(Integer, default=0)
    figure_count: Mapped[int] = mapped_column(Integer, default=0)

    # Continuation flags
    continues_from_previous: Mapped[bool] = mapped_column(Boolean, default=False)
    continues_to_next: Mapped[bool] = mapped_column(Boolean, default=False)
    continuation_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Quality metrics
    avg_ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    needs_audit: Mapped[bool] = mapped_column(Boolean, default=False)
    audit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    document: Mapped["DocumentORM"] = relationship(back_populates="pages")
    blocks: Mapped[list["BlockORM"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_pages_document_page", "document_id", "page_number"),
        Index("ix_pages_status", "status"),
    )


class BlockORM(Base):
    """Block table - detected regions within pages."""

    __tablename__ = "blocks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Detection info
    block_type: Mapped[str] = mapped_column(Enum(BlockType), nullable=False)
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detector_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Reading order
    reading_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Orientation
    detected_orientation: Mapped[int] = mapped_column(
        Enum(Orientation), default=Orientation.DEG_0
    )
    applied_orientation: Mapped[int] = mapped_column(
        Enum(Orientation), default=Orientation.DEG_0
    )

    # OCR result
    ocr_result_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Semantic labels
    semantic_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    semantic_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cross-references
    table_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tables.id", ondelete="SET NULL"), nullable=True
    )
    parent_block_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("blocks.id", ondelete="SET NULL"), nullable=True
    )
    linked_block_ids: Mapped[Optional[list]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=True
    )

    # Audit
    needs_audit: Mapped[bool] = mapped_column(Boolean, default=False)
    audit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    page: Mapped["PageORM"] = relationship(back_populates="blocks")
    entities: Mapped[list["EntityORM"]] = relationship(
        back_populates="block", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_blocks_document_page", "document_id", "page_id"),
        Index("ix_blocks_type", "block_type"),
        Index("ix_blocks_needs_audit", "needs_audit"),
    )


class TableORM(Base):
    """Table storage - first-class relational data."""

    __tablename__ = "tables"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    block_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False
    )
    start_page: Mapped[int] = mapped_column(Integer, nullable=False)
    end_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Structure
    num_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    num_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Classification
    table_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    table_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    table_caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Header info
    header_rows: Mapped[Optional[list]] = mapped_column(ARRAY(Integer), nullable=True)
    header_cols: Mapped[Optional[list]] = mapped_column(ARRAY(Integer), nullable=True)
    column_names: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    column_types: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)

    # Multi-page handling
    is_continuation: Mapped[bool] = mapped_column(Boolean, default=False)
    continues_on_next: Mapped[bool] = mapped_column(Boolean, default=False)
    continuation_evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parent_table_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tables.id", ondelete="SET NULL"), nullable=True
    )

    # Quality
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extractor_model: Mapped[str] = mapped_column(String(100), nullable=False)
    needs_audit: Mapped[bool] = mapped_column(Boolean, default=False)
    audit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    document: Mapped["DocumentORM"] = relationship(back_populates="tables")
    cells: Mapped[list["TableCellORM"]] = relationship(
        back_populates="table", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_tables_document", "document_id"),
        Index("ix_tables_type", "table_type"),
    )


class TableCellORM(Base):
    """Table cell storage."""

    __tablename__ = "table_cells"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    table_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tables.id", ondelete="CASCADE")
    )

    # Position
    row: Mapped[int] = mapped_column(Integer, nullable=False)
    col: Mapped[int] = mapped_column(Integer, nullable=False)
    row_span: Mapped[int] = mapped_column(Integer, default=1)
    col_span: Mapped[int] = mapped_column(Integer, default=1)

    # Content
    raw_text: Mapped[str] = mapped_column(Text, default="")
    normalized_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Classification
    is_header: Mapped[bool] = mapped_column(Boolean, default=False)
    is_row_header: Mapped[bool] = mapped_column(Boolean, default=False)
    data_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Confidence
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_level: Mapped[str] = mapped_column(
        Enum(ConfidenceLevel), default=ConfidenceLevel.MEDIUM
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    table: Mapped["TableORM"] = relationship(back_populates="cells")

    __table_args__ = (
        Index("ix_table_cells_table_row_col", "table_id", "row", "col"),
    )


class EntityORM(Base):
    """Entity storage for extracted structured data."""

    __tablename__ = "entities"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE")
    )
    block_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("blocks.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entity classification
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    sub_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Extracted value
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_value_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Source location
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    char_span: Mapped[Optional[list]] = mapped_column(ARRAY(Integer), nullable=True)

    # Confidence
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_level: Mapped[str] = mapped_column(
        Enum(ConfidenceLevel), default=ConfidenceLevel.MEDIUM
    )
    extraction_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Relationships
    related_entity_ids: Mapped[Optional[list]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=True
    )

    # Validation
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_errors: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    needs_audit: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    block: Mapped["BlockORM"] = relationship(back_populates="entities")

    __table_args__ = (
        Index("ix_entities_document", "document_id"),
        Index("ix_entities_type", "entity_type"),
        Index("ix_entities_type_subtype", "entity_type", "sub_type"),
    )


class ChunkORM(Base):
    """Chunk storage for embeddings and retrieval."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Source blocks
    source_block_ids: Mapped[list] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False
    )
    source_type: Mapped[str] = mapped_column(Enum(BlockType), nullable=False)

    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Embedding - pgvector for similarity search
    embedding: Mapped[Optional[list]] = mapped_column(
        Vector(384), nullable=True  # all-MiniLM-L6-v2 = 384 dims
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Full-text search
    text_tsvector = mapped_column(
        "text_tsv",
        nullable=True,
    )

    # Metadata
    semantic_labels: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    contains_table: Mapped[bool] = mapped_column(Boolean, default=False)
    contains_handwriting: Mapped[bool] = mapped_column(Boolean, default=False)
    contains_entities: Mapped[bool] = mapped_column(Boolean, default=False)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)

    # Context
    page_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    section_header: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Quality
    avg_ocr_confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    document: Mapped["DocumentORM"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document", "document_id"),
        Index("ix_chunks_page", "page_id"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
