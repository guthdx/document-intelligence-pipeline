"""Initial schema with all tables and pgvector support.

Revision ID: 001
Revises: None
Create Date: 2026-01-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Create enum types
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE processingstatus AS ENUM (
                'pending', 'rendering', 'layout_detection', 'ocr',
                'extraction', 'complete', 'failed', 'needs_audit'
            );
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE blocktype AS ENUM (
                'text', 'table', 'figure', 'handwriting', 'stamp',
                'signature', 'form_field', 'header', 'footer', 'page_number'
            );
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE orientation AS ENUM ('0', '90', '180', '270');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE confidencelevel AS ENUM ('high', 'medium', 'low', 'very_low');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_path", sa.String(1024), nullable=False),
        sa.Column("source_filename", sa.String(255), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True),
        sa.Column("page_count", sa.Integer, nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "rendering", "layout_detection", "ocr",
                          "extraction", "complete", "failed", "needs_audit",
                          name="processingstatus", create_type=False),
            server_default="pending",
        ),
        sa.Column("current_page", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("output_dir", sa.String(1024), nullable=True),
        sa.Column("render_dir", sa.String(1024), nullable=True),
        sa.Column("needs_manual_review", sa.Boolean, server_default="false"),
        sa.Column("review_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_source_hash", "documents", ["source_hash"])
    op.create_index("ix_documents_status", "documents", ["status"])

    # Create pages table
    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("render_info_json", postgresql.JSONB, nullable=True),
        sa.Column("original_width", sa.Integer, nullable=True),
        sa.Column("original_height", sa.Integer, nullable=True),
        sa.Column("rotation", sa.Integer, server_default="0"),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "rendering", "layout_detection", "ocr",
                          "extraction", "complete", "failed", "needs_audit",
                          name="processingstatus", create_type=False),
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("block_count", sa.Integer, server_default="0"),
        sa.Column("text_block_count", sa.Integer, server_default="0"),
        sa.Column("table_count", sa.Integer, server_default="0"),
        sa.Column("handwriting_count", sa.Integer, server_default="0"),
        sa.Column("figure_count", sa.Integer, server_default="0"),
        sa.Column("continues_from_previous", sa.Boolean, server_default="false"),
        sa.Column("continues_to_next", sa.Boolean, server_default="false"),
        sa.Column("continuation_type", sa.String(50), nullable=True),
        sa.Column("avg_ocr_confidence", sa.Float, server_default="0.0"),
        sa.Column("needs_audit", sa.Boolean, server_default="false"),
        sa.Column("audit_reason", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_pages_document_page", "pages", ["document_id", "page_number"])
    op.create_index("ix_pages_status", "pages", ["status"])

    # Create tables table (for extracted tables)
    op.create_table(
        "tables",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("block_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start_page", sa.Integer, nullable=False),
        sa.Column("end_page", sa.Integer, nullable=True),
        sa.Column("num_rows", sa.Integer, nullable=False),
        sa.Column("num_cols", sa.Integer, nullable=False),
        sa.Column("bbox_json", postgresql.JSONB, nullable=False),
        sa.Column("table_type", sa.String(100), nullable=True),
        sa.Column("table_title", sa.Text, nullable=True),
        sa.Column("table_caption", sa.Text, nullable=True),
        sa.Column("header_rows", postgresql.ARRAY(sa.Integer), nullable=True),
        sa.Column("header_cols", postgresql.ARRAY(sa.Integer), nullable=True),
        sa.Column("column_names", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("column_types", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("is_continuation", sa.Boolean, server_default="false"),
        sa.Column("continues_on_next", sa.Boolean, server_default="false"),
        sa.Column("continuation_evidence", sa.Text, nullable=True),
        sa.Column(
            "parent_table_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tables.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("extraction_confidence", sa.Float, server_default="0.0"),
        sa.Column("extractor_model", sa.String(100), nullable=False),
        sa.Column("needs_audit", sa.Boolean, server_default="false"),
        sa.Column("audit_reason", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_tables_document", "tables", ["document_id"])
    op.create_index("ix_tables_type", "tables", ["table_type"])

    # Create blocks table
    op.create_table(
        "blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "page_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column(
            "block_type",
            postgresql.ENUM("text", "table", "figure", "handwriting", "stamp",
                          "signature", "form_field", "header", "footer", "page_number",
                          name="blocktype", create_type=False),
            nullable=False,
        ),
        sa.Column("bbox_json", postgresql.JSONB, nullable=False),
        sa.Column("detection_confidence", sa.Float, nullable=False),
        sa.Column("detector_model", sa.String(100), nullable=False),
        sa.Column("reading_order", sa.Integer, nullable=False),
        sa.Column(
            "detected_orientation",
            postgresql.ENUM("0", "90", "180", "270", name="orientation", create_type=False),
            server_default="0",
        ),
        sa.Column(
            "applied_orientation",
            postgresql.ENUM("0", "90", "180", "270", name="orientation", create_type=False),
            server_default="0",
        ),
        sa.Column("ocr_result_json", postgresql.JSONB, nullable=True),
        sa.Column("semantic_label", sa.String(100), nullable=True),
        sa.Column("semantic_confidence", sa.Float, nullable=True),
        sa.Column(
            "table_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tables.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "parent_block_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("blocks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("linked_block_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column("needs_audit", sa.Boolean, server_default="false"),
        sa.Column("audit_reason", sa.Text, nullable=True),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("verified_by", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_blocks_document_page", "blocks", ["document_id", "page_id"])
    op.create_index("ix_blocks_type", "blocks", ["block_type"])
    op.create_index("ix_blocks_needs_audit", "blocks", ["needs_audit"])

    # Create table_cells table
    op.create_table(
        "table_cells",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "table_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tables.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("row", sa.Integer, nullable=False),
        sa.Column("col", sa.Integer, nullable=False),
        sa.Column("row_span", sa.Integer, server_default="1"),
        sa.Column("col_span", sa.Integer, server_default="1"),
        sa.Column("raw_text", sa.Text, server_default=""),
        sa.Column("normalized_text", sa.Text, nullable=True),
        sa.Column("bbox_json", postgresql.JSONB, nullable=False),
        sa.Column("is_header", sa.Boolean, server_default="false"),
        sa.Column("is_row_header", sa.Boolean, server_default="false"),
        sa.Column("data_type", sa.String(50), nullable=True),
        sa.Column("ocr_confidence", sa.Float, server_default="0.0"),
        sa.Column(
            "confidence_level",
            postgresql.ENUM("high", "medium", "low", "very_low",
                          name="confidencelevel", create_type=False),
            server_default="medium",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_table_cells_table_row_col", "table_cells", ["table_id", "row", "col"])

    # Create entities table
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "page_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "block_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("blocks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("sub_type", sa.String(50), nullable=True),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("normalized_value", sa.Text, nullable=True),
        sa.Column("parsed_value_json", postgresql.JSONB, nullable=True),
        sa.Column("bbox_json", postgresql.JSONB, nullable=False),
        sa.Column("char_span", postgresql.ARRAY(sa.Integer), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column(
            "confidence_level",
            postgresql.ENUM("high", "medium", "low", "very_low",
                          name="confidencelevel", create_type=False),
            server_default="medium",
        ),
        sa.Column("extraction_model", sa.String(100), nullable=False),
        sa.Column("related_entity_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column("is_validated", sa.Boolean, server_default="false"),
        sa.Column("validation_errors", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("needs_audit", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_entities_document", "entities", ["document_id"])
    op.create_index("ix_entities_type", "entities", ["entity_type"])
    op.create_index("ix_entities_type_subtype", "entities", ["entity_type", "sub_type"])

    # Create chunks table with vector embeddings
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "page_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("source_block_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column(
            "source_type",
            postgresql.ENUM("text", "table", "figure", "handwriting", "stamp",
                          "signature", "form_field", "header", "footer", "page_number",
                          name="blocktype", create_type=False),
            nullable=False,
        ),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),  # all-MiniLM-L6-v2 = 384 dims
        sa.Column("embedding_model", sa.String(100), nullable=True),
        sa.Column("embedding_dim", sa.Integer, nullable=True),
        sa.Column("text_tsv", postgresql.TSVECTOR, nullable=True),
        sa.Column("semantic_labels", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("contains_table", sa.Boolean, server_default="false"),
        sa.Column("contains_handwriting", sa.Boolean, server_default="false"),
        sa.Column("contains_entities", sa.Boolean, server_default="false"),
        sa.Column("entity_count", sa.Integer, server_default="0"),
        sa.Column("page_context", sa.Text, nullable=True),
        sa.Column("section_header", sa.String(255), nullable=True),
        sa.Column("avg_ocr_confidence", sa.Float, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_document", "chunks", ["document_id"])
    op.create_index("ix_chunks_page", "chunks", ["page_id"])

    # Create vector similarity index (IVFFlat)
    op.execute("""
        CREATE INDEX ix_chunks_embedding ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # Create full-text search index
    op.execute("""
        CREATE INDEX ix_chunks_text_tsv ON chunks
        USING gin (text_tsv)
    """)

    # Create trigram index for fuzzy search
    op.execute("""
        CREATE INDEX ix_chunks_text_trgm ON chunks
        USING gin (text gin_trgm_ops)
    """)

    # Create function to update text_tsv on insert/update
    op.execute("""
        CREATE OR REPLACE FUNCTION chunks_text_tsv_trigger()
        RETURNS trigger AS $$
        BEGIN
            NEW.text_tsv := to_tsvector('english', NEW.text);
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER chunks_text_tsv_update
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION chunks_text_tsv_trigger();
    """)


def downgrade() -> None:
    """Drop all tables and extensions."""
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS chunks_text_tsv_update ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_text_tsv_trigger()")

    # Drop tables in reverse order of creation
    op.drop_table("chunks")
    op.drop_table("entities")
    op.drop_table("table_cells")
    op.drop_table("blocks")
    op.drop_table("tables")
    op.drop_table("pages")
    op.drop_table("documents")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS confidencelevel")
    op.execute("DROP TYPE IF EXISTS orientation")
    op.execute("DROP TYPE IF EXISTS blocktype")
    op.execute("DROP TYPE IF EXISTS processingstatus")

    # Note: Not dropping vector or pg_trgm extensions as they may be used by other apps
