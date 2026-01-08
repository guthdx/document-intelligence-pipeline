"""Repository layer for database CRUD operations."""

from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from docint.models import (
    Block,
    Chunk,
    Document,
    Entity,
    Page,
    Table,
    TableCell,
)

from .orm_models import (
    BlockORM,
    ChunkORM,
    DocumentORM,
    EntityORM,
    PageORM,
    TableCellORM,
    TableORM,
)


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, doc: Document) -> DocumentORM:
        """Create a new document record."""
        orm_doc = DocumentORM(
            id=doc.id,
            source_path=doc.source_path,
            source_filename=doc.source_filename,
            source_hash=doc.source_hash,
            metadata_json=doc.metadata.model_dump() if doc.metadata else None,
            page_count=doc.page_count,
            status=doc.status,
            current_page=doc.current_page,
            error_message=doc.error_message,
            processing_started_at=doc.processing_started_at,
            processing_completed_at=doc.processing_completed_at,
            output_dir=doc.output_dir,
            render_dir=doc.render_dir,
            needs_manual_review=doc.needs_manual_review,
            review_notes=doc.review_notes,
        )
        self.session.add(orm_doc)
        await self.session.flush()
        return orm_doc

    async def get_by_id(self, doc_id: UUID) -> Optional[DocumentORM]:
        """Get document by ID."""
        result = await self.session.execute(
            select(DocumentORM).where(DocumentORM.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, source_hash: str) -> Optional[DocumentORM]:
        """Get document by source file hash (deduplication)."""
        result = await self.session.execute(
            select(DocumentORM).where(DocumentORM.source_hash == source_hash)
        )
        return result.scalar_one_or_none()

    async def list_pending(self, limit: int = 100) -> Sequence[DocumentORM]:
        """Get documents pending processing."""
        result = await self.session.execute(
            select(DocumentORM)
            .where(DocumentORM.status == "pending")
            .order_by(DocumentORM.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def list_needing_audit(self, limit: int = 100) -> Sequence[DocumentORM]:
        """Get documents needing manual review."""
        result = await self.session.execute(
            select(DocumentORM)
            .where(DocumentORM.needs_manual_review == True)
            .order_by(DocumentORM.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def update_status(
        self, doc_id: UUID, status: str, error: Optional[str] = None
    ) -> None:
        """Update document processing status."""
        doc = await self.get_by_id(doc_id)
        if doc:
            doc.status = status
            doc.error_message = error
            await self.session.flush()

    async def count_all(self) -> int:
        """Count total documents."""
        result = await self.session.execute(
            select(func.count()).select_from(DocumentORM)
        )
        return result.scalar_one()


class PageRepository:
    """Repository for Page operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, page: Page) -> PageORM:
        """Create a new page record."""
        orm_page = PageORM(
            id=page.id,
            document_id=page.document_id,
            page_number=page.page_number,
            render_info_json=page.render_info.model_dump() if page.render_info else None,
            original_width=page.original_width,
            original_height=page.original_height,
            rotation=page.rotation,
            status=page.status,
            block_count=page.block_count,
            text_block_count=page.text_block_count,
            table_count=page.table_count,
            handwriting_count=page.handwriting_count,
            figure_count=page.figure_count,
            continues_from_previous=page.continues_from_previous,
            continues_to_next=page.continues_to_next,
            continuation_type=page.continuation_type,
            avg_ocr_confidence=page.avg_ocr_confidence,
            needs_audit=page.needs_audit,
            audit_reason=page.audit_reason,
        )
        self.session.add(orm_page)
        await self.session.flush()
        return orm_page

    async def create_batch(self, pages: list[Page]) -> list[PageORM]:
        """Batch create pages for efficiency."""
        orm_pages = [
            PageORM(
                id=p.id,
                document_id=p.document_id,
                page_number=p.page_number,
                status=p.status,
            )
            for p in pages
        ]
        self.session.add_all(orm_pages)
        await self.session.flush()
        return orm_pages

    async def get_by_id(self, page_id: UUID) -> Optional[PageORM]:
        """Get page by ID."""
        result = await self.session.execute(
            select(PageORM).where(PageORM.id == page_id)
        )
        return result.scalar_one_or_none()

    async def get_document_pages(
        self, doc_id: UUID, include_blocks: bool = False
    ) -> Sequence[PageORM]:
        """Get all pages for a document."""
        query = select(PageORM).where(PageORM.document_id == doc_id)
        if include_blocks:
            query = query.options(selectinload(PageORM.blocks))
        query = query.order_by(PageORM.page_number)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_page_window(
        self, doc_id: UUID, start_page: int, end_page: int
    ) -> Sequence[PageORM]:
        """Get a bounded window of pages (for multi-page operations)."""
        result = await self.session.execute(
            select(PageORM)
            .where(PageORM.document_id == doc_id)
            .where(PageORM.page_number >= start_page)
            .where(PageORM.page_number <= end_page)
            .options(selectinload(PageORM.blocks))
            .order_by(PageORM.page_number)
        )
        return result.scalars().all()


class BlockRepository:
    """Repository for Block operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, block: Block) -> BlockORM:
        """Create a new block record."""
        orm_block = BlockORM(
            id=block.id,
            document_id=block.document_id,
            page_id=block.page_id,
            page_number=block.page_number,
            block_type=block.block_type,
            bbox_json=block.bbox.model_dump(),
            detection_confidence=block.detection_confidence,
            detector_model=block.detector_model,
            reading_order=block.reading_order,
            detected_orientation=block.detected_orientation,
            applied_orientation=block.applied_orientation,
            ocr_result_json=block.ocr_result.model_dump() if block.ocr_result else None,
            semantic_label=block.semantic_label,
            semantic_confidence=block.semantic_confidence,
            table_id=block.table_id,
            parent_block_id=block.parent_block_id,
            linked_block_ids=block.linked_block_ids or None,
            needs_audit=block.needs_audit,
            audit_reason=block.audit_reason,
            is_verified=block.is_verified,
            verified_by=block.verified_by,
        )
        self.session.add(orm_block)
        await self.session.flush()
        return orm_block

    async def create_batch(self, blocks: list[Block]) -> list[BlockORM]:
        """Batch create blocks for efficiency."""
        orm_blocks = [
            BlockORM(
                id=b.id,
                document_id=b.document_id,
                page_id=b.page_id,
                page_number=b.page_number,
                block_type=b.block_type,
                bbox_json=b.bbox.model_dump(),
                detection_confidence=b.detection_confidence,
                detector_model=b.detector_model,
                reading_order=b.reading_order,
                ocr_result_json=b.ocr_result.model_dump() if b.ocr_result else None,
                needs_audit=b.needs_audit,
            )
            for b in blocks
        ]
        self.session.add_all(orm_blocks)
        await self.session.flush()
        return orm_blocks

    async def get_page_blocks(
        self, page_id: UUID, include_entities: bool = False
    ) -> Sequence[BlockORM]:
        """Get all blocks for a page in reading order."""
        query = select(BlockORM).where(BlockORM.page_id == page_id)
        if include_entities:
            query = query.options(selectinload(BlockORM.entities))
        query = query.order_by(BlockORM.reading_order)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_needing_audit(
        self, doc_id: Optional[UUID] = None, limit: int = 100
    ) -> Sequence[BlockORM]:
        """Get blocks needing manual review."""
        query = select(BlockORM).where(BlockORM.needs_audit == True)
        if doc_id:
            query = query.where(BlockORM.document_id == doc_id)
        query = query.limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()


class TableRepository:
    """Repository for Table operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, table: Table) -> TableORM:
        """Create a new table record with cells."""
        orm_table = TableORM(
            id=table.id,
            document_id=table.document_id,
            block_id=table.block_id,
            start_page=table.start_page,
            end_page=table.end_page,
            num_rows=table.num_rows,
            num_cols=table.num_cols,
            bbox_json=table.bbox.model_dump(),
            table_type=table.table_type,
            table_title=table.table_title,
            table_caption=table.table_caption,
            header_rows=table.header_rows or None,
            header_cols=table.header_cols or None,
            column_names=table.column_names or None,
            column_types=table.column_types or None,
            is_continuation=table.is_continuation,
            continues_on_next=table.continues_on_next,
            continuation_evidence=table.continuation_evidence,
            parent_table_id=table.parent_table_id,
            extraction_confidence=table.extraction_confidence,
            extractor_model=table.extractor_model,
            needs_audit=table.needs_audit,
            audit_reason=table.audit_reason,
        )
        self.session.add(orm_table)

        # Add cells
        for cell in table.cells:
            orm_cell = TableCellORM(
                id=cell.id,
                table_id=table.id,
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                raw_text=cell.raw_text,
                normalized_text=cell.normalized_text,
                bbox_json=cell.bbox.model_dump(),
                is_header=cell.is_header,
                is_row_header=cell.is_row_header,
                data_type=cell.data_type,
                ocr_confidence=cell.ocr_confidence,
                confidence_level=cell.confidence_level,
            )
            self.session.add(orm_cell)

        await self.session.flush()
        return orm_table

    async def get_by_id(
        self, table_id: UUID, include_cells: bool = True
    ) -> Optional[TableORM]:
        """Get table by ID with optional cells."""
        query = select(TableORM).where(TableORM.id == table_id)
        if include_cells:
            query = query.options(selectinload(TableORM.cells))
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_document_tables(
        self, doc_id: UUID, include_cells: bool = False
    ) -> Sequence[TableORM]:
        """Get all tables for a document."""
        query = select(TableORM).where(TableORM.document_id == doc_id)
        if include_cells:
            query = query.options(selectinload(TableORM.cells))
        query = query.order_by(TableORM.start_page)
        result = await self.session.execute(query)
        return result.scalars().all()


class ChunkRepository:
    """Repository for Chunk operations (retrieval layer)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, chunk: Chunk) -> ChunkORM:
        """Create a new chunk record."""
        orm_chunk = ChunkORM(
            id=chunk.id,
            document_id=chunk.document_id,
            page_id=chunk.page_id,
            page_number=chunk.page_number,
            source_block_ids=chunk.source_block_ids,
            source_type=chunk.source_type,
            text=chunk.text,
            token_count=chunk.token_count,
            embedding=chunk.embedding,
            embedding_model=chunk.embedding_model,
            embedding_dim=chunk.embedding_dim,
            semantic_labels=chunk.semantic_labels or None,
            contains_table=chunk.contains_table,
            contains_handwriting=chunk.contains_handwriting,
            contains_entities=chunk.contains_entities,
            entity_count=chunk.entity_count,
            page_context=chunk.page_context,
            section_header=chunk.section_header,
            avg_ocr_confidence=chunk.avg_ocr_confidence,
        )
        self.session.add(orm_chunk)
        await self.session.flush()
        return orm_chunk

    async def create_batch(self, chunks: list[Chunk]) -> list[ChunkORM]:
        """Batch create chunks with embeddings."""
        orm_chunks = [
            ChunkORM(
                id=c.id,
                document_id=c.document_id,
                page_id=c.page_id,
                page_number=c.page_number,
                source_block_ids=c.source_block_ids,
                source_type=c.source_type,
                text=c.text,
                token_count=c.token_count,
                embedding=c.embedding,
                embedding_model=c.embedding_model,
                semantic_labels=c.semantic_labels or None,
            )
            for c in chunks
        ]
        self.session.add_all(orm_chunks)
        await self.session.flush()
        return orm_chunks

    async def get_document_chunks(self, doc_id: UUID) -> Sequence[ChunkORM]:
        """Get all chunks for a document."""
        result = await self.session.execute(
            select(ChunkORM)
            .where(ChunkORM.document_id == doc_id)
            .order_by(ChunkORM.page_number)
        )
        return result.scalars().all()
