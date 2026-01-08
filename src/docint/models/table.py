"""Table IR models - first-class relational data per ADR-002."""

from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import BaseIRModel, BoundingBox, ConfidenceLevel, Provenance


class TableCell(BaseIRModel):
    """
    Individual cell in a table.

    Stores cell content with full provenance to source coordinates.
    """

    table_id: UUID
    row: int = Field(..., ge=0, description="0-indexed row number")
    col: int = Field(..., ge=0, description="0-indexed column number")
    row_span: int = Field(default=1, ge=1, description="Number of rows this cell spans")
    col_span: int = Field(default=1, ge=1, description="Number of columns this cell spans")

    # Content
    raw_text: str = Field(default="", description="Verbatim OCR text")
    normalized_text: Optional[str] = Field(None, description="Normalized text")
    bbox: BoundingBox

    # Classification (from LLM stage)
    is_header: bool = Field(default=False, description="Is this a header cell")
    is_row_header: bool = Field(default=False, description="Is this a row header (first column)")
    data_type: Optional[str] = Field(
        None, description="Inferred type: numeric, date, text, code, etc."
    )

    # Confidence
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

    @property
    def text(self) -> str:
        """Get normalized text if available, else raw text."""
        return self.normalized_text or self.raw_text

    @property
    def is_merged(self) -> bool:
        """Check if cell spans multiple rows or columns."""
        return self.row_span > 1 or self.col_span > 1

    def get_provenance(self, document_id: UUID, page_number: int) -> Provenance:
        """Get provenance object for this cell."""
        return Provenance(
            document_id=document_id,
            page_number=page_number,
            block_id=self.table_id,
            bbox=self.bbox,
            confidence=self.ocr_confidence,
            stage="table_extraction",
        )


class Table(BaseIRModel):
    """
    First-class table representation.

    Tables are stored relationally (not flattened to text) per ADR-002.
    Supports multi-page tables through continuation tracking.
    """

    # Parent references
    document_id: UUID
    block_id: UUID  # Reference to the Block that detected this table
    start_page: int = Field(..., ge=1)
    end_page: Optional[int] = Field(
        None, ge=1, description="End page for multi-page tables"
    )

    # Structure
    num_rows: int = Field(..., ge=1)
    num_cols: int = Field(..., ge=1)
    bbox: BoundingBox  # Table bounding box

    # Cells (stored separately for large tables)
    cells: list[TableCell] = Field(default_factory=list)

    # Table classification (from LLM stage)
    table_type: Optional[str] = Field(
        None,
        description="Semantic type: lab_results, vitals, medications, billing, etc.",
    )
    table_title: Optional[str] = Field(None, description="Detected or inferred title")
    table_caption: Optional[str] = None

    # Header info
    header_rows: list[int] = Field(
        default_factory=list, description="Row indices that are headers"
    )
    header_cols: list[int] = Field(
        default_factory=list, description="Column indices that are row headers"
    )
    column_names: list[str] = Field(
        default_factory=list, description="Extracted column headers"
    )
    column_types: list[str] = Field(
        default_factory=list, description="Inferred column data types"
    )

    # Multi-page handling (bounded context per plan)
    is_continuation: bool = Field(
        default=False, description="This table continues from previous page"
    )
    continues_on_next: bool = Field(
        default=False, description="This table continues on next page"
    )
    continuation_evidence: Optional[str] = Field(
        None, description="Evidence for continuation (repeated headers, markers, etc.)"
    )
    parent_table_id: Optional[UUID] = Field(
        None, description="ID of table this continues from"
    )

    # Quality
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extractor_model: str = Field(..., description="Table extraction model used")
    needs_audit: bool = Field(default=False)
    audit_reason: Optional[str] = None

    @property
    def is_multi_page(self) -> bool:
        """Check if table spans multiple pages."""
        return self.end_page is not None and self.end_page > self.start_page

    @property
    def page_span(self) -> int:
        """Number of pages this table spans."""
        if self.end_page is None:
            return 1
        return self.end_page - self.start_page + 1

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at specified row and column."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
            # Check if this cell spans to include the requested position
            if (
                cell.row <= row < cell.row + cell.row_span
                and cell.col <= col < cell.col + cell.col_span
            ):
                return cell
        return None

    def get_row(self, row: int) -> list[TableCell]:
        """Get all cells in a row."""
        return sorted(
            [c for c in self.cells if c.row == row],
            key=lambda c: c.col,
        )

    def get_column(self, col: int) -> list[TableCell]:
        """Get all cells in a column."""
        return sorted(
            [c for c in self.cells if c.col == col],
            key=lambda c: c.row,
        )

    def get_column_values(self, col: int, skip_headers: bool = True) -> list[str]:
        """Get all values in a column, optionally skipping headers."""
        cells = self.get_column(col)
        if skip_headers:
            cells = [c for c in cells if not c.is_header]
        return [c.text for c in cells]

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.cells:
            return ""

        # Build grid
        grid: list[list[str]] = [[""] * self.num_cols for _ in range(self.num_rows)]
        for cell in self.cells:
            grid[cell.row][cell.col] = cell.text

        # Build markdown
        lines = []
        for i, row in enumerate(grid):
            line = "| " + " | ".join(row) + " |"
            lines.append(line)
            # Add separator after first row (header)
            if i == 0:
                sep = "| " + " | ".join(["---"] * self.num_cols) + " |"
                lines.append(sep)

        return "\n".join(lines)

    def to_dict_records(self) -> list[dict]:
        """Convert table to list of dicts (one per data row)."""
        if not self.column_names or not self.cells:
            return []

        records = []
        data_rows = [r for r in range(self.num_rows) if r not in self.header_rows]

        for row_idx in data_rows:
            record = {}
            for col_idx, col_name in enumerate(self.column_names):
                cell = self.get_cell(row_idx, col_idx)
                record[col_name] = cell.text if cell else ""
            records.append(record)

        return records
