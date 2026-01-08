"""Continuation Detection Stage - Detect multi-page content.

Identifies content that continues across page boundaries:
- "Continued on next page" markers
- Truncated sentences at page boundaries
- Repeated table headers (indicating table continuation)
- Matching column structures between pages

Used to enable bounded-context LLM operations (2-10 page windows).
"""

import re
from dataclasses import dataclass
from typing import Optional

from docint.models import Block, BlockType, Page, Table


# Common continuation marker patterns
CONTINUATION_MARKERS = [
    r"continued\s+on\s+(next\s+)?page",
    r"cont(?:inued|\.)\s*$",
    r"\(continued\)",
    r"see\s+next\s+page",
    r"turn\s+page",
    r"continued\s+from\s+(previous\s+)?page",
    r"cont(?:inued|\.)\s+from\s+page\s*\d+",
    r"\(cont(?:inued)?\.?\)",
    r"---\s*continued\s*---",
]

# Patterns for incomplete content
INCOMPLETE_PATTERNS = [
    r",\s*$",  # Ends with comma
    r":\s*$",  # Ends with colon
    r";\s*$",  # Ends with semicolon
    r"-\s*$",  # Ends with hyphen (word break)
    r"\band\s*$",  # Ends with "and"
    r"\bor\s*$",  # Ends with "or"
    r"\bthe\s*$",  # Ends with "the"
    r"\ba\s*$",  # Ends with "a"
]


@dataclass
class ContinuationInfo:
    """Information about detected continuation."""

    continues_to_next: bool = False
    continues_from_previous: bool = False
    continuation_type: Optional[str] = None  # "table", "paragraph", "section"
    evidence: list[str] = None  # Reasons for detection
    confidence: float = 0.0

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class ContinuationDetector:
    """Detects content continuation across pages.

    Analyzes page content to identify multi-page elements
    for bounded-context LLM processing.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
    ):
        """Initialize continuation detector.

        Args:
            confidence_threshold: Minimum confidence to flag continuation.
        """
        self.confidence_threshold = confidence_threshold

        # Compile regex patterns
        self.continuation_patterns = [
            re.compile(p, re.IGNORECASE) for p in CONTINUATION_MARKERS
        ]
        self.incomplete_patterns = [
            re.compile(p) for p in INCOMPLETE_PATTERNS
        ]

    def detect_page_continuations(
        self,
        current_page: Page,
        current_blocks: list[Block],
        previous_page: Optional[Page] = None,
        previous_blocks: Optional[list[Block]] = None,
        next_page: Optional[Page] = None,
        next_blocks: Optional[list[Block]] = None,
        current_tables: Optional[list[Table]] = None,
        previous_tables: Optional[list[Table]] = None,
        next_tables: Optional[list[Table]] = None,
    ) -> ContinuationInfo:
        """Detect continuation relationships for a page.

        Args:
            current_page: Page to analyze.
            current_blocks: Blocks on current page.
            previous_page: Previous page (if exists).
            previous_blocks: Blocks on previous page.
            next_page: Next page (if exists).
            next_blocks: Blocks on next page.
            current_tables: Tables on current page.
            previous_tables: Tables on previous page.
            next_tables: Tables on next page.

        Returns:
            ContinuationInfo with detection results.
        """
        info = ContinuationInfo()

        # Check continuation from previous page
        if previous_page and previous_blocks:
            from_prev = self._check_from_previous(
                current_blocks,
                previous_blocks,
                current_tables or [],
                previous_tables or [],
            )
            if from_prev.continues_from_previous:
                info.continues_from_previous = True
                info.evidence.extend(from_prev.evidence)
                if from_prev.continuation_type:
                    info.continuation_type = from_prev.continuation_type

        # Check continuation to next page
        if next_page and next_blocks:
            to_next = self._check_to_next(
                current_blocks,
                next_blocks,
                current_tables or [],
                next_tables or [],
            )
            if to_next.continues_to_next:
                info.continues_to_next = True
                info.evidence.extend(to_next.evidence)
                if to_next.continuation_type and not info.continuation_type:
                    info.continuation_type = to_next.continuation_type

        # Calculate confidence
        if info.evidence:
            info.confidence = min(1.0, len(info.evidence) * 0.3)

        return info

    def _check_from_previous(
        self,
        current_blocks: list[Block],
        previous_blocks: list[Block],
        current_tables: list[Table],
        previous_tables: list[Table],
    ) -> ContinuationInfo:
        """Check if content continues from previous page."""
        info = ContinuationInfo()

        # Check for continuation markers at start of current page
        first_blocks = self._get_top_blocks(current_blocks, n=3)
        for block in first_blocks:
            text = self._get_block_text(block).lower()
            for pattern in self.continuation_patterns:
                if "from" in text and pattern.search(text):
                    info.continues_from_previous = True
                    info.continuation_type = "paragraph"
                    info.evidence.append(f"Continuation marker found: '{text[:50]}...'")
                    break

        # Check for table continuation (repeated headers)
        if current_tables and previous_tables:
            table_cont = self._check_table_continuation(
                current_tables[0],
                previous_tables[-1],
            )
            if table_cont:
                info.continues_from_previous = True
                info.continuation_type = "table"
                info.evidence.append("Table structure matches previous page")

        # Check if first block starts mid-sentence
        if first_blocks:
            first_text = self._get_block_text(first_blocks[0])
            if first_text and first_text[0].islower():
                info.continues_from_previous = True
                info.continuation_type = info.continuation_type or "paragraph"
                info.evidence.append("Page starts with lowercase (mid-sentence)")

        return info

    def _check_to_next(
        self,
        current_blocks: list[Block],
        next_blocks: list[Block],
        current_tables: list[Table],
        next_tables: list[Table],
    ) -> ContinuationInfo:
        """Check if content continues to next page."""
        info = ContinuationInfo()

        # Check for continuation markers at end of current page
        last_blocks = self._get_bottom_blocks(current_blocks, n=3)
        for block in last_blocks:
            text = self._get_block_text(block).lower()
            for pattern in self.continuation_patterns:
                if "next" in text and pattern.search(text):
                    info.continues_to_next = True
                    info.continuation_type = "paragraph"
                    info.evidence.append(f"Continuation marker found: '{text[:50]}...'")
                    break

        # Check for incomplete content
        if last_blocks:
            last_text = self._get_block_text(last_blocks[-1])
            for pattern in self.incomplete_patterns:
                if pattern.search(last_text):
                    info.continues_to_next = True
                    info.continuation_type = info.continuation_type or "paragraph"
                    info.evidence.append("Page ends with incomplete content")
                    break

        # Check for table continuation
        if current_tables and next_tables:
            table_cont = self._check_table_continuation(
                next_tables[0],
                current_tables[-1],
            )
            if table_cont:
                info.continues_to_next = True
                info.continuation_type = "table"
                info.evidence.append("Table continues on next page")

        return info

    def _check_table_continuation(
        self,
        table_after: Table,
        table_before: Table,
    ) -> bool:
        """Check if two tables are continuations of each other.

        Args:
            table_after: Table on later page.
            table_before: Table on earlier page.

        Returns:
            True if tables appear to be continuations.
        """
        # Check column count match
        if table_before.num_cols != table_after.num_cols:
            return False

        # Check column names match (if available)
        if table_before.column_names and table_after.column_names:
            if table_before.column_names == table_after.column_names:
                return True

        # Check for repeated header row (common in continued tables)
        if table_after.header_rows and table_before.column_names:
            # Get first row of table_after
            first_row_cells = table_after.get_row(0)
            first_row_values = [c.text.strip().lower() for c in first_row_cells]
            prev_headers = [h.strip().lower() for h in table_before.column_names]

            if first_row_values == prev_headers:
                return True

        # Check if table_after has is_continuation flag
        if table_after.is_continuation:
            return True

        return False

    def _get_top_blocks(
        self,
        blocks: list[Block],
        n: int = 3,
    ) -> list[Block]:
        """Get top n blocks by reading order."""
        sorted_blocks = sorted(blocks, key=lambda b: b.reading_order)
        return sorted_blocks[:n]

    def _get_bottom_blocks(
        self,
        blocks: list[Block],
        n: int = 3,
    ) -> list[Block]:
        """Get bottom n blocks by reading order."""
        sorted_blocks = sorted(blocks, key=lambda b: b.reading_order, reverse=True)
        return sorted_blocks[:n]

    def _get_block_text(self, block: Block) -> str:
        """Get text content of a block."""
        if block.ocr_result:
            return block.ocr_result.normalized_text or block.ocr_result.raw_text
        return ""

    def update_pages(
        self,
        pages: list[Page],
        blocks_by_page: dict,
        tables_by_page: dict = None,
    ) -> list[Page]:
        """Update pages with continuation information.

        Args:
            pages: List of pages to update (sorted by page number).
            blocks_by_page: Dict mapping page_id to list of blocks.
            tables_by_page: Dict mapping page_id to list of tables.

        Returns:
            Updated pages with continuation flags.
        """
        if tables_by_page is None:
            tables_by_page = {}

        for i, page in enumerate(pages):
            previous_page = pages[i - 1] if i > 0 else None
            next_page = pages[i + 1] if i < len(pages) - 1 else None

            current_blocks = blocks_by_page.get(page.id, [])
            previous_blocks = blocks_by_page.get(previous_page.id, []) if previous_page else None
            next_blocks = blocks_by_page.get(next_page.id, []) if next_page else None

            current_tables = tables_by_page.get(page.id, [])
            previous_tables = tables_by_page.get(previous_page.id, []) if previous_page else None
            next_tables = tables_by_page.get(next_page.id, []) if next_page else None

            info = self.detect_page_continuations(
                current_page=page,
                current_blocks=current_blocks,
                previous_page=previous_page,
                previous_blocks=previous_blocks,
                next_page=next_page,
                next_blocks=next_blocks,
                current_tables=current_tables,
                previous_tables=previous_tables,
                next_tables=next_tables,
            )

            # Update page
            page.continues_from_previous = info.continues_from_previous
            page.continues_to_next = info.continues_to_next
            page.continuation_type = info.continuation_type

        return pages


def find_continuation_groups(
    pages: list[Page],
    max_group_size: int = 10,
) -> list[list[Page]]:
    """Group pages that are connected by continuations.

    Respects bounded-context limit per ADR.

    Args:
        pages: List of pages sorted by page number.
        max_group_size: Maximum pages in a group (default 10 per ADR).

    Returns:
        List of page groups.
    """
    groups = []
    current_group = []

    for page in pages:
        if not current_group:
            current_group.append(page)
        elif page.continues_from_previous and len(current_group) < max_group_size:
            current_group.append(page)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [page]

    # Don't forget last group
    if current_group:
        groups.append(current_group)

    return groups
