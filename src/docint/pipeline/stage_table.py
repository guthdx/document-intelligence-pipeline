"""Table Extraction Stage - Extract structured table data.

Uses Table Transformer (TATR) for table structure recognition.
Tables are stored as first-class relational data per ADR-002.

Flow:
1. Detect table structure (rows, columns, cells)
2. Extract cell bounding boxes
3. OCR each cell
4. Build relational Table and TableCell objects
"""

from pathlib import Path
from typing import Optional
from uuid import UUID

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)

from docint.models import (
    Block,
    BlockType,
    BoundingBox,
    ConfidenceLevel,
    Table,
    TableCell,
)
from docint.pipeline.stage_ocr import TesseractOCR


def get_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TableTransformerExtractor:
    """Table structure extraction using Table Transformer.

    Uses Microsoft's Table Transformer model for:
    - Table detection
    - Row/column structure recognition
    - Cell detection
    """

    # Model variants
    MODELS = {
        "detection": "microsoft/table-transformer-detection",
        "structure": "microsoft/table-transformer-structure-recognition",
        "structure-v1.1": "microsoft/table-transformer-structure-recognition-v1.1-all",
    }

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize table extractor.

        Args:
            device: Device for inference.
            confidence_threshold: Minimum confidence for detections.
        """
        self.device = device or get_device()
        self.confidence_threshold = confidence_threshold

        # Load structure recognition model
        model_name = self.MODELS["structure-v1.1"]
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Class labels for structure recognition
        self.id2label = self.model.config.id2label

    def extract_structure(
        self,
        image: np.ndarray,
    ) -> dict:
        """Extract table structure from image.

        Args:
            image: Table image as numpy array.

        Returns:
            Dict with detected rows, columns, cells, and headers.
        """
        # Convert to PIL RGB
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        # Get original size
        orig_width, orig_height = pil_image.size

        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([[orig_height, orig_width]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        # Organize by category
        structure = {
            "rows": [],
            "columns": [],
            "cells": [],
            "headers": [],
            "spanning_cells": [],
        }

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            label_name = self.id2label[label.item()]
            bbox = box.cpu().numpy()

            detection = {
                "bbox": [float(b) for b in bbox],  # [x1, y1, x2, y2]
                "confidence": float(score.cpu().numpy()),
            }

            if "row" in label_name.lower():
                structure["rows"].append(detection)
            elif "column" in label_name.lower():
                structure["columns"].append(detection)
            elif "cell" in label_name.lower():
                if "spanning" in label_name.lower():
                    structure["spanning_cells"].append(detection)
                else:
                    structure["cells"].append(detection)
            elif "header" in label_name.lower():
                structure["headers"].append(detection)

        # Sort rows top-to-bottom, columns left-to-right
        structure["rows"].sort(key=lambda x: x["bbox"][1])
        structure["columns"].sort(key=lambda x: x["bbox"][0])

        return structure


class TableExtractor:
    """Extracts structured tables from document pages.

    Combines table structure detection with OCR to produce
    Table and TableCell objects with full provenance.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        ocr_language: str = "eng",
    ):
        """Initialize table extractor.

        Args:
            device: Device for inference.
            ocr_language: Tesseract language for cell OCR.
        """
        self.structure_extractor = TableTransformerExtractor(device=device)
        self.ocr = TesseractOCR(language=ocr_language, psm=6)

    def extract_table(
        self,
        block: Block,
        page_image_path: Path,
        document_id: UUID,
    ) -> Optional[Table]:
        """Extract structured table from a table block.

        Args:
            block: Block of type TABLE.
            page_image_path: Path to page image.
            document_id: Parent document UUID.

        Returns:
            Table with cells, or None if extraction fails.
        """
        if block.block_type != BlockType.TABLE:
            return None

        # Extract table image
        table_image = self._extract_block_image(block, page_image_path)
        if table_image is None:
            return None

        # Get table structure
        structure = self.structure_extractor.extract_structure(table_image)

        # Build cell grid
        cells = self._build_cells(
            structure,
            table_image,
            block,
        )

        if not cells:
            return None

        # Determine grid dimensions
        max_row = max(c.row for c in cells) + 1
        max_col = max(c.col for c in cells) + 1

        # Identify header rows
        header_rows = self._identify_headers(structure, cells)

        # Create table
        table = Table(
            document_id=document_id,
            block_id=block.id,
            start_page=block.page_number,
            num_rows=max_row,
            num_cols=max_col,
            bbox=block.bbox,
            cells=cells,
            header_rows=header_rows,
            extraction_confidence=block.detection_confidence,
            extractor_model="table-transformer",
        )

        # Extract column names from first header row
        if header_rows:
            header_row = header_rows[0]
            column_names = []
            for col in range(max_col):
                cell = table.get_cell(header_row, col)
                column_names.append(cell.text if cell else "")
            table.column_names = column_names

        return table

    def _build_cells(
        self,
        structure: dict,
        table_image: np.ndarray,
        block: Block,
    ) -> list[TableCell]:
        """Build TableCell objects from detected structure.

        Args:
            structure: Detected table structure.
            table_image: Table image for OCR.
            block: Parent block.

        Returns:
            List of TableCell objects.
        """
        cells = []
        img_height, img_width = table_image.shape[:2]

        # Use detected cells if available
        if structure["cells"]:
            for i, cell_det in enumerate(structure["cells"]):
                bbox = cell_det["bbox"]
                x1, y1, x2, y2 = [int(b) for b in bbox]

                # Extract cell image for OCR
                cell_image = table_image[
                    max(0, y1):min(img_height, y2),
                    max(0, x1):min(img_width, x2),
                ]

                # OCR the cell
                if cell_image.size > 0:
                    text = self.ocr.extract_text(cell_image)
                else:
                    text = ""

                # Determine row/col position based on bbox center
                row = self._find_row_index(
                    (y1 + y2) / 2,
                    structure["rows"],
                )
                col = self._find_col_index(
                    (x1 + x2) / 2,
                    structure["columns"],
                )

                # Create normalized bbox relative to table
                cell_bbox = BoundingBox(
                    x=x1 / img_width,
                    y=y1 / img_height,
                    width=(x2 - x1) / img_width,
                    height=(y2 - y1) / img_height,
                    unit="normalized",
                )

                cell = TableCell(
                    table_id=block.id,  # Will be updated when table is created
                    row=row,
                    col=col,
                    raw_text=text,
                    bbox=cell_bbox,
                    ocr_confidence=0.8,  # Placeholder
                )
                cells.append(cell)
        else:
            # Fall back to grid-based cell detection
            cells = self._grid_based_extraction(
                structure,
                table_image,
                block,
            )

        return cells

    def _grid_based_extraction(
        self,
        structure: dict,
        table_image: np.ndarray,
        block: Block,
    ) -> list[TableCell]:
        """Extract cells using row/column intersections.

        Args:
            structure: Detected rows and columns.
            table_image: Table image.
            block: Parent block.

        Returns:
            List of TableCell objects.
        """
        cells = []
        img_height, img_width = table_image.shape[:2]

        rows = structure["rows"]
        cols = structure["columns"]

        if not rows or not cols:
            return cells

        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                # Calculate cell bounds from row/column intersection
                x1 = int(col["bbox"][0])
                y1 = int(row["bbox"][1])
                x2 = int(col["bbox"][2])
                y2 = int(row["bbox"][3])

                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract and OCR
                cell_image = table_image[y1:y2, x1:x2]
                text = self.ocr.extract_text(cell_image) if cell_image.size > 0 else ""

                cell_bbox = BoundingBox(
                    x=x1 / img_width,
                    y=y1 / img_height,
                    width=(x2 - x1) / img_width,
                    height=(y2 - y1) / img_height,
                    unit="normalized",
                )

                cell = TableCell(
                    table_id=block.id,
                    row=row_idx,
                    col=col_idx,
                    raw_text=text,
                    bbox=cell_bbox,
                    ocr_confidence=0.8,
                )
                cells.append(cell)

        return cells

    def _find_row_index(
        self,
        y_center: float,
        rows: list[dict],
    ) -> int:
        """Find row index for a y-coordinate."""
        for i, row in enumerate(rows):
            y1, y2 = row["bbox"][1], row["bbox"][3]
            if y1 <= y_center <= y2:
                return i
        # Fallback: find closest row
        if rows:
            distances = [
                abs((r["bbox"][1] + r["bbox"][3]) / 2 - y_center)
                for r in rows
            ]
            return distances.index(min(distances))
        return 0

    def _find_col_index(
        self,
        x_center: float,
        cols: list[dict],
    ) -> int:
        """Find column index for an x-coordinate."""
        for i, col in enumerate(cols):
            x1, x2 = col["bbox"][0], col["bbox"][2]
            if x1 <= x_center <= x2:
                return i
        # Fallback: find closest column
        if cols:
            distances = [
                abs((c["bbox"][0] + c["bbox"][2]) / 2 - x_center)
                for c in cols
            ]
            return distances.index(min(distances))
        return 0

    def _identify_headers(
        self,
        structure: dict,
        cells: list[TableCell],
    ) -> list[int]:
        """Identify header rows."""
        header_rows = []

        # Use detected headers
        if structure["headers"]:
            for header in structure["headers"]:
                y_center = (header["bbox"][1] + header["bbox"][3]) / 2
                row_idx = self._find_row_index(y_center, structure["rows"])
                if row_idx not in header_rows:
                    header_rows.append(row_idx)
        else:
            # Assume first row is header
            header_rows = [0]

        # Mark cells in header rows
        for cell in cells:
            if cell.row in header_rows:
                cell.is_header = True

        return sorted(header_rows)

    def _extract_block_image(
        self,
        block: Block,
        page_image_path: Path,
    ) -> Optional[np.ndarray]:
        """Extract table region from page image."""
        try:
            page_image = cv2.imread(str(page_image_path))
            if page_image is None:
                return None

            height, width = page_image.shape[:2]

            x1 = int(block.bbox.x * width)
            y1 = int(block.bbox.y * height)
            x2 = int((block.bbox.x + block.bbox.width) * width)
            y2 = int((block.bbox.y + block.bbox.height) * height)

            # Small padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                return None

            return page_image[y1:y2, x1:x2]

        except Exception:
            return None

    def extract_page_tables(
        self,
        blocks: list[Block],
        page_image_path: Path,
        document_id: UUID,
    ) -> list[Table]:
        """Extract all tables from page blocks.

        Args:
            blocks: All blocks on the page.
            page_image_path: Path to page image.
            document_id: Parent document UUID.

        Returns:
            List of extracted Table objects.
        """
        tables = []

        for block in blocks:
            if block.block_type == BlockType.TABLE:
                table = self.extract_table(block, page_image_path, document_id)
                if table:
                    # Link table to block
                    block.table_id = table.id
                    tables.append(table)

        return tables
