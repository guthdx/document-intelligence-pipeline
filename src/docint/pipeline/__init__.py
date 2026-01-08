"""Pipeline stages for Document Intelligence processing.

Deterministic Stages (No LLM):
1. stage_render - PDF to PNG at 300 DPI
2. stage_layout - Layout detection (YOLO/DINO)
3. stage_orient - Orientation detection (0/90/180/270)
4. stage_ocr - Printed text OCR
5. stage_htr - Handwriting recognition
6. stage_table - Table extraction
7. stage_cont - Continuation detection
8. stage_understand - Layout understanding (LayoutLMv3)

Each stage is independent and can be run separately or
orchestrated through the main pipeline.
"""

from .stage_render import PDFRenderer

__all__ = [
    "PDFRenderer",
]
