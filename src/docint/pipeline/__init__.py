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

from .stage_layout import LayoutDetector, LayoutDetectorConfig
from .stage_ocr import OCRProcessor, TesseractOCR
from .stage_orient import OrientationDetector, correct_orientation
from .stage_render import PDFRenderer

__all__ = [
    # Render
    "PDFRenderer",
    # Layout
    "LayoutDetector",
    "LayoutDetectorConfig",
    # Orientation
    "OrientationDetector",
    "correct_orientation",
    # OCR
    "OCRProcessor",
    "TesseractOCR",
]
