"""OCR Stage - Extract text from printed content blocks.

Uses Tesseract OCR as primary engine for printed text extraction.
Produces word-level bounding boxes and confidence scores.

For handwriting recognition, see stage_htr.py.
"""

import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import cv2
import numpy as np
import pytesseract
from PIL import Image

from docint.models import (
    Block,
    BlockType,
    BoundingBox,
    ConfidenceLevel,
    OCRResult,
    WordBox,
)
from docint.pipeline.stage_orient import correct_orientation


def confidence_to_level(confidence: float) -> ConfidenceLevel:
    """Convert numeric confidence to confidence level.

    Args:
        confidence: Confidence score (0-100 or 0-1).

    Returns:
        ConfidenceLevel enum value.
    """
    # Normalize to 0-1 range
    if confidence > 1:
        confidence = confidence / 100.0

    if confidence >= 0.9:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.7:
        return ConfidenceLevel.MEDIUM
    elif confidence >= 0.5:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


class TesseractOCR:
    """OCR engine using Tesseract.

    Extracts text with word-level bounding boxes and confidence scores.
    """

    def __init__(
        self,
        language: str = "eng",
        psm: int = 6,
        oem: int = 3,
        config: Optional[str] = None,
    ):
        """Initialize Tesseract OCR.

        Args:
            language: Tesseract language code(s), e.g., 'eng', 'eng+spa'.
            psm: Page segmentation mode (6 = assume uniform block of text).
            oem: OCR Engine mode (3 = default, based on what's available).
            config: Additional Tesseract config string.
        """
        self.language = language
        self.psm = psm
        self.oem = oem
        self.config = config or ""

    def _build_config(self) -> str:
        """Build Tesseract configuration string."""
        config_parts = [
            f"--psm {self.psm}",
            f"--oem {self.oem}",
        ]
        if self.config:
            config_parts.append(self.config)
        return " ".join(config_parts)

    def extract_text(self, image: np.ndarray) -> str:
        """Extract plain text from image.

        Args:
            image: Image as numpy array (grayscale or BGR).

        Returns:
            Extracted text string.
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        text = pytesseract.image_to_string(
            pil_image,
            lang=self.language,
            config=self._build_config(),
        )

        return text.strip()

    def extract_with_boxes(
        self,
        image: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> OCRResult:
        """Extract text with word-level bounding boxes.

        Args:
            image: Image as numpy array.
            image_width: Original image width (for normalization).
            image_height: Original image height (for normalization).

        Returns:
            OCRResult with text, words, and confidence.
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        # Get detailed word data
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.language,
            config=self._build_config(),
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confidences = []
        text_parts = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = data["conf"][i]

            # Skip empty or low-confidence noise
            if not text or conf < 0:
                continue

            # Normalize confidence (Tesseract returns 0-100)
            conf_normalized = conf / 100.0
            confidences.append(conf_normalized)

            # Get bounding box (Tesseract returns absolute pixels)
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Normalize to 0-1 coordinates
            bbox = BoundingBox(
                x=x / image_width,
                y=y / image_height,
                width=w / image_width,
                height=h / image_height,
                unit="normalized",
            )

            word_box = WordBox(
                text=text,
                bbox=bbox,
                confidence=conf_normalized,
            )
            words.append(word_box)
            text_parts.append(text)

        # Calculate overall confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        raw_text = " ".join(text_parts)

        return OCRResult(
            raw_text=raw_text,
            words=words,
            confidence=avg_confidence,
            confidence_level=confidence_to_level(avg_confidence),
            ocr_engine="tesseract",
            ocr_version=pytesseract.get_tesseract_version().vstring,
        )

    def extract_hocr(
        self,
        image: np.ndarray,
    ) -> str:
        """Extract text in hOCR format (XML with coordinates).

        Args:
            image: Image as numpy array.

        Returns:
            hOCR XML string.
        """
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        hocr = pytesseract.image_to_pdf_or_hocr(
            pil_image,
            lang=self.language,
            config=self._build_config(),
            extension="hocr",
        )

        return hocr.decode("utf-8")


class OCRProcessor:
    """Processes blocks through OCR pipeline.

    Handles orientation correction and OCR for multiple block types.
    """

    def __init__(
        self,
        language: str = "eng",
        min_confidence: float = 0.3,
        audit_threshold: float = 0.5,
    ):
        """Initialize OCR processor.

        Args:
            language: Tesseract language code(s).
            min_confidence: Minimum confidence to accept OCR result.
            audit_threshold: Confidence below which to flag for audit.
        """
        self.ocr_engine = TesseractOCR(language=language)
        self.min_confidence = min_confidence
        self.audit_threshold = audit_threshold

    def process_block(
        self,
        block: Block,
        page_image_path: Path,
    ) -> Block:
        """Process a single block through OCR.

        Args:
            block: Block to process.
            page_image_path: Path to page image.

        Returns:
            Block with OCR result populated.
        """
        # Skip non-text blocks
        if block.block_type in [BlockType.FIGURE, BlockType.SIGNATURE]:
            return block

        # For handwriting, should use HTR stage instead
        if block.block_type == BlockType.HANDWRITING:
            return block

        # Extract block image
        block_image = self._extract_block_image(block, page_image_path)
        if block_image is None:
            return block

        # Apply orientation correction if detected
        if block.detected_orientation.value != 0:
            block_image = correct_orientation(block_image, block.detected_orientation)

        # Get image dimensions
        height, width = block_image.shape[:2]

        # Run OCR
        ocr_result = self.ocr_engine.extract_with_boxes(block_image, width, height)

        # Update block
        block.ocr_result = ocr_result

        # Flag for audit if low confidence
        if ocr_result.confidence < self.audit_threshold:
            block.flag_for_audit(f"Low OCR confidence: {ocr_result.confidence:.2f}")

        return block

    def process_page_blocks(
        self,
        blocks: list[Block],
        page_image_path: Path,
    ) -> list[Block]:
        """Process all text blocks on a page.

        Args:
            blocks: List of blocks to process.
            page_image_path: Path to page image.

        Returns:
            Blocks with OCR results.
        """
        for block in blocks:
            # Process text-containing blocks
            if block.block_type in [
                BlockType.TEXT,
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.FORM_FIELD,
                BlockType.TABLE,  # Tables need OCR too
            ]:
                self.process_block(block, page_image_path)

        return blocks

    def _extract_block_image(
        self,
        block: Block,
        page_image_path: Path,
    ) -> Optional[np.ndarray]:
        """Extract block region from page image.

        Args:
            block: Block with bounding box.
            page_image_path: Path to page image.

        Returns:
            Block image as numpy array, or None if extraction fails.
        """
        try:
            page_image = cv2.imread(str(page_image_path))
            if page_image is None:
                return None

            height, width = page_image.shape[:2]

            # Convert normalized bbox to pixels
            x1 = int(block.bbox.x * width)
            y1 = int(block.bbox.y * height)
            x2 = int((block.bbox.x + block.bbox.width) * width)
            y2 = int((block.bbox.y + block.bbox.height) * height)

            # Add small padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                return None

            block_region = page_image[y1:y2, x1:x2]

            # Convert to grayscale
            if len(block_region.shape) == 3:
                block_region = cv2.cvtColor(block_region, cv2.COLOR_BGR2GRAY)

            return block_region

        except Exception:
            return None


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing to improve OCR quality.

    Args:
        image: Grayscale image.

    Returns:
        Preprocessed image.
    """
    # Binarization using Otsu's method
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # Dilation to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)

    return dilated


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Correct skew in image.

    Args:
        image: Grayscale image.

    Returns:
        Deskewed image.
    """
    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(image > 0))

    if len(coords) < 10:
        return image

    # Get minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Rotate if significant skew
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    return image
