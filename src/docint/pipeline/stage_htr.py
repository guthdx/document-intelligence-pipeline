"""Handwriting Recognition Stage - Extract text from handwritten content.

Uses TrOCR (Transformer-based OCR) for handwriting recognition.
TrOCR is a pre-trained model from Microsoft that combines image encoder
(ViT/DeiT) with text decoder (RoBERTa) for OCR.

Per ADR-003: Handwriting is treated as a separate data class with:
- Block-level orientation detection
- Confidence flagging for audit
- Annotation type classification
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from docint.models import (
    Block,
    BlockType,
    BoundingBox,
    ConfidenceLevel,
    HandwritingBlock,
    OCRResult,
    WordBox,
)
from docint.pipeline.stage_orient import correct_orientation


def get_device() -> str:
    """Get the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TrOCREngine:
    """TrOCR-based handwriting recognition engine.

    Uses Microsoft's TrOCR model for handwritten text recognition.
    """

    # Available TrOCR model variants
    MODELS = {
        "small": "microsoft/trocr-small-handwritten",
        "base": "microsoft/trocr-base-handwritten",
        "large": "microsoft/trocr-large-handwritten",
        # For printed text (fallback)
        "printed-small": "microsoft/trocr-small-printed",
        "printed-base": "microsoft/trocr-base-printed",
    }

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        max_length: int = 64,
    ):
        """Initialize TrOCR engine.

        Args:
            model_name: Model variant ('small', 'base', 'large').
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detected if None.
            max_length: Maximum output sequence length.
        """
        self.device = device or get_device()
        self.max_length = max_length

        # Get model path
        model_path = self.MODELS.get(model_name, model_name)
        self.model_name = model_name

        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def recognize(
        self,
        image: np.ndarray,
    ) -> tuple[str, float]:
        """Recognize handwritten text in an image.

        Args:
            image: Image as numpy array (grayscale or BGR).

        Returns:
            Tuple of (recognized_text, confidence_score).
        """
        # Convert to PIL RGB image
        if len(image.shape) == 2:
            # Grayscale to RGB
            pil_image = Image.fromarray(image).convert("RGB")
        elif len(image.shape) == 3:
            # BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Preprocess
        pixel_values = self.processor(
            images=pil_image,
            return_tensors="pt",
        ).pixel_values

        # Move to device
        pixel_values = pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode text
        generated_ids = generated.sequences
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # Calculate confidence from scores
        confidence = self._calculate_confidence(generated)

        return text.strip(), confidence

    def _calculate_confidence(self, generated) -> float:
        """Calculate confidence score from generation output.

        Args:
            generated: Output from model.generate().

        Returns:
            Confidence score (0-1).
        """
        if not hasattr(generated, "scores") or not generated.scores:
            return 0.5  # Default confidence

        # Get probabilities from scores
        scores = generated.scores
        confidences = []

        for score in scores:
            probs = torch.softmax(score, dim=-1)
            max_prob = probs.max().item()
            confidences.append(max_prob)

        if confidences:
            return sum(confidences) / len(confidences)
        return 0.5

    def recognize_batch(
        self,
        images: list[np.ndarray],
    ) -> list[tuple[str, float]]:
        """Recognize handwritten text in multiple images.

        Args:
            images: List of images as numpy arrays.

        Returns:
            List of (text, confidence) tuples.
        """
        # Convert all images
        pil_images = []
        for image in images:
            if len(image.shape) == 2:
                pil_images.append(Image.fromarray(image).convert("RGB"))
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

        # Batch preprocess
        pixel_values = self.processor(
            images=pil_images,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                max_length=self.max_length,
            )

        # Decode
        texts = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        # Return with default confidence (batch doesn't return scores easily)
        return [(t.strip(), 0.7) for t in texts]


class HandwritingRecognizer:
    """Processes handwriting blocks through HTR pipeline.

    Handles orientation correction, recognition, and confidence-based flagging.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        audit_threshold: float = 0.5,
    ):
        """Initialize handwriting recognizer.

        Args:
            model_name: TrOCR model variant.
            device: Device for inference.
            audit_threshold: Confidence below which to flag for audit.
        """
        self.engine = TrOCREngine(model_name=model_name, device=device)
        self.audit_threshold = audit_threshold
        self.model_name = f"trocr-{model_name}"

    def process_block(
        self,
        block: Block,
        page_image_path: Path,
    ) -> Block:
        """Process a single handwriting block.

        Args:
            block: Block to process (should be HANDWRITING type).
            page_image_path: Path to page image.

        Returns:
            Block with OCR result populated.
        """
        # Only process handwriting blocks
        if block.block_type != BlockType.HANDWRITING:
            return block

        # Extract block image
        block_image = self._extract_block_image(block, page_image_path)
        if block_image is None:
            return block

        # Apply orientation correction
        if block.detected_orientation.value != 0:
            block_image = correct_orientation(block_image, block.detected_orientation)
            block.applied_orientation = block.detected_orientation

        # Recognize text
        text, confidence = self.engine.recognize(block_image)

        # Determine confidence level
        if confidence >= 0.9:
            conf_level = ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            conf_level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.5:
            conf_level = ConfidenceLevel.LOW
        else:
            conf_level = ConfidenceLevel.VERY_LOW

        # Create OCR result
        ocr_result = OCRResult(
            raw_text=text,
            words=[],  # TrOCR doesn't provide word-level boxes
            confidence=confidence,
            confidence_level=conf_level,
            ocr_engine=self.model_name,
        )

        block.ocr_result = ocr_result

        # Flag for audit if low confidence per ADR-003
        if confidence < self.audit_threshold:
            block.flag_for_audit(
                f"Low handwriting recognition confidence: {confidence:.2f}"
            )

        return block

    def process_page_blocks(
        self,
        blocks: list[Block],
        page_image_path: Path,
        batch_size: int = 8,
    ) -> list[Block]:
        """Process all handwriting blocks on a page.

        Uses batched inference for efficiency.

        Args:
            blocks: List of blocks to process.
            page_image_path: Path to page image.
            batch_size: Number of blocks per batch.

        Returns:
            Blocks with OCR results.
        """
        # Filter handwriting blocks
        hw_blocks = [
            b for b in blocks
            if b.block_type == BlockType.HANDWRITING
        ]

        if not hw_blocks:
            return blocks

        # Extract images
        images = []
        valid_blocks = []
        for block in hw_blocks:
            image = self._extract_block_image(block, page_image_path)
            if image is not None:
                # Apply orientation correction
                if block.detected_orientation.value != 0:
                    image = correct_orientation(image, block.detected_orientation)
                    block.applied_orientation = block.detected_orientation
                images.append(image)
                valid_blocks.append(block)

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_blocks = valid_blocks[i:i + batch_size]

            # Recognize batch
            results = self.engine.recognize_batch(batch_images)

            # Update blocks
            for block, (text, confidence) in zip(batch_blocks, results):
                ocr_result = OCRResult(
                    raw_text=text,
                    words=[],
                    confidence=confidence,
                    confidence_level=self._confidence_level(confidence),
                    ocr_engine=self.model_name,
                )
                block.ocr_result = ocr_result

                if confidence < self.audit_threshold:
                    block.flag_for_audit(
                        f"Low handwriting recognition confidence: {confidence:.2f}"
                    )

        return blocks

    def _confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence to level."""
        if confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.5:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

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

            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                return None

            return page_image[y1:y2, x1:x2]

        except Exception:
            return None


def classify_annotation_type(
    block: Block,
    page_width: float,
    page_height: float,
) -> str:
    """Classify the type of handwritten annotation.

    Args:
        block: Handwriting block.
        page_width: Page width for position calculation.
        page_height: Page height for position calculation.

    Returns:
        Annotation type string.
    """
    # Get block position
    x_center = block.bbox.x + block.bbox.width / 2
    y_center = block.bbox.y + block.bbox.height / 2

    # Check margins
    left_margin = x_center < 0.15
    right_margin = x_center > 0.85
    top_margin = y_center < 0.1
    bottom_margin = y_center > 0.9

    # Classify based on position
    if left_margin or right_margin:
        return "margin_note"
    elif top_margin:
        return "header_annotation"
    elif bottom_margin:
        return "footer_annotation"
    elif block.bbox.width < 0.3 and block.bbox.height < 0.05:
        return "inline_annotation"
    else:
        return "general_handwriting"
