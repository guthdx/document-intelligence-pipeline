"""Orientation Detection Stage - Detect and correct block rotation.

Detects rotation (0°, 90°, 180°, 270°) for each block.
Critical for handwriting blocks per ADR-003.

Primary method: Tesseract OSD (Orientation and Script Detection)
Fallback: CNN-based rotation classifier for low-confidence cases.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from docint.models import Block, BlockType, Orientation


def tesseract_osd(image_path: Path) -> tuple[Orientation, float]:
    """Detect orientation using Tesseract OSD.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (detected_orientation, confidence)
    """
    try:
        result = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "-",
                "--psm", "0",  # OSD only mode
                "-c", "min_characters_to_try=5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse Tesseract OSD output
        output = result.stdout + result.stderr
        orientation = Orientation.DEG_0
        confidence = 0.0

        for line in output.split("\n"):
            line = line.strip()
            if "Orientation in degrees:" in line:
                try:
                    degrees = int(line.split(":")[-1].strip())
                    orientation = _degrees_to_orientation(degrees)
                except (ValueError, IndexError):
                    pass
            elif "Orientation confidence:" in line:
                try:
                    confidence = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass

        return orientation, confidence

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return Orientation.DEG_0, 0.0


def _degrees_to_orientation(degrees: int) -> Orientation:
    """Convert degrees to Orientation enum.

    Args:
        degrees: Rotation in degrees (0, 90, 180, 270)

    Returns:
        Corresponding Orientation enum value
    """
    # Normalize to 0-360 range
    degrees = degrees % 360

    if degrees < 45 or degrees >= 315:
        return Orientation.DEG_0
    elif 45 <= degrees < 135:
        return Orientation.DEG_90
    elif 135 <= degrees < 225:
        return Orientation.DEG_180
    else:
        return Orientation.DEG_270


def analyze_text_lines(image: np.ndarray) -> tuple[Orientation, float]:
    """Analyze text line angles to detect orientation.

    Uses Hough transform to detect dominant line angles.

    Args:
        image: Grayscale image as numpy array.

    Returns:
        Tuple of (detected_orientation, confidence)
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return Orientation.DEG_0, 0.0

    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Find dominant angle
    angles = np.array(angles)
    median_angle = np.median(angles)

    # Determine orientation based on dominant angle
    # Horizontal text (0°) has angles near 0 or 180
    # Vertical text (90°) has angles near 90 or -90
    abs_angle = abs(median_angle)

    if abs_angle < 20 or abs_angle > 160:
        # Horizontal - check if upside down
        if abs_angle > 160:
            return Orientation.DEG_180, 0.7
        return Orientation.DEG_0, 0.7
    elif 70 < abs_angle < 110:
        # Vertical
        if median_angle > 0:
            return Orientation.DEG_90, 0.7
        return Orientation.DEG_270, 0.7

    return Orientation.DEG_0, 0.3


class OrientationDetector:
    """Detects and corrects block orientation.

    Uses Tesseract OSD as primary method with fallback to
    geometric analysis for low-confidence cases.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        use_tesseract: bool = True,
        use_geometric: bool = True,
    ):
        """Initialize orientation detector.

        Args:
            confidence_threshold: Minimum confidence to accept detection.
            use_tesseract: Use Tesseract OSD for detection.
            use_geometric: Use geometric analysis as fallback.
        """
        self.confidence_threshold = confidence_threshold
        self.use_tesseract = use_tesseract
        self.use_geometric = use_geometric

    def detect_block_orientation(
        self,
        block: Block,
        page_image_path: Path,
    ) -> tuple[Orientation, float]:
        """Detect orientation for a single block.

        Args:
            block: Block to detect orientation for.
            page_image_path: Path to the full page image.

        Returns:
            Tuple of (detected_orientation, confidence)
        """
        # Extract block region from page image
        block_image = self._extract_block_image(block, page_image_path)

        if block_image is None:
            return Orientation.DEG_0, 0.0

        # Try Tesseract OSD first
        if self.use_tesseract:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), block_image)

                orientation, confidence = tesseract_osd(tmp_path)
                tmp_path.unlink()  # Clean up

                if confidence >= self.confidence_threshold:
                    return orientation, confidence

        # Fallback to geometric analysis
        if self.use_geometric:
            orientation, confidence = analyze_text_lines(block_image)
            if confidence >= self.confidence_threshold:
                return orientation, confidence

        # Default to no rotation with low confidence
        return Orientation.DEG_0, 0.0

    def detect_and_update_blocks(
        self,
        blocks: list[Block],
        page_image_path: Path,
        prioritize_handwriting: bool = True,
    ) -> list[Block]:
        """Detect orientation for multiple blocks and update them.

        Args:
            blocks: List of blocks to process.
            page_image_path: Path to the page image.
            prioritize_handwriting: Process handwriting blocks even if others are skipped.

        Returns:
            Updated blocks with orientation information.
        """
        for block in blocks:
            # Always check handwriting blocks
            # Skip printed text blocks if they appear normally oriented
            should_check = (
                block.block_type == BlockType.HANDWRITING
                or block.block_type == BlockType.SIGNATURE
                or prioritize_handwriting is False
            )

            if not should_check:
                # Quick heuristic: if block is in normal reading position, skip
                # (tall blocks might be rotated, wide blocks are probably normal)
                aspect_ratio = block.bbox.width / block.bbox.height if block.bbox.height > 0 else 1
                if 0.3 < aspect_ratio < 3:
                    continue

            # Detect orientation
            orientation, confidence = self.detect_block_orientation(block, page_image_path)

            block.detected_orientation = orientation

            # Flag low-confidence detections for audit
            if block.block_type == BlockType.HANDWRITING and confidence < 0.5:
                block.flag_for_audit(f"Low orientation confidence: {confidence:.2f}")

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
            Grayscale image of block region, or None if extraction fails.
        """
        try:
            # Load page image
            page_image = cv2.imread(str(page_image_path))
            if page_image is None:
                return None

            height, width = page_image.shape[:2]

            # Convert normalized bbox to pixels
            x1 = int(block.bbox.x * width)
            y1 = int(block.bbox.y * height)
            x2 = int((block.bbox.x + block.bbox.width) * width)
            y2 = int((block.bbox.y + block.bbox.height) * height)

            # Ensure valid bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Extract and convert to grayscale
            block_region = page_image[y1:y2, x1:x2]
            gray = cv2.cvtColor(block_region, cv2.COLOR_BGR2GRAY)

            return gray

        except Exception:
            return None

    def apply_orientation(
        self,
        block: Block,
        page_image_path: Path,
        output_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """Apply detected orientation correction to block image.

        Args:
            block: Block with detected orientation.
            page_image_path: Path to page image.
            output_path: Optional path to save corrected image.

        Returns:
            Corrected image as numpy array, or None if correction fails.
        """
        block_image = self._extract_block_image(block, page_image_path)
        if block_image is None:
            return None

        # Apply rotation if needed
        if block.detected_orientation == Orientation.DEG_0:
            corrected = block_image
        elif block.detected_orientation == Orientation.DEG_90:
            corrected = cv2.rotate(block_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif block.detected_orientation == Orientation.DEG_180:
            corrected = cv2.rotate(block_image, cv2.ROTATE_180)
        elif block.detected_orientation == Orientation.DEG_270:
            corrected = cv2.rotate(block_image, cv2.ROTATE_90_CLOCKWISE)
        else:
            corrected = block_image

        # Update block's applied orientation
        block.applied_orientation = block.detected_orientation

        # Save if output path provided
        if output_path is not None:
            cv2.imwrite(str(output_path), corrected)

        return corrected


def correct_orientation(
    image: np.ndarray,
    orientation: Orientation,
) -> np.ndarray:
    """Apply orientation correction to an image.

    Args:
        image: Image as numpy array.
        orientation: Current orientation to correct from.

    Returns:
        Corrected image.
    """
    if orientation == Orientation.DEG_0:
        return image
    elif orientation == Orientation.DEG_90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == Orientation.DEG_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == Orientation.DEG_270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image
