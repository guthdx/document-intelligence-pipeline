"""Layout Detection Stage - Detect document regions using YOLO.

Uses YOLOv8 for object detection to identify document regions:
- text blocks, tables, figures, handwriting, stamps, signatures, form fields

Supports MPS (Metal Performance Shaders) acceleration on Apple Silicon.
"""

from pathlib import Path
from typing import Optional
from uuid import UUID

import torch
from PIL import Image
from ultralytics import YOLO

from docint.config import settings
from docint.models import Block, BlockType, BoundingBox, Page


# Mapping from model class IDs to BlockType
# This mapping depends on the training dataset (e.g., DocLayNet, PubLayNet)
DOCLAYNET_CLASS_MAP = {
    0: BlockType.TEXT,
    1: BlockType.TEXT,  # Title -> TEXT
    2: BlockType.TEXT,  # List -> TEXT
    3: BlockType.TABLE,
    4: BlockType.FIGURE,
    5: BlockType.TEXT,  # Caption -> TEXT
    6: BlockType.HEADER,
    7: BlockType.FOOTER,
    8: BlockType.PAGE_NUMBER,
    9: BlockType.TEXT,  # Section header -> TEXT
    10: BlockType.FORM_FIELD,
}

# Extended mapping for custom-trained models
DOCINT_CLASS_MAP = {
    0: BlockType.TEXT,
    1: BlockType.TABLE,
    2: BlockType.FIGURE,
    3: BlockType.HANDWRITING,
    4: BlockType.STAMP,
    5: BlockType.SIGNATURE,
    6: BlockType.FORM_FIELD,
    7: BlockType.HEADER,
    8: BlockType.FOOTER,
    9: BlockType.PAGE_NUMBER,
}


def get_device() -> str:
    """Get the best available device for inference.

    Returns:
        Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, else 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LayoutDetector:
    """Detects document layout regions using YOLO.

    Uses YOLOv8 model (optionally fine-tuned on document datasets like DocLayNet)
    to detect text blocks, tables, figures, handwriting, and other regions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        class_map: Optional[dict[int, BlockType]] = None,
    ):
        """Initialize the layout detector.

        Args:
            model_path: Path to YOLO model weights. If None, uses default YOLOv8n.
            confidence_threshold: Minimum confidence for detections (0-1).
            iou_threshold: IoU threshold for NMS.
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detected if None.
            class_map: Mapping from model class IDs to BlockType.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or get_device()
        self.class_map = class_map or DOCINT_CLASS_MAP

        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.model_name = Path(model_path).stem
        else:
            # Use pretrained YOLOv8 as fallback
            # In production, use a document-specific model
            self.model = YOLO("yolov8n.pt")
            self.model_name = "yolov8n-generic"

        # Move model to device
        self.model.to(self.device)

    def detect_page(
        self,
        page: Page,
        document_id: UUID,
    ) -> list[Block]:
        """Detect layout regions in a single page.

        Args:
            page: Page object with render_info containing image path.
            document_id: Parent document UUID.

        Returns:
            List of Block objects with detected regions.
        """
        if page.render_info is None:
            raise ValueError(f"Page {page.page_number} has no render info")

        image_path = Path(page.render_info.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Page image not found: {image_path}")

        # Load image to get dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Run detection
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        blocks = []
        if results and len(results) > 0:
            result = results[0]

            # Sort by reading order (top-to-bottom, left-to-right)
            detections = []
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    detections.append({
                        "xyxy": xyxy,
                        "conf": conf,
                        "cls_id": cls_id,
                        "center_y": (xyxy[1] + xyxy[3]) / 2,
                        "center_x": (xyxy[0] + xyxy[2]) / 2,
                    })

            # Sort by reading order: primary by Y (top to bottom), secondary by X (left to right)
            detections.sort(key=lambda d: (d["center_y"] // 50, d["center_x"]))

            for reading_order, det in enumerate(detections):
                xyxy = det["xyxy"]
                conf = det["conf"]
                cls_id = det["cls_id"]

                # Convert to normalized coordinates
                x1, y1, x2, y2 = xyxy
                bbox = BoundingBox(
                    x=float(x1) / img_width,
                    y=float(y1) / img_height,
                    width=float(x2 - x1) / img_width,
                    height=float(y2 - y1) / img_height,
                    unit="normalized",
                )

                # Map class ID to BlockType
                block_type = self.class_map.get(cls_id, BlockType.TEXT)

                # Create block
                block = Block(
                    document_id=document_id,
                    page_id=page.id,
                    page_number=page.page_number,
                    block_type=block_type,
                    bbox=bbox,
                    detection_confidence=conf,
                    detector_model=self.model_name,
                    reading_order=reading_order,
                )

                # Flag low-confidence detections
                if conf < 0.5:
                    block.flag_for_audit(f"Low detection confidence: {conf:.2f}")

                blocks.append(block)

        return blocks

    def detect_batch(
        self,
        pages: list[Page],
        document_id: UUID,
        batch_size: int = 8,
    ) -> dict[UUID, list[Block]]:
        """Detect layout regions in multiple pages.

        Uses batched inference for efficiency.

        Args:
            pages: List of Page objects to process.
            document_id: Parent document UUID.
            batch_size: Number of images per batch.

        Returns:
            Dict mapping page_id to list of detected Blocks.
        """
        results_map: dict[UUID, list[Block]] = {}

        # Process in batches
        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i:i + batch_size]
            image_paths = []
            page_dims = []

            for page in batch_pages:
                if page.render_info is None:
                    continue
                image_path = Path(page.render_info.image_path)
                if image_path.exists():
                    image_paths.append(str(image_path))
                    with Image.open(image_path) as img:
                        page_dims.append(img.size)

            if not image_paths:
                continue

            # Batched prediction
            batch_results = self.model.predict(
                source=image_paths,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            # Process results
            for j, (page, result) in enumerate(zip(batch_pages, batch_results)):
                if page.render_info is None:
                    continue

                img_width, img_height = page_dims[j]
                blocks = self._process_result(
                    result,
                    page,
                    document_id,
                    img_width,
                    img_height,
                )
                results_map[page.id] = blocks

        return results_map

    def _process_result(
        self,
        result,
        page: Page,
        document_id: UUID,
        img_width: int,
        img_height: int,
    ) -> list[Block]:
        """Process a single YOLO result into Block objects.

        Args:
            result: YOLO detection result.
            page: Source page.
            document_id: Parent document UUID.
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            List of Block objects.
        """
        blocks = []

        if result.boxes is None:
            return blocks

        # Collect and sort detections
        detections = []
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            detections.append({
                "xyxy": xyxy,
                "conf": conf,
                "cls_id": cls_id,
                "center_y": (xyxy[1] + xyxy[3]) / 2,
                "center_x": (xyxy[0] + xyxy[2]) / 2,
            })

        # Sort by reading order
        detections.sort(key=lambda d: (d["center_y"] // 50, d["center_x"]))

        for reading_order, det in enumerate(detections):
            xyxy = det["xyxy"]
            x1, y1, x2, y2 = xyxy

            bbox = BoundingBox(
                x=float(x1) / img_width,
                y=float(y1) / img_height,
                width=float(x2 - x1) / img_width,
                height=float(y2 - y1) / img_height,
                unit="normalized",
            )

            block_type = self.class_map.get(det["cls_id"], BlockType.TEXT)

            block = Block(
                document_id=document_id,
                page_id=page.id,
                page_number=page.page_number,
                block_type=block_type,
                bbox=bbox,
                detection_confidence=det["conf"],
                detector_model=self.model_name,
                reading_order=reading_order,
            )

            if det["conf"] < 0.5:
                block.flag_for_audit(f"Low detection confidence: {det['conf']:.2f}")

            blocks.append(block)

        return blocks

    def detect_handwriting_regions(
        self,
        page: Page,
        document_id: UUID,
        existing_blocks: list[Block],
    ) -> list[Block]:
        """Identify potential handwriting regions within detected blocks.

        Uses secondary analysis to detect handwriting that may be
        mixed with printed content or in margins.

        Args:
            page: Page to analyze.
            document_id: Parent document UUID.
            existing_blocks: Already detected blocks.

        Returns:
            Additional handwriting blocks found.
        """
        # This is a placeholder for more sophisticated handwriting detection
        # In practice, would use a specialized handwriting detector or
        # analyze ink density, stroke patterns, etc.
        handwriting_blocks = []

        for block in existing_blocks:
            if block.block_type == BlockType.HANDWRITING:
                # Already identified as handwriting
                continue

            # Check if block might contain handwriting annotations
            # based on characteristics like:
            # - Low OCR confidence (later stage)
            # - Unusual aspect ratios
            # - Marginal positions

            # For now, flag blocks in margin areas for potential handwriting
            if block.bbox.x < 0.1 or block.bbox.x + block.bbox.width > 0.9:
                # Near margins - might be annotation
                # Will be refined in OCR stage based on recognition results
                pass

        return handwriting_blocks


class LayoutDetectorConfig:
    """Configuration for layout detection models."""

    # Pre-trained model options
    MODELS = {
        "yolov8n-doclaynet": {
            "description": "YOLOv8 nano trained on DocLayNet",
            "classes": 11,
            "class_map": DOCLAYNET_CLASS_MAP,
        },
        "yolov8s-doclaynet": {
            "description": "YOLOv8 small trained on DocLayNet",
            "classes": 11,
            "class_map": DOCLAYNET_CLASS_MAP,
        },
        "yolov8m-docint": {
            "description": "YOLOv8 medium trained on custom DocInt dataset",
            "classes": 10,
            "class_map": DOCINT_CLASS_MAP,
        },
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get configuration for a specific model."""
        return cls.MODELS.get(model_name, {})


def download_model(model_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a pre-trained layout detection model.

    Args:
        model_name: Name of the model to download.
        cache_dir: Directory to cache models. Defaults to ~/.cache/docint/models.

    Returns:
        Path to the downloaded model.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "docint" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / f"{model_name}.pt"

    if model_path.exists():
        return model_path

    # Model download URLs would be configured here
    # For now, return path where model should be placed
    raise FileNotFoundError(
        f"Model {model_name} not found. Please download and place at {model_path}"
    )
