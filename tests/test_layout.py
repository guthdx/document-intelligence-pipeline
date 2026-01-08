"""Tests for layout detection stage."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from docint.models import BlockType, Page, PageRenderInfo, ProcessingStatus
from docint.pipeline.stage_layout import (
    DOCINT_CLASS_MAP,
    DOCLAYNET_CLASS_MAP,
    LayoutDetector,
    LayoutDetectorConfig,
    get_device,
)


class TestGetDevice:
    """Tests for device detection."""

    @patch("docint.pipeline.stage_layout.torch")
    def test_mps_available(self, mock_torch):
        """Test MPS device detection on Apple Silicon."""
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        assert get_device() == "mps"

    @patch("docint.pipeline.stage_layout.torch")
    def test_cuda_available(self, mock_torch):
        """Test CUDA device detection."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True

        assert get_device() == "cuda"

    @patch("docint.pipeline.stage_layout.torch")
    def test_cpu_fallback(self, mock_torch):
        """Test CPU fallback when no GPU available."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        assert get_device() == "cpu"


class TestClassMaps:
    """Tests for class ID to BlockType mappings."""

    def test_doclaynet_map_completeness(self):
        """DocLayNet map should have expected classes."""
        assert BlockType.TEXT in DOCLAYNET_CLASS_MAP.values()
        assert BlockType.TABLE in DOCLAYNET_CLASS_MAP.values()
        assert BlockType.FIGURE in DOCLAYNET_CLASS_MAP.values()

    def test_docint_map_has_handwriting(self):
        """DocInt map should include handwriting class."""
        assert BlockType.HANDWRITING in DOCINT_CLASS_MAP.values()
        assert BlockType.STAMP in DOCINT_CLASS_MAP.values()
        assert BlockType.SIGNATURE in DOCINT_CLASS_MAP.values()


class TestLayoutDetector:
    """Tests for LayoutDetector class."""

    @pytest.fixture
    def mock_page(self, tmp_path):
        """Create a mock page with render info."""
        # Create a dummy image file
        image_path = tmp_path / "page_0001.png"
        # Create minimal PNG (1x1 pixel)
        png_header = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D,  # IHDR length
            0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01,  # width
            0x00, 0x00, 0x00, 0x01,  # height
            0x08, 0x02,  # bit depth, color type
            0x00, 0x00, 0x00,  # compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE,  # CRC
            0x00, 0x00, 0x00, 0x0C,  # IDAT length
            0x49, 0x44, 0x41, 0x54,  # IDAT
            0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF, 0x00,
            0x05, 0xFE, 0x02, 0xFE,
            0xA3, 0x6B, 0xC0, 0x01,  # CRC placeholder
            0x00, 0x00, 0x00, 0x00,  # IEND length
            0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82,  # CRC
        ])
        image_path.write_bytes(png_header)

        render_info = PageRenderInfo(
            image_path=str(image_path),
            width_pixels=2550,
            height_pixels=3300,
            dpi=300,
        )

        return Page(
            document_id=uuid4(),
            page_number=1,
            render_info=render_info,
            status=ProcessingStatus.COMPLETE,
        )

    @patch("docint.pipeline.stage_layout.YOLO")
    @patch("docint.pipeline.stage_layout.get_device")
    def test_detector_initialization(self, mock_device, mock_yolo):
        """Test detector initializes with default settings."""
        mock_device.return_value = "cpu"
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = LayoutDetector()

        assert detector.confidence_threshold == 0.25
        assert detector.iou_threshold == 0.45
        mock_model.to.assert_called_once_with("cpu")

    @patch("docint.pipeline.stage_layout.YOLO")
    @patch("docint.pipeline.stage_layout.get_device")
    def test_detector_custom_thresholds(self, mock_device, mock_yolo):
        """Test detector with custom confidence thresholds."""
        mock_device.return_value = "cpu"
        mock_yolo.return_value = MagicMock()

        detector = LayoutDetector(
            confidence_threshold=0.5,
            iou_threshold=0.3,
        )

        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.3

    @patch("docint.pipeline.stage_layout.YOLO")
    @patch("docint.pipeline.stage_layout.get_device")
    @patch("docint.pipeline.stage_layout.Image")
    def test_detect_page_no_render_info(self, mock_image, mock_device, mock_yolo):
        """Test error when page has no render info."""
        mock_device.return_value = "cpu"
        mock_yolo.return_value = MagicMock()

        detector = LayoutDetector()
        page = Page(
            document_id=uuid4(),
            page_number=1,
            render_info=None,
        )

        with pytest.raises(ValueError, match="no render info"):
            detector.detect_page(page, uuid4())

    @patch("docint.pipeline.stage_layout.YOLO")
    @patch("docint.pipeline.stage_layout.get_device")
    @patch("docint.pipeline.stage_layout.Image")
    def test_detect_page_empty_results(self, mock_image, mock_device, mock_yolo, mock_page):
        """Test detection with no results."""
        mock_device.return_value = "cpu"

        # Mock YOLO model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        # Mock image open
        mock_img = MagicMock()
        mock_img.size = (2550, 3300)
        mock_image.open.return_value.__enter__.return_value = mock_img

        detector = LayoutDetector()
        blocks = detector.detect_page(mock_page, uuid4())

        assert len(blocks) == 0

    @patch("docint.pipeline.stage_layout.YOLO")
    @patch("docint.pipeline.stage_layout.get_device")
    @patch("docint.pipeline.stage_layout.Image")
    def test_detect_page_with_detections(self, mock_image, mock_device, mock_yolo, mock_page):
        """Test detection with results."""
        import numpy as np

        mock_device.return_value = "cpu"

        # Mock detection boxes
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 200, 500, 400])
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.85)
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)  # TEXT class

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        # Mock image
        mock_img = MagicMock()
        mock_img.size = (2550, 3300)
        mock_image.open.return_value.__enter__.return_value = mock_img

        detector = LayoutDetector()
        doc_id = uuid4()
        blocks = detector.detect_page(mock_page, doc_id)

        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.TEXT
        assert blocks[0].detection_confidence == 0.85
        assert blocks[0].reading_order == 0
        assert blocks[0].document_id == doc_id


class TestLayoutDetectorConfig:
    """Tests for LayoutDetectorConfig."""

    def test_models_defined(self):
        """Test that model configurations are defined."""
        assert len(LayoutDetectorConfig.MODELS) > 0
        assert "yolov8n-doclaynet" in LayoutDetectorConfig.MODELS

    def test_get_model_config(self):
        """Test getting model configuration."""
        config = LayoutDetectorConfig.get_model_config("yolov8n-doclaynet")
        assert "classes" in config
        assert "class_map" in config

    def test_get_unknown_model_config(self):
        """Test getting unknown model returns empty dict."""
        config = LayoutDetectorConfig.get_model_config("nonexistent")
        assert config == {}
