from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from find_anything.mask.generator.fastsam import FastSAMMaskGenerator
from find_anything.mask.generator.sam import SAMMaskGenerator


class TestFastSAMMaskGenerator:
    @patch("find_anything.mask.generator.fastsam.FastSAM")
    def test_initialization(self, mock_fastsam_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_fastsam_cls.return_value = mock_model

        generator = FastSAMMaskGenerator(model_path="test.pt", device="cpu")

        mock_fastsam_cls.assert_called_once_with("test.pt")
        mock_model.model.float.assert_called_once()
        assert generator._device == "cpu"

    @patch("find_anything.mask.generator.fastsam.FastSAMPrompt")
    @patch("find_anything.mask.generator.fastsam.FastSAM")
    def test_generate_returns_tensor(
        self, mock_fastsam_cls: MagicMock, mock_prompt_cls: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_fastsam_cls.return_value = mock_model

        mock_results = MagicMock()
        mock_model.return_value = mock_results

        expected_masks = torch.ones(5, 256, 256)
        mock_prompt = MagicMock()
        mock_prompt.everything_prompt.return_value = expected_masks
        mock_prompt_cls.return_value = mock_prompt

        generator = FastSAMMaskGenerator(model_path="test.pt", device="cpu")
        result = generator.generate(sample_image)

        assert torch.equal(result, expected_masks)

    @patch("find_anything.mask.generator.fastsam.FastSAMPrompt")
    @patch("find_anything.mask.generator.fastsam.FastSAM")
    def test_generate_calls_model_with_correct_params(
        self, mock_fastsam_cls: MagicMock, mock_prompt_cls: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_fastsam_cls.return_value = mock_model

        mock_results = MagicMock()
        mock_model.return_value = mock_results

        mock_prompt = MagicMock()
        mock_prompt.everything_prompt.return_value = torch.ones(1, 64, 64)
        mock_prompt_cls.return_value = mock_prompt

        generator = FastSAMMaskGenerator(model_path="test.pt", device="cpu")
        generator.generate(sample_image)

        mock_model.assert_called_once_with(
            sample_image,
            device="cpu",
            retina_masks=True,
            imgsz=1024,
            conf=0.7,
            iou=0.5,
        )

    @patch("find_anything.mask.generator.fastsam.FastSAMPrompt")
    @patch("find_anything.mask.generator.fastsam.FastSAM")
    def test_generate_creates_prompt_with_results(
        self, mock_fastsam_cls: MagicMock, mock_prompt_cls: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_fastsam_cls.return_value = mock_model

        mock_results = MagicMock()
        mock_model.return_value = mock_results

        mock_prompt = MagicMock()
        mock_prompt.everything_prompt.return_value = torch.ones(1, 64, 64)
        mock_prompt_cls.return_value = mock_prompt

        generator = FastSAMMaskGenerator(model_path="test.pt", device="cpu")
        generator.generate(sample_image)

        mock_prompt_cls.assert_called_once_with(sample_image, mock_results, device="cpu")
        mock_prompt.everything_prompt.assert_called_once()


class TestSAMMaskGenerator:
    @patch("find_anything.mask.generator.sam.sam_model_registry")
    def test_initialization(self, mock_registry: MagicMock) -> None:
        mock_sam = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_sam))

        generator = SAMMaskGenerator(model_path="test.pth", device="cpu", model_type="vit_h")

        mock_registry.__getitem__.assert_called_once_with("vit_h")
        mock_sam.to.assert_called_once_with(device="cpu")
        assert generator._device == "cpu"

    @patch("find_anything.mask.generator.sam.SamAutomaticMaskGenerator")
    @patch("find_anything.mask.generator.sam.sam_model_registry")
    def test_initialization_with_custom_params(
        self, mock_registry: MagicMock, mock_mask_generator_cls: MagicMock
    ) -> None:
        mock_sam = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_sam))

        SAMMaskGenerator(
            model_path="test.pth",
            device="cpu",
            model_type="vit_l",
            points_per_side=32,
            min_mask_region_area=100,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            box_nms_thresh=0.5,
        )

        mock_mask_generator_cls.assert_called_once_with(
            mock_sam,
            points_per_side=32,
            min_mask_region_area=100,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            box_nms_thresh=0.5,
        )

    @patch("find_anything.mask.generator.sam.SamAutomaticMaskGenerator")
    @patch("find_anything.mask.generator.sam.sam_model_registry")
    def test_generate_returns_tensor(
        self,
        mock_registry: MagicMock,
        mock_mask_generator_cls: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        mock_sam = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_sam))

        mock_mask_generator = MagicMock()
        mock_mask_generator.generate.return_value = [
            {"segmentation": np.ones((64, 64), dtype=bool)},
            {"segmentation": np.zeros((64, 64), dtype=bool)},
        ]
        mock_mask_generator_cls.return_value = mock_mask_generator

        generator = SAMMaskGenerator(model_path="test.pth", device="cpu")
        result = generator.generate(sample_image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 64, 64)
        assert result.dtype == torch.float32

    @patch("find_anything.mask.generator.sam.SamAutomaticMaskGenerator")
    @patch("find_anything.mask.generator.sam.sam_model_registry")
    def test_generate_empty_masks(
        self,
        mock_registry: MagicMock,
        mock_mask_generator_cls: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        mock_sam = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_sam))

        mock_mask_generator = MagicMock()
        mock_mask_generator.generate.return_value = []
        mock_mask_generator_cls.return_value = mock_mask_generator

        generator = SAMMaskGenerator(model_path="test.pth", device="cpu")
        result = generator.generate(sample_image)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0
        assert result.dtype == torch.float32

    @patch("find_anything.mask.generator.sam.SamAutomaticMaskGenerator")
    @patch("find_anything.mask.generator.sam.sam_model_registry")
    def test_generate_calls_mask_generator_with_rgb_array(
        self,
        mock_registry: MagicMock,
        mock_mask_generator_cls: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        mock_sam = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=MagicMock(return_value=mock_sam))

        mock_mask_generator = MagicMock()
        mock_mask_generator.generate.return_value = [
            {"segmentation": np.ones((64, 64), dtype=bool)},
        ]
        mock_mask_generator_cls.return_value = mock_mask_generator

        generator = SAMMaskGenerator(model_path="test.pth", device="cpu")
        generator.generate(sample_image)

        mock_mask_generator.generate.assert_called_once()
        call_arg = mock_mask_generator.generate.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)
        assert call_arg.ndim == 3
        assert call_arg.shape[2] == 3
