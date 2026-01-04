from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from find_anything.mask.generator.fastsam import FastSAMMaskGenerator


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
