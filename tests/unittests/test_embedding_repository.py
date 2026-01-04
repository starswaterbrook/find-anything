from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from find_anything.embedding_repository.base import BaseEmbeddingRepository


@pytest.fixture
def mock_encoder() -> Mock:
    return Mock()


class TestBaseEmbeddingRepository:
    def test_get_returns_empty_tensor_initially(self, mock_encoder: Mock) -> None:
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        result = repo.get()

        assert result.numel() == 0

    def test_add_images_stores_embeddings(
        self, sample_image: Image.Image, mock_encoder: Mock
    ) -> None:
        mock_encoder.encode_mean.return_value = torch.randn(1, 384)
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        repo.add_images([sample_image])

        assert repo.get().shape == (1, 384)

    def test_add_multiple_images(self, sample_image: Image.Image, mock_encoder: Mock) -> None:
        mock_encoder.encode_mean.return_value = torch.randn(1, 384)
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        repo.add_images([sample_image, sample_image, sample_image])

        assert repo.get().shape == (3, 384)
        assert mock_encoder.encode_mean.call_count == 3

    def test_add_images_converts_rgba_to_rgb(
        self, sample_image_rgba: Image.Image, mock_encoder: Mock
    ) -> None:
        mock_encoder.encode_mean.return_value = torch.randn(1, 384)
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        repo.add_images([sample_image_rgba])

        call_args = mock_encoder.encode_mean.call_args[0][0]
        assert call_args.mode == "RGB"

    def test_add_images_preserves_rgb_mode(
        self, sample_image: Image.Image, mock_encoder: Mock
    ) -> None:
        mock_encoder.encode_mean.return_value = torch.randn(1, 384)
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        repo.add_images([sample_image])

        call_args = mock_encoder.encode_mean.call_args[0][0]
        assert call_args.mode == "RGB"

    def test_add_images_from_paths(self, tmp_path: Path, mock_encoder: Mock) -> None:
        mock_encoder.encode_mean.return_value = torch.randn(1, 384)
        repo = BaseEmbeddingRepository(encoder=mock_encoder)

        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)

        repo.add_images_from_paths([str(img_path)])

        assert repo.get().shape == (1, 384)
        assert len(repo._image_paths) == 1
        assert repo._image_paths[0] == str(img_path)
