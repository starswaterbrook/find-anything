from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from find_anything.embedding_repository.base import BaseEmbeddingRepository
from find_anything.mask.pooler.dense_feature import DenseFeatureMaskPooler
from find_anything.mask.selector.topk import TopKMaskSelector


@pytest.fixture
def mock_feature_encoder() -> MagicMock:
    encoder = MagicMock()
    encoder.encode_mean.return_value = torch.nn.functional.normalize(torch.randn(1, 384), dim=-1)
    encoder.encode_dense.return_value = torch.randn(1, 28, 28, 384)

    def mock_encode_patches(masks: torch.Tensor, image: Image.Image) -> dict[int, torch.Tensor]:  # noqa: ARG001
        embeddings = {}
        for i in range(len(masks)):
            embeddings[i] = torch.nn.functional.normalize(torch.randn(384), dim=-1)
        return embeddings

    encoder.encode_patches.side_effect = mock_encode_patches
    return encoder


@pytest.fixture
def mock_mask_generator() -> MagicMock:
    generator = MagicMock()

    def mock_generate(image: Image.Image) -> torch.Tensor:  # noqa: ARG001
        masks = torch.zeros(3, 256, 256)
        masks[0, 50:150, 50:150] = 1.0
        masks[1, 100:200, 100:200] = 1.0
        masks[2, 30:100, 30:100] = 1.0
        return masks

    generator.generate.side_effect = mock_generate
    return generator


@pytest.fixture
def real_mask_pooler(device: str) -> DenseFeatureMaskPooler:
    return DenseFeatureMaskPooler(device=device, min_mask_area=5)


@pytest.fixture
def real_embedding_repository(mock_feature_encoder: MagicMock) -> BaseEmbeddingRepository:
    return BaseEmbeddingRepository(encoder=mock_feature_encoder)


@pytest.fixture
def real_mask_selector(
    real_embedding_repository: BaseEmbeddingRepository,
) -> TopKMaskSelector:
    return TopKMaskSelector(base_embeddings=real_embedding_repository, top_k=3)
