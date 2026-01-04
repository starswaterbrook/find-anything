import pytest
import torch
from PIL import Image

from find_anything.models import MaskEmbedding


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


@pytest.fixture
def sample_image_rgba() -> Image.Image:
    return Image.new("RGBA", (256, 256), color=(128, 128, 128, 255))


@pytest.fixture
def sample_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (256, 256), color=(255, 0, 0)),
        Image.new("RGB", (256, 256), color=(0, 255, 0)),
        Image.new("RGB", (256, 256), color=(0, 0, 255)),
    ]


@pytest.fixture
def sample_masks() -> torch.Tensor:
    masks = torch.zeros(3, 64, 64)
    masks[0, 10:30, 10:30] = 1.0
    masks[1, 20:50, 20:50] = 1.0
    masks[2, 5:15, 5:15] = 1.0
    return masks


@pytest.fixture
def sample_dense_features() -> torch.Tensor:
    return torch.randn(1, 28, 28, 384)


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    embeddings = torch.randn(3, 384)
    return torch.nn.functional.normalize(embeddings, dim=-1)


@pytest.fixture
def sample_mask_embeddings(sample_masks: torch.Tensor) -> list[MaskEmbedding]:
    embeddings = []
    for i in range(3):
        emb = torch.nn.functional.normalize(torch.randn(384), dim=-1)
        embeddings.append(MaskEmbedding(mask=sample_masks[i], embedding=emb))
    return embeddings
