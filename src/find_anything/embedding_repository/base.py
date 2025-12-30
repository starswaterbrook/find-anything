import torch
from PIL import Image

from find_anything.embedding_repository.protocol import EmbeddingRepository
from find_anything.feature_encoder.dinov2 import FeatureEncoder


class BaseEmbeddingRepository(EmbeddingRepository):
    def __init__(self, encoder: FeatureEncoder) -> None:
        self.encoder = encoder
        self._embeddings = torch.empty(0)
        self._image_paths: list[str] = []

    def add_images_from_paths(self, image_paths: list[str]) -> None:
        embeddings = []
        for path in image_paths:
            self._image_paths.append(path)
            image = Image.open(path).convert("RGB")
            emb = self.encoder.encode_mean(image)
            embeddings.append(emb)
        self._embeddings = torch.cat(embeddings, dim=0)

    def add_images(self, images: list[Image.Image]) -> None:
        embeddings = []
        for image in images:
            image_obj = image if image.mode == "RGB" else image.convert("RGB")
            emb = self.encoder.encode_mean(image_obj)
            embeddings.append(emb)
        self._embeddings = torch.cat(embeddings, dim=0)

    def get(self) -> torch.Tensor:
        return self._embeddings
