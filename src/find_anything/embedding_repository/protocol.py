from typing import Protocol

import torch
from PIL import Image


class EmbeddingRepository(Protocol):
    def add_images(self, image_paths: list[str]) -> None: ...

    def add_images_binary(self, images: list[Image.Image]) -> None: ...

    def get(self) -> torch.Tensor: ...
