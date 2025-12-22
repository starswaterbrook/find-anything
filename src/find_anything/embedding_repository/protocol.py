from typing import Protocol

import torch


class EmbeddingRepository(Protocol):
    def add_images(self, image_paths: list[str]) -> None: ...

    def get(self) -> torch.Tensor: ...
