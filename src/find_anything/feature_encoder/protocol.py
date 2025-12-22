from typing import Protocol

import torch
from PIL import Image


class FeatureEncoder(Protocol):
    @torch.no_grad()
    def encode_mean(self, image: Image.Image) -> torch.Tensor: ...

    @torch.no_grad()
    def encode_dense(self, image: Image.Image) -> torch.Tensor: ...

    @torch.no_grad()
    def encode_patches(
        self, masks: torch.Tensor, image: Image.Image
    ) -> dict[int, torch.Tensor]: ...
