from typing import Protocol

import torch
from PIL import Image


class MaskGenerator(Protocol):
    def generate(self, image: Image.Image) -> torch.Tensor: ...
