from typing import Protocol

import torch

from find_anything.models import MaskEmbedding


class MaskPooler(Protocol):
    @torch.no_grad()
    def pool(
        self,
        dense_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> list[MaskEmbedding]: ...
