from dataclasses import dataclass

import torch


@dataclass
class MaskEmbedding:
    mask: torch.Tensor
    embedding: torch.Tensor
