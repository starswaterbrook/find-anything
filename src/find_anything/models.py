from dataclasses import dataclass

import torch


@dataclass
class MatcherResult:
    mask_id: int
    mask: torch.Tensor
    similarity: float
    matched_base: int


@dataclass
class MaskEmbedding:
    mask: torch.Tensor
    embedding: torch.Tensor
