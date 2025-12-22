import torch

from find_anything.embedding_repository.base import BaseEmbeddingRepository
from find_anything.mask.selector.protocol import MaskSelector
from find_anything.models import MaskEmbedding


class TopKMaskSelector(MaskSelector):
    def __init__(self, base_embeddings: BaseEmbeddingRepository, top_k: int = 5) -> None:
        self._top_k = top_k
        self._base_embeddings = base_embeddings

    @torch.no_grad()
    def select(self, mask_embeddings: list[MaskEmbedding]) -> list[MaskEmbedding]:
        if not mask_embeddings:
            return []

        Z = torch.stack([m.embedding for m in mask_embeddings])
        sim = Z @ self._base_embeddings.get().T
        max_sim, _ = sim.max(dim=1)

        k = min(self._top_k, max_sim.numel())
        topk_idx = torch.topk(max_sim, k=k).indices.tolist()

        return [mask_embeddings[i] for i in topk_idx]
