import torch
import torch.nn.functional as F

from find_anything.mask.pooler.protocol import MaskPooler
from find_anything.models import MaskEmbedding


class DenseFeatureMaskPooler(MaskPooler):
    def __init__(self, device: str = "cuda", min_mask_area: int = 5, eps: float = 1e-6) -> None:
        self._device = device
        self._min_mask_area = min_mask_area
        self._eps = eps

    @torch.no_grad()
    def pool(
        self,
        dense_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> list[MaskEmbedding]:
        _, Hf, Wf, D = dense_features.shape

        if len(masks) == 0:
            return []

        masks_t = masks.float().to(self._device)

        masks_resized = F.interpolate(masks_t.unsqueeze(1), size=(Hf, Wf), mode="nearest").squeeze(
            1
        )

        M_flat = masks_resized.view(len(masks), Hf * Wf)
        F_flat = dense_features.view(Hf * Wf, D)

        masked_sum = M_flat @ F_flat
        mask_area = M_flat.sum(dim=1)

        outputs = []

        for i in range(len(masks)):
            if mask_area[i] < self._min_mask_area:
                continue

            z = masked_sum[i] / (mask_area[i] + self._eps)
            z = F.normalize(z, dim=-1)

            outputs.append(
                MaskEmbedding(
                    mask=masks[i],
                    embedding=z,
                )
            )

        return outputs
