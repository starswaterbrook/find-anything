from typing import Literal

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from find_anything.mask.generator.protocol import MaskGenerator

SamModelType = Literal["vit_h", "vit_l", "vit_b"]


class SAMMaskGenerator(MaskGenerator):
    def __init__(  # noqa: PLR0913
        self,
        model_path: str,
        device: str = "cuda",
        model_type: SamModelType = "vit_h",
        *,
        points_per_side: int = 16,
        min_mask_region_area: int = 0,
        pred_iou_thresh: float = 0.6,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 0,
        box_nms_thresh: float = 0.7,
    ) -> None:
        self._device = device
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            min_mask_region_area=min_mask_region_area,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
        )

    @torch.no_grad()
    def generate(self, image: Image.Image) -> torch.Tensor:
        image_array = np.array(image.convert("RGB"))

        masks_data = self._mask_generator.generate(image_array)

        if len(masks_data) == 0:
            h, w = image_array.shape[:2]
            return torch.zeros((0, h, w), dtype=torch.float32, device=self._device)

        masks = [mask_data["segmentation"].astype(np.float32) for mask_data in masks_data]

        return torch.from_numpy(np.stack(masks, axis=0)).to(device=self._device)
