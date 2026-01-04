import torch
from fastsam import FastSAM, FastSAMPrompt
from PIL import Image

from find_anything.mask.generator.protocol import MaskGenerator


class FastSAMMaskGenerator(MaskGenerator):
    def __init__(self, model_path: str = "FastSAM-s.pt", device: str = "cuda") -> None:
        self._model = FastSAM(model_path)
        self._device = device
        self._model.model.float()

    @torch.no_grad()
    def generate(self, image: Image.Image) -> torch.Tensor:
        results = self._model(
            image,
            device=self._device,
            retina_masks=True,
            imgsz=1024,
            conf=0.7,
            iou=0.5,
        )
        prompt = FastSAMPrompt(image, results, device=self._device)
        return prompt.everything_prompt()  # type: ignore[no-any-return]
