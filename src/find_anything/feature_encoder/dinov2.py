import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from find_anything.feature_encoder.protocol import FeatureEncoder


class DinoV2FeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        repo_name: str = "facebookresearch/dinov2",
        model_name: str = "dinov2_vits14",
        device: str = "cuda",
    ) -> None:
        self.device = device
        self._model = torch.hub.load(repo_name, model_name)
        self._model = self._model.float().eval().to(device)
        self._preprocess = T.Compose(
            [
                T.Resize((392, 392)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.patch_size = self._model.patch_size
        self.embed_dim = self._model.embed_dim

    @torch.no_grad()
    def encode_mean(self, image: Image.Image) -> torch.Tensor:
        x = self._preprocess(image).unsqueeze(0).to(self.device)

        features = self._model.get_intermediate_layers(x, n=1, return_class_token=False)[0]
        features = features.mean(dim=1)

        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def encode_dense(self, image: Image.Image) -> torch.Tensor:
        x = self._preprocess(image).unsqueeze(0).to(self.device)

        features = self._model.get_intermediate_layers(x, n=1, return_class_token=False)[0]

        B, N, D = features.shape
        H = W = int(N**0.5)

        return features.reshape(B, H, W, D)  # type: ignore[no-any-return]

    @torch.no_grad()
    def encode_patches(self, masks: torch.Tensor, image: Image.Image) -> dict[int, torch.Tensor]:
        image_np = np.array(image.convert("RGB"))
        embeddings = {}

        for idx, mask in enumerate(masks):
            mask_bool = mask.bool()
            ys, xs = torch.where(mask_bool)
            if ys.numel() == 0 or xs.numel() == 0:
                continue

            y1, y2 = ys.min().item(), ys.max().item()
            x1, x2 = xs.min().item(), xs.max().item()
            patch_np = image_np[y1 : y2 + 1, x1 : x2 + 1]  # type: ignore[misc, index]
            patch_img = Image.fromarray(patch_np)

            feat = self.encode_mean(patch_img).squeeze(0)
            feat = F.normalize(feat, dim=-1)

            embeddings[idx] = feat

        return embeddings
