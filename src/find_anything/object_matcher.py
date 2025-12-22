import torch
from PIL import Image

from find_anything.embedding_repository import EmbeddingRepository
from find_anything.feature_encoder import FeatureEncoder
from find_anything.mask import MaskGenerator, MaskPooler, MaskSelector
from find_anything.models import MatcherResult


class ZeroShotObjectMatcher:
    def __init__(  # noqa: PLR0913
        self,
        encoder: FeatureEncoder,
        mask_generator: MaskGenerator,
        mask_pooler: MaskPooler,
        mask_selector: MaskSelector,
        base_embeddings: EmbeddingRepository,
        similarity_threshold: float,
    ) -> None:
        self.encoder = encoder
        self.mask_generator = mask_generator
        self.mask_pooler = mask_pooler
        self.mask_selector = mask_selector
        self.base_embeddings = base_embeddings
        self.threshold = similarity_threshold

    def set_base_images(self, image_paths: list[str]) -> None:
        self.base_embeddings.add_images(image_paths)

    @torch.no_grad()
    def forward(self, target_image_path: str) -> list[MatcherResult]:
        target_image = Image.open(target_image_path).convert("RGB")
        dense_features = self.encoder.encode_dense(target_image)

        masks = self.mask_generator.generate(target_image)
        if len(masks) == 0:
            return []

        mask_embeddings = self.mask_pooler.pool(dense_features, masks)
        if len(mask_embeddings) == 0:
            return []

        top_mask_embeddings = self.mask_selector.select(mask_embeddings)
        if len(top_mask_embeddings) == 0:
            return []

        patch_embeddings = self.encoder.encode_patches(
            torch.stack([m.mask for m in top_mask_embeddings]), target_image
        )

        embeddings_tensor = torch.stack(list(patch_embeddings.values()))
        base_embeddings = self.base_embeddings.get()
        similarity_matrix = embeddings_tensor @ base_embeddings.T

        matches = []
        max_sim, best_idx = similarity_matrix.max(dim=1)
        for i, sim in enumerate(max_sim):
            if sim.item() >= self.threshold:
                mask_obj = top_mask_embeddings[i]
                matches.append(
                    MatcherResult(
                        mask_id=i,
                        mask=mask_obj.mask,
                        similarity=sim.item(),
                        matched_base=int(best_idx[i].item()),
                    )
                )

        return matches
