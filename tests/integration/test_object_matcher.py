from pathlib import Path
from unittest.mock import MagicMock

import torch
from PIL import Image

from find_anything.embedding_repository.base import BaseEmbeddingRepository
from find_anything.mask.pooler.dense_feature import DenseFeatureMaskPooler
from find_anything.mask.selector.topk import TopKMaskSelector
from find_anything.object_matcher import ZeroShotObjectMatcher


class TestZeroShotObjectMatcherIntegration:
    def test_matcher_basic_integration(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.5,
        )

        matcher.set_base_images(sample_images)
        results = matcher.forward_from_image(sample_image)

        assert isinstance(results, list)
        for result in results:
            assert result.mask_id >= 0
            assert 0 <= result.similarity <= 1
            assert result.matched_base >= 0

    def test_matcher_set_base_images_from_paths(
        self,
        tmp_path: Path,
        sample_image: Image.Image,
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.5,
        )

        base_image_paths = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            path = tmp_path / f"base_{i}.png"
            Image.new("RGB", (256, 256), color=color).save(path)
            base_image_paths.append(str(path))

        matcher.set_base_images_from_paths(base_image_paths)
        results = matcher.forward_from_image(sample_image)

        assert isinstance(results, list)
        assert len(real_embedding_repository.get()) == 3

    def test_matcher_threshold_filtering(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        high_threshold_matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.95,
        )

        high_threshold_matcher.set_base_images(sample_images)
        high_threshold_results = high_threshold_matcher.forward_from_image(sample_image)

        low_threshold_matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.1,
        )

        low_threshold_matcher.set_base_images(sample_images)
        low_threshold_results = low_threshold_matcher.forward_from_image(sample_image)

        assert len(high_threshold_results) <= len(low_threshold_results)

    def test_matcher_no_masks_generated(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        empty_mask_generator = MagicMock()
        empty_mask_generator.generate.return_value = torch.zeros(0, 256, 256)

        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=empty_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.5,
        )

        matcher.set_base_images(sample_images)
        results = matcher.forward_from_image(sample_image)

        assert results == []

    def test_matcher_small_masks_filtered_by_pooler(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        device: str,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        strict_pooler = DenseFeatureMaskPooler(device=device, min_mask_area=10000)

        small_mask_generator = MagicMock()
        small_masks = torch.zeros(2, 256, 256)
        small_masks[0, 0:5, 0:5] = 1.0
        small_masks[1, 0:10, 0:10] = 1.0
        small_mask_generator.generate.return_value = small_masks

        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=small_mask_generator,
            mask_pooler=strict_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.5,
        )

        matcher.set_base_images(sample_images)
        results = matcher.forward_from_image(sample_image)

        assert results == []

    def test_matcher_component_interaction_pipeline(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.5,
        )

        matcher.set_base_images(sample_images)

        assert mock_feature_encoder.encode_mean.call_count == 3
        assert real_embedding_repository.get().shape == (3, 384)

        results = matcher.forward_from_image(sample_image)

        assert mock_mask_generator.generate.called
        assert mock_feature_encoder.encode_dense.called
        assert mock_feature_encoder.encode_patches.called

        for result in results:
            assert isinstance(result.mask_id, int)
            assert isinstance(result.similarity, float)
            assert isinstance(result.matched_base, int)

    def test_matcher_with_different_top_k_values(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
    ) -> None:
        matcher_k1 = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=TopKMaskSelector(real_embedding_repository, top_k=1),
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.0,
        )

        matcher_k3 = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=TopKMaskSelector(real_embedding_repository, top_k=3),
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.0,
        )

        matcher_k1.set_base_images(sample_images)
        matcher_k3.set_base_images(sample_images)

        results_k1 = matcher_k1.forward_from_image(sample_image)
        results_k3 = matcher_k3.forward_from_image(sample_image)

        assert len(results_k1) <= len(results_k3)

    def test_matcher_result_properties(
        self,
        sample_image: Image.Image,
        sample_images: list[Image.Image],
        mock_feature_encoder: MagicMock,
        mock_mask_generator: MagicMock,
        real_mask_pooler: DenseFeatureMaskPooler,
        real_embedding_repository: BaseEmbeddingRepository,
        real_mask_selector: TopKMaskSelector,
    ) -> None:
        matcher = ZeroShotObjectMatcher(
            encoder=mock_feature_encoder,
            mask_generator=mock_mask_generator,
            mask_pooler=real_mask_pooler,
            mask_selector=real_mask_selector,
            base_embeddings=real_embedding_repository,
            similarity_threshold=0.0,
        )

        matcher.set_base_images(sample_images)
        results = matcher.forward_from_image(sample_image)

        for result in results:
            assert isinstance(result.mask_id, int)
            assert result.mask_id >= 0
            assert isinstance(result.mask, torch.Tensor)
            assert result.mask.ndim == 2
            assert 0 <= result.similarity <= 1.0
            assert 0 <= result.matched_base < len(sample_images)
