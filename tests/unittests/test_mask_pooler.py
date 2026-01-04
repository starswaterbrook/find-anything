import torch

from find_anything.mask.pooler.dense_feature import DenseFeatureMaskPooler


class TestDenseFeatureMaskPooler:
    def test_pool_returns_mask_embeddings(
        self, device: str, sample_dense_features: torch.Tensor, sample_masks: torch.Tensor
    ) -> None:
        pooler = DenseFeatureMaskPooler(device=device, min_mask_area=1)

        results = pooler.pool(sample_dense_features, sample_masks)

        assert len(results) > 0
        for result in results:
            assert result.mask is not None
            assert result.embedding is not None

    def test_pool_returns_normalized_embeddings(
        self, device: str, sample_dense_features: torch.Tensor, sample_masks: torch.Tensor
    ) -> None:
        pooler = DenseFeatureMaskPooler(device=device, min_mask_area=1)

        results = pooler.pool(sample_dense_features, sample_masks)

        for result in results:
            norm = torch.norm(result.embedding).item()
            assert abs(norm - 1.0) < 1e-5

    def test_pool_filters_small_masks(self, device: str) -> None:
        dense_features = torch.randn(1, 8, 8, 384)
        masks = torch.zeros(2, 64, 64)
        masks[0, 0:2, 0:2] = 1.0
        masks[1, 0:40, 0:40] = 1.0

        pooler = DenseFeatureMaskPooler(device=device, min_mask_area=10)

        results = pooler.pool(dense_features, masks)

        assert len(results) == 1

    def test_pool_empty_masks_returns_empty_list(
        self, device: str, sample_dense_features: torch.Tensor
    ) -> None:
        pooler = DenseFeatureMaskPooler(device=device)
        empty_masks = torch.zeros(0, 64, 64)

        results = pooler.pool(sample_dense_features, empty_masks)

        assert results == []

    def test_pool_embedding_dimension_matches_features(self, device: str) -> None:
        embed_dim = 256
        dense_features = torch.randn(1, 8, 8, embed_dim)
        masks = torch.zeros(1, 64, 64)
        masks[0, 10:50, 10:50] = 1.0

        pooler = DenseFeatureMaskPooler(device=device, min_mask_area=1)

        results = pooler.pool(dense_features, masks)

        assert results[0].embedding.shape == (embed_dim,)

    def test_pool_preserves_original_mask(
        self, device: str, sample_dense_features: torch.Tensor
    ) -> None:
        original_mask = torch.zeros(1, 64, 64)
        original_mask[0, 10:30, 10:30] = 1.0

        pooler = DenseFeatureMaskPooler(device=device, min_mask_area=1)

        results = pooler.pool(sample_dense_features, original_mask)

        assert torch.equal(results[0].mask, original_mask[0])
