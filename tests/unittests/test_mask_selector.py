from unittest.mock import Mock

import torch

from find_anything.mask.selector.topk import TopKMaskSelector
from find_anything.models import MaskEmbedding


class TestTopKMaskSelector:
    def test_select_returns_top_k_masks(self, sample_mask_embeddings: list[MaskEmbedding]) -> None:
        base_repo = Mock()
        base_repo.get.return_value = torch.nn.functional.normalize(torch.randn(2, 384), dim=-1)

        selector = TopKMaskSelector(base_embeddings=base_repo, top_k=2)

        results = selector.select(sample_mask_embeddings)

        assert len(results) == 2

    def test_select_returns_fewer_when_less_than_k_available(self) -> None:
        base_repo = Mock()
        base_repo.get.return_value = torch.nn.functional.normalize(torch.randn(2, 384), dim=-1)

        mask_embeddings = [
            MaskEmbedding(
                mask=torch.ones(64, 64),
                embedding=torch.nn.functional.normalize(torch.randn(384), dim=-1),
            )
        ]

        selector = TopKMaskSelector(base_embeddings=base_repo, top_k=5)

        results = selector.select(mask_embeddings)

        assert len(results) == 1

    def test_select_empty_list_returns_empty(self) -> None:
        base_repo = Mock()
        base_repo.get.return_value = torch.randn(2, 384)

        selector = TopKMaskSelector(base_embeddings=base_repo, top_k=5)

        results = selector.select([])

        assert results == []

    def test_select_returns_most_similar_masks(self) -> None:
        base_embedding = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
        base_repo = Mock()
        base_repo.get.return_value = base_embedding

        similar_emb = torch.nn.functional.normalize(torch.tensor([0.9, 0.1, 0.0]), dim=-1)
        dissimilar_emb = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=-1)
        medium_emb = torch.nn.functional.normalize(torch.tensor([0.5, 0.5, 0.0]), dim=-1)

        mask_embeddings = [
            MaskEmbedding(mask=torch.ones(64, 64), embedding=dissimilar_emb),
            MaskEmbedding(mask=torch.ones(64, 64), embedding=similar_emb),
            MaskEmbedding(mask=torch.ones(64, 64), embedding=medium_emb),
        ]

        selector = TopKMaskSelector(base_embeddings=base_repo, top_k=1)

        results = selector.select(mask_embeddings)

        assert len(results) == 1
        assert torch.allclose(results[0].embedding, similar_emb)

    def test_select_considers_all_base_embeddings(self) -> None:
        base_embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dim=-1
        )
        base_repo = Mock()
        base_repo.get.return_value = base_embeddings

        matches_first = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=-1)
        matches_second = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0]), dim=-1)
        matches_neither = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=-1)

        mask_embeddings = [
            MaskEmbedding(mask=torch.ones(64, 64), embedding=matches_neither),
            MaskEmbedding(mask=torch.ones(64, 64), embedding=matches_first),
            MaskEmbedding(mask=torch.ones(64, 64), embedding=matches_second),
        ]

        selector = TopKMaskSelector(base_embeddings=base_repo, top_k=2)

        results = selector.select(mask_embeddings)

        result_embeddings = {tuple(r.embedding.tolist()) for r in results}
        assert tuple(matches_first.tolist()) in result_embeddings
        assert tuple(matches_second.tolist()) in result_embeddings
