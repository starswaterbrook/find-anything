from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from find_anything.feature_encoder.dinov2 import DinoV2FeatureEncoder


def create_mock_dinov2_model() -> MagicMock:
    model = MagicMock()
    model.patch_size = 14
    model.embed_dim = 384
    model.float.return_value = model
    model.eval.return_value = model
    model.to.return_value = model
    return model


class TestDinoV2FeatureEncoder:
    @patch("torch.hub.load")
    def test_encoder_initialization(self, mock_hub_load: MagicMock) -> None:
        mock_model = create_mock_dinov2_model()
        mock_hub_load.return_value = mock_model

        encoder = DinoV2FeatureEncoder(device="cpu")

        mock_hub_load.assert_called_once_with("facebookresearch/dinov2", "dinov2_vits14")
        assert encoder.patch_size == 14
        assert encoder.embed_dim == 384

    @patch("torch.hub.load")
    def test_encode_mean_returns_normalized_tensor(
        self, mock_hub_load: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = create_mock_dinov2_model()
        features = torch.randn(1, 784, 384)
        mock_model.get_intermediate_layers.return_value = [features]
        mock_hub_load.return_value = mock_model

        encoder = DinoV2FeatureEncoder(device="cpu")
        result = encoder.encode_mean(sample_image)

        assert result.shape == (1, 384)
        norm = torch.norm(result).item()
        assert abs(norm - 1.0) < 1e-5

    @patch("torch.hub.load")
    def test_encode_dense_returns_spatial_features(
        self, mock_hub_load: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = create_mock_dinov2_model()
        features = torch.randn(1, 784, 384)
        mock_model.get_intermediate_layers.return_value = [features]
        mock_hub_load.return_value = mock_model

        encoder = DinoV2FeatureEncoder(device="cpu")
        result = encoder.encode_dense(sample_image)

        assert result.shape == (1, 28, 28, 384)

    @patch("torch.hub.load")
    def test_encode_patches_returns_dict_of_embeddings(
        self, mock_hub_load: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = create_mock_dinov2_model()
        features = torch.randn(1, 784, 384)
        mock_model.get_intermediate_layers.return_value = [features]
        mock_hub_load.return_value = mock_model

        encoder = DinoV2FeatureEncoder(device="cpu")

        masks = torch.zeros(2, 256, 256)
        masks[0, 50:150, 50:150] = 1.0
        masks[1, 100:200, 100:200] = 1.0

        result = encoder.encode_patches(masks, sample_image)

        assert isinstance(result, dict)
        assert len(result) == 2
        for idx, embedding in result.items():
            assert isinstance(idx, int)
            assert embedding.shape == (384,)
            norm = torch.norm(embedding).item()
            assert abs(norm - 1.0) < 1e-5

    @patch("torch.hub.load")
    def test_encode_patches_skips_empty_masks(
        self, mock_hub_load: MagicMock, sample_image: Image.Image
    ) -> None:
        mock_model = create_mock_dinov2_model()
        features = torch.randn(1, 784, 384)
        mock_model.get_intermediate_layers.return_value = [features]
        mock_hub_load.return_value = mock_model

        encoder = DinoV2FeatureEncoder(device="cpu")

        masks = torch.zeros(3, 256, 256)
        masks[0, 50:150, 50:150] = 1.0
        masks[2, 100:200, 100:200] = 1.0

        result = encoder.encode_patches(masks, sample_image)

        assert 0 in result
        assert 1 not in result
        assert 2 in result
