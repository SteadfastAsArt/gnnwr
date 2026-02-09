import torch
import torch.nn as nn

from gnnwr.networks import DistanceProjection, SWNN, default_dense_layer


class TestDistanceProjection:
    """Tests for the DistanceProjection module."""

    def test_output_shape(self):
        proj = DistanceProjection(insize=500, embed_dim=128)
        x = torch.randn(32, 500)
        out = proj(x)
        assert out.shape == (32, 128)

    def test_output_shape_various_dims(self):
        for insize, embed_dim in [(50, 32), (200, 64), (1000, 128)]:
            proj = DistanceProjection(insize=insize, embed_dim=embed_dim)
            x = torch.randn(16, insize)
            out = proj(x)
            assert out.shape == (16, embed_dim), \
                f"Expected (16, {embed_dim}), got {out.shape}"

    def test_kaiming_init(self):
        proj = DistanceProjection(insize=200, embed_dim=64)
        # Bias should be zero
        assert torch.all(proj.linear.bias == 0)

    def test_dropout_zero(self):
        proj = DistanceProjection(insize=100, embed_dim=64, drop_out=0)
        assert isinstance(proj.dropout, nn.Identity)

    def test_gradient_flows(self):
        proj = DistanceProjection(insize=200, embed_dim=128)
        x = torch.randn(8, 200, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestDefaultDenseLayer:
    """Tests for default_dense_layer backward compatibility."""

    def test_backward_compat_small(self):
        """default_dense_layer(64, 8) should match upstream behavior."""
        layers = default_dense_layer(64, 8)
        assert layers == [64, 32, 16]

    def test_backward_compat_large(self):
        """default_dense_layer(5000, 7) should match upstream behavior.

        With DistanceProjection, this function is only called with
        effective_insize=embed_dim (e.g. 128), not with raw knn_k.
        But the function itself must still work correctly for any input.
        """
        layers = default_dense_layer(5000, 7)
        # upstream: 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8
        assert layers[0] == 4096
        assert layers[-1] == 8
        assert all(layers[i] > layers[i + 1] for i in range(len(layers) - 1))

    def test_backward_compat_128(self):
        """default_dense_layer(128, 7) — typical embed_dim use case."""
        layers = default_dense_layer(128, 7)
        assert layers == [128, 64, 32, 16, 8]

    def test_basic_properties(self):
        layers = default_dense_layer(256, 10)
        assert len(layers) > 0
        assert layers[-1] > 10
        assert all(isinstance(s, int) for s in layers)


class TestSWNNEmbedDim:
    """Tests for SWNN with embed_dim parameter."""

    def test_no_embed_dim_backward_compat(self):
        """Without embed_dim, SWNN should behave exactly as before."""
        swnn = SWNN(insize=200, outsize=7)
        assert swnn.projection is None
        x = torch.randn(8, 200)
        out = swnn(x)
        assert out.shape == (8, 7)

    def test_embed_dim_creates_projection(self):
        """When insize > embed_dim, projection layer should be created."""
        swnn = SWNN(insize=500, outsize=7, embed_dim=128)
        assert swnn.projection is not None
        assert isinstance(swnn.projection, DistanceProjection)

    def test_embed_dim_no_projection_when_small(self):
        """When insize <= embed_dim, no projection should be created."""
        swnn = SWNN(insize=64, outsize=7, embed_dim=128)
        assert swnn.projection is None

    def test_different_knn_k_same_hidden_structure(self):
        """Different knn_k values with same embed_dim should produce same hidden layer structure."""
        SWNN(insize=50, outsize=7, embed_dim=128)  # no projection (50 < 128)
        swnn_200 = SWNN(insize=200, outsize=7, embed_dim=128)
        swnn_1000 = SWNN(insize=1000, outsize=7, embed_dim=128)

        # Both project to 128, so hidden layers should be identical
        assert swnn_200.dense_layer == swnn_1000.dense_layer
        assert swnn_200.projection is not None
        assert swnn_1000.projection is not None

    def test_forward_with_embed_dim(self):
        """Forward pass should work correctly with projection."""
        swnn = SWNN(insize=500, outsize=7, embed_dim=128)
        x = torch.randn(16, 500)
        out = swnn(x)
        assert out.shape == (16, 7)

    def test_forward_various_batch_sizes(self):
        swnn = SWNN(insize=200, outsize=7, embed_dim=64)
        for bs in [8, 32, 128]:
            x = torch.randn(bs, 200)
            out = swnn(x)
            assert out.shape == (bs, 7)
        # batch_size=1 requires eval mode (BatchNorm1d limitation)
        swnn.eval()
        x = torch.randn(1, 200)
        out = swnn(x)
        assert out.shape == (1, 7)

    def test_gradient_flows_through_projection(self):
        swnn = SWNN(insize=300, outsize=5, embed_dim=64)
        x = torch.randn(8, 300, requires_grad=True)
        out = swnn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_custom_dense_layer_with_embed_dim(self):
        """Custom dense_layer should override auto-generated layers."""
        swnn = SWNN(dense_layer=[64, 32], insize=500, outsize=7, embed_dim=128)
        assert swnn.dense_layer == [64, 32]
        assert swnn.projection is not None
        x = torch.randn(8, 500)
        out = swnn(x)
        assert out.shape == (8, 7)


