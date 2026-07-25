"""Unit tests for the layer registry in ``src/app/models.py``.

Imports the *real* production ``LAYER_REGISTRY`` / ``build_layer`` /
``init_module_weights`` / ``SubNetwork`` (no test-side copies) and verifies:

    * every registered layer type constructs and materializes on a correctly
      shaped input, producing the expected output shape;
    * shape validation raises precise errors on rank mismatches;
    * missing required params raise ``ValueError``;
    * unknown layer types raise ``ValueError`` listing the available types;
    * kernel initialization reaches EVERY weight tensor of multi-tensor
      modules (LSTM / GRU / MultiheadAttention / TransformerEncoderLayer) and
      zeroes their biases (legacy convention);
    * rank-1 weights (LayerNorm/BatchNorm scales) are left at their defaults.

All tests are CPU-only and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch as T
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.models import (  # noqa: E402
    LAYER_REGISTRY,
    LazyRecurrent,
    PositionalEncoding,
    SelfAttention,
    SubNetwork,
    TransformerEncoderBlock,
    build_layer,
    init_module_weights,
)

T.manual_seed(0)
np.random.seed(0)


# -----------------------------------------------------------------------------
# Construction + forward-shape table.
# Each case: (layer type, params, input factory, expected output shape or None)
# ``None`` expected shape = just assert the forward succeeds and is finite.
# -----------------------------------------------------------------------------
def _f(*shape):
    return lambda: T.randn(*shape)


LAYER_CASES = [
    # type, params, input factory, expected output shape (None = just run)
    ("dense", {"units": 8}, _f(4, 5), (4, 8)),
    ("linear", {"in_features": 5, "out_features": 8}, _f(4, 5), (4, 8)),
    ("conv1d", {"out_channels": 4, "kernel_size": 3}, _f(2, 3, 16), (2, 4, 14)),
    ("conv2d", {"out_channels": 4, "kernel_size": 3}, _f(2, 3, 16, 16), (2, 4, 14, 14)),
    ("conv2d", {"out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1}, _f(2, 3, 16, 16), (2, 4, 8, 8)),
    ("conv3d", {"out_channels": 4, "kernel_size": 3}, _f(2, 3, 8, 8, 8), (2, 4, 6, 6, 6)),
    ("convtranspose2d", {"out_channels": 4, "kernel_size": 3}, _f(2, 3, 8, 8), (2, 4, 10, 10)),
    ("pool", {"kernel_size": 2}, _f(2, 3, 8, 8), (2, 3, 4, 4)),
    ("maxpool1d", {"kernel_size": 2}, _f(2, 3, 8), (2, 3, 4)),
    ("maxpool2d", {"kernel_size": 2}, _f(2, 3, 8, 8), (2, 3, 4, 4)),
    ("avgpool1d", {"kernel_size": 2}, _f(2, 3, 8), (2, 3, 4)),
    ("avgpool2d", {"kernel_size": 2}, _f(2, 3, 8, 8), (2, 3, 4, 4)),
    ("adaptiveavgpool2d", {"output_size": 1}, _f(2, 3, 8, 8), (2, 3, 1, 1)),
    ("adaptivemaxpool2d", {"output_size": 2}, _f(2, 3, 8, 8), (2, 3, 2, 2)),
    ("batchnorm1d", {}, _f(4, 5), (4, 5)),
    ("batchnorm2d", {}, _f(2, 3, 8, 8), (2, 3, 8, 8)),
    ("batchnorm3d", {}, _f(2, 3, 4, 4, 4), (2, 3, 4, 4, 4)),
    ("layernorm", {"normalized_shape": 5}, _f(4, 5), (4, 5)),
    ("groupnorm", {"num_groups": 2, "num_channels": 4}, _f(2, 4, 8, 8), (2, 4, 8, 8)),
    ("relu", {}, _f(4, 5), (4, 5)),
    ("leakyrelu", {"negative_slope": 0.2}, _f(4, 5), (4, 5)),
    ("tanh", {}, _f(4, 5), (4, 5)),
    ("sigmoid", {}, _f(4, 5), (4, 5)),
    ("gelu", {}, _f(4, 5), (4, 5)),
    ("silu", {}, _f(4, 5), (4, 5)),
    ("elu", {}, _f(4, 5), (4, 5)),
    ("mish", {}, _f(4, 5), (4, 5)),
    ("softmax", {"dim": -1}, _f(4, 5), (4, 5)),
    ("softplus", {}, _f(4, 5), (4, 5)),
    ("dropout", {"p": 0.5}, _f(4, 5), (4, 5)),
    ("dropout2d", {"p": 0.5}, _f(2, 3, 8, 8), (2, 3, 8, 8)),
    ("flatten", {}, _f(2, 3, 4), (2, 12)),
    ("unflatten", {"dim": -1, "sizes": [2, 3]}, _f(4, 6), (4, 2, 3)),
    ("embedding", {"num_embeddings": 10, "embedding_dim": 6},
     lambda: T.randint(0, 10, (4, 3)), (4, 3, 6)),
    ("positional_encoding", {"d_model": 6}, _f(4, 3, 6), (4, 3, 6)),
    ("positional_encoding", {"d_model": 6, "learned": True}, _f(4, 3, 6), (4, 3, 6)),
    ("mha", {"embed_dim": 8, "num_heads": 2}, _f(2, 5, 8), (2, 5, 8)),
    ("transformer_encoder", {"d_model": 8, "nhead": 2}, _f(2, 5, 8), (2, 5, 8)),
    ("transformer_encoder", {"d_model": 8, "nhead": 2, "num_layers": 2, "causal": True},
     _f(2, 5, 8), (2, 5, 8)),
]

if hasattr(nn, "RMSNorm"):
    LAYER_CASES.append(("rmsnorm", {"normalized_shape": 5}, _f(4, 5), (4, 5)))


class TestLayerRegistryConstruction:
    @pytest.mark.parametrize(
        "layer_type,params,input_factory,expected_shape",
        LAYER_CASES,
        ids=[f"{i}-{c[0]}" for i, c in enumerate(LAYER_CASES)],
    )
    def test_layer_builds_and_forwards(self, layer_type, params, input_factory, expected_shape):
        layer = build_layer(layer_type, params)
        assert isinstance(layer, nn.Module)
        layer.train(False)  # deterministic (dropout off)
        x = input_factory()
        out = layer(x)
        if isinstance(out, tuple):  # recurrent layers return (out, hidden)
            out = out[0]
        assert T.isfinite(out.float()).all()
        if expected_shape is not None:
            assert tuple(out.shape) == expected_shape, (
                f"{layer_type}: expected {expected_shape}, got {tuple(out.shape)}"
            )

    @pytest.mark.parametrize("mode", ["lstm", "gru"])
    def test_recurrent_layers(self, mode):
        layer = build_layer(mode, {"hidden_size": 6})
        x = T.randn(2, 5, 4)
        out, hidden = layer(x)
        assert out.shape == (2, 5, 6)
        if mode == "lstm":
            assert isinstance(hidden, tuple) and hidden[0].shape == (1, 2, 6)
        else:
            assert hidden.shape == (1, 2, 6)
        # hidden init/mask helpers
        h0 = layer.init_hidden(2, "cpu")
        out2, _ = layer(x, h0)
        assert T.allclose(out, out2, atol=1e-6)  # zero init == default init
        masked = layer.mask_hidden(hidden, T.tensor([True, False]))
        if mode == "lstm":
            assert T.equal(masked[0][:, 0], hidden[0][:, 0])
            assert T.all(masked[0][:, 1] == 0) and T.all(masked[1][:, 1] == 0)
        else:
            assert T.equal(masked[:, 0], hidden[:, 0])
            assert T.all(masked[:, 1] == 0)

    def test_every_registered_type_is_covered(self):
        covered = {c[0] for c in LAYER_CASES} | {"lstm", "gru", "rmsnorm"}
        assert covered >= set(LAYER_REGISTRY.keys()), (
            f"Registry types missing coverage: {set(LAYER_REGISTRY) - covered}"
        )

    def test_unknown_type_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Unsupported layer type"):
            build_layer("totally_bogus", {})

    def test_missing_required_param_raises(self):
        with pytest.raises(ValueError, match="missing required param"):
            build_layer("dense", {})  # 'units' required
        with pytest.raises(ValueError, match="missing required param"):
            build_layer("lstm", {})  # 'hidden_size' required


class TestShapeValidation:
    def test_mha_rejects_rank2(self):
        layer = build_layer("mha", {"embed_dim": 8, "num_heads": 2})
        with pytest.raises(ValueError, match="rank-3"):
            layer(T.randn(4, 8))

    def test_transformer_rejects_rank2(self):
        layer = build_layer("transformer_encoder", {"d_model": 8, "nhead": 2})
        with pytest.raises(ValueError, match="rank-3"):
            layer(T.randn(4, 8))

    def test_positional_encoding_rejects_rank2_and_overlong(self):
        layer = build_layer("positional_encoding", {"d_model": 6, "max_len": 4})
        with pytest.raises(ValueError, match="rank-3"):
            layer(T.randn(4, 6))
        with pytest.raises(ValueError, match="max_len"):
            layer(T.randn(2, 5, 6))

    def test_recurrent_rejects_rank2(self):
        layer = build_layer("lstm", {"hidden_size": 6})
        with pytest.raises(ValueError, match="rank-3"):
            layer(T.randn(4, 6))


class TestCausalMasking:
    def test_causal_mask_blocks_future(self):
        """Changing a future token must not change past outputs under causal
        masking (and must change them without it)."""
        T.manual_seed(1)
        causal = build_layer(
            "transformer_encoder", {"d_model": 8, "nhead": 2, "causal": True, "dropout": 0.0}
        ).eval()
        x = T.randn(1, 6, 8)
        y1 = causal(x)
        x2 = x.clone()
        x2[:, -1] += 10.0  # perturb only the last (future) token
        y2 = causal(x2)
        assert T.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5), "future leaked into past"
        assert not T.allclose(y1[:, -1], y2[:, -1], atol=1e-5)

    def test_segment_mask_blocks_across_episode_starts(self):
        """With a start flag at t=3, tokens after the boundary must not attend
        to tokens before it: outputs for the second segment must equal running
        the second segment alone."""
        T.manual_seed(2)
        causal = build_layer(
            "transformer_encoder", {"d_model": 8, "nhead": 2, "causal": True, "dropout": 0.0}
        ).eval()
        x = T.randn(2, 6, 8)
        start = T.zeros(2, 6, dtype=T.bool)
        start[:, 3] = True
        y = causal(x, start_mask=start)
        y_second_alone = causal(x[:, 3:], start_mask=T.zeros(2, 3, dtype=T.bool))
        assert T.allclose(y[:, 3:], y_second_alone, atol=1e-5)


class TestWeightInitialization:
    def test_dense_kernel_and_bias(self):
        subnet = SubNetwork([
            {"type": "dense", "params": {"units": 16, "kernel": "constant", "kernel_params": {"val": 0.5}}},
        ])
        subnet(T.randn(4, 8))  # materialize lazy
        subnet.init_weights()
        layer = subnet.layers["dense_0"]
        assert T.all(layer.weight == 0.5)
        assert T.all(layer.bias == 0.0)

    def test_lstm_kernel_reaches_all_weight_tensors(self):
        subnet = SubNetwork([
            {"type": "lstm", "params": {"hidden_size": 8, "num_layers": 2,
                                        "kernel": "constant", "kernel_params": {"val": 0.25}}},
        ])
        subnet(T.randn(2, 3, 4), mode="sequence")
        subnet.init_weights()
        rnn = subnet.layers["lstm_0"].rnn
        weight_names = [n for n, _ in rnn.named_parameters() if "weight" in n]
        assert set(weight_names) == {"weight_ih_l0", "weight_hh_l0", "weight_ih_l1", "weight_hh_l1"}
        for name, param in rnn.named_parameters():
            if "weight" in name:
                assert T.all(param == 0.25), f"{name} not initialized by kernel"
            else:
                assert T.all(param == 0.0), f"{name} (bias) not zeroed"

    def test_mha_kernel_reaches_projections(self):
        subnet = SubNetwork([
            {"type": "mha", "params": {"embed_dim": 8, "num_heads": 2,
                                       "kernel": "constant", "kernel_params": {"val": 0.1}}},
        ])
        subnet(T.randn(2, 5, 8))
        subnet.init_weights()
        attn = subnet.layers["mha_0"].attn
        assert T.all(attn.in_proj_weight == 0.1)
        assert T.all(attn.out_proj.weight == 0.1)
        assert T.all(attn.in_proj_bias == 0.0)
        assert T.all(attn.out_proj.bias == 0.0)

    def test_transformer_kernel_reaches_all_linears(self):
        subnet = SubNetwork([
            {"type": "transformer_encoder",
             "params": {"d_model": 8, "nhead": 2, "dim_feedforward": 16,
                        "kernel": "constant", "kernel_params": {"val": 0.2}}},
        ])
        subnet(T.randn(2, 5, 8))
        subnet.init_weights()
        block = subnet.layers["transformer_encoder_0"]
        for name, param in block.named_parameters():
            if "weight" in name and param.dim() >= 2:
                assert T.all(param == 0.2), f"{name} untouched by kernel"
            elif "weight" in name:  # rank-1 LayerNorm scales stay at default (ones)
                assert T.all(param == 1.0), f"{name} (norm scale) should be untouched"
            elif "bias" in name:
                assert T.all(param == 0.0), f"{name} not zeroed"

    def test_layernorm_scale_untouched_by_kernel(self):
        subnet = SubNetwork([
            {"type": "layernorm", "params": {"normalized_shape": 8,
                                             "kernel": "constant", "kernel_params": {"val": 9.0}}},
        ])
        subnet(T.randn(4, 8))
        subnet.init_weights()
        ln = subnet.layers["layernorm_0"]
        assert T.all(ln.weight == 1.0)  # rank-1 weight: kernel must NOT apply
        assert T.all(ln.bias == 0.0)

    def test_orthogonal_kernel_orthogonality(self):
        subnet = SubNetwork([
            {"type": "dense", "params": {"units": 8, "kernel": "orthogonal", "kernel_params": {"gain": 1.0}}},
        ])
        subnet(T.randn(4, 16))
        subnet.init_weights()
        w = subnet.layers["dense_0"].weight  # (8, 16), rows orthonormal
        eye = w @ w.t()
        assert T.allclose(eye, T.eye(8), atol=1e-5)

    def test_unknown_kernel_raises(self):
        subnet = SubNetwork([
            {"type": "dense", "params": {"units": 4, "kernel": "not_a_kernel"}},
        ])
        subnet(T.randn(2, 3))
        with pytest.raises(ValueError, match="Unsupported initialization"):
            subnet.init_weights()

    def test_default_kernel_keeps_pytorch_init(self):
        T.manual_seed(7)
        subnet = SubNetwork([{"type": "dense", "params": {"units": 4}}])
        subnet(T.randn(2, 3))
        before = subnet.layers["dense_0"].weight.clone()
        subnet.init_weights()  # kernel 'default' -> weights untouched, bias zeroed
        assert T.equal(subnet.layers["dense_0"].weight, before)
        assert T.all(subnet.layers["dense_0"].bias == 0.0)


class TestSubNetworkTemporalIntrospection:
    def test_flags(self):
        plain = SubNetwork([{"type": "dense", "params": {"units": 4}}])
        assert not plain.is_recurrent and not plain.is_causal and not plain.is_temporal

        rec = SubNetwork([{"type": "lstm", "params": {"hidden_size": 4}}])
        assert rec.is_recurrent and rec.is_temporal and not rec.is_causal

        caus = SubNetwork([{"type": "transformer_encoder",
                            "params": {"d_model": 8, "nhead": 2, "causal": True}}])
        assert caus.is_causal and caus.is_temporal and not caus.is_recurrent

        intra = SubNetwork([{"type": "transformer_encoder",
                             "params": {"d_model": 8, "nhead": 2, "causal": False}}])
        assert not intra.is_temporal

    def test_expects_image_and_tokens(self):
        conv = SubNetwork([{"type": "conv2d", "params": {"out_channels": 4}}])
        assert conv.expects_image and not conv.expects_tokens
        emb = SubNetwork([{"type": "embedding", "params": {"num_embeddings": 5, "embedding_dim": 4}}])
        assert emb.expects_tokens and not emb.expects_image
        mlp = SubNetwork([{"type": "dense", "params": {"units": 4}}])
        assert not mlp.expects_image and not mlp.expects_tokens
