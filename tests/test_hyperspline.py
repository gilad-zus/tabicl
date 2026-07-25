import inspect

import numpy as np
import pytest
import torch
from torch import nn

from tabicl._hyperspline.preprocessing import HyperSplineEnsembleGenerator
from tabicl._sklearn.classifier import TabICLClassifier
from tabicl._sklearn.preprocessing import EncodedTable, TransformToNumerical

from tabicl._hyperspline import (
    DirectSplineTransform,
    FrozenTabICLHyperSpline,
    HyperSplineTransform,
    backbone_state_dict_hash,
    load_hyperspline_checkpoint,
    save_hyperspline_checkpoint,
    summarize_context,
)


def test_shapes_identity_and_feature_permutation():
    torch.manual_seed(0)
    module = HyperSplineTransform(n_control_points=8)
    context = torch.randn(2, 7, 3)
    query = torch.randn(2, 4, 3)
    context_out, query_out, parameters = module(context, query, return_parameters=True)
    expected_context = (context - context.mean(1, keepdim=True)) / context.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    expected_query = (query - context.mean(1, keepdim=True)) / context.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    assert context_out.shape == context.shape
    assert query_out.shape == query.shape
    assert parameters.control_points.shape == (2, 3, 8)
    assert torch.allclose(context_out, expected_context, atol=2e-5, rtol=2e-5)
    assert torch.allclose(query_out, expected_query, atol=2e-5, rtol=2e-5)
    permutation = torch.tensor([2, 0, 1])
    permuted_out, _ = module(context[..., permutation], query[..., permutation])
    assert torch.allclose(permuted_out, context_out[..., permutation], atol=2e-5, rtol=2e-5)


def test_summaries_are_context_only_and_handle_missing():
    context = torch.tensor([[[1.0, float("nan")], [2.0, float("nan")], [3.0, float("nan")]]])
    missing = torch.isnan(context)
    context = context.nan_to_num(0.0)
    a = summarize_context(context, missing)
    b = summarize_context(context, missing)
    assert torch.equal(a.summary, b.summary)
    assert a.all_missing.tolist() == [[False, True]]
    module = HyperSplineTransform()
    query_a = torch.tensor([[[10.0, 4.0]]])
    query_b = torch.tensor([[[-1000.0, 99.0]]])
    _, _, params_a = module(context, query_a, missing, return_parameters=True)
    _, _, params_b = module(context, query_b, missing, return_parameters=True)
    assert torch.equal(params_a.control_points, params_b.control_points)
    assert torch.equal(params_a.location, params_b.location)


def test_gradients_dtype_and_checkpoint_round_trip(tmp_path):
    module = HyperSplineTransform(n_control_points=7, generate_location=True, generate_scale=True)
    context = torch.randn(1, 6, 2)
    query = torch.randn(1, 3, 2)
    out_context, out_query = module(context, query)
    (out_context.square().mean() + out_query.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in module.parameters())
    assert module.knots.dtype == torch.float32
    assert out_context.dtype == context.dtype
    assert out_query.dtype == query.dtype
    path = tmp_path / "hyperspline.ckpt"
    config = {"n_control_points": 7, "generate_location": True, "generate_scale": True}
    save_hyperspline_checkpoint(
        path, module, config, backbone_reference="official.ckpt", backbone_hash="abc123", step=2
    )
    restored, metadata = load_hyperspline_checkpoint(
        path, expected_backbone_reference="official.ckpt", expected_backbone_hash="abc123"
    )
    original = module(context, query)
    recovered = restored(context, query)
    assert metadata["backbone_reference"] == "official.ckpt"
    assert torch.allclose(original[0], recovered[0])
    assert torch.allclose(original[1], recovered[1])
    with pytest.raises(ValueError, match="different backbone reference"):
        load_hyperspline_checkpoint(path, expected_backbone_reference="other.ckpt")
    hash_probe = nn.Linear(2, 1)
    original_hash = backbone_state_dict_hash(hash_probe)
    with torch.no_grad():
        hash_probe.weight.add_(1.0)
    assert original_hash != backbone_state_dict_hash(hash_probe)


def test_direct_spline_has_trainable_per_column_parameters():
    context = torch.randn(1, 5, 4)
    direct = DirectSplineTransform(context, n_control_points=8)
    output = direct.transform(context)
    output.square().mean().backward()
    assert direct.gap_logits.grad is not None
    assert direct.gate_logits.grad is not None


def test_frozen_adapter_preserves_categorical_columns_and_backbone_freezing():
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, y_train):
            self.last_x = x
            return (x[:, y_train.shape[1] :].sum(dim=-1) * self.weight).unsqueeze(-1)

    backbone = ToyBackbone()
    adapter = FrozenTabICLHyperSpline(backbone, HyperSplineTransform(n_control_points=8))
    context = torch.randn(1, 5, 3, requires_grad=True)
    query = torch.randn(1, 2, 3, requires_grad=True)
    output, _ = adapter(context, query, torch.zeros(1, 5), torch.tensor([False, True, True]), return_parameters=True)
    output.mean().backward()
    merged = backbone.last_x
    assert torch.equal(merged[:, :, 0], torch.cat((context, query), dim=1)[:, :, 0])
    assert backbone.weight.grad is None
    assert any(parameter.grad is not None for parameter in adapter.hyperspline.parameters())
    assert context.grad is not None
    assert query.grad is not None
    adapter.train()
    assert not adapter.backbone.training
    assert all(parameter is not backbone.weight for parameter in adapter.parameters())


def test_target_aware_statistics_affect_parameters_and_are_class_id_invariant():
    torch.manual_seed(0)
    context = torch.tensor([[[0.0, 1.0], [0.2, 1.2], [3.0, 4.0], [3.2, 4.2]]])
    query = torch.zeros(1, 2, 2)
    labels_a = torch.tensor([[0, 0, 1, 1]])
    labels_b = torch.tensor([[0, 1, 0, 1]])

    # The target-aware summary has no dependence on the arbitrary class IDs.
    assert torch.allclose(
        summarize_context(context, y_context=labels_a).summary,
        summarize_context(context, y_context=torch.tensor([[7, 7, 3, 3]])).summary,
    )

    unaware = HyperSplineTransform(hidden_dim=8)
    _, _, unaware_a = unaware(context, query, y_context=labels_a, return_parameters=True)
    _, _, unaware_b = unaware(context, query, y_context=labels_b, return_parameters=True)
    assert torch.equal(unaware_a.control_points, unaware_b.control_points)

    aware = HyperSplineTransform(hidden_dim=8, target_aware=True)
    # The zero-initialized output layer starts at identity, so make its first
    # output explicitly depend on a supervised summary coordinate.
    with torch.no_grad():
        aware.mlp[-1].weight[0, 0] = 1.0
        aware.mlp[1].weight[0, -1] = 1.0
    _, _, aware_a = aware(context, query, y_context=labels_a, return_parameters=True)
    _, _, aware_b = aware(context, query, y_context=labels_b, return_parameters=True)
    assert not torch.equal(aware_a.control_points, aware_b.control_points)


def test_supervised_residual_starts_as_and_preserves_a_frozen_marginal_policy():
    torch.manual_seed(0)
    context = torch.tensor([[[0.0], [0.2], [3.0], [3.2]]])
    query = torch.zeros(1, 1, 1)
    labels_a = torch.tensor([[0, 0, 1, 1]])
    labels_b = torch.tensor([[0, 1, 0, 1]])
    marginal = HyperSplineTransform(hidden_dim=8)
    residual = HyperSplineTransform(hidden_dim=8, target_aware=True, supervised_residual=True)
    residual.initialize_supervised_residual_from(marginal)
    assert all(not parameter.requires_grad for parameter in residual.mlp.parameters())

    _, _, marginal_parameters = marginal(context, query, return_parameters=True)
    _, _, initial_parameters = residual(context, query, y_context=labels_a, return_parameters=True)
    assert torch.equal(initial_parameters.control_points, marginal_parameters.control_points)
    assert torch.equal(initial_parameters.gate, marginal_parameters.gate)

    # The residual is label-only: after enabling one residual output path, a
    # label permutation changes generated parameters while the base stays frozen.
    with torch.no_grad():
        residual.supervised_residual_mlp[-1].weight[0, 0] = 1.0
        residual.supervised_residual_mlp[1].weight[0, -1] = 1.0
    _, _, parameters_a = residual(context, query, y_context=labels_a, return_parameters=True)
    _, _, parameters_b = residual(context, query, y_context=labels_b, return_parameters=True)
    assert not torch.equal(parameters_a.control_points, parameters_b.control_points)


def test_query_labels_never_enter_parameter_generation():
    context = torch.randn(1, 4, 2)
    query = torch.randn(1, 3, 2)
    y_context = torch.tensor([[0, 1, 0, 1]])
    module = HyperSplineTransform(target_aware=True)
    _, _, first = module(context, query, y_context=y_context, return_parameters=True)
    _, _, second = module(context, query, y_context=y_context, return_parameters=True)
    assert torch.equal(first.control_points, second.control_points)
    assert torch.equal(first.gate, second.gate)
    assert "y_query" not in inspect.signature(module.forward).parameters


def test_ensemble_preserves_categorical_pipeline_and_aligns_feature_permutations():
    context = EncodedTable(
        categorical=np.array([[0.0], [1.0], [0.0], [2.0]]),
        numerical=np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 30.0], [8.0, 40.0]]),
        numerical_missing=np.zeros((4, 2), dtype=bool),
    )
    query = EncodedTable(
        categorical=np.array([[2.0], [1.0]]),
        numerical=np.array([[16.0, 50.0], [32.0, 60.0]]),
        numerical_missing=np.zeros((2, 2), dtype=bool),
    )
    generator = HyperSplineEnsembleGenerator(
        classification=True,
        n_estimators=2,
        norm_methods=["none"],
        feat_shuffle_method="latin",
        random_state=0,
    ).fit(context, np.array([0, 1, 0, 1]))
    context_num = torch.as_tensor(generator.context_numerical_).unsqueeze(0)
    query_num = torch.as_tensor(generator.query_numerical(query)[0]).unsqueeze(0)
    assembled = generator.build(query, context_num, query_num)
    n_cat = generator.context_categorical_.shape[1]
    for method, (variants, _) in assembled.items():
        baseline = generator.ensemble_.preprocessors_[method]
        expected_context_cat = baseline.X_transformed_[:, :n_cat]
        query_filtered = generator.ensemble_.unique_filter_.transform(generator._merge(query))
        expected_query_cat = baseline.transform(query_filtered)[:, :n_cat]
        canonical_expected = np.concatenate(
            (
                np.concatenate((expected_context_cat, generator.context_numerical_), axis=1),
                np.concatenate((expected_query_cat, generator.query_numerical(query)[0]), axis=1),
            ),
            axis=0,
        )
        for variant, permutation in zip(variants, generator.feature_shuffles_[method]):
            recovered_canonical = variant[:, np.argsort(permutation)]
            assert np.allclose(recovered_canonical, canonical_expected)

    categorical_only = EncodedTable(
        categorical=np.array([[0.0], [1.0], [0.0]]),
        numerical=np.empty((3, 0)),
        numerical_missing=np.empty((3, 0), dtype=bool),
    )
    cat_generator = HyperSplineEnsembleGenerator(
        classification=True,
        n_estimators=1,
        norm_methods=["none"],
        feat_shuffle_method="latin",
        random_state=0,
    ).fit(categorical_only, np.array([0, 1, 0]))
    empty_context = torch.empty((1, 3, 0))
    empty_query = torch.empty((1, 2, 0))
    cat_query = EncodedTable(np.array([[1.0], [0.0]]), np.empty((2, 0)), np.empty((2, 0), dtype=bool))
    cat_variants = cat_generator.build(cat_query, empty_context, empty_query)
    assert next(iter(cat_variants.values()))[0].shape == (1, 5, 1)


def test_classifier_hyperspline_opt_in_uses_context_and_freezes_backbone(monkeypatch):
    class TinyClassifier(nn.Module):
        max_classes = 2

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, X, y_train, **kwargs):
            self.calls.append((X.detach().clone(), y_train.detach().clone()))
            n_query = X.shape[1] - y_train.shape[1]
            return torch.full((X.shape[0], n_query, 2), 0.5, device=X.device) * self.weight

    def load_tiny_model(self):
        self.model_ = TinyClassifier()
        self.model_config_ = {}

    monkeypatch.setattr(TabICLClassifier, "_load_model", load_tiny_model)
    estimator = TabICLClassifier(
        n_estimators=2,
        norm_methods=["none"],
        numerical_preprocessing="hyperspline",
        hyperspline_config={"n_control_points": 6, "hidden_dim": 8},
        device="cpu",
        random_state=0,
    ).fit(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]), np.array([0, 1, 0, 1]))
    probabilities = estimator.predict_proba(np.array([[5.0, 50.0], [6.0, 60.0]]))
    assert probabilities.shape == (2, 2)
    assert estimator.hyperspline_parameters_.control_points.shape == (1, 2, 6)
    assert not estimator.model_.training
    assert all(not parameter.requires_grad for parameter in estimator.model_.parameters())
    assert estimator.model_.calls[0][0].shape[1] == 6


def test_typed_encoder_keeps_numerical_missing_mask():
    x = torch.tensor([[1.0, float("nan")], [2.0, 3.0]]).numpy()
    encoded = TransformToNumerical().fit_transform_parts(x)
    assert encoded.categorical.shape == (2, 0)
    assert encoded.numerical.shape == (2, 2)
    assert encoded.numerical_missing.tolist() == [[False, True], [False, False]]
