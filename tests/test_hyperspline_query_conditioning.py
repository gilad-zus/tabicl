import pytest
import torch
from argparse import Namespace
from torch import nn

from tabicl._hyperspline import HyperSplineTransform, summarize_context
from scripts.hyperspline_query_conditioning import Episode, RowPool, run_model_seed


def test_query_conditioner_is_per_column_and_never_uses_query_labels():
    context = torch.tensor(
        [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0], [5.0, 15.0]]]
    )
    query = torch.tensor([[[20.0, 1.0], [21.0, 2.0], [22.0, 3.0], [23.0, 4.0]]])
    labels = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    relabelled = 1.0 - labels
    model = HyperSplineTransform(n_control_points=6, hidden_dim=8, target_aware=True, conditioning_mode="context_query_shift")
    context_stats = summarize_context(context, y_context=labels)
    relabelled_stats = summarize_context(context, y_context=relabelled)
    query_stats = summarize_context(query)
    feature = model.conditioning_summary(context_stats, query_stats)
    relabelled_feature = model.conditioning_summary(relabelled_stats, query_stats)
    assert feature.shape == (1, 2, 31 + 3 * 23 + 2)
    # Class IDs are arbitrary, so a class permutation cannot alter the input.
    assert torch.allclose(feature, relabelled_feature)
    reshaped_query = query.clone()
    reshaped_query[:, 0] += 100.0
    changed_query = model.conditioning_summary(context_stats, summarize_context(reshaped_query))
    assert not torch.allclose(feature, changed_query)


def test_query_marginal_mode_rejects_complex_residual_variants():
    with pytest.raises(ValueError, match="query-aware conditioning"):
        HyperSplineTransform(
            n_control_points=6, hidden_dim=8, target_aware=True,
            conditioning_mode="query_marginal", supervised_residual=True,
        )


def test_capacity_matched_query_arms_have_identical_input_width():
    context = torch.randn(1, 8, 3)
    query = torch.randn(1, 5, 3)
    labels = torch.tensor([[0.0, 1.0] * 4])
    context_stats = summarize_context(context, y_context=labels)
    query_stats = summarize_context(query)
    models = [
        HyperSplineTransform(n_control_points=6, hidden_dim=8, target_aware=True,
                             conditioning_mode=mode, capacity_matched_conditioning=True)
        for mode in ("context", "query_marginal", "context_query_shift")
    ]
    summaries = [model.conditioning_summary(context_stats, query_stats) for model in models]
    assert {summary.shape[-1] for summary in summaries} == {102}
    assert {model.mlp[1].weight.shape for model in models} == {torch.Size((8, 102))}


def test_query_conditioning_runner_smoke_test_uses_only_task_nll(tmp_path):
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def clear_cache(self):
            return None

        def forward(self, x, y_context):
            self.calls += 1
            query = x[:, y_context.shape[1] :]
            score = query.mean(-1)
            return torch.stack((-score, score), dim=-1)

    generator = torch.Generator().manual_seed(7)
    pools, validation, test = [], [], []
    for dataset_index, dataset in enumerate(("alpha", "beta")):
        x = torch.randn(40, 3, generator=generator).numpy()
        y = torch.tensor([0, 1] * 20).numpy()
        pools.append(RowPool(dataset, "train", x, y))
        for stage, output in (("validation", validation), ("test", test)):
            output.append(Episode(
                dataset=dataset, stage=stage, episode_id=0, source_seed=dataset_index,
                x_context=torch.as_tensor(x[:12]), y_context=torch.as_tensor(y[:12], dtype=torch.float32),
                x_query=torch.as_tensor(x[12:20]), y_query=torch.as_tensor(y[12:20], dtype=torch.long),
            ))
    args = Namespace(
        output_dir=tmp_path, resume=False, arm="context_query_shift", n_control_points=6, hidden_dim=8,
        generate_location=False, generate_scale=False, gate_initial_probability=0.01, target_aware=True,
        lr=1e-3, train_seed=19, tasks_per_step=2, max_backbone_batch_size=1, steps=1,
        context_rows=12, query_rows=8, log_every=1, gradient_clip=1.0, validate_every=1,
        patience_validations=0, evaluation_batch_size=1,
    )
    backbone = TinyBackbone()
    summary = run_model_seed(
        args, backbone=backbone, pools=pools, validation=validation, test=test,
        model_seed=0, device=torch.device("cpu"),
    )
    assert summary["status"] == "complete"
    assert summary["protocol"].endswith("TabICL query NLL only")
    # 2 identity-validation + 2 identity-test + 2 training microbatches +
    # 2 validation + 2 selected-validation + 2 selected-test calls.  The
    # old implementation silently used one task here despite tasks_per_step=2.
    assert backbone.calls == 12
    for name in ("manifest.json", "training.csv", "evaluations.csv", "paired_test.csv", "per_dataset.csv", "best.pt", "summary.json"):
        assert (tmp_path / "model_seed_0" / name).is_file()
