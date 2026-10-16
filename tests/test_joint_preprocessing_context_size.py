import argparse
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_context_size as runner
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, read_rows, real_episode, source_family
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original_threads)


def arguments(root, **changes):
    values = dict(output_dir=root, reuse_bank_dir=root / "original", source_fingerprint=runner.SOURCE_FINGERPRINT,
        candidate_manifest=runner.residual.DEFAULT_CANDIDATES, expected_revision=None, checkpoint=None,
        device="cpu", arm="small", continuation_seed=0, steps=2, evaluate_every=1,
        save_every=1, probe_sources=2, lr=.0003, resume=False, max_steps=None)
    values.update(changes)
    return argparse.Namespace(**values)


def test_restricted_model_has_only_requested_heads_and_exact_native_identity():
    model, episode, backbone = runner.new_model(), real_episode(), tiny()
    names = set(dict(model.named_children()))
    assert not any("attention" in name or "neural" in name or "mix" in name for name, _ in model.named_parameters())
    assert {"encoder", "affine_head", "spline_head", "spline_gate_head"} <= names
    assert sum(parameter.numel() for parameter in model.parameters()) < 8000
    torch.testing.assert_close(runner.residual.episode_logits(backbone, model, episode),
                               runner.residual.episode_logits(backbone, None, episode), rtol=0, atol=0)
    parameters = model.generate(episode["x_context"][..., episode["numerical_mask"]], episode["y_context"],
                                episode["context_missing"][None])
    assert parameters.mixing is None and parameters.neural_first_weight is None


def test_restricted_staged_backward_matches_full_ensemble_gradient():
    model, episode, backbone = runner.new_model(), real_episode(), tiny()
    with torch.no_grad():
        model.affine_head.weight.fill_(.002)
        model.spline_head.weight.fill_(.003)
    staged = copy.deepcopy(model)
    context, views = runner.residual.prepared_episode(episode, torch.device("cpu"))
    numerical_values, _ = runner.residual.numeric_views(model, backbone, context, views)
    logits = runner.residual.mean_logits(backbone, views, episode["n_classes"], numerical_values)
    expected_loss = torch.nn.functional.cross_entropy((logits / .9).flatten(0, 1), episode["y_query"])
    expected_loss.backward()
    actual_loss = runner.residual.ensemble_backward(backbone, staged, episode, scale=1.)
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    for (name, parameter), (_, staged_parameter) in zip(model.named_parameters(), staged.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, staged_parameter.grad, rtol=5e-4, atol=5e-6, msg=name)
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_full_size_pairs_are_nested_and_queries_match_without_probe_leakage():
    labels = np.resize(np.array([0, 0, 0, 1, 2]), 12000)
    fitting_positions, probe_queries = runner.partition_source(labels, 12)
    pair = runner.paired_positions(labels, fitting_positions, 99)
    repeated = runner.paired_positions(labels, fitting_positions, 99)
    assert len(pair["small"]) == 512 and len(pair["large"]) == 2048 and len(pair["query"]) == 256
    assert set(pair["small"]) < set(pair["large"])
    assert not set(pair["large"]) & set(pair["query"])
    assert not set(probe_queries) & set(np.concatenate(list(pair.values())))
    assert pair["large"].max() > 1024
    for role in pair:
        np.testing.assert_array_equal(pair[role], repeated[role])
        assert set(labels[pair[role]]) == {0, 1, 2}
    probe = runner.paired_positions(labels, fitting_positions, 66, probe_queries)
    np.testing.assert_array_equal(probe["query"], probe_queries)


def test_insufficient_rows_or_rare_classes_are_rejected_instead_of_capped():
    with pytest.raises(ValueError):
        runner.partition_source(np.arange(2400) % 2, 10)
    with pytest.raises(ValueError):
        runner.partition_source(np.r_[np.zeros(3000), [1, 1]], 10)


def test_query_values_do_not_fit_encoding_or_conditioner():
    source = source_family(rows=3000)
    labels = np.asarray(source["labels"])
    fitting_positions, query_positions = runner.partition_source(labels, 77)
    pair = runner.paired_positions(labels, fitting_positions, 88, query_positions)
    original = runner.episode_from_rows(source, pair["small"], pair["query"], 88)
    changed = copy.deepcopy(source)
    for column in changed["columns"]:
        if column["kind"] == "numerical":
            for query_position in pair["query"]:
                column["values"][int(query_position)] = 1e20
    for query_position in pair["query"]:
        changed["labels"][int(query_position)] = 1 - changed["labels"][int(query_position)]
    actual = runner.episode_from_rows(changed, pair["small"], pair["query"], 88)
    for field_name in ("x_context", "y_context", "context_missing"):
        torch.testing.assert_close(actual[field_name], original[field_name], rtol=0, atol=0)
    model = runner.new_model()
    for episode in (original, actual):
        context = episode["x_context"][..., episode["numerical_mask"]]
        parameters = model.generate(context, episode["y_context"], episode["context_missing"][None])
        if episode is original:
            expected_parameters = parameters
        else:
            for field_name in parameters.__dataclass_fields__:
                expected_value = getattr(expected_parameters, field_name)
                actual_value = getattr(parameters, field_name)
                if expected_value is not None:
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def fixture_source(root, name):
    source = source_family(name, rows=3000, mixed=False)
    cache_path = root / f"{name}.pt"
    runner.pilot.atomic_save(cache_path, source)
    fitting_positions, probe_queries = runner.partition_source(np.asarray(source["labels"]), 21)
    entry = dict(family=name, source_group=name, panel="train", cache_path=str(cache_path),
        cache_sha256=runner.pilot.hash_file(cache_path), fitting_positions=torch.tensor(fitting_positions),
        probe_query_positions=torch.tensor(probe_queries))
    return source, entry


def test_training_arms_share_queries_and_full_source_pool(tmp_path):
    _, entry = fixture_source(tmp_path, "source")
    cache = runner.SourceCache([entry])
    small = runner.training_batch("small", 1, [entry], cache)
    large = runner.training_batch("large", 1, [entry], cache)
    for small_episode, large_episode in zip(small, large, strict=True):
        assert small_episode["n_context"] == 512 and large_episode["n_context"] == 2048
        assert small_episode["full_fitting_rows"] == 2744
        torch.testing.assert_close(small_episode["query_indices"], large_episode["query_indices"])
        assert set(small_episode["context_indices"].tolist()) < set(large_episode["context_indices"].tolist())
    Path(entry["cache_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        runner.SourceCache([entry]).get("source")


def test_train_resume_has_identical_weights_logs_and_coverage(tmp_path, monkeypatch):
    _, entry = fixture_source(tmp_path, "source")
    manifest = dict(fingerprint="fixture", counts=dict(train=1, validation=1))
    probe = real_episode(mixed=False)
    evaluation = [dict(family="source", panel="source_probe", episodes={arm: probe for arm in runner.ARMS}),
                  dict(family="unseen", panel="real_validation", episodes={arm: dict(probe, family="unseen") for arm in runner.ARMS})]
    monkeypatch.setattr(runner, "setup", lambda args: (tiny(), runner.new_model(), manifest))
    monkeypatch.setattr(runner, "evaluation_bank", lambda args: evaluation)

    def tiny_batch(arm, step, entries, cache):
        # Small episodes keep the end-to-end resume test CPU-only and fast.
        episodes = [copy.deepcopy(probe) for _ in range(4)]
        for episode in episodes:
            episode.update(source_seed=step, full_fitting_rows=len(entry["fitting_positions"]))
        return episodes

    monkeypatch.setattr(runner, "training_batch", tiny_batch)
    for name, max_steps in (("full", None), ("resumed", 1)):
        root = tmp_path / name
        runner.pilot.atomic_save(root / "sources.pt", dict(entries=[entry]))
        runner.train(arguments(root, max_steps=max_steps))
    runner.train(arguments(tmp_path / "resumed", resume=True))
    original_state = torch.load(tmp_path / "full/runs/small_seed0/state.pt", weights_only=True)
    resumed_state = torch.load(tmp_path / "resumed/runs/small_seed0/state.pt", weights_only=True)
    assert_nested_equal(original_state, resumed_state)
    assert len(read_rows(tmp_path / "resumed/runs/small_seed0/presentations.csv")) == 8
    assert len(read_rows(tmp_path / "resumed/runs/small_seed0/evaluation_episodes.csv")) == 12
    monkeypatch.setattr(runner, "checked_manifest", lambda args: manifest)
    runner.train(arguments(tmp_path / "full", arm="large"))
    monkeypatch.setattr(runner, "CONTEXT_ROWS", dict(small=probe["n_context"], large=probe["n_context"]))
    monkeypatch.setattr(runner, "QUERY_ROWS", probe["n_query"])
    runner.report(arguments(tmp_path / "full"))
    result = runner.previous.read(tmp_path / "full/complete.json")
    assert result["comparisons"]["final"]["real_validation/large"]["large_vs_small"]["mean_nll_reduction"] == 0.
    monkeypatch.setattr(runner, "checked_manifest", lambda args: manifest)
    runner.train(arguments(tmp_path / "full", arm="large"))
    runner.report(arguments(tmp_path / "full"))
    result = runner.previous.read(tmp_path / "full/complete.json")
    assert result["comparisons"]["final"]["real_validation/large"]["large_vs_small"]["mean_nll_reduction"] == 0.


def test_prepare_uses_full_cached_rows_and_preserves_original_source_assignment(tmp_path, monkeypatch):
    args = arguments(tmp_path / "experiment")
    original_root = args.reuse_bank_dir
    selection = dict(large=[], validation=[])
    for panel, count in (("large", 16), ("validation", 5)):
        for source_position in range(count):
            family = f"{panel}_{source_position}"
            source = source_family(family, rows=3000, mixed=False)
            source_entry = dict(source="openml", data_id=source_position + (100 if panel == "validation" else 0), name=family)
            identity = runner.bank._entry_identity(source_entry)
            filename = runner.hashlib.sha256(identity.encode()).hexdigest()[:20] + ".pt"
            cache_path = original_root / "source_cache" / filename
            runner.pilot.atomic_save(cache_path, dict(source, entry=source_entry))
            fingerprint = runner.bank.content_fingerprints(runner.bank.unpack_frame(source["columns"]), np.asarray(source["labels"]))
            selection[panel].append(dict(name=family, source_group=family, source_entry=source_entry,
                descriptors=dict(rows=3000), inputs_sha256=fingerprint["inputs_sha256"]))
    monkeypatch.setattr(runner.residual.objective.real, "_prepared_views", lambda *args: None)
    runner.pilot.json_write(original_root / "banks_manifest.json", dict(selection=selection))
    original_manifest = dict(banks_manifest_sha256=runner.pilot.hash_file(original_root / "banks_manifest.json"))
    original_manifest["fingerprint"] = runner.previous.digest(original_manifest)
    args.source_fingerprint = original_manifest["fingerprint"]
    runner.pilot.json_write(original_root / "manifest.json", original_manifest)
    runner.prepare(args)
    manifest = runner.checked_manifest(args)
    assert manifest["counts"] == dict(train=16, validation=5)
    entries = torch.load(args.output_dir / "sources.pt", weights_only=True)["entries"]
    assert all(entry["full_rows"] == 3000 for entry in entries)
    assert all(len(entry["fitting_positions"]) == 2744 for entry in entries)
    assert all(not set(entry["fitting_positions"].tolist()) & set(entry["probe_query_positions"].tolist()) for entry in entries)
    assert len(runner.evaluation_bank(args)) == 7
    runner.prepare(args)
    assert runner.checked_manifest(args) == manifest


def test_settings_pin_code_and_exclude_other_heads(tmp_path):
    specification = runner.settings(arguments(tmp_path))
    assert specification["context_rows"] == dict(small=512, large=2048)
    assert specification["steps"] == 2
    assert "scripts/joint_preprocessing_context_size.py" in specification["code_hashes"]
    assert "src/tabicl/_hyperspline/statistics_preprocessing.py" in specification["code_hashes"]
