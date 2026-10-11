"""Exercise row isolation, the paired control, and resumable adapter fitting."""
import argparse
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import joint_preprocessing_benchmark_sampling as runner
from tabicl._experiments.direct_spline_openml import OpenMLTaskData
from tests.test_joint_preprocessing_residual_conditioning import tiny
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, read_rows


@pytest.fixture(autouse=True)
def one_thread():
    previous_count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_count)


def args_for(root, **changes):
    values = dict(output_dir=root, steps=3, evaluate_every=1, outer_train_rows=60,
        query_chunk_rows=16, lr=.001, expected_revision=None, device="cpu", checkpoint=None,
        resume=False, max_steps=None)
    values.update(changes)
    return argparse.Namespace(**values)


def public_task():
    random_generator = np.random.default_rng(7)
    frame = pd.DataFrame({f"feature_{column}": random_generator.normal(size=96) for column in range(5)})
    frame["category"] = pd.Series(np.resize(["a", "b"], 96), dtype=object)
    frame.loc[::7, "feature_1"] = np.nan
    labels = np.arange(96) % 2
    return OpenMLTaskData(363613, 91, "fixture", "binary", 2,
        frame.iloc[:72].reset_index(drop=True), labels[:72],
        frame.iloc[72:].reset_index(drop=True), labels[72:], "published-split")


def entry_for(root):
    return runner.make_entry(public_task(), runner.settings(args_for(root)))


def test_partition_roles_are_disjoint_class_complete_and_step_deterministic(tmp_path):
    entry, test = entry_for(tmp_path)
    fitting_rows = set(entry["fitting"]["row_ids"].tolist())
    validation_rows = set(entry["validation"]["row_ids"].tolist())
    assert not fitting_rows & validation_rows
    assert not (fitting_rows | validation_rows) & set(test["row_ids"].tolist())
    fixed_first = runner.training_episode(entry, "fixed", 1)
    assert_nested_equal(fixed_first, runner.training_episode(entry, "fixed", 9))
    resampled_first = runner.training_episode(entry, "resampled", 1)
    assert_nested_equal(resampled_first, runner.training_episode(entry, "resampled", 1))
    assert not torch.equal(resampled_first["query_indices"], runner.training_episode(entry, "resampled", 2)["query_indices"])
    for episode in (fixed_first, resampled_first):
        context_rows, query_rows = set(episode["context_indices"].tolist()), set(episode["query_indices"].tolist())
        assert not context_rows & query_rows
        assert context_rows | query_rows == fitting_rows
        assert episode["n_context"] == 36 and episode["n_query"] == 12
        assert episode["y_context"].unique().numel() == episode["y_query"].unique().numel() == 2
    validation_episode = runner.prediction_episode(entry, entry["validation"])
    test_episode = runner.prediction_episode(entry, test)
    for key in ("x_context", "y_context", "context_missing", "context_indices"):
        torch.testing.assert_close(validation_episode[key], test_episode[key], rtol=0, atol=0)


def test_encoding_ignores_validation_test_features_and_labels(tmp_path):
    task = public_task()
    specification = runner.settings(args_for(tmp_path))
    original, _ = runner.make_entry(task, specification)
    validation_positions = original["validation"]["row_ids"].numpy()
    changed = copy.deepcopy(task)
    changed.x_train.loc[validation_positions, "feature_1"] = 1e6
    changed.x_test.loc[:, "feature_1"] = -1e7
    changed.y_test = 1 - changed.y_test
    actual, _ = runner.make_entry(changed, specification)
    assert_nested_equal(actual["fitting"], original["fitting"])
    assert_nested_equal(runner.training_episode(actual, "resampled", 3),
                        runner.training_episode(original, "resampled", 3))


@pytest.mark.parametrize("arm", runner.ARMS)
def test_resume_reproduces_maps_optimizer_and_query_exposure(tmp_path, arm):
    entry, _ = entry_for(tmp_path)
    manifest = dict(fingerprint="fixture")
    initial = runner.diagnostic.new_model("statistics")
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    runner.fit_arm(args_for(uninterrupted_root), tiny(), initial, manifest, entry, arm)
    runner.fit_arm(args_for(resumed_root, max_steps=1), tiny(), initial, manifest, entry, arm)
    runner.fit_arm(args_for(resumed_root, resume=True), tiny(), initial, manifest, entry, arm)
    for filename in ("state.pt", "selected.pt", "final.pt"):
        expected = torch.load(runner.task_folder(args_for(uninterrupted_root), 363613) / arm / filename, weights_only=True)
        actual = torch.load(runner.task_folder(args_for(resumed_root), 363613) / arm / filename, weights_only=True)
        assert_nested_equal(actual, expected)
    rows = read_rows(runner.task_folder(args_for(resumed_root), 363613) / arm / "validation.csv")
    assert [int(row["step"]) for row in rows] == [0, 1, 2, 3]
    assert all(parameter.grad is None for parameter in tiny().parameters())


def test_identity_scores_all_queries_with_same_context_and_stable_chunks(tmp_path):
    entry, test = entry_for(tmp_path)
    backbone = tiny()
    model = runner.diagnostic.direct_model(runner.diagnostic.new_model("statistics"),
        runner.prediction_episode(entry, entry["validation"]), torch.device("cpu"))
    baseline = runner.score_partition(backbone, None, entry, test, 7)
    identity = runner.score_partition(backbone, model, entry, test, 7)
    assert identity == baseline and identity["rows"] == len(test["labels"])
    full_query = runner.score_partition(backbone, None, entry, test, len(test["labels"]))
    assert baseline["nll"] == pytest.approx(full_query["nll"], abs=2e-6)


def test_test_scoring_requires_both_arms_locked(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    entry, _ = entry_for(tmp_path)
    manifest = dict(fingerprint="fixture", accepted=[dict(task_id=363613)])
    runner.fit_arm(args, tiny(), runner.diagnostic.new_model("statistics"), manifest, entry, "fixed")
    monkeypatch.setattr(runner, "score_partition", lambda *arguments: pytest.fail("test scoring must not start"))
    with pytest.raises(FileNotFoundError):
        runner.report(args, tiny(), runner.diagnostic.new_model("statistics"), manifest)


def test_manifest_rejects_changed_settings_or_bank(tmp_path):
    args = args_for(tmp_path)
    folder = runner.task_folder(args, 363613)
    entry, test = entry_for(tmp_path)
    runner.pilot.atomic_save(folder / "fit_validation.pt", entry)
    runner.pilot.atomic_save(folder / "test.pt", test)
    record = dict(task_id=363613, fitting_sha256=runner.pilot.hash_file(folder / "fit_validation.pt"),
                  test_sha256=runner.pilot.hash_file(folder / "test.pt"))
    manifest = dict(settings=runner.settings(args), accepted=[record], excluded=[])
    manifest["fingerprint"] = runner.previous.digest(manifest)
    runner.pilot.json_write(tmp_path / "manifest.json", manifest)
    assert runner.checked_manifest(args) == manifest
    with pytest.raises(ValueError, match="settings/code"):
        runner.checked_manifest(args_for(tmp_path, steps=4))
    (folder / "test.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="test.pt changed"):
        runner.checked_manifest(args)
