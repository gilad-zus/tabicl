from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_openml_regression_confirmation_bank import (
    _dataset_ids_for_task_ids,
    _task_ids_from_exclusion_file,
    audit_candidate_task,
    select_distinct_regression_candidates,
)
from scripts.direct_spline_openml_lite import _task_ids_from_file


def test_task_id_file_accepts_frozen_bank_and_rejects_duplicate_ids(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"selected_task_ids": [30, 10, 20]}), encoding="utf-8")
    assert _task_ids_from_file(bank) == [30, 10, 20]

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("30\n10\n30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _task_ids_from_file(duplicate)

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"selected_task_ids": [30], "is_complete": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _task_ids_from_file(incomplete)


def test_exclusion_file_reads_the_prior_experiment_manifest(tmp_path):
    manifest = tmp_path / "experiment_manifest.json"
    manifest.write_text(
        json.dumps({"immutable_run": {"data_source": {"task_ids": [11, 12, 13]}}}), encoding="utf-8"
    )
    assert _task_ids_from_exclusion_file(manifest) == [11, 12, 13]


def test_exclusion_dataset_lookup_is_cross_problem_and_metadata_only(monkeypatch):
    calls = []

    def get_task(task_id):
        calls.append(task_id)
        return SimpleNamespace(dataset_id={11: 101, 12: 102, 13: 101}[task_id])

    fake_openml = types.SimpleNamespace(tasks=types.SimpleNamespace(get_task=get_task))
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    assert _dataset_ids_for_task_ids({13, 11, 12}) == {101, 102}
    assert calls == [11, 12, 13]


def test_candidate_selection_excludes_existing_and_deduplicates_datasets():
    records = [
        {"tid": 1, "did": 101, "name": "already-used-task", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
        {"tid": 2, "did": 102, "name": "already-used-dataset", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
        {"tid": 3, "did": 103, "name": "first-version", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
        {"tid": 4, "did": 103, "name": "other-split-same-dataset", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
        {"tid": 5, "did": 105, "name": "too-small", "status": "active", "NumberOfInstances": 99, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
        {"tid": 6, "did": 106, "name": "not-numeric", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 0},
        {"tid": 7, "did": 107, "name": "kept", "status": "active", "NumberOfInstances": 1000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8},
    ]

    selected, rejected = select_distinct_regression_candidates(
        records,
        excluded_task_ids={1},
        excluded_dataset_ids={102},
        min_total_rows=600,
        max_total_rows=18_000,
        max_features=200,
        selection_seed=17,
    )

    selected_ids = {item["task_id"] for item in selected}
    assert {7} <= selected_ids
    assert len({item["dataset_id"] for item in selected}) == len(selected)
    assert len(selected_ids.intersection({3, 4})) == 1
    assert rejected["tabarena_task"] == 1
    assert rejected["tabarena_dataset"] == 1
    assert rejected["too_few_total_rows"] == 1
    assert rejected["listed_no_numerical_features"] == 1
    assert rejected["duplicate_dataset_task"] == 1


def test_split_audit_requires_a_trainable_numerical_feature_and_never_calls_test_metric():
    candidate = {
        "task_id": 77,
        "dataset_id": 88,
        "listed_dataset_name": "candidate",
        "listed_status": "active",
        "listed_total_rows": 900,
        "listed_features": 3,
        "listed_numerical_features": 2,
        "listed_categorical_features": 1,
        "selection_key": "frozen-rank",
    }
    fake_task = SimpleNamespace(
        task_id=77,
        dataset_id=88,
        dataset_name="candidate",
        problem_type="regression",
        y_train=np.array([0.0, 1.0, 0.0, 1.0] * 125),
        y_test=np.array(["intentionally", "not", "inspected"], dtype=object),
        x_train=pd.DataFrame({"useful": np.tile([0.0, 1.0], 250), "constant": 1.0, "category": ["a", "b"] * 250}),
        x_test=pd.DataFrame({"useful": [0.0, 1.0, 1.0], "constant": 1.0, "category": ["a", "b", "a"]}),
        outer_split_hash="split-hash",
    )

    def loader(task_id, **kwargs):
        assert task_id == 77
        assert kwargs == {"outer_repeat": 0, "outer_fold": 0, "outer_sample": 0}
        return fake_task

    audited = audit_candidate_task(
        candidate,
        outer_repeat=0,
        outer_fold=0,
        outer_sample=0,
        min_outer_train_rows=400,
        min_outer_test_rows=3,
        max_outer_train_rows=12_000,
        max_features=200,
        task_loader=loader,
    )

    assert audited["task_id"] == 77
    assert audited["n_trainable_raw_numerical_features"] == 1
    assert audited["trainable_raw_numerical_columns"] == ["useful"]
