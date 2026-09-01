from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_openml_binary_confirmation_bank import audit_binary_candidate_task
from scripts.build_openml_regression_confirmation_bank import select_distinct_openml_candidates


def _candidate() -> dict:
    return {
        "task_id": 77,
        "dataset_id": 88,
        "listed_dataset_name": "candidate",
        "listed_status": "active",
        "listed_total_rows": 900,
        "listed_features": 3,
        "listed_numerical_features": 2,
        "listed_categorical_features": 1,
        "listed_classes": 2,
        "selection_key": "frozen-rank",
    }


def _binary_task(*, counts: tuple[int, int] = (130, 140)):
    labels = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(counts)]
    )
    return SimpleNamespace(
        task_id=77,
        dataset_id=88,
        dataset_name="candidate",
        problem_type="binary",
        n_classes=2,
        y_train=labels,
        y_test=np.asarray(["outer", "labels", "not", "scored"], dtype=object),
        x_train=pd.DataFrame(
            {
                "useful": np.arange(labels.size, dtype=float),
                "constant": 1.0,
                "category": np.resize(np.asarray(["a", "b"]), labels.size),
            }
        ),
        x_test=pd.DataFrame(
            {
                "useful": [0.0, 1.0, 2.0, 3.0],
                "constant": 1.0,
                "category": ["a", "b", "a", "b"],
            }
        ),
        outer_split_hash="split-hash",
    )


def test_metadata_selector_keeps_only_binary_tasks():
    records = [
        {"tid": 1, "did": 101, "name": "binary", "status": "active", "NumberOfInstances": 1_000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8, "NumberOfClasses": 2},
        {"tid": 2, "did": 102, "name": "multiclass", "status": "active", "NumberOfInstances": 1_000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8, "NumberOfClasses": 3},
        {"tid": 3, "did": 103, "name": "one-class", "status": "active", "NumberOfInstances": 1_000, "NumberOfFeatures": 8, "NumberOfNumericFeatures": 8, "NumberOfClasses": 1},
    ]

    selected, rejected = select_distinct_openml_candidates(
        records,
        excluded_task_ids=set(),
        excluded_dataset_ids=set(),
        min_total_rows=600,
        max_total_rows=18_000,
        max_features=200,
        selection_seed=17,
        selection_namespace="binary-test",
        min_listed_classes=2,
        max_listed_classes=2,
    )

    assert [item["task_id"] for item in selected] == [1]
    assert rejected["too_few_listed_classes"] == 1
    assert rejected["too_many_listed_classes"] == 1


def test_binary_split_audit_requires_eight_bag_class_support():
    audited = audit_binary_candidate_task(
        _candidate(),
        outer_repeat=0,
        outer_fold=0,
        outer_sample=0,
        min_outer_train_rows=200,
        min_outer_test_rows=4,
        max_outer_train_rows=12_000,
        max_features=200,
        min_outer_train_class_rows=8,
        task_loader=lambda *_args, **_kwargs: _binary_task(),
    )

    assert audited["problem_type"] == "binary"
    assert audited["n_classes"] == 2
    assert audited["min_outer_train_class_count"] == 130
    assert audited["n_trainable_raw_numerical_features"] == 1

    with pytest.raises(ValueError, match="smallest outer-training class has 7 rows"):
        audit_binary_candidate_task(
            _candidate(),
            outer_repeat=0,
            outer_fold=0,
            outer_sample=0,
            min_outer_train_rows=200,
            min_outer_test_rows=4,
            max_outer_train_rows=12_000,
            max_features=200,
            min_outer_train_class_rows=8,
            task_loader=lambda *_args, **_kwargs: _binary_task(counts=(7, 263)),
        )
