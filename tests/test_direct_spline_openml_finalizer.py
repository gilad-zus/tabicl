import numpy as np

from scripts.finalize_direct_spline_openml_standard import rebuild_standard_config_from_bags
from tabicl._experiments.direct_spline_openml import (
    BagPredictions,
    OpenMLTaskData,
    _bag_splits,
    _config_dir,
    _save_bag,
    _seed,
)


def test_finalizer_rebuilds_standard_config_from_saved_bags_only(tmp_path):
    task = OpenMLTaskData(
        task_id=77,
        dataset_id=88,
        dataset_name="saved-bags-only",
        problem_type="binary",
        n_classes=2,
        x_train=np.zeros((8, 2)),
        y_train=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        x_test=np.zeros((3, 2)),
        y_test=np.asarray([0, 1, 0]),
        outer_split_hash="split",
    )
    label = "D"
    config = {"guard_relative_improvement": 0.005, "max_context_rows": None}
    fingerprint = "immutable"
    config_dir = _config_dir(tmp_path, task, label)
    test_prediction = np.tile(np.asarray([[0.7, 0.3]]), (len(task.y_test), 1))
    for bag, (_, validation_indices) in enumerate(
        _bag_splits(task, requested_bags=2, seed=_seed(0, task.task_id, 0))
    ):
        validation_prediction = np.tile(np.asarray([[0.7, 0.3]]), (len(validation_indices), 1))
        _save_bag(
            config_dir / f"bag_{bag}.npz",
            BagPredictions(
                validation_indices=validation_indices,
                identity_validation=validation_prediction,
                adapted_validation=validation_prediction,
                guarded_validation=validation_prediction,
                identity_test=test_prediction,
                adapted_test=test_prediction,
                guarded_test=test_prediction,
                metadata={
                    "bag": bag,
                    "run_fingerprint_hash": fingerprint,
                    "guard_selected_adapted": False,
                    "adapter_has_valid_learned_checkpoint": False,
                    "train_seconds": 1.0,
                    "peak_allocated_gib": 0.0,
                    "identity_parity_max_abs_validation": 0.0,
                    "identity_parity_max_abs_test": 0.0,
                },
            ),
        )

    summary = rebuild_standard_config_from_bags(
        task=task,
        label=label,
        config=config,
        output_dir=tmp_path,
        requested_bags=2,
        protocol_seed=0,
        run_fingerprint_hash=fingerprint,
        checkpoint_metadata={"path": "frozen.ckpt", "sha256": "abc", "bytes": 1},
    )

    assert summary["effective_bags"] == 2
    assert summary["run_fingerprint_hash"] == fingerprint
    with np.load(config_dir / "config_predictions.npz", allow_pickle=False) as predictions:
        np.testing.assert_allclose(predictions["identity_test"], test_prediction)
