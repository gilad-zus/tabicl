import json

import torch

from scripts.direct_spline_openml_lite import (
    _event_reporter,
    _is_cuda_out_of_memory,
    _persisted_cuda_oom_skips,
)


def test_cuda_oom_detection_does_not_swallow_unrelated_runtime_errors():
    assert _is_cuda_out_of_memory(torch.OutOfMemoryError("CUDA capacity exhausted"))
    assert _is_cuda_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not _is_cuda_out_of_memory(RuntimeError("identity parity failed"))


def test_persisted_cuda_oom_skips_recover_only_resource_failures(tmp_path):
    oom = {
        "task_id": 363624,
        "dataset_name": "California-Housing-Classification",
        "reason": "cuda_out_of_memory",
        "stage": "config_D",
    }
    (tmp_path / "run_progress.json").write_text(
        json.dumps(
            {
                "completed_task_ids": [363612],
                "skipped_tasks": [
                    {"task_id": 363616, "reason": "n_features_exceeds_max_features"},
                    oom,
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _persisted_cuda_oom_skips(tmp_path) == {363624: oom}


def test_cuda_oom_skip_event_is_logged_clearly(tmp_path, capsys):
    report = _event_reporter(tmp_path / "progress.jsonl")
    report(
        {
            "event": "task_skipped",
            "task_id": 363624,
            "dataset_name": "California-Housing-Classification",
            "reason": "cuda_out_of_memory",
            "stage": "config_D",
        }
    )

    assert "skipped after CUDA OOM during config_D; continuing" in capsys.readouterr().out
    record = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8"))
    assert record["reason"] == "cuda_out_of_memory"
