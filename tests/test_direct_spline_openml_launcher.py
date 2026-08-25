import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import scripts.direct_spline_openml_lite as launcher
from scripts.direct_spline_openml_lite import (
    EXPERIMENT_SEMANTICS_VERSION,
    _checkpoint_fingerprint,
    _configs,
    _event_reporter,
    _experiment_source_hashes,
    _is_cuda_out_of_memory,
    _persisted_cuda_oom_skips,
    _persisted_task_skips,
    _repair_interrupted_config_summaries,
    _resolve_execution_environment,
    _restore_run_checkpoints,
    _same_equivalent_hardware_resume_semantics,
    _same_experimental_semantics,
    _validate,
    _write_all_skipped_summary,
    _write_run_progress,
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


def test_interrupted_resume_progress_preserves_later_prior_skips(tmp_path):
    prior_skips = [
        {"task_id": 363624, "dataset_name": "large-a", "reason": "cuda_out_of_memory"},
        {"task_id": 363628, "dataset_name": "wide", "reason": "n_features_exceeds_max_features"},
        {"task_id": 363631, "dataset_name": "large-b", "reason": "cuda_out_of_memory"},
    ]
    (tmp_path / "run_progress.json").write_text(
        json.dumps({"completed_task_ids": [363612], "skipped_tasks": prior_skips}),
        encoding="utf-8",
    )

    recovered = _persisted_task_skips(tmp_path)
    _write_run_progress(
        tmp_path,
        task_summaries=[{"task_id": 363612}],
        skipped_tasks=list(recovered.values()),
    )

    rewritten = json.loads((tmp_path / "run_progress.json").read_text(encoding="utf-8"))
    assert rewritten["skipped_tasks"] == prior_skips
    assert set(_persisted_cuda_oom_skips(tmp_path)) == {363624, 363631}


def test_all_skipped_run_writes_terminal_summary_instead_of_raising(tmp_path):
    skipped = [{"task_id": 363624, "reason": "cuda_out_of_memory"}]

    result = _write_all_skipped_summary(
        tmp_path,
        skipped_tasks=skipped,
        task_eligibility={"max_features": 100},
    )

    assert result["status"] == "no_completed_tasks"
    assert result["paired_results"] == {}
    assert result["n_skipped_tasks"] == 1
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == result


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


def test_resume_quarantines_only_malformed_config_summaries(tmp_path):
    config_dir = tmp_path / "raw" / "task_363612_example" / "config_D"
    config_dir.mkdir(parents=True)
    broken = config_dir / "config_summary.json"
    broken.write_text("", encoding="utf-8")
    valid_dir = tmp_path / "raw" / "task_363613_example" / "config_D"
    valid_dir.mkdir(parents=True)
    valid = valid_dir / "config_summary.json"
    valid_payload = {"run_fingerprint_hash": "fixed", "validation": {}}
    valid.write_text(json.dumps(valid_payload), encoding="utf-8")

    recovered = _repair_interrupted_config_summaries(tmp_path)

    assert len(recovered) == 1
    assert not broken.exists()
    quarantine = Path(recovered[0]["quarantine"])
    assert quarantine.read_text(encoding="utf-8") == ""
    assert json.loads(valid.read_text(encoding="utf-8")) == valid_payload
    audit = [
        json.loads(line)
        for line in (tmp_path / "artifact_recoveries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert audit == recovered


def test_compatible_resume_ignores_only_revision_metadata_not_source_or_semantics():
    previous = {
        "experiment_semantics_version": EXPERIMENT_SEMANTICS_VERSION,
        "repository_revision": "old",
        "source_sha256": {"old.py": "a"},
        "pipeline": "standard",
    }
    current = {
        **previous,
        "repository_revision": "new",
    }
    assert _same_experimental_semantics(previous, current)

    current["source_sha256"] = {"new.py": "b"}
    assert not _same_experimental_semantics(previous, current)
    current["source_sha256"] = previous["source_sha256"]
    current["experiment_semantics_version"] += 1
    assert not _same_experimental_semantics(previous, current)
    legacy = {key: value for key, value in previous.items() if key != "experiment_semantics_version"}
    assert not _same_experimental_semantics(legacy, previous)


def test_equivalent_hardware_resume_ignores_only_scheduler_allocation_identity():
    previous = {
        "experiment_semantics_version": EXPERIMENT_SEMANTICS_VERSION,
        "repository_revision": "old",
        "source_sha256": {
            "scripts/direct_spline_openml_lite.py": "old-launcher",
            "src/tabicl/_model/tabicl.py": "frozen-model",
        },
        "pipeline": "standard",
        "configs": [{"adapter_steps": 500}],
        "execution_environment": {
            "requested_device": "cuda:0",
            "resolved_device": "cuda:0",
            "resolution_status": "resolved",
            "resolution_error": None,
            "cuda": {
                "available": True,
                "torch_cuda_runtime_version": "13.0",
                "cudnn_version": 90100,
                "visible_device_count": 1,
                "selected_hardware": {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "total_memory_bytes": 80_000,
                    "compute_capability": [8, 0],
                    "multi_processor_count": 108,
                    "uuid": "GPU-old",
                },
            },
            "python_hash_seed": {"value": "0", "fixed_before_process_start": True},
            "numeric_environment": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "precision": {"float32_matmul_precision": "high"},
        },
    }
    current = {
        **previous,
        "repository_revision": "new",
        "source_sha256": {
            "scripts/direct_spline_openml_lite.py": "new-launcher",
            "src/tabicl/_model/tabicl.py": "frozen-model",
        },
        "execution_environment": {
            **previous["execution_environment"],
            "cuda": {
                **previous["execution_environment"]["cuda"],
                "visible_device_count": 4,
                "selected_hardware": {
                    **previous["execution_environment"]["cuda"]["selected_hardware"],
                    "index": 3,
                    "uuid": "GPU-new",
                },
            },
        },
    }

    assert _same_equivalent_hardware_resume_semantics(previous, current)

    current["execution_environment"]["cuda"]["selected_hardware"]["compute_capability"] = [9, 0]
    assert not _same_equivalent_hardware_resume_semantics(previous, current)
    current["execution_environment"]["cuda"]["selected_hardware"]["compute_capability"] = [8, 0]
    current["source_sha256"]["src/tabicl/_model/tabicl.py"] = "changed-model"
    assert not _same_equivalent_hardware_resume_semantics(previous, current)


def test_source_hashes_cover_public_model_and_spline_implementation():
    hashes = _experiment_source_hashes()
    expected = {
        "scripts/direct_spline_openml_lite.py",
        "scripts/direct_spline_openml_standard.py",
        "src/tabicl/__init__.py",
        "src/tabicl/_experiments/direct_spline_openml.py",
        "src/tabicl/_experiments/direct_spline_openml_standard.py",
        "src/tabicl/_experiments/tabarena_direct_spline_protocol.py",
        "src/tabicl/_sklearn/classifier.py",
        "src/tabicl/_sklearn/preprocessing.py",
        "src/tabicl/_sklearn/regressor.py",
        "src/tabicl/_model/tabicl.py",
        "src/tabicl/_hyperspline/module.py",
    }
    assert expected <= hashes.keys()
    assert all(len(digest) == 64 for digest in hashes.values())


def test_resume_restores_manifest_checkpoints_without_openml_metadata(monkeypatch, tmp_path):
    checkpoint = tmp_path / "classifier.ckpt"
    checkpoint.write_bytes(b"immutable weights")
    fingerprint = _checkpoint_fingerprint(checkpoint)
    previous = {
        "immutable_run": {
            "classifier_checkpoint_argument": str(checkpoint.resolve()),
            "regressor_checkpoint_argument": None,
            "checkpoint_fingerprints": {
                "classifier": fingerprint,
                "regressor": None,
            },
        }
    }
    args = Namespace(classifier_checkpoint=None, regressor_checkpoint=None)
    monkeypatch.setattr(
        launcher,
        "_required_checkpoint_kinds",
        lambda _task_ids: (_ for _ in ()).throw(AssertionError("OpenML metadata was queried")),
    )

    restored = _restore_run_checkpoints(args, previous)

    assert restored == previous["immutable_run"]["checkpoint_fingerprints"]
    assert args.classifier_checkpoint == checkpoint
    assert args.regressor_checkpoint is None


def test_dry_run_records_unavailable_cuda_instead_of_failing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device, record = _resolve_execution_environment("cuda:7", dry_run=True)

    assert device is None
    assert record["requested_device"] == "cuda:7"
    assert record["resolved_device"] is None
    assert record["resolution_status"] == "cuda_unavailable_dry_run"
    assert record["cuda"]["available"] is False
    assert "float32_matmul_precision" in record["precision"]
    assert "value" in record["python_hash_seed"]


def test_dry_run_prints_preview_without_writing_manifest(monkeypatch, tmp_path):
    output_dir = tmp_path / "preview"
    monkeypatch.setattr(
        "sys.argv",
        [
            "direct_spline_openml_standard.py",
            "--output-dir",
            str(output_dir),
            "--task-id",
            "363621",
            "--pipeline",
            "standard",
            "--device",
            "cpu",
            "--dry-run",
        ],
    )

    launcher.main(default_pipeline="standard", required_pipeline="standard")

    assert not output_dir.exists()


def test_retry_cuda_oom_skips_requires_resume():
    args = Namespace(
        bags=8,
        n_random_configs=0,
        bootstrap_rounds=1,
        max_tasks=None,
        max_features=None,
        context_cap=0,
        train_context_rows=0,
        outer_repeat=0,
        outer_fold=0,
        outer_sample=0,
        allow_compatible_code_resume=False,
        retry_cuda_oom_skips=True,
        resume=False,
    )

    try:
        _validate(args)
    except ValueError as error:
        assert str(error) == "--retry-cuda-oom-skips requires --resume"
    else:  # pragma: no cover - clearer failure than a bare assertion
        raise AssertionError("_validate accepted an OOM retry without --resume")


def test_zero_train_context_rows_maps_to_all_available_rows():
    args = Namespace(
        pipeline="standard",
        context_cap=0,
        train_context_rows=0,
        n_random_configs=1,
        tuning_seed=123,
    )

    _labels, configs = _configs(args)

    assert all(config["train_context_rows"] is None for config in configs)


def test_zero_train_context_rows_matches_an_explicit_deployment_cap():
    args = Namespace(
        pipeline="standard",
        context_cap=512,
        train_context_rows=0,
        n_random_configs=1,
        tuning_seed=123,
    )

    _labels, configs = _configs(args)

    assert all(config["train_context_rows"] == 512 for config in configs)


def test_standard_launcher_forwards_explicit_adapter_schedule(monkeypatch, tmp_path):
    from scripts.direct_spline_openml_lite import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "direct_spline_openml_standard.py",
            "--output-dir",
            str(tmp_path),
            "--adapter-steps",
            "500",
            "--adapter-patience",
            "10",
            "--validation-interval",
            "10",
        ],
    )

    args = parse_args(default_pipeline="standard", required_pipeline="standard")
    _validate(args)
    labels, configs = _configs(args)

    assert labels == ["D"]
    assert configs[0]["adapter_steps"] == 500
    assert configs[0]["adapter_patience"] == 10
    assert configs[0]["validation_interval"] == 10


def test_checkpoint_audit_launcher_defaults_to_a_500_step_curve(monkeypatch, tmp_path):
    from scripts.direct_spline_openml_lite import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "direct_spline_openml_full_refit_checkpoint_audit.py",
            "--output-dir",
            str(tmp_path),
            "--pipeline",
            "standard",
        ],
    )

    args = parse_args(
        default_pipeline="standard",
        required_pipeline="standard",
        checkpoint_audit=True,
    )
    _validate(args, full_context_refit=True, checkpoint_audit=True)
    labels, configs = _configs(args)

    assert args.checkpoint_steps == (0, 25, 50, 100, 200, 300, 500)
    assert args.adapter_steps == 500
    assert labels == ["D"]
    assert configs[0]["adapter_steps"] == 500


def test_standard_launcher_pipeline_can_be_enforced(monkeypatch, tmp_path):
    from scripts.direct_spline_openml_lite import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["direct_spline_openml_standard.py", "--output-dir", str(tmp_path), "--pipeline", "lite"],
    )
    with pytest.raises(SystemExit):
        parse_args(default_pipeline="standard", required_pipeline="standard")


def test_identity_view_parity_failure_event_uses_current_schema(tmp_path, capsys):
    report = _event_reporter(tmp_path / "progress.jsonl")
    report(
        {
            "event": "identity_view_parity_failed",
            "task_id": 363631,
            "bag": 2,
            "split": "validation",
            "query_rows": 819,
            "diagnostics": {},
        }
    )

    output = capsys.readouterr().out
    assert "validation identity input-view parity failed on 819 query rows" in output
    assert "public repeat drift" not in output
