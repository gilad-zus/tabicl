"""Inference-only, one-column curvature removal from saved staged splines.

Project unmixed outputs onto a line in the adapter's arctan coordinate using T
only. Keep the saved mixer, preprocessing, A/B states and full-context protocol.
OOF selects at most one column (or no intervention); test never selects anything.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from direct_spline_openml_support_audit import _find_source_cases, _load_json
from direct_spline_openml_crossfit_blend import _load_source_task, _validate_case, _source_standard_prediction
from direct_spline_openml_crossfit_context_expansion import _load_bag, _append_prediction
from tabicl._experiments.direct_spline_openml import (
    _bag_splits, _metric_bundle, _safe_name, _seed, load_frozen_backbone,
)
from tabicl._experiments.direct_spline_openml_standard import _AdapterSet, _fit_standard_bag, _make_adapters
from tabicl._hyperspline.module import DirectSplineTransform
from tabicl._hyperspline.bspline import greville_abscissae


TASKS = {"multiclass": [4602, 75158, 167186, 361539], "regression": [4999, 5042, 362096, 362343]}
SOURCES = {
    "multiclass": "openml_direct_spline_adaptive_retouche/multiclass_seed20260828",
    "regression": "openml_direct_spline_regression_confirmation/full_D_500_v1",
}
STAGED = "openml_direct_spline_staged_curvature_ablation/dev8_seed20260915"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bind(path, value):
    if path.exists():
        if _canonical(_load_json(path, label="audit provenance")) != _canonical(value):
            raise ValueError(f"provenance changed: {path}")
    else:
        _write(path, value)


def _runtime(device):
    stable = {"python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "device_type": device.type, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    if device.type == "cuda":
        stable.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    return {"stable": stable, "host": platform.node(), "device": str(device)}


def _validate_resume(previous, semantic, runtime, *, resume, equivalent):
    if not resume or _canonical(previous["semantic"]) != _canonical(semantic):
        raise ValueError("resume requires identical source and inference semantics")
    if previous["runtime"] != runtime and not (equivalent and previous["runtime"]["stable"] == runtime["stable"]):
        raise ValueError("resume runtime differs; equivalent stable hardware requires --allow-equivalent-hardware-resume")


def _coordinate(adapter, x):
    if not isinstance(adapter, DirectSplineTransform) or not adapter.direct_spline_output or adapter.coordinate_mapping != "arctan":
        raise ValueError("audit requires a direct-output arctan DirectSpline")
    params = adapter.parameters_for_transform()
    _, _, radius = adapter._location_scale_range()
    z = (x.float() - params.location.unsqueeze(1)) / params.scale.unsqueeze(1)
    return (2.0 / torch.pi) * torch.atan(torch.pi * z / (2.0 * radius.unsqueeze(1)))


def _fit_lines(u, output):
    """Empirical least squares, independent per column; no labels or query rows."""
    if u.shape != output.shape or u.ndim != 3 or u.shape[1] == 0:
        raise ValueError("projection requires equally shaped, nonempty [batch, rows, columns] tensors")
    if not torch.isfinite(u).all() or not torch.isfinite(output).all():
        raise ValueError("nonfinite training coordinates or spline outputs")
    u64, y64 = u.double(), output.double()
    mean_u, mean_y = u64.mean(1), y64.mean(1)
    centered = u64 - mean_u.unsqueeze(1)
    variance = centered.square().mean(1)
    covariance = (centered * (y64 - mean_y.unsqueeze(1))).mean(1)
    slope = torch.where(variance > 1e-12, covariance / variance.clamp_min(1e-12), torch.zeros_like(variance))
    intercept = mean_y - slope * mean_u
    residual = y64 - (slope.unsqueeze(1) * u64 + intercept.unsqueeze(1))
    return slope.float(), intercept.float(), residual.square().mean(1).sqrt().float()


class _ProjectedColumn(nn.Module):
    """Replace selected unmixed functions, then apply the original fixed mixer."""
    def __init__(self, original, slope, intercept, columns):
        super().__init__()
        self.original = original
        self.register_buffer("slope", slope.detach().clone())
        self.register_buffer("intercept", intercept.detach().clone())
        self.columns = tuple(int(column) for column in columns)
        if any(column < 0 or column >= slope.shape[-1] for column in self.columns):
            raise ValueError("projection column out of range")
        controls = original.parameters_for_transform().control_points
        straight = greville_abscissae(original.knots_for_transform(), original.degree, controls.shape[-1])
        curved = torch.any(controls != straight, dim=-1).any(dim=0)
        # Removing curvature from an already straight column must be bit-exact.
        self.columns = tuple(column for column in self.columns if bool(curved[column]))

    def effective_mixing_matrix(self):
        return self.original.effective_mixing_matrix()

    def unmixed_transform(self, x):
        if not self.columns:
            return self.original.unmixed_transform(x)
        output = self.original.unmixed_transform(x).clone()
        line = self.slope.unsqueeze(1) * _coordinate(self.original, x) + self.intercept.unsqueeze(1)
        output[..., list(self.columns)] = line[..., list(self.columns)]
        return output

    def transform(self, x):
        # Preserve the exact unmodified path, including its floating point order.
        if not self.columns:
            return self.original.transform(x)
        output = self.unmixed_transform(x)
        mixer = self.effective_mixing_matrix()
        if mixer is not None:
            output = output + torch.matmul(output, mixer)
        return output.to(x.dtype)


@torch.no_grad()
def _project(bundle, adapters, device):
    coefficients, diagnostics = {}, {}
    for method, preprocessor in bundle.estimator.ensemble_generator_.preprocessors_.items():
        adapter = adapters.for_method(method)
        values = torch.as_tensor(preprocessor.X_transformed_[:, bundle.numerical_indices], device=device, dtype=torch.float32).unsqueeze(0)
        slope, intercept, residual = _fit_lines(_coordinate(adapter, values), adapter.unmixed_transform(values))
        coefficients[method] = (slope, intercept)
        diagnostics[method] = {"slope": slope.cpu().tolist(), "intercept": intercept.cpu().tolist(),
                               "curvature_rms_on_T": residual.cpu().tolist(), "fit_rows": values.shape[1]}
    return coefficients, diagnostics


def _intervene(adapters, coefficients, columns):
    return _AdapterSet(OrderedDict(
        (method, _ProjectedColumn(adapters.for_method(method), *values, columns))
        for method, values in coefficients.items()
    ))


def _column_map(bundle):
    """Use original input positions, not bag-dependent filtered-column indices."""
    encoder = bundle.estimator.X_encoder_
    encoded_to_input = dict(zip(encoder.numeric_output_positions_, encoder.numeric_input_positions_))
    kept = np.flatnonzero(bundle.estimator.ensemble_generator_.unique_filter_.features_to_keep_)
    return {int(encoded_to_input[kept[filtered]]): local
            for local, filtered in enumerate(bundle.numerical_indices)}


def _validate_splits(fit, validation, a, b, n_train):
    arrays = [np.asarray(value, dtype=int) for value in (fit, validation, a, b)]
    for value in arrays:
        if value.ndim != 1 or not value.size or len(np.unique(value)) != len(value) or np.any(value < 0) or np.any(value >= n_train):
            raise ValueError("invalid or duplicated source split indices")
    fit, validation, a, b = arrays
    if np.intersect1d(a, b).size or not np.array_equal(np.sort(np.r_[a, b]), np.sort(validation)):
        raise ValueError("A/B are not a disjoint partition of validation")
    if np.intersect1d(fit, validation).size or not np.array_equal(np.sort(np.r_[fit, validation]), np.arange(n_train)):
        raise ValueError("T/validation are not a disjoint partition of outer training")


def _check_prediction(value, shape):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid audit prediction: {value.shape}, expected {shape}")


def _replay(actual, expected, *, atol, rtol):
    _check_prediction(actual, expected.shape)
    difference = float(np.max(np.abs(np.asarray(actual) - expected), initial=0.0))
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"unchanged-checkpoint replay mismatch: max_abs={difference}; atol={atol}, rtol={rtol}. No intervention is accepted.")
    return difference


def _cached(path, fingerprint, oof_shape, test_shape):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            raise ValueError(f"prediction provenance changed: {path}")
        result = {key: np.array(data[key]) for key in ("oof", "test")}
    _check_prediction(result["oof"], oof_shape)
    _check_prediction(result["test"], test_shape)
    return result


def _save_predictions(path, fingerprint, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, fingerprint=np.asarray(fingerprint), **values)
    os.replace(temporary, path)


def _error(task, labels, predictions):
    error = float(_metric_bundle(task.problem_type, labels, predictions, task.n_classes)["benchmark_error"])
    if not np.isfinite(error) or error < 0:
        raise ValueError("nonfinite/negative benchmark error")
    return error


def _gain(reference, candidate):
    return {"error_reduction": reference - candidate,
            "relative_error_reduction": None if reference == 0 else (reference - candidate) / reference}


def _select(oof_errors):
    """No test argument; ties retain the untouched spline before any column."""
    allowed = {key: value for key, value in oof_errors.items() if key == "unchanged" or key.startswith("column_")}
    if "unchanged" not in allowed or not all(np.isfinite(value) for value in allowed.values()):
        raise ValueError("selection requires finite OOF errors and unchanged reference")
    best = "unchanged"
    for key in sorted(allowed):
        if allowed[key] < allowed[best] - 1e-12:
            best = key
    return best


def _task_path(root, task):
    return root / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _payload(path, task, bag, seed):
    value = torch.load(path, map_location="cpu", weights_only=True)
    provenance = value["provenance"]
    expected = {"task_id": task.task_id, "outer_split_hash": task.outer_split_hash, "bag": bag, "protocol_seed": seed}
    if any(provenance.get(key) != item for key, item in expected.items()):
        raise ValueError(f"adapter provenance mismatch: {path}")
    if set(value["checkpoints"]) != {"original_a", "original_b"}:
        raise ValueError("both selected A/B states are required")
    if any(not item["valid"] for item in value["checkpoints"].values()):
        raise ValueError("source contains an invalid selected checkpoint")
    return value


def _run_task(args, case, task, staged_dir, output, device, fingerprint):
    source = _task_path(staged_dir / "continued_spline", task)
    destination = _task_path(output, task)
    manifest = _load_json(staged_dir / "experiment_manifest.json", label="staged manifest")
    seed = int(manifest["protocol_seed"])
    splits = list(_bag_splits(task, requested_bags=int(manifest["bags"]), seed=_seed(seed, task.task_id, 0)))
    artifacts = [source / "task_provenance.json"]
    for bag in range(len(splits)):
        artifacts.extend((source / f"bag_{bag}.npz", source / f"bag_{bag}.adapters.pt"))
    provenance = {"fingerprint": fingerprint, "outer_split_hash": task.outer_split_hash,
                  "files": {str(path): _hash(path) for path in artifacts}}
    _bind(destination / "source_provenance.json", provenance)
    task_fingerprint = hashlib.sha256(_canonical(provenance).encode()).hexdigest()
    report_path = destination / "column_curvature_summary.json"
    if report_path.exists():
        return _load_json(report_path, label="completed task")
    source_provenance = _load_json(source / "task_provenance.json", label="source backbone")
    backbone, _, backbone_meta = load_frozen_backbone(problem_type=task.problem_type, device=device,
                                                    classifier_checkpoint=None, regressor_checkpoint=None)
    if backbone_meta["sha256"] != source_provenance["checkpoint"]["sha256"]:
        raise ValueError("backbone differs from saved adapter experiment")
    _bind(destination / "backbone.json", {"sha256": backbone_meta["sha256"]})
    # Dataset-level numeric IDs are stable even when a bag drops a constant column.
    # pandas handles categorical/nullable dtypes safely, unlike np.issubdtype.
    from pandas.api.types import is_numeric_dtype, is_bool_dtype
    column_ids = [i for i, dtype in enumerate(task.x_train.dtypes) if is_numeric_dtype(dtype) and not is_bool_dtype(dtype)]
    candidates = ["unchanged", *[f"column_{i}" for i in column_ids], "all_columns_line"]
    records = []
    for bag, (fit, validation) in enumerate(splits):
        saved = _load_bag(source / f"bag_{bag}.npz")
        if not np.array_equal(validation, saved.validation_indices):
            raise ValueError("reconstructed bag differs from persisted split")
        a, b = saved.selection_a_indices, saved.selection_b_indices
        _validate_splits(fit, validation, a, b, len(task.y_train))
        payload = _payload(source / f"bag_{bag}.adapters.pt", task, bag, seed)
        bundle = _fit_standard_bag(task=task, fit_indices=np.asarray(fit, dtype=int), config=payload["config"],
                                   protocol_seed=seed, bag=bag, backbone=backbone, device=device)
        adapters = _make_adapters(bundle, payload["config"], device)
        if adapters is None:
            raise ValueError("no numerical adapter to audit")
        mapping = _column_map(bundle)
        if not set(mapping).issubset(column_ids):
            raise ValueError("numeric column mapping changed")
        if list(bundle.numerical_indices) != payload["provenance"]["numerical_indices"]:
            raise ValueError("filtered numerical indices changed")
        if list(bundle.estimator.ensemble_generator_.preprocessors_) != payload["provenance"]["normalization_methods"]:
            raise ValueError("normalization branches changed")
        for selected, evaluation, appended in (("a", b, a), ("b", a, b)):
            adapters.load_state_dict(payload["checkpoints"][f"original_{selected}"]["state_dict"], strict=True)
            adapters.eval().requires_grad_(False)
            coefficients, diagnostics = _project(bundle, adapters, device)
            state_dir = destination / f"bag_{bag}" / f"selected_{selected}"
            projection_path = state_dir / "projection.json"
            if not projection_path.exists():
                _write(projection_path, {"input_to_local": mapping, "branches": diagnostics,
                                        "step": payload["checkpoints"][f"original_{selected}"]["step"]})
            opposite = "b" if selected == "a" else "a"
            expected_oof = getattr(saved, f"expanded_spline_selected_on_{selected}_selection_{opposite}")
            expected_test = getattr(saved, f"expanded_spline_selected_on_{selected}_test")
            query = task.x_train.iloc[evaluation].reset_index(drop=True)
            for position, candidate in enumerate(candidates):
                path = state_dir / f"{candidate}.npz"
                values = _cached(path, task_fingerprint, expected_oof.shape, expected_test.shape)
                if values is None:
                    print(f"task {task.task_id} bag {bag} selected {selected}: {position + 1}/{len(candidates)} {candidate}", flush=True)
                    if candidate == "unchanged":
                        chosen = adapters
                    else:
                        columns = list(mapping.values()) if candidate == "all_columns_line" else ([mapping[int(candidate[7:])]] if int(candidate[7:]) in mapping else [])
                        chosen = _intervene(adapters, coefficients, columns)
                    with torch.no_grad():
                        values = {
                            "oof": _append_prediction(bundle=bundle, task=task, query_x=query, appended_indices=appended, adapters=chosen, device=device),
                            "test": _append_prediction(bundle=bundle, task=task, query_x=task.x_test, appended_indices=validation, adapters=chosen, device=device),
                        }
                    _check_prediction(values["oof"], expected_oof.shape)
                    _check_prediction(values["test"], expected_test.shape)
                    if candidate == "unchanged":
                        replay = {split: _replay(values[split], expected, atol=args.replay_atol, rtol=args.replay_rtol)
                                  for split, expected in (("oof", expected_oof), ("test", expected_test))}
                        _write(state_dir / "replay.json", replay)
                    _save_predictions(path, task_fingerprint, values)
                records.append((candidate, evaluation, path, expected_oof.shape, expected_test.shape))
            del coefficients
        del adapters, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    oof_errors, predictions = {}, {}
    for candidate in candidates:
        parts = [(indices, _cached(path, task_fingerprint, oof_shape, test_shape))
                 for key, indices, path, oof_shape, test_shape in records if key == candidate]
        first = parts[0][1]["oof"]
        oof = np.full((len(task.y_train), *first.shape[1:]), np.nan)
        coverage = np.zeros(len(task.y_train), dtype=int)
        for indices, values in parts:
            oof[indices] = values["oof"]
            coverage[indices] += 1
        if not np.all(coverage == 1) or not np.isfinite(oof).all():
            raise ValueError("OOF rows must be covered exactly once")
        predictions[candidate] = np.mean([values["test"] for _, values in parts], axis=0)
        oof_errors[candidate] = _error(task, task.y_train, oof)
    selected = _select(oof_errors)
    # Persist the choice before any test label is scored.
    _bind(destination / "oof_selection.json", {"selected": selected, "oof_errors": oof_errors,
          "rule": "minimum pooled cross-fitted OOF error, unchanged wins ties; all-columns line excluded"})
    test_errors = {key: _error(task, task.y_test, prediction) for key, prediction in predictions.items()}
    reference_oof, reference_test = oof_errors["unchanged"], test_errors["unchanged"]
    rows = [{"candidate": key, "column_name": str(task.x_train.columns[int(key[7:])]) if key.startswith("column_") else None,
             "oof_error": oof_errors[key], "test_error": test_errors[key],
             "oof_removal_effect": _gain(reference_oof, oof_errors[key]),
             "test_removal_effect": _gain(reference_test, test_errors[key])} for key in candidates]
    column_rows = [row for row in rows if row["candidate"].startswith("column_")]
    counts = {}
    for split in ("oof", "test"):
        effects = [row[f"{split}_removal_effect"]["error_reduction"] for row in column_rows]
        counts[split] = {"removal_helps": sum(x > 1e-12 for x in effects), "removal_hurts": sum(x < -1e-12 for x in effects),
                         "ties": sum(abs(x) <= 1e-12 for x in effects)}
    transfer = {"helps_both": 0, "hurts_both": 0, "helps_oof_hurts_test": 0,
                "hurts_oof_helps_test": 0, "tie_on_either": 0}
    for row in column_rows:
        val, test = (row[f"{split}_removal_effect"]["error_reduction"] for split in ("oof", "test"))
        key = ("tie_on_either" if min(abs(val), abs(test)) <= 1e-12 else
               "helps_both" if val > 0 and test > 0 else "hurts_both" if val < 0 and test < 0 else
               "helps_oof_hurts_test" if val > 0 else "hurts_oof_helps_test")
        transfer[key] += 1
    ordinary = _source_standard_prediction(source_dir=case.source_dir, task=task)
    continued_line = next(row for row in json.loads((staged_dir / "continued_line/task_summaries.json").read_text(encoding="utf-8")) if row["task_id"] == task.task_id)
    report = {"task_id": task.task_id, "dataset_name": task.dataset_name, "problem_type": task.problem_type,
              "n_columns": len(column_ids), "n_bags": len(splits), "candidates": rows, "column_effect_counts": counts,
              "column_sign_transfer": transfer,
              "validation_selected_candidate": selected, "selected_test_effect": _gain(reference_test, test_errors[selected]),
              "ordinary_tabiclv2_test_error": None if ordinary is None else _error(task, task.y_test, ordinary),
              "matched_continued_line_test_error": continued_line["expanded_context"]["outer_test"]["raw_spline"]["benchmark_error"],
              "interpretation": "Positive removal effect means this intervention improves the frozen model. Column effects are conditional and nonadditive. OOF is a selection score, not an unbiased score for the selected intervention. No retraining; no test-based column choice."}
    _write(report_path, report)
    return report


def _prepare(args, kind, device):
    root = Path(__file__).resolve().parents[1]
    source = args.results_root / SOURCES[kind]
    staged = args.results_root / STAGED / kind
    # Hash all package inference code, plus the imported experiment helpers.
    code = [Path(__file__), *sorted((root / "src/tabicl").rglob("*.py"))]
    code += [root / "scripts" / name for name in ("direct_spline_openml_support_audit.py", "direct_spline_openml_crossfit_blend.py", "direct_spline_openml_crossfit_context_expansion.py")]
    semantic = {"schema": 1, "experiment": "saved staged K20 one-column curvature removal",
                "source_dir": str(source), "staged_dir": str(staged), "task_ids": TASKS[kind],
                "projection": "least-squares unmixed output on arctan u, fitted only on all T coordinates per branch",
                "selection": "OOF best single column or unchanged; no joint mask and no test selection",
                "replay_atol": args.replay_atol, "replay_rtol": args.replay_rtol,
                "code_sha256": {str(path.relative_to(root)): _hash(path) for path in code},
                "manifests": {str(path): _hash(path) for path in (source / "experiment_manifest.json", staged / "experiment_manifest.json",
                              staged / "continued_spline/experiment_manifest.json", staged / "continued_line/task_summaries.json")}}
    runtime = _runtime(device)
    output = args.output_dir / kind
    path = output / "experiment_manifest.json"
    if path.exists():
        previous = _load_json(path, label="audit manifest")
        _validate_resume(previous, semantic, runtime, resume=args.resume, equivalent=args.allow_equivalent_hardware_resume)
    else:
        _write(path, {"semantic": semantic, "runtime": runtime})
    return source, staged, output, hashlib.sha256(_canonical(semantic).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-type", choices=("both", "multiclass", "regression"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-atol", type=float, default=2e-5)
    parser.add_argument("--replay-rtol", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-equivalent-hardware-resume", action="store_true")
    args = parser.parse_args()
    if args.allow_equivalent_hardware_resume and not args.resume:
        parser.error("equivalent-hardware resume requires --resume")
    if any(not np.isfinite(x) or x < 0 for x in (args.replay_atol, args.replay_rtol)):
        parser.error("replay tolerances must be finite and nonnegative")
    args.results_root, args.output_dir = args.results_root.resolve(), args.output_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    kinds = tuple(TASKS) if args.problem_type == "both" else (args.problem_type,)
    for kind in kinds:
        source, staged, output, fingerprint = _prepare(args, kind, device)
        manifest = _load_json(source / "experiment_manifest.json", label="source manifest")
        cases = _find_source_cases(source_dir=source, manifest=manifest, config_label="D", requested_task_ids=set(TASKS[kind]))
        if {case.task_id for case in cases} != set(TASKS[kind]):
            raise ValueError("one or more development tasks is missing")
        rows = []
        for case in cases:
            _validate_case(case)
            task = _load_source_task(case=case, immutable_run=manifest["immutable_run"])
            rows.append(_run_task(args, case, task, staged, output, device, fingerprint))
            _write(output / "column_curvature_summary.json", {"n_completed": len(rows), "n_expected": len(cases), "task_results": rows})
        print(f"{kind}: completed {len(rows)} column-curvature audits", flush=True)


if __name__ == "__main__":
    main()
