"""Audit and selectively deploy a fixed HyperSpline on new synthetic tasks.

This is deliberately a *second* experiment after the synthetic zero-shot
result.  It asks a narrower, falsifiable question: can label-free task
descriptors predict whether an already-frozen HyperSpline will improve a new
task over identity preprocessing?  If so, a validation-selected router can
abstain to identity on likely harms.

The script creates a new, disjoint train/validation/test benchmark.  The
frozen test split is not evaluated until ``report --bank test``.  Candidate
outcomes use query labels only to create meta-training labels and metrics;
router descriptors use context labels and unlabeled query features only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:  # Supports tests and ``python scripts/...`` invocation.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_scheduled_episodes,
        load_episode_bank,
        save_episode_bank,
        validate_episode_classes,
    )
    from scripts.hyperspline_synthetic_zero_shot import (
        aggregate_report,
        forward_hyperspline,
        forward_identity,
        normalise_probability_rows,
        paired_elo_delta,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation.
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_scheduled_episodes,
        load_episode_bank,
        save_episode_bank,
        validate_episode_classes,
    )
    from hyperspline_synthetic_zero_shot import (
        aggregate_report,
        forward_hyperspline,
        forward_identity,
        normalise_probability_rows,
        paired_elo_delta,
    )

from tabicl._hyperspline import backbone_state_dict_hash, load_hyperspline_checkpoint, summarize_context


MANIFEST_VERSION = 1
ROUTER_VERSION = 1
BANK_NAMES = ("train", "validation", "test")
_CHUNK_RE = re.compile(r"chunk_(\d+)_(\d+)\.npz$")


def parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or min(values) <= 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or not all(0.0 <= item <= 1.01 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated thresholds in [0, 1.01]")
    return values


def parse_run_name(value: str) -> str:
    """Keep router variants in one output directory without allowing paths."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, numbers, dot, underscore, and hyphen"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "router_manifest.json"


def bank_path(output_dir: Path, split: str) -> Path:
    return output_dir / "banks" / f"{split}.pt"


def collection_dir(output_dir: Path, split: str) -> Path:
    return output_dir / "collected" / split


def router_dir(output_dir: Path, run_name: str) -> Path:
    return output_dir / "router" / run_name


def router_path(output_dir: Path, run_name: str) -> Path:
    return router_dir(output_dir, run_name) / "router.joblib"


def router_selection_path(output_dir: Path, run_name: str) -> Path:
    return router_dir(output_dir, run_name) / "selection.json"


def router_run_name(args: argparse.Namespace) -> str:
    """Obtain the selected router name from either collection or reporting CLI."""
    return getattr(args, "router_run_name", getattr(args, "run_name", "default"))


def _bank_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prior_type": args.prior_type,
        "min_features": args.min_features,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
        "prior_n_jobs": args.prior_n_jobs,
        "synthetic_observation_mode": args.synthetic_observation_mode,
        "sequence_lengths": list(args.sequence_lengths),
        "context_fractions": list(args.context_fractions),
    }


def _bank_specs(args: argparse.Namespace) -> dict[str, tuple[int, int, int]]:
    counts = (args.train_tasks, args.validation_tasks, args.test_tasks)
    seeds = (args.train_seed, args.validation_seed, args.test_seed)
    if any(count <= 0 for count in counts) or len(set(seeds)) != len(seeds):
        raise ValueError("bank task counts must be positive and bank seeds must be distinct")
    offset, result = 0, {}
    for name, count, seed in zip(BANK_NAMES, counts, seeds, strict=True):
        result[name] = (count, seed, offset)
        offset += count
    return result


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"missing routing manifest; run prepare first: {path}")
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported routing manifest: {path}")
    return manifest


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    manifest_file = manifest_path(output_dir)
    candidate = args.candidate_checkpoint.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"candidate checkpoint does not exist: {candidate}")
    paths = [manifest_file, *(bank_path(output_dir, name) for name in BANK_NAMES)]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to replace frozen routing artifacts; use a new --output-dir or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    stale_outputs = [
        path
        for path in (output_dir / "collected", output_dir / "router", output_dir / "reports")
        if path.exists()
    ]
    if stale_outputs:
        raise FileExistsError(
            "refusing to create or overwrite banks alongside existing routing results; "
            "choose a new --output-dir: " + ", ".join(str(path) for path in stale_outputs)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    specs, metadata = _bank_specs(args), {}
    print("Generating three disjoint frozen routing banks on CPU...", flush=True)
    for split, (count, seed, offset) in specs.items():
        episodes = generate_scheduled_episodes(
            args,
            count,
            source_seed=seed,
            task_offset=offset,
            device=torch.device("cpu"),
            sequence_lengths=args.sequence_lengths,
            context_fractions=args.context_fractions,
            observation_mode=args.synthetic_observation_mode,
        )
        path = bank_path(output_dir, split)
        save_episode_bank(path, episodes, source_seed=seed)
        metadata[split] = {
            "path": str(path.relative_to(output_dir)), "count": count, "seed": seed,
            "task_id_offset": offset, "sha256": sha256_file(path),
        }
    write_json(
        manifest_file,
        {
            "format_version": MANIFEST_VERSION,
            "created_unix": time.time(),
            "purpose": "fixed-HyperSpline win/harm predictability audit and selective router",
            "candidate": {"path": str(candidate), "sha256": sha256_file(candidate)},
            "bank_generation": _bank_config(args),
            "banks": metadata,
            "protocol": {
                "candidate": "frozen before routing-bank generation",
                "descriptors": "context labels and unlabeled query features only",
                "targets": "query labels create meta-training candidate-versus-identity outcomes only",
                "selection": "router threshold selected only on routing validation",
                "test_access": "report --bank test only after router fitting",
            },
        },
    )
    print(f"Prepared frozen routing benchmark: {manifest_file}", flush=True)


def load_bank(output_dir: Path, manifest: dict[str, Any], split: str) -> list[SyntheticEpisode]:
    entry = manifest["banks"].get(split)
    if entry is None:
        raise ValueError(f"unknown routing split: {split}")
    path = output_dir / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"frozen routing bank hash changed: {path}")
    return load_episode_bank(
        path,
        expected_seed=int(entry["seed"]),
        expected_count=int(entry["count"]),
        device=torch.device("cpu"),
        expected_observation_mode=manifest["bank_generation"]["synthetic_observation_mode"],
    )


def episode_to_device(episode: SyntheticEpisode, device: torch.device) -> SyntheticEpisode:
    return replace(
        episode,
        x_context=episode.x_context.to(device), x_query=episode.x_query.to(device),
        y_context=episode.y_context.to(device), y_query=episode.y_query.to(device),
    )


def load_candidate(args: argparse.Namespace, manifest: dict[str, Any], device: torch.device):
    candidate_path = Path(manifest["candidate"]["path"])
    if not candidate_path.is_file() or sha256_file(candidate_path) != manifest["candidate"]["sha256"]:
        raise ValueError("candidate checkpoint differs from the checkpoint frozen in router_manifest.json")
    backbone, _ = load_backbone(args, device)
    candidate, metadata = load_hyperspline_checkpoint(
        candidate_path,
        device=device,
        expected_backbone_reference=args.checkpoint_version,
        expected_backbone_hash=backbone_state_dict_hash(backbone),
    )
    candidate.eval()
    return backbone, candidate, metadata


def logits_metric(logits: torch.Tensor, episode: SyntheticEpisode) -> dict[str, float]:
    probabilities = normalise_probability_rows(torch.softmax(logits.flatten(0, 1), dim=-1).detach().cpu().numpy())
    labels = episode.y_query.detach().cpu().numpy().astype(int)
    nll = float(log_loss(labels, probabilities, labels=list(range(episode.n_classes))))
    result = {"nll": nll, "accuracy": float((probabilities.argmax(axis=1) == labels).mean())}
    if episode.n_classes == 2 and np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities[:, 1]))
        result.update({"auc": auc, "deployment_error": 1.0 - auc})
    else:
        result.update({"auc": float("nan"), "deployment_error": nll})
    return result


def _pool_columns(values: torch.Tensor, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Pool (D, S) label-free column descriptors into a fixed vector."""
    if values.ndim != 2:
        raise ValueError("column descriptor must have shape (features, values)")
    values = values.float()
    pooled = torch.cat(
        (values.mean(0), values.std(0, unbiased=False), torch.quantile(values, 0.10, dim=0), torch.quantile(values, 0.90, dim=0))
    )
    width = values.shape[1]
    names = [f"{prefix}_{stat}_{index}" for stat in ("mean", "std", "q10", "q90") for index in range(width)]
    return pooled.detach().cpu().numpy().astype(np.float32), names


def _task_global_features(episode: SyntheticEpisode) -> tuple[np.ndarray, list[str]]:
    labels = episode.y_context.long().flatten()
    counts = torch.bincount(labels, minlength=episode.n_classes).float()
    frequencies = counts / counts.sum().clamp_min(1.0)
    entropy = -(frequencies * frequencies.clamp_min(1e-12).log()).sum() / math.log(max(episode.n_classes, 2))
    values = episode.x_context.float().squeeze(0)
    centered = values - values.mean(0, keepdim=True)
    normalised = centered / centered.square().mean(0, keepdim=True).sqrt().clamp_min(1e-6)
    correlation = (normalised.T @ normalised / max(values.shape[0], 1)).abs()
    if correlation.shape[0] > 1:
        off_diagonal = correlation[~torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)]
        correlation_mean, correlation_max = off_diagonal.mean(), off_diagonal.max()
    else:
        correlation_mean = correlation_max = values.new_zeros(())
    features = torch.tensor(
        [
            math.log1p(episode.x_context.shape[1]), math.log1p(episode.x_query.shape[1]),
            math.log1p(episode.x_context.shape[2]), math.log1p(episode.n_classes),
            episode.x_context.shape[1] / (episode.x_context.shape[1] + episode.x_query.shape[1]),
            float(entropy), float(frequencies.min()), float(frequencies.max()),
            float(correlation_mean), float(correlation_max),
        ],
        dtype=torch.float32,
    )
    names = [
        "global_log_context_rows", "global_log_query_rows", "global_log_features", "global_log_classes",
        "global_context_fraction", "global_class_entropy", "global_class_min_frequency", "global_class_max_frequency",
        "global_abs_correlation_mean", "global_abs_correlation_max",
    ]
    return features.numpy(), names


def _identity_probe(backbone, episode: SyntheticEpisode) -> tuple[np.ndarray, list[str]]:
    """One label-safe held-out-context probe of the frozen identity backbone."""
    labels = episode.y_context.long().flatten()
    validation_indices = []
    for class_label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == class_label)[0]
        if indices.numel() >= 2:
            validation_indices.append(indices[::5])
    if not validation_indices:
        return np.zeros(4, dtype=np.float32), ["probe_nll", "probe_accuracy", "probe_entropy", "probe_margin"]
    held_out = torch.unique(torch.cat(validation_indices), sorted=True)
    if held_out.numel() == 0 or held_out.numel() >= labels.numel():
        return np.zeros(4, dtype=np.float32), ["probe_nll", "probe_accuracy", "probe_entropy", "probe_margin"]
    keep = torch.ones(labels.numel(), dtype=torch.bool, device=labels.device)
    keep[held_out] = False
    probe = SyntheticEpisode(
        task_id=episode.task_id, source_seed=episode.source_seed,
        x_context=episode.x_context[:, keep], x_query=episode.x_context[:, held_out],
        y_context=episode.y_context[:, keep], y_query=labels[held_out], n_classes=episode.n_classes,
        observation_mode=episode.observation_mode,
    )
    with torch.no_grad():
        backbone.clear_cache()
        _, logits = forward_identity(backbone, probe)
        probabilities = torch.softmax(logits.flatten(0, 1), dim=-1)
        labels_probe = probe.y_query
        nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels_probe)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        sorted_probabilities = probabilities.sort(dim=-1, descending=True).values
        margin = (sorted_probabilities[:, 0] - sorted_probabilities[:, 1]).mean()
        accuracy = (probabilities.argmax(-1) == labels_probe).float().mean()
    return np.asarray([float(nll), float(accuracy), float(entropy), float(margin)], dtype=np.float32), [
        "probe_nll", "probe_accuracy", "probe_entropy", "probe_margin"
    ]


def router_descriptor(backbone, candidate, episode: SyntheticEpisode) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Return feature groups without ever reading ``episode.y_query``."""
    with torch.no_grad():
        context_statistics = summarize_context(
            episode.x_context.float(), y_context=episode.y_context if candidate.target_aware else None, eps=candidate.eps
        )
        query_statistics = summarize_context(episode.x_query.float(), eps=candidate.eps)
        if candidate.conditioning_mode == "cdf":
            current = torch.cat(
                (query_statistics.summary[..., :23], context_statistics.summary[..., -8:]), dim=-1
            )
        else:
            current = candidate.conditioning_summary(context_statistics, query_statistics)
        current_values, current_names = _pool_columns(current.squeeze(0), "current")
        query_marginal = query_statistics.summary[..., :23]
        context_marginal = context_statistics.summary[..., :23]
        shift = query_marginal - context_marginal
        shift_values = torch.cat((context_marginal, shift, shift.abs()), dim=-1)
        shift_values, shift_names = _pool_columns(shift_values.squeeze(0), "shift")
        global_values, global_names = _task_global_features(episode)
        probe_values, probe_names = _identity_probe(backbone, episode)
        _, _, parameters = candidate(
            episode.x_context, episode.x_query,
            y_context=episode.y_context if candidate.target_aware else None,
            return_parameters=True,
        )
        identity = candidate.identity_control_points.to(parameters.control_points).view(1, 1, -1)
        deformation = parameters.control_points - identity
        deformation_values, deformation_names = _pool_columns(deformation.squeeze(0), "transform_deformation")
        gate_values, gate_names = _pool_columns(parameters.gate.squeeze(0).unsqueeze(-1), "transform_gate")
        transform_values = np.concatenate((deformation_values, gate_values, np.asarray([
            float(candidate.grid_deformation_penalty(parameters))
        ], dtype=np.float32)))
        transform_names = deformation_names + gate_names + ["transform_grid_penalty"]
    values = np.concatenate((current_values, shift_values, global_values, probe_values, transform_values)).astype(np.float32)
    groups = (
        ["current"] * len(current_values)
        + ["shift_global"] * (len(shift_values) + len(global_values))
        + ["probe"] * len(probe_values)
        + ["transform"] * len(transform_values)
    )
    names = current_names + shift_names + global_names + probe_names + transform_names
    return values, names, {"groups": groups}


def feature_indices(groups: Sequence[str], feature_set: str) -> np.ndarray:
    allowed = {
        "current": {"current"},
        "shift_global": {"current", "shift_global"},
        "shift_global_probe": {"current", "shift_global", "probe"},
        "all": {"current", "shift_global", "probe", "transform"},
    }[feature_set]
    indices = np.asarray([index for index, group in enumerate(groups) if group in allowed], dtype=np.int64)
    if not indices.size:
        raise ValueError(f"feature set {feature_set!r} selected no features")
    return indices


def _chunk_path(output_dir: Path, split: str, start: int, stop: int) -> Path:
    return collection_dir(output_dir, split) / f"chunk_{start:06d}_{stop:06d}.npz"


def _save_chunk(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _metric_arrays(identity: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    row = {f"identity_{key}": value for key, value in identity.items()}
    row.update({f"candidate_{key}": value for key, value in candidate.items()})
    return row


def evaluate_episode(backbone, candidate, episode: SyntheticEpisode) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    descriptor, feature_names, descriptor_info = router_descriptor(backbone, candidate, episode)
    with torch.no_grad():
        backbone.clear_cache()
        _, identity_logits = forward_identity(backbone, episode)
        identity = logits_metric(identity_logits, episode)
        backbone.clear_cache()
        _, candidate_logits, _ = forward_hyperspline(backbone, candidate, episode)
        candidate_metrics = logits_metric(candidate_logits, episode)
    row: dict[str, Any] = {
        "task_id": episode.task_id, "n_context": episode.x_context.shape[1], "n_query": episode.x_query.shape[1],
        "n_features": episode.x_context.shape[2], "n_classes": episode.n_classes,
        **_metric_arrays(identity, candidate_metrics),
    }
    row["deployment_gain"] = row["identity_deployment_error"] - row["candidate_deployment_error"]
    row["candidate_win"] = int(row["deployment_gain"] > 0.0)
    row["candidate_material_harm"] = int(material_harm(row))
    row["candidate_material_win"] = int(material_win(row))
    row["descriptor_groups"] = descriptor_info["groups"]
    return descriptor, feature_names, row


def material_harm(row: dict[str, Any]) -> bool:
    if int(row["n_classes"]) == 2:
        return float(row["candidate_auc"]) - float(row["identity_auc"]) <= -0.005
    return (float(row["candidate_nll"]) - float(row["identity_nll"])) / max(float(row["identity_nll"]), 1e-12) >= 0.01


def material_win(row: dict[str, Any]) -> bool:
    if int(row["n_classes"]) == 2:
        return float(row["candidate_auc"]) - float(row["identity_auc"]) >= 0.005
    return (float(row["candidate_nll"]) - float(row["identity_nll"])) / max(float(row["identity_nll"]), 1e-12) <= -0.01


def collect_split(args: argparse.Namespace, manifest: dict[str, Any], split: str) -> None:
    if split == "test" and not router_path(args.output_dir, router_run_name(args)).is_file():
        raise RuntimeError("fit the router before collecting the frozen routing test split")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    episodes = load_bank(args.output_dir, manifest, split)
    backbone, candidate, _ = load_candidate(args, manifest, device)
    validate_episode_classes(episodes, backbone.max_classes)
    schema_path = args.output_dir / "collected" / "feature_schema.json"
    for start in range(0, len(episodes), args.chunk_size):
        stop, path = min(start + args.chunk_size, len(episodes)), _chunk_path(args.output_dir, split, start, min(start + args.chunk_size, len(episodes)))
        if path.is_file():
            with np.load(path, allow_pickle=False) as existing:
                if existing["task_id"].shape[0] == stop - start:
                    print(f"[{split} {start}:{stop}] already collected", flush=True)
                    continue
            raise ValueError(f"incomplete or invalid existing collection chunk: {path}")
        features, rows, names, groups = [], [], None, None
        for local_index, episode_cpu in enumerate(episodes[start:stop], start):
            episode = episode_to_device(episode_cpu, device)
            descriptor, feature_names, row = evaluate_episode(backbone, candidate, episode)
            if names is None:
                names, groups = feature_names, row.pop("descriptor_groups")
            else:
                if feature_names != names:
                    raise AssertionError("descriptor schema changed across routing tasks")
                row.pop("descriptor_groups")
            features.append(descriptor)
            rows.append(row)
            if (local_index + 1) % args.progress_every == 0 or local_index + 1 == stop:
                print(f"[{split} {local_index + 1}/{len(episodes)}] collected", flush=True)
        assert names is not None and groups is not None
        schema = {"feature_names": names, "groups": groups, "schema_hash": stable_digest({"names": names, "groups": groups})}
        if schema_path.is_file():
            old = json.loads(schema_path.read_text(encoding="utf8"))
            if old["schema_hash"] != schema["schema_hash"]:
                raise ValueError("routing descriptor schema differs from existing collection")
        else:
            write_json(schema_path, schema)
        scalar_keys = sorted(rows[0])
        arrays: dict[str, np.ndarray] = {"features": np.stack(features).astype(np.float32)}
        for key in scalar_keys:
            arrays[key] = np.asarray([row[key] for row in rows])
        _save_chunk(path, arrays)
        print(f"[{split} {start}:{stop}] wrote {path}", flush=True)


def collection_complete(output_dir: Path, manifest: dict[str, Any], split: str) -> bool:
    expected, cursor = int(manifest["banks"][split]["count"]), 0
    for path in sorted(collection_dir(output_dir, split).glob("chunk_*.npz")):
        match = _CHUNK_RE.match(path.name)
        if match is None:
            continue
        start, stop = map(int, match.groups())
        if start != cursor or stop <= start:
            return False
        with np.load(path, allow_pickle=False) as values:
            if values["task_id"].shape[0] != stop - start:
                return False
        cursor = stop
    return cursor == expected


def load_collection(output_dir: Path, manifest: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    if not collection_complete(output_dir, manifest, split):
        raise FileNotFoundError(f"collection for {split!r} is incomplete; run collect first")
    chunks = []
    for path in sorted(collection_dir(output_dir, split).glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            chunks.append({key: source[key] for key in source.files})
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def prediction_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(np.int64)
    if labels.min() == labels.max():
        raise ValueError("win/harm target contains only one class")
    return {
        "n_tasks": int(labels.size), "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def fit_classifier(args: argparse.Namespace, x: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0))
    if args.classifier == "logistic":
        classifier = LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.model_seed)
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden_dims), alpha=args.alpha, batch_size=min(args.batch_size, len(x)),
            learning_rate_init=args.router_lr, max_iter=args.max_iter, early_stopping=True,
            validation_fraction=0.10, n_iter_no_change=args.patience, random_state=args.model_seed,
        )
    classifier.fit(x_scaled, y)
    return scaler, classifier


def router_rows(data: dict[str, np.ndarray], probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for index, probability in enumerate(probabilities):
        applied = bool(probability >= threshold)
        row = {key: data[key][index].item() for key in data if key != "features"}
        row["router_probability"] = float(probability)
        row["router_applied"] = int(applied)
        for metric in ("nll", "accuracy", "auc", "deployment_error"):
            row[f"router_{metric}"] = row[f"candidate_{metric}"] if applied else row[f"identity_{metric}"]
        rows.append(row)
    return rows


def router_summary(rows: Sequence[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int) -> dict[str, Any]:
    report_rows = []
    for row in rows:
        report_rows.append(
            {
                "identity_nll": row["identity_nll"], "identity_accuracy": row["identity_accuracy"],
                "identity_auc": row["identity_auc"], "identity_deployment_error": row["identity_deployment_error"],
                "candidate_nll": row["router_nll"], "candidate_accuracy": row["router_accuracy"],
                "candidate_auc": row["router_auc"], "candidate_deployment_error": row["router_deployment_error"],
            }
        )
    summary = aggregate_report(report_rows, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    summary["applied_tasks"] = int(sum(int(row["router_applied"]) for row in rows))
    summary["applied_fraction"] = float(summary["applied_tasks"] / len(rows))
    summary["material_harms"] = int(sum(material_harm({
        "n_classes": row["n_classes"], "identity_nll": row["identity_nll"], "identity_auc": row["identity_auc"],
        "candidate_nll": row["router_nll"], "candidate_auc": row["router_auc"],
    }) for row in rows))
    summary["material_wins"] = int(sum(material_win({
        "n_classes": row["n_classes"], "identity_nll": row["identity_nll"], "identity_auc": row["identity_auc"],
        "candidate_nll": row["router_nll"], "candidate_auc": row["router_auc"],
    }) for row in rows))
    return summary


def choose_threshold(
    data: dict[str, np.ndarray], probabilities: np.ndarray, thresholds: Sequence[float], max_material_harm_ratio: float
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    raw_rows = router_rows(data, probabilities, threshold=0.0)
    raw_harms = router_summary(raw_rows, bootstrap_seed=0, bootstrap_samples=0)["material_harms"]
    allowed_harms = math.floor(raw_harms * max_material_harm_ratio)
    candidates = []
    for threshold in sorted(set(float(value) for value in thresholds) | {1.01}):
        rows = router_rows(data, probabilities, threshold)
        summary = router_summary(rows, bootstrap_seed=0, bootstrap_samples=0)
        candidates.append({
            "threshold": threshold, "eligible": summary["material_harms"] <= allowed_harms,
            "material_harms": summary["material_harms"], "material_wins": summary["material_wins"],
            "applied_fraction": summary["applied_fraction"], "elo_score": summary["elo"]["score"],
            "elo_delta": summary["elo"]["elo_delta"], "mean_nll_delta": summary["mean_nll_delta"],
            "summary": summary,
        })
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["elo_score"], -item["material_harms"], -item["mean_nll_delta"]))
    return float(selected["threshold"]), selected["summary"], candidates


def fit(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output_dir)
    train, validation = load_collection(args.output_dir, manifest, "train"), load_collection(args.output_dir, manifest, "validation")
    schema = json.loads((args.output_dir / "collected" / "feature_schema.json").read_text(encoding="utf8"))
    indices = feature_indices(schema["groups"], args.feature_set)
    x_train, x_validation = train["features"][:, indices], validation["features"][:, indices]
    y_train, y_validation = train["candidate_win"].astype(np.int64), validation["candidate_win"].astype(np.int64)
    scaler, classifier = fit_classifier(args, x_train, y_train)
    train_probability = classifier.predict_proba(scaler.transform(np.nan_to_num(x_train, nan=0.0)))[:, 1]
    validation_probability = classifier.predict_proba(scaler.transform(np.nan_to_num(x_validation, nan=0.0)))[:, 1]
    threshold, validation_router, threshold_rows = choose_threshold(
        validation, validation_probability, args.thresholds, args.max_material_harm_ratio
    )
    payload = {
        "format_version": ROUTER_VERSION, "classifier": args.classifier, "feature_set": args.feature_set,
        "feature_indices": indices, "feature_names": [schema["feature_names"][index] for index in indices],
        "scaler": scaler, "model": classifier, "candidate_sha256": manifest["candidate"]["sha256"],
    }
    run_dir = router_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, router_path(args.output_dir, args.run_name))
    write_csv(run_dir / "validation_predictions.csv", [
        {
            "task_id": int(validation["task_id"][index]), "candidate_win": int(y_validation[index]),
            "candidate_material_harm": int(validation["candidate_material_harm"][index]),
            "router_probability": float(validation_probability[index]),
        }
        for index in range(len(y_validation))
    ])
    write_csv(run_dir / "thresholds.csv", [{key: value for key, value in row.items() if key != "summary"} for row in threshold_rows])
    selection = {
        "protocol": "router fit on train only; threshold selected on routing validation only",
        "run_name": args.run_name,
        "candidate": manifest["candidate"], "feature_set": args.feature_set,
        "train_predictability": prediction_summary(y_train, train_probability),
        "validation_predictability": prediction_summary(y_validation, validation_probability),
        "max_material_harm_ratio": args.max_material_harm_ratio, "selected_threshold": threshold,
        "validation_router": validation_router,
    }
    write_json(router_selection_path(args.output_dir, args.run_name), selection)
    print(
        f"Fitted {args.classifier} router ({args.run_name}); validation AUROC={selection['validation_predictability']['roc_auc']:.3f}; "
        f"selected threshold={threshold:.3f}; validation router Elo={validation_router['elo']['elo_delta']:+.1f}", flush=True
    )


def report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output_dir)
    if not router_path(args.output_dir, args.run_name).is_file() or not router_selection_path(args.output_dir, args.run_name).is_file():
        raise FileNotFoundError("fit the router before reporting")
    output_dir = args.output_dir / "reports" / args.bank / args.run_name
    if args.bank == "test" and not args.allow_existing_test_report:
        existing_test_reports = list((args.output_dir / "reports" / "test").glob("*/summary.json"))
        if existing_test_reports:
            raise FileExistsError(
                "frozen routing test was already reported for a router variant: "
                + ", ".join(str(path) for path in existing_test_reports)
            )
    if not collection_complete(args.output_dir, manifest, args.bank):
        collect_split(args, manifest, args.bank)
    data = load_collection(args.output_dir, manifest, args.bank)
    router = joblib.load(router_path(args.output_dir, args.run_name))
    if router["candidate_sha256"] != manifest["candidate"]["sha256"]:
        raise ValueError("router belongs to a different frozen candidate")
    indices = np.asarray(router["feature_indices"], dtype=np.int64)
    features = np.nan_to_num(data["features"][:, indices], nan=0.0, posinf=20.0, neginf=-20.0)
    probability = router["model"].predict_proba(router["scaler"].transform(features))[:, 1]
    selection = json.loads(router_selection_path(args.output_dir, args.run_name).read_text(encoding="utf8"))
    rows = router_rows(data, probability, float(selection["selected_threshold"]))
    raw = router_summary(router_rows(data, probability, 0.0), bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples)
    routed = router_summary(rows, bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples)
    result = {
        "evaluated_bank": args.bank, "is_final_test": args.bank == "test",
        "protocol": manifest["protocol"], "selection": selection,
        "predictability": prediction_summary(data["candidate_win"].astype(np.int64), probability),
        "always_apply_candidate": raw, "router": routed,
    }
    write_csv(output_dir / "per_table_metrics.csv", rows)
    write_json(output_dir / "summary.json", result)
    print(
        f"[{args.bank}/{args.run_name}] prediction AUROC={result['predictability']['roc_auc']:.3f}; "
        f"candidate Elo={raw['elo']['elo_delta']:+.1f}; router Elo={routed['elo']['elo_delta']:+.1f}; "
        f"material harms={raw['material_harms']}->{routed['material_harms']}", flush=True
    )


def add_shared_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--synthetic-observation-mode", choices=SYNTHETIC_OBSERVATION_MODES, default="coverage_expanded")
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--sequence-lengths", type=parse_int_csv, default=parse_int_csv("128,256,512,1024"))
    parser.add_argument("--context-fractions", type=parse_float_csv, default=parse_float_csv("0.50,0.70,0.85"))


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Create disjoint frozen routing banks.")
    add_shared_generation_arguments(prepare_parser)
    prepare_parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--train-tasks", type=int, default=5_000)
    prepare_parser.add_argument("--validation-tasks", type=int, default=1_000)
    prepare_parser.add_argument("--test-tasks", type=int, default=2_000)
    prepare_parser.add_argument("--train-seed", type=int, default=81_001)
    prepare_parser.add_argument("--validation-seed", type=int, default=82_001)
    prepare_parser.add_argument("--test-seed", type=int, default=83_001)
    prepare_parser.add_argument("--overwrite", action="store_true")

    collect_parser = commands.add_parser("collect", help="Evaluate the frozen candidate in resumable routing-data chunks.")
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    add_execution_arguments(collect_parser)
    collect_parser.add_argument(
        "--split", choices=("train", "validation", "development"), default="train",
        help="Development bank to collect. The held-out test is collected only by report --bank test.",
    )
    collect_parser.add_argument("--chunk-size", type=int, default=50)
    collect_parser.add_argument("--progress-every", type=int, default=10)

    fit_parser = commands.add_parser("fit", help="Fit the win-predictability classifier and select the router threshold.")
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--run-name", type=parse_run_name, default="default")
    fit_parser.add_argument("--feature-set", choices=("current", "shift_global", "shift_global_probe", "all"), default="current")
    fit_parser.add_argument("--classifier", choices=("logistic", "mlp"), default="mlp")
    fit_parser.add_argument("--hidden-dims", type=parse_int_csv, default=parse_int_csv("64,32"))
    fit_parser.add_argument("--router-lr", type=float, default=1e-3)
    fit_parser.add_argument("--alpha", type=float, default=1e-4)
    fit_parser.add_argument("--batch-size", type=int, default=256)
    fit_parser.add_argument("--max-iter", type=int, default=300)
    fit_parser.add_argument("--patience", type=int, default=20)
    fit_parser.add_argument("--model-seed", type=int, default=0)
    fit_parser.add_argument("--thresholds", type=parse_float_csv, default=parse_float_csv(",".join(f"{value / 100:.2f}" for value in range(0, 101))))
    fit_parser.add_argument("--max-material-harm-ratio", type=float, default=0.75)

    report_parser = commands.add_parser("report", help="Report validation diagnostics or unlock the frozen routing test once.")
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.add_argument("--run-name", type=parse_run_name, default="default")
    add_execution_arguments(report_parser)
    report_parser.add_argument("--bank", choices=("validation", "test"), default="test")
    report_parser.add_argument("--bootstrap-seed", type=int, default=9_101)
    report_parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    report_parser.add_argument("--allow-existing-test-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "collect":
        manifest = load_manifest(args.output_dir)
        if args.chunk_size <= 0 or args.progress_every <= 0:
            raise ValueError("chunk size and progress interval must be positive")
        for split in ("train", "validation") if args.split == "development" else (args.split,):
            collect_split(args, manifest, split)
    elif args.command == "fit":
        if not 0.0 <= args.max_material_harm_ratio <= 1.0:
            raise ValueError("max material harm ratio must be in [0, 1]")
        fit(args)
    elif args.command == "report":
        report(args)
    else:  # pragma: no cover - argparse protects this branch.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
