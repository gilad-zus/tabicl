"""Test whether different tabular task splits need different spline policies.

This is a *diagnostic*, not a HyperSpline training run.  For every independent
split of a PMLB dataset we fit a DirectSpline using only the frozen TabICL
classification loss on a labelled adaptation partition.  We then transplant
that spline residual into every other split of the same dataset and evaluate it
on that target split's untouched outer query partition.

The resulting matrix answers two questions separately:

1. Does a split-specific DirectSpline beat a spline learned for another split
   (or a leave-one-split-out average)?
2. Can observable task statistics choose a useful source spline?  We compare
   nearest-source retrieval with (a) context-only descriptors and (b)
   descriptors that use context labels plus *unlabelled* outer-query features.

No query label enters a descriptor, a retrieval decision, or the transformed
input.  Adaptation labels are used only to make each DirectSpline an oracle
teacher for the diagnostic; they are never a future HyperSpline target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import summarize_context
from tabicl._hyperspline.statistics import UNSUPERVISED_SUMMARY_DIM

try:  # Support ``python scripts/...`` and package-style imports.
    from scripts.direct_spline_dataset_headroom import (
        clone_with_shape,
        evaluate_chunked,
        make_direct_spline,
        make_direct_spline_optimizer,
        optimize_direct_spline,
        release_cuda,
        spline_diagnostics,
        stratified_subset,
        to_device,
    )
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    from direct_spline_dataset_headroom import (
        clone_with_shape, evaluate_chunked, make_direct_spline,
        make_direct_spline_optimizer, optimize_direct_spline, release_cuda,
        spline_diagnostics, stratified_subset, to_device,
    )
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_pmlb_frame


@dataclass
class SplitEpisode:
    dataset: str
    seed: int
    episode: int
    support_x: np.ndarray
    support_y: np.ndarray
    adaptation_x: np.ndarray
    adaptation_y: np.ndarray
    evaluation_x: np.ndarray
    evaluation_y: np.ndarray
    context_descriptor: np.ndarray
    transductive_descriptor: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True,
                        help="Repeat for each dataset to diagnose.")
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes", type=int, default=12,
                        help="Independent outer splits per dataset/seed; 10--16 is usually enough.")
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--adaptation-fraction", type=float, default=0.50,
                        help="Fraction of an outer-training split used as labelled DirectSpline adaptation queries.")
    parser.add_argument("--max-context-rows", type=int, default=512)
    parser.add_argument("--train-context-rows", type=int, default=384)
    parser.add_argument("--query-batch-rows", type=int, default=256)
    parser.add_argument("--evaluation-query-chunk-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1_250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--trainable-location-scale", action="store_true")
    parser.add_argument("--transform-regularization", type=float, default=0.0,
                        help="Kept at zero by default: this experiment isolates TabICL NLL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf8")


def descriptor_from_statistics(context_x: np.ndarray, context_y: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make permutation-invariant task descriptors without looking at query labels.

    Context-only uses all 31 current HyperSpline statistics.  The transductive
    descriptor uses the 23 unsupervised columns from context+query features,
    and the 8 label-aware columns from context only.  Five symmetric moments
    over feature columns make both vectors invariant to column order.
    """
    context = torch.as_tensor(context_x, dtype=torch.float32).unsqueeze(0)
    labels = torch.as_tensor(context_y, dtype=torch.float32).unsqueeze(0)
    query = torch.as_tensor(query_x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        context_summary = summarize_context(context, y_context=labels).summary[0]
        all_x_summary = summarize_context(torch.cat((context, query), dim=1)).summary[0]
        transductive_summary = torch.cat(
            (all_x_summary[:, :UNSUPERVISED_SUMMARY_DIM], context_summary[:, UNSUPERVISED_SUMMARY_DIM:]), dim=-1
        )

        def aggregate(summary: torch.Tensor) -> np.ndarray:
            return torch.cat((
                summary.mean(0), summary.std(0, unbiased=False),
                torch.quantile(summary, 0.25, dim=0), torch.quantile(summary, 0.50, dim=0),
                torch.quantile(summary, 0.75, dim=0),
            )).cpu().numpy().astype(np.float64, copy=False)

        return aggregate(context_summary), aggregate(transductive_summary)


def make_split_episode(
    x: np.ndarray, y: np.ndarray, *, dataset: str, seed: int, episode: int, args: argparse.Namespace
) -> SplitEpisode:
    """Create a labelled fit episode and a never-optimized outer query set."""
    indices = np.arange(y.size)
    split_seed = seed * 1_000_000 + episode
    fit_idx, evaluation_idx = train_test_split(
        indices, test_size=args.outer_test_size, random_state=split_seed, stratify=y
    )
    support_idx, adaptation_idx = train_test_split(
        fit_idx, test_size=args.adaptation_fraction, random_state=split_seed + 17, stratify=y[fit_idx]
    )
    if args.max_context_rows > 0 and support_idx.size > args.max_context_rows:
        local = stratified_subset(y[support_idx], args.max_context_rows, np.random.default_rng(split_seed + 31))
        support_idx = support_idx[local]
    context_descriptor, transductive_descriptor = descriptor_from_statistics(
        x[support_idx], y[support_idx], x[evaluation_idx]
    )
    return SplitEpisode(
        dataset=dataset, seed=seed, episode=episode,
        support_x=np.asarray(x[support_idx], dtype=np.float32), support_y=np.asarray(y[support_idx], dtype=np.int64),
        adaptation_x=np.asarray(x[adaptation_idx], dtype=np.float32), adaptation_y=np.asarray(y[adaptation_idx], dtype=np.int64),
        evaluation_x=np.asarray(x[evaluation_idx], dtype=np.float32), evaluation_y=np.asarray(y[evaluation_idx], dtype=np.int64),
        context_descriptor=context_descriptor, transductive_descriptor=transductive_descriptor,
    )


def robust_distances(descriptors: list[np.ndarray]) -> np.ndarray:
    """Pairwise descriptor distance with robust per-coordinate scaling."""
    values = np.stack(descriptors)
    median = np.median(values, axis=0)
    scale = np.median(np.abs(values - median), axis=0) * 1.4826
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = (values - median) / scale
    return np.sqrt(((standardized[:, None] - standardized[None, :]) ** 2).mean(axis=-1))


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 3 or x_values[valid].std() == 0 or y_values[valid].std() == 0:
        return None
    return float(np.corrcoef(x_values[valid], y_values[valid])[0, 1])


def state_from(spline) -> dict[str, torch.Tensor | None | str]:
    return {
        "gap_logits": spline.gap_logits.detach().cpu().clone(),
        "gate_logits": spline.gate_logits.detach().cpu().clone(),
        "location_offsets": spline.location_offsets.detach().cpu().clone(),
        "log_scale_offsets": spline.log_scale_offsets.detach().cpu().clone(),
        "range_logits": spline.range_logits.detach().cpu().clone(),
        "knot_width_logits": spline.knot_width_logits.detach().cpu().clone(),
        "mixing_left": None if spline.mixing_left is None else spline.mixing_left.detach().cpu().clone(),
        "mixing_right": None if spline.mixing_right is None else spline.mixing_right.detach().cpu().clone(),
        "mixing_weight_logits": None if spline.mixing_weight_logits is None else spline.mixing_weight_logits.detach().cpu().clone(),
        "mixing_gate": None if spline.mixing_gate is None else spline.mixing_gate.detach().cpu().clone(),
        "knot_placement": spline.knot_placement,
    }


def transplant(identity, state: dict):
    return clone_with_shape(identity, **state)


def average_state(states: list[dict]) -> dict:
    if not states:
        raise ValueError("cannot average no spline states")
    result: dict[str, torch.Tensor | None | str] = {"knot_placement": str(states[0]["knot_placement"])}
    for name in (
        "gap_logits", "gate_logits", "location_offsets", "log_scale_offsets", "range_logits", "knot_width_logits",
        "mixing_left", "mixing_right", "mixing_weight_logits", "mixing_gate",
    ):
        values = [item[name] for item in states]
        result[name] = None if values[0] is None else torch.stack(values).mean(0)  # type: ignore[arg-type]
    return result


def direct_args(args: argparse.Namespace) -> SimpleNamespace:
    """The existing DirectSpline helpers deliberately retain their shared API."""
    return SimpleNamespace(
        n_control_points=args.n_control_points,
        freeze_spline_shape=False,
        trainable_range=False,
        trainable_location_scale=args.trainable_location_scale,
        control_mode="monotone", free_control_bound=1.0, knot_placement="uniform",
        cross_column_mixing_rank=0, cross_column_mixing_bound=0.1,
        lr=args.lr, train_context_rows=args.train_context_rows, query_batch_rows=args.query_batch_rows,
        transform_regularization=args.transform_regularization,
        free_control_curvature_regularization=0.0, free_control_reference_regularization=0.0,
    )


def fit_teacher(backbone, episode: SplitEpisode, args: argparse.Namespace, device: torch.device):
    support_x, support_y = to_device(episode.support_x, episode.support_y, np.arange(episode.support_y.size), device)
    identity = make_direct_spline(support_x, direct_args(args)).to(device).eval()
    spline = copy.deepcopy(identity).train()
    optimizer = make_direct_spline_optimizer(spline, direct_args(args))
    initial, final = optimize_direct_spline(
        backbone, spline, optimizer, support_x, support_y, episode.adaptation_x, episode.adaptation_y,
        steps=args.steps, sample_rng=np.random.default_rng(10_000_000 * episode.seed + episode.episode),
        args=direct_args(args),
    )
    spline.eval()
    identity_loss, identity_accuracy, _ = evaluate_chunked(
        backbone, identity, support_x, support_y, episode.evaluation_x, episode.evaluation_y,
        chunk_rows=args.evaluation_query_chunk_rows, device=device,
    )
    own_loss, own_accuracy, _ = evaluate_chunked(
        backbone, spline, support_x, support_y, episode.evaluation_x, episode.evaluation_y,
        chunk_rows=args.evaluation_query_chunk_rows, device=device,
    )
    state, diagnostics = state_from(spline), spline_diagnostics(spline)
    del optimizer, spline
    return support_x.detach().cpu(), support_y.detach().cpu(), identity, state, initial, final, identity_loss, identity_accuracy, own_loss, own_accuracy, diagnostics


def run_dataset_seed(args: argparse.Namespace, backbone, *, dataset: str, seed: int, device: torch.device):
    x, raw_y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
    _, y = np.unique(raw_y, return_inverse=True)
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2 or labels.size > backbone.max_classes or counts.min() < 4:
        raise ValueError(f"{dataset}: incompatible class count/support for this protocol")
    episodes = [make_split_episode(x, y, dataset=dataset, seed=seed, episode=index, args=args)
                for index in range(args.episodes)]
    artifacts = []
    print(f"[{dataset} seed={seed}] fitting {len(episodes)} split-specific DirectSpline teachers", flush=True)
    for episode in episodes:
        values = fit_teacher(backbone, episode, args, device)
        (support_x, support_y, identity, state, initial, final, identity_loss, identity_accuracy,
         own_loss, own_accuracy, diagnostics) = values
        artifacts.append({
            "episode": episode, "support_x": support_x, "support_y": support_y, "state": state,
            "identity_loss": identity_loss, "identity_accuracy": identity_accuracy,
            "own_loss": own_loss, "own_accuracy": own_accuracy,
        })
        print(
            f"[{dataset} seed={seed} split={episode.episode}] train={initial:.4f}->{final:.4f}; "
            f"own_delta={own_loss - identity_loss:+.5f}; gate={diagnostics[0]:.3f}", flush=True,
        )
        del identity
        release_cuda(device)

    context_distances = robust_distances([episode.context_descriptor for episode in episodes])
    transductive_distances = robust_distances([episode.transductive_descriptor for episode in episodes])
    matrix_rows: list[dict[str, object]] = []
    by_pair: dict[tuple[int, int], dict[str, object]] = {}
    print(f"[{dataset} seed={seed}] cross-applying {len(artifacts)}x{len(artifacts)} spline residuals", flush=True)
    for target_index, target in enumerate(artifacts):
        target_episode: SplitEpisode = target["episode"]
        target_x, target_y = target["support_x"].to(device), target["support_y"].to(device)
        target_identity = make_direct_spline(target_x, direct_args(args)).to(device).eval()
        for source_index, source in enumerate(artifacts):
            candidate = transplant(target_identity, source["state"])
            loss, accuracy, _ = evaluate_chunked(
                backbone, candidate, target_x, target_y, target_episode.evaluation_x, target_episode.evaluation_y,
                chunk_rows=args.evaluation_query_chunk_rows, device=device,
            )
            row = {
                "dataset": dataset, "seed": seed, "target_episode": target_index, "source_episode": source_index,
                "candidate": "own_oracle" if source_index == target_index else "cross_split",
                "identity_loss": target["identity_loss"], "identity_accuracy": target["identity_accuracy"],
                "own_oracle_loss": target["own_loss"], "own_oracle_accuracy": target["own_accuracy"],
                "candidate_loss": loss, "candidate_accuracy": accuracy,
                "candidate_minus_identity_loss": loss - target["identity_loss"],
                "candidate_minus_own_oracle_loss": loss - target["own_loss"],
                "context_descriptor_distance": float(context_distances[target_index, source_index]),
                "transductive_descriptor_distance": float(transductive_distances[target_index, source_index]),
            }
            matrix_rows.append(row); by_pair[target_index, source_index] = row
            del candidate
        del target_identity, target_x, target_y
        release_cuda(device)
        print(f"[{dataset} seed={seed}] cross target {target_index + 1}/{len(artifacts)} complete", flush=True)

    selection_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(artifacts):
        sources = [index for index in range(len(artifacts)) if index != target_index]
        for descriptor_name, distances in (("context_only", context_distances), ("context_plus_unlabelled_query", transductive_distances)):
            source_index = min(sources, key=lambda index: float(distances[target_index, index]))
            matrix = by_pair[target_index, source_index]
            selection_rows.append({
                "dataset": dataset, "seed": seed, "target_episode": target_index, "selector": descriptor_name,
                "selected_source_episode": source_index, "selected_descriptor_distance": float(distances[target_index, source_index]),
                "identity_loss": target["identity_loss"], "own_oracle_loss": target["own_loss"],
                "selected_loss": matrix["candidate_loss"], "selected_accuracy": matrix["candidate_accuracy"],
                "selected_minus_identity_loss": float(matrix["candidate_loss"]) - float(target["identity_loss"]),
                "selected_minus_own_oracle_loss": float(matrix["candidate_loss"]) - float(target["own_loss"]),
            })
        mean_candidate = transplant(
            make_direct_spline(target["support_x"].to(device), direct_args(args)).to(device).eval(),
            average_state([source["state"] for index, source in enumerate(artifacts) if index != target_index]),
        )
        target_x, target_y = target["support_x"].to(device), target["support_y"].to(device)
        mean_loss, mean_accuracy, _ = evaluate_chunked(
            backbone, mean_candidate, target_x, target_y, target["episode"].evaluation_x, target["episode"].evaluation_y,
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
        )
        selection_rows.append({
            "dataset": dataset, "seed": seed, "target_episode": target_index, "selector": "leave_one_split_mean",
            "selected_source_episode": None, "selected_descriptor_distance": None,
            "identity_loss": target["identity_loss"], "own_oracle_loss": target["own_loss"],
            "selected_loss": mean_loss, "selected_accuracy": mean_accuracy,
            "selected_minus_identity_loss": mean_loss - target["identity_loss"],
            "selected_minus_own_oracle_loss": mean_loss - target["own_loss"],
        })
        del mean_candidate, target_x, target_y
        release_cuda(device)

    cross_rows = [row for row in matrix_rows if row["candidate"] == "cross_split"]
    summary = {
        "dataset": dataset, "seed": seed, "episodes": len(episodes), "n_features": int(x.shape[1]), "n_classes": int(labels.size),
        "mean_own_oracle_minus_identity_loss": float(np.mean([item["own_loss"] - item["identity_loss"] for item in artifacts])),
        "own_oracle_improves_fraction": float(np.mean([item["own_loss"] < item["identity_loss"] for item in artifacts])),
        "mean_cross_minus_identity_loss": float(np.mean([float(row["candidate_minus_identity_loss"]) for row in cross_rows])),
        "mean_cross_regret_vs_own_loss": float(np.mean([float(row["candidate_minus_own_oracle_loss"]) for row in cross_rows])),
        "context_distance_vs_cross_regret_pearson": pearson_correlation(
            [float(row["context_descriptor_distance"]) for row in cross_rows],
            [float(row["candidate_minus_own_oracle_loss"]) for row in cross_rows],
        ),
        "transductive_distance_vs_cross_regret_pearson": pearson_correlation(
            [float(row["transductive_descriptor_distance"]) for row in cross_rows],
            [float(row["candidate_minus_own_oracle_loss"]) for row in cross_rows],
        ),
        "protocol": "independent outer splits; DirectSpline teachers fit on support+labelled adaptation; matrix/selection evaluate untouched outer query; descriptors never use query labels",
    }
    for selector in ("context_only", "context_plus_unlabelled_query", "leave_one_split_mean"):
        selected = [row for row in selection_rows if row["selector"] == selector]
        summary[f"{selector}_mean_minus_identity_loss"] = float(np.mean([float(row["selected_minus_identity_loss"]) for row in selected]))
        summary[f"{selector}_mean_regret_vs_own_loss"] = float(np.mean([float(row["selected_minus_own_oracle_loss"]) for row in selected]))
        summary[f"{selector}_beats_identity_fraction"] = float(np.mean([float(row["selected_minus_identity_loss"]) < 0 for row in selected]))
    return matrix_rows, selection_rows, summary


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.steps <= 0 or not 0 < args.outer_test_size < 1 or not 0 < args.adaptation_fraction < 1:
        raise ValueError("need --episodes >= 3, positive steps, and fractions strictly in (0, 1)")
    if min(args.max_context_rows, args.train_context_rows, args.query_batch_rows, args.evaluation_query_chunk_rows) <= 0:
        raise ValueError("all row budgets must be positive")
    if args.transform_regularization < 0 or args.lr <= 0 or args.n_control_points <= 3:
        raise ValueError("invalid DirectSpline optimization configuration")
    if len(set(args.pmlb_dataset)) != len(args.pmlb_dataset) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("datasets and seeds must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone, _ = load_backbone(args, device)
    matrix_rows, selection_rows, summaries = [], [], []
    completed: set[tuple[str, int]] = set()
    summary_path = args.output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf8"))
        summaries = list(payload.get("runs", []))
        completed = {(str(row["dataset"]), int(row["seed"])) for row in summaries}
        for name, target in (("transfer_matrix.csv", matrix_rows), ("selection.csv", selection_rows)):
            path = args.output_dir / name
            if path.is_file():
                with path.open(newline="", encoding="utf8") as handle:
                    target.extend(csv.DictReader(handle))
    for dataset in args.pmlb_dataset:
        for seed in args.seeds:
            if (dataset, seed) in completed:
                print(f"[{dataset} seed={seed}] complete; skipping due to --resume", flush=True)
                continue
            rows, selected, summary = run_dataset_seed(args, backbone, dataset=dataset, seed=seed, device=device)
            matrix_rows.extend(rows); selection_rows.extend(selected); summaries.append(summary)
            write_csv(args.output_dir / "transfer_matrix.csv", matrix_rows)
            write_csv(args.output_dir / "selection.csv", selection_rows)
            write_json(summary_path, {"runs": summaries, "protocol": summary["protocol"]})
            print(json.dumps(summary, indent=2), flush=True)
    write_csv(args.output_dir / "transfer_matrix.csv", matrix_rows)
    write_csv(args.output_dir / "selection.csv", selection_rows)
    macro = {}
    if summaries:
        numeric = [key for key, value in summaries[0].items() if isinstance(value, (float, int)) and key not in {"seed", "episodes", "n_features", "n_classes"}]
        macro = {key: float(np.mean([float(row[key]) for row in summaries if row.get(key) is not None])) for key in numeric}
    write_json(summary_path, {"runs": summaries, "macro_mean_over_dataset_seed_runs": macro,
                              "protocol": "DirectSpline split-specificity oracle diagnostic; no HyperSpline is trained."})
    print(f"Wrote {len(matrix_rows)} transfer rows, {len(selection_rows)} selection rows, and {len(summaries)} run summaries.", flush=True)


if __name__ == "__main__":
    main()
