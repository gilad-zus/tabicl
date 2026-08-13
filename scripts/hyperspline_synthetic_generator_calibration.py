"""Calibrate structural synthetic candidates without touching external validation/test data.

This is a generator-coverage experiment, not HyperSpline training.  It answers
whether a richer family of synthetic *tasks* can cover the exact 33-dimensional
``query_marginal`` inputs that the current HyperSpline receives.

Only ``--real-meta-bank`` is read.  Its dataset identities are deterministically
split into generator-fit and generator-selection identities.  Candidate
synthetic profiles are ranked using *fit identities only*.  The held-out
selection identities are reported once after the profile is frozen, and no
external real validation or final-test bank is accepted by this script.

The score deliberately does not optimize classifier AUC alone.  It combines
real-to-synthetic coverage, synthetic-to-real realism, coordinate-wise
distribution agreement, and source separability.  This prevents a synthetic
cloud from winning merely by confusing a source classifier while failing to
cover useful real descriptor regions.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

try:  # Supports tests and direct ``python scripts/...`` invocation.
    from scripts.hyperspline_query_marginal_synthetic_coverage_audit import (
        QUERY_MARGINAL_DIM,
        ColumnPoint,
        RobustScaler,
        column_rows,
        descriptor_matrix,
        extract_points,
        fit_robust_scaler,
        nearest_distances,
        source_auc,
        source_profile,
        write_csv,
        write_json,
    )
    from scripts.hyperspline_real_task_bank import load_bank
    from scripts.hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_scheduled_episodes,
        save_episode_bank,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation.
    from hyperspline_query_marginal_synthetic_coverage_audit import (
        QUERY_MARGINAL_DIM,
        ColumnPoint,
        RobustScaler,
        column_rows,
        descriptor_matrix,
        extract_points,
        fit_robust_scaler,
        nearest_distances,
        source_auc,
        source_profile,
        write_csv,
        write_json,
    )
    from hyperspline_real_task_bank import load_bank
    from hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_scheduled_episodes,
        save_episode_bank,
    )


DEFAULT_PROFILES = (
    "native",
    "coverage_expanded",
    "coverage_structural_light",
    "coverage_structural_broad",
)

# The 33 coordinates must remain in exactly the ordering used by
# query_marginal_descriptors.  These names make it possible to identify which
# part of the actual HyperSpline input still separates real from synthetic.
DESCRIPTOR_NAMES = (
    "observed_fraction", "missing_fraction", "unique_fraction",
    "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
    "z_mean", "z_std", "mad", "iqr", "skew", "kurtosis",
    "tail_gt1", "tail_gt2", "tail_gt4", "low_count", "all_missing",
    "class_count_norm", "class_entropy", "minimum_class_frequency",
    "between_class_spread", "within_class_spread", "between_within_ratio",
    "max_pairwise_class_mean_separation", "class_conditional_iqr",
    "context_query_relative_location", "context_query_log_scale_ratio",
)
if len(DESCRIPTOR_NAMES) != QUERY_MARGINAL_DIM:  # Guard against silent API drift.
    raise RuntimeError("descriptor coordinate names no longer match query_marginal input")

DESCRIPTOR_BLOCKS = {
    "marginal_shape": tuple(range(3, 21)),
    "discrete_missing": (0, 1, 2, 21, 22),
    "supervised_context": tuple(range(23, 31)),
    "context_query_shift": (31, 32),
    "all": tuple(range(QUERY_MARGINAL_DIM)),
}


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or not all(0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("expected unique context fractions in (0, 1)")
    return values


def parse_profile_csv(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(profiles).difference(SYNTHETIC_OBSERVATION_MODES))
    if not profiles or unknown:
        raise argparse.ArgumentTypeError(
            f"profiles must be a non-empty subset of {SYNTHETIC_OBSERVATION_MODES}; unknown={unknown}"
        )
    if len(profiles) != len(set(profiles)):
        raise argparse.ArgumentTypeError("profiles must be unique")
    return profiles


def split_episodes_by_dataset(
    episodes: Sequence[object], *, fit_fraction: float, seed: int
) -> tuple[list[object], list[object], tuple[str, ...], tuple[str, ...]]:
    """Split whole real dataset identities, never individual episodes.

    A dataset may have several stored splits.  Keeping all of them in one side
    prevents a sibling split from leaking its numerical distribution into the
    held-out generator-selection identities.
    """
    groups: dict[str, list[object]] = {}
    for episode in episodes:
        dataset = str(getattr(episode, "dataset"))
        groups.setdefault(dataset, []).append(episode)
    identities = sorted(groups)
    if len(identities) < 6:
        raise ValueError("the real meta bank needs at least six distinct dataset identities for a fit/selection split")
    if not 0.50 <= fit_fraction < 1.0:
        raise ValueError("--fit-fraction must be in [0.50, 1.0)")
    rng = np.random.default_rng(seed)
    ordered = np.asarray(identities, dtype=object)
    rng.shuffle(ordered)
    fit_count = int(round(len(identities) * fit_fraction))
    fit_count = min(max(fit_count, 4), len(identities) - 2)
    fit_ids = tuple(sorted(str(item) for item in ordered[:fit_count]))
    selection_ids = tuple(sorted(str(item) for item in ordered[fit_count:]))
    fit_set = set(fit_ids)
    fit = [episode for episode in episodes if str(getattr(episode, "dataset")) in fit_set]
    selection = [episode for episode in episodes if str(getattr(episode, "dataset")) not in fit_set]
    if not fit or not selection or set(fit_ids).intersection(selection_ids):
        raise AssertionError("invalid generator dataset split")
    return fit, selection, fit_ids, selection_ids


def _with_descriptor_subset(points: Sequence[ColumnPoint], indices: Sequence[int]) -> list[ColumnPoint]:
    return [replace(point, descriptor=point.descriptor[np.asarray(indices, dtype=int)]) for point in points]


def macro_mean(values: np.ndarray, points: Sequence[ColumnPoint]) -> float:
    """Equal-weight a distance per real/synthetic task identity."""
    by_identity: dict[str, list[float]] = {}
    for point, value in zip(points, values, strict=True):
        by_identity.setdefault(point.identity, []).append(float(value))
    return float(np.mean([np.mean(group) for group in by_identity.values()]))


def descriptor_quantile_gaps(
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
) -> tuple[float, list[dict[str, float | str]]]:
    """Return one robust per-coordinate shape gap in real-fit-scaled units."""
    levels = np.linspace(0.05, 0.95, 19)
    real_quantiles = np.quantile(real_values, levels, axis=0)
    synthetic_quantiles = np.quantile(synthetic_values, levels, axis=0)
    gaps = np.mean(np.abs(real_quantiles - synthetic_quantiles), axis=0)
    rows = [
        {"descriptor_index": index, "descriptor": DESCRIPTOR_NAMES[index], "quantile_l1_gap": float(gap)}
        for index, gap in enumerate(gaps)
    ]
    return float(np.mean(gaps)), rows


def profile_metrics(
    real_points: Sequence[ColumnPoint],
    synthetic_points: Sequence[ColumnPoint],
    *,
    scaler: RobustScaler,
    auc_seed: int,
) -> tuple[dict[str, float | int], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    """Compute coverage and source diagnostics for one frozen candidate."""
    real_values = scaler.transform(descriptor_matrix(real_points))
    synthetic_values = scaler.transform(descriptor_matrix(synthetic_points))
    real_to_synthetic = nearest_distances(synthetic_values, real_values)
    synthetic_to_real = nearest_distances(real_values, synthetic_values)
    quantile_gap, gap_rows = descriptor_quantile_gaps(real_values, synthetic_values)
    all_auc = source_auc(real_points, synthetic_points, seed=auc_seed)
    identity_rows = []
    for identity in sorted({point.identity for point in real_points}):
        mask = np.asarray([point.identity == identity for point in real_points])
        identity_rows.append(
            {
                "identity": identity,
                "n_columns": int(mask.sum()),
                "mean_nearest_synthetic": float(real_to_synthetic[mask].mean()),
                "median_nearest_synthetic": float(np.median(real_to_synthetic[mask])),
            }
        )
    summary = {
        "n_real_columns": len(real_points),
        "n_synthetic_columns": len(synthetic_points),
        "macro_real_to_synthetic_nn": macro_mean(real_to_synthetic, real_points),
        "mean_real_to_synthetic_nn": float(real_to_synthetic.mean()),
        "median_real_to_synthetic_nn": float(np.median(real_to_synthetic)),
        "macro_synthetic_to_real_nn": macro_mean(synthetic_to_real, synthetic_points),
        "mean_synthetic_to_real_nn": float(synthetic_to_real.mean()),
        "descriptor_quantile_l1_gap": quantile_gap,
        "source_auc": float(all_auc["auc"]),
        "source_auc_n_splits": int(all_auc["n_splits"]),
        "real_covered_by_distance_2": float(np.mean(real_to_synthetic <= 2.0)),
        "real_covered_by_distance_3": float(np.mean(real_to_synthetic <= 3.0)),
    }
    return summary, gap_rows, identity_rows + [{"identity": "__synthetic_profile__", "n_columns": len(synthetic_points), "mean_nearest_synthetic": float(synthetic_to_real.mean()), "median_nearest_synthetic": float(np.median(synthetic_to_real))}]


def calibrate_profiles(rows: Sequence[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Rank candidates using only fit metrics, with native as a scale reference."""
    by_profile = {str(row["profile"]): row for row in rows}
    if "native" not in by_profile:
        raise ValueError("--profiles must include native so calibration has an interpretable reference")
    native = by_profile["native"]

    def ratio(value: float, baseline: float) -> float:
        return value / max(baseline, 1e-8)

    baseline_auc_gap = max(float(native["source_auc"]) - 0.5, 1e-4)
    ranked = []
    for row in rows:
        scored = dict(row)
        source_gap_ratio = max(float(row["source_auc"]) - 0.5, 0.0) / baseline_auc_gap
        scored["real_to_synthetic_ratio_to_native"] = ratio(
            float(row["macro_real_to_synthetic_nn"]), float(native["macro_real_to_synthetic_nn"])
        )
        scored["synthetic_to_real_ratio_to_native"] = ratio(
            float(row["macro_synthetic_to_real_nn"]), float(native["macro_synthetic_to_real_nn"])
        )
        scored["quantile_gap_ratio_to_native"] = ratio(
            float(row["descriptor_quantile_l1_gap"]), float(native["descriptor_quantile_l1_gap"])
        )
        scored["source_auc_gap_ratio_to_native"] = source_gap_ratio
        # AUC is only 15%: the remaining terms demand bidirectional coverage
        # and coordinate-wise distribution agreement in the actual input space.
        scored["fit_calibration_score"] = (
            0.45 * float(scored["real_to_synthetic_ratio_to_native"])
            + 0.20 * float(scored["synthetic_to_real_ratio_to_native"])
            + 0.20 * float(scored["quantile_gap_ratio_to_native"])
            + 0.15 * source_gap_ratio
        )
        ranked.append(scored)
    ranked.sort(key=lambda row: (float(row["fit_calibration_score"]), str(row["profile"])))
    for rank, row in enumerate(ranked, start=1):
        row["fit_rank"] = rank
    return ranked


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if materialized:
        write_csv(path, materialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-meta-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--dataset-split-seed", type=int, default=73_001)
    parser.add_argument("--profiles", type=parse_profile_csv, default=list(DEFAULT_PROFILES))
    parser.add_argument("--synthetic-tasks", type=int, default=384)
    parser.add_argument("--synthetic-sequence-lengths", type=parse_int_csv, default=parse_int_csv("128,256,512,1024"))
    parser.add_argument("--synthetic-context-fractions", type=parse_float_csv, default=parse_float_csv("0.50,0.70,0.85"))
    parser.add_argument("--synthetic-seed", type=int, default=81_001)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="Descriptor extraction device; episode banks stay on CPU.")
    parser.add_argument("--summary-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--scaler-clip", type=float, default=10.0)
    parser.add_argument("--auc-seed", type=int, default=17)
    parser.add_argument("--write-column-descriptors", action="store_true")
    parser.add_argument(
        "--selected-bank-output", type=Path, default=None,
        help="Optional CPU bank of the profile selected using generator-fit identities only.",
    )
    args = parser.parse_args()
    if args.synthetic_tasks <= 0 or args.summary_batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("synthetic task count, summary batch size, and progress interval must be positive")
    if not 0 < args.min_features <= args.max_features or args.max_classes < 2:
        raise ValueError("invalid synthetic feature/class limits")
    if args.scaler_clip <= 0:
        raise ValueError("--scaler-clip must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requested CUDA but CUDA is unavailable")
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        current = torch.cuda.current_device()
        expected = current if device.index is None else device.index
        if current != expected:
            raise RuntimeError(f"failed to activate requested --device {device}; current CUDA device is cuda:{current}")
        print(f"CUDA routing: descriptor mini-batches={device}; stored episodes remain on CPU.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # External validation/test banks are purposefully not command-line inputs:
    # they cannot leak their numerical statistics into candidate selection.
    _, real_all = load_bank(args.real_meta_bank, device=torch.device("cpu"))
    fit_episodes, selection_episodes, fit_ids, selection_ids = split_episodes_by_dataset(
        real_all, fit_fraction=args.fit_fraction, seed=args.dataset_split_seed
    )
    print(
        f"Generator split: fit={len(fit_ids)} datasets/{len(fit_episodes)} episodes; "
        f"selection={len(selection_ids)} datasets/{len(selection_episodes)} episodes. "
        "Candidate choice will use fit only.",
        flush=True,
    )
    real_sources = {"real_fit": fit_episodes, "real_selection": selection_episodes}
    real_points = {
        name: extract_points(items, source=name, batch_size=args.summary_batch_size,
                             progress_every=args.progress_every, device=device)
        for name, items in real_sources.items()
    }
    for source, points in real_points.items():
        print(f"[{source}] extracted {len(points)} numerical-column descriptors", flush=True)
        if args.write_column_descriptors:
            _write_rows(args.output_dir / f"{source}_columns.csv", column_rows(points))
    scaler = fit_robust_scaler(real_points["real_fit"], clip=args.scaler_clip)

    schedule = {
        "sequence_lengths": args.synthetic_sequence_lengths,
        "context_fractions": args.synthetic_context_fractions,
    }
    candidate_episodes: dict[str, list[SyntheticEpisode]] = {}
    candidate_points: dict[str, list[ColumnPoint]] = {}
    for profile_index, profile in enumerate(args.profiles):
        print(
            f"[candidate {profile_index + 1}/{len(args.profiles)}:{profile}] generating "
            f"{args.synthetic_tasks} fixed tasks (prior={args.prior_type})...",
            flush=True,
        )
        episodes = generate_scheduled_episodes(
            args,
            args.synthetic_tasks,
            source_seed=args.synthetic_seed,
            task_offset=0,
            device=torch.device("cpu"),
            observation_mode=profile,
            **schedule,
        )
        candidate_episodes[profile] = episodes
        points = extract_points(
            episodes, source=f"synthetic_{profile}", batch_size=args.summary_batch_size,
            progress_every=args.progress_every, device=device,
        )
        candidate_points[profile] = points
        print(f"[candidate:{profile}] extracted {len(points)} numerical-column descriptors", flush=True)
        if args.write_column_descriptors:
            _write_rows(args.output_dir / f"synthetic_{profile}_columns.csv", column_rows(points))

    fit_rows, selection_rows = [], []
    fit_gap_rows, selection_gap_rows = [], []
    fit_identity_rows, selection_identity_rows = [], []
    fit_auc_rows, selection_auc_rows = [], []
    for profile in args.profiles:
        for partition, points, destination, gap_destination, identity_destination, auc_destination in (
            ("fit", real_points["real_fit"], fit_rows, fit_gap_rows, fit_identity_rows, fit_auc_rows),
            ("selection", real_points["real_selection"], selection_rows, selection_gap_rows, selection_identity_rows, selection_auc_rows),
        ):
            metrics, gaps, identity_rows = profile_metrics(
                points, candidate_points[profile], scaler=scaler, auc_seed=args.auc_seed
            )
            row = {"partition": partition, "profile": profile, **metrics}
            destination.append(row)
            gap_destination.extend({"partition": partition, "profile": profile, **gap} for gap in gaps)
            identity_destination.extend({"partition": partition, "profile": profile, **value} for value in identity_rows)
            for block_name, indices in DESCRIPTOR_BLOCKS.items():
                auc = source_auc(
                    _with_descriptor_subset(points, indices),
                    _with_descriptor_subset(candidate_points[profile], indices),
                    seed=args.auc_seed,
                )
                auc_destination.append({"partition": partition, "profile": profile, "descriptor_block": block_name, **auc})
            print(
                f"[{partition}:{profile}] real->synthetic={metrics['macro_real_to_synthetic_nn']:.4f}, "
                f"synthetic->real={metrics['macro_synthetic_to_real_nn']:.4f}, "
                f"qgap={metrics['descriptor_quantile_l1_gap']:.4f}, auc={metrics['source_auc']:.4f}",
                flush=True,
            )

    ranked_fit_rows = calibrate_profiles(fit_rows)
    selected_profile = str(ranked_fit_rows[0]["profile"])
    selection_by_profile = {str(row["profile"]): row for row in selection_rows}
    selected_selection = selection_by_profile[selected_profile]
    _write_rows(args.output_dir / "fit_profile_metrics.csv", ranked_fit_rows)
    _write_rows(args.output_dir / "selection_profile_metrics.csv", selection_rows)
    _write_rows(args.output_dir / "fit_descriptor_quantile_gaps.csv", fit_gap_rows)
    _write_rows(args.output_dir / "selection_descriptor_quantile_gaps.csv", selection_gap_rows)
    _write_rows(args.output_dir / "fit_coverage_by_identity.csv", fit_identity_rows)
    _write_rows(args.output_dir / "selection_coverage_by_identity.csv", selection_identity_rows)
    _write_rows(args.output_dir / "fit_source_auc_blocks.csv", fit_auc_rows)
    _write_rows(args.output_dir / "selection_source_auc_blocks.csv", selection_auc_rows)

    if args.selected_bank_output is not None:
        save_episode_bank(args.selected_bank_output, candidate_episodes[selected_profile], source_seed=args.synthetic_seed)
    summary = {
        "protocol": {
            "purpose": "leakage-safe structural synthetic-generator calibration; no HyperSpline/teacher training",
            "descriptor": "current 33D query_marginal HyperSpline input",
            "real_data_access": "only --real-meta-bank; split by entire dataset identity",
            "candidate_selection": "fit identities only; selection identities are report-only",
            "external_validation_or_test_banks": "not accepted or read",
            "score": {
                "weights": {
                    "real_to_synthetic_coverage": 0.45,
                    "synthetic_to_real_realism": 0.20,
                    "per_coordinate_quantile_agreement": 0.20,
                    "source_auc_gap": 0.15,
                },
                "interpretation": "lower is better; each term is normalized to native fit performance",
            },
            "source_auc": "group-disjoint logistic-regression diagnostic; it is not the sole selection objective",
            "label_use": "structural candidates may generate class imbalance/noise before the ICL split; query labels never enter descriptors",
            "missingness": "not injected because the current numeric episode protocol has no missingness mask",
        },
        "arguments": vars(args),
        "dataset_split": {"fit_identities": fit_ids, "selection_identities": selection_ids},
        "selected_profile": selected_profile,
        "fit_ranking": ranked_fit_rows,
        "selected_profile_heldout_selection_metrics": selected_selection,
        "source_profiles": {
            "real_fit": source_profile(real_points["real_fit"], scaler=scaler),
            "real_selection": source_profile(real_points["real_selection"], scaler=scaler),
            **{f"synthetic_{profile}": source_profile(points, scaler=scaler) for profile, points in candidate_points.items()},
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Calibration complete: selected_profile={selected_profile} using fit identities only; "
        f"held-out selection real->synthetic={selected_selection['macro_real_to_synthetic_nn']:.4f}, "
        f"AUC={selected_selection['source_auc']:.4f}. Outputs: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
