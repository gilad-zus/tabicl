"""Joint multi-dataset HyperSpline overfit pilot.

The same datasets and fixed context/query splits are used for optimization and
evaluation.  This deliberately tests whether one shared statistics-to-spline
MLP can amortize DirectSpline-like improvements across several tasks.  It is
not a held-out zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import (
    HyperSplineTransform,
    backbone_state_dict_hash,
    save_hyperspline_checkpoint,
    summarize_context,
)
from tabicl._model.tabicl import TabICL


DatasetLoader = Callable[[], tuple[np.ndarray, np.ndarray]]


def _load_sklearn_dataset(loader: Callable) -> tuple[np.ndarray, np.ndarray]:
    data = loader()
    return np.asarray(data.data, dtype=np.float32), np.asarray(data.target)


def _load_openml_dataset(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a numeric OpenML classification dataset only when it is selected.

    OpenML data are cached by scikit-learn after the first download.  Keeping
    these loaders lazy means the bundled four-dataset pilot remains offline.
    """
    data = fetch_openml(data_id=data_id, as_frame=False)
    x = data.data.toarray() if hasattr(data.data, "toarray") else data.data
    return np.asarray(x, dtype=np.float32), np.asarray(data.target)


DATASETS: dict[str, DatasetLoader] = {
    "iris": lambda: _load_sklearn_dataset(load_iris),
    "wine": lambda: _load_sklearn_dataset(load_wine),
    "breast_cancer": lambda: _load_sklearn_dataset(load_breast_cancer),
    "digits": lambda: _load_sklearn_dataset(load_digits),
    # Harder numeric OpenML classification tasks.  They download on first use.
    "banknote_authentication": lambda: _load_openml_dataset(1462),
    "blood_transfusion": lambda: _load_openml_dataset(1464),
    "climate_model_crashes": lambda: _load_openml_dataset(1467),
    "phoneme": lambda: _load_openml_dataset(1489),
    "qsar_biodeg": lambda: _load_openml_dataset(1494),
    "spambase": lambda: _load_openml_dataset(44),
}


@dataclass(frozen=True)
class Episode:
    dataset: str
    split_seed: int
    x_context: torch.Tensor  # (1, N_C, D)
    x_query: torch.Tensor  # (1, N_Q, D)
    missing_context: torch.Tensor  # (1, N_C, D)
    missing_query: torch.Tensor  # (1, N_Q, D)
    y_context: torch.Tensor  # (1, N_C), float for TabICL
    y_query: torch.Tensor  # (N_Q,), long for cross entropy


def parse_csv(value: str, converter: Callable[[str], object]) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return [converter(item) for item in items]


def load_backbone(args: argparse.Namespace, device: torch.device) -> tuple[TabICL, Path]:
    if args.checkpoint is None:
        print(
            f"No checkpoint provided; resolving {args.checkpoint_version!r} from jingang/TabICL.",
            flush=True,
        )
        checkpoint_path = Path(
            hf_hub_download(repo_id="jingang/TabICL", filename=args.checkpoint_version)
        )
    else:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TabICL checkpoint does not exist: {checkpoint_path}")

    print(f"Loading TabICL checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain 'config' and 'state_dict'")

    backbone = TabICL(**checkpoint["config"])
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    # TabICL eval routes through no_grad inference managers. Train mode selects
    # its differentiable route; frozen parameters remain outside the optimizer.
    backbone.to(device).train()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    print(
        f"Loaded frozen differentiable TabICL backbone "
        f"({sum(parameter.numel() for parameter in backbone.parameters()):,} parameters; dropout disabled).",
        flush=True,
    )
    return backbone, checkpoint_path


def build_episode(name: str, split_seed: int, args: argparse.Namespace, device: torch.device) -> Episode:
    x, y = DATASETS[name]()
    _, y = np.unique(y, return_inverse=True)
    print(f"[{name} seed={split_seed}] loaded: X={x.shape}, classes={np.unique(y).size}", flush=True)
    if args.max_rows > 0 and x.shape[0] > args.max_rows:
        x, _, y, _ = train_test_split(x, y, train_size=args.max_rows, random_state=split_seed, stratify=y)
        print(f"[{name} seed={split_seed}] stratified subsample: X={x.shape}", flush=True)
    x_context, x_query, y_context, y_query = train_test_split(
        x, y, test_size=args.test_size, random_state=split_seed, stratify=y
    )
    print(
        f"[{name} seed={split_seed}] fixed overfit split: context={x_context.shape}, query={x_query.shape}",
        flush=True,
    )
    return Episode(
        dataset=name,
        split_seed=split_seed,
        x_context=torch.as_tensor(x_context, dtype=torch.float32, device=device).unsqueeze(0),
        x_query=torch.as_tensor(x_query, dtype=torch.float32, device=device).unsqueeze(0),
        missing_context=torch.zeros((1, *x_context.shape), dtype=torch.bool, device=device),
        missing_query=torch.zeros((1, *x_query.shape), dtype=torch.bool, device=device),
        y_context=torch.as_tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0),
        y_query=torch.as_tensor(y_query, dtype=torch.long, device=device),
    )


def forward_episode(
    backbone: TabICL,
    hyperspline: HyperSplineTransform,
    episode: Episode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Statistics depend only on context data. They have no learnable weights,
    # so retaining an autograd graph for their computation is unnecessary.
    with torch.no_grad():
        statistics = summarize_context(
            episode.x_context,
            episode.missing_context,
            y_context=None,
            eps=hyperspline.eps,
        )
    parameters = hyperspline.generate_parameters(statistics)
    transformed_context = hyperspline.apply_transform(
        episode.x_context, parameters, episode.missing_context
    )
    transformed_query = hyperspline.apply_transform(
        episode.x_query, parameters, episode.missing_query
    )
    transformed = torch.cat((transformed_context, transformed_query), dim=1)
    logits = backbone(transformed, episode.y_context)
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    return loss, logits, parameters.gate


def evaluate_episode(
    backbone: TabICL,
    hyperspline: HyperSplineTransform,
    episode: Episode,
) -> dict[str, float | int | str]:
    with torch.no_grad():
        loss, logits, gate = forward_episode(backbone, hyperspline, episode)
        accuracy = (logits.argmax(dim=-1).flatten() == episode.y_query).float().mean()
    return {
        "dataset": episode.dataset,
        "split_seed": episode.split_seed,
        "n_context": int(episode.x_context.shape[1]),
        "n_query": int(episode.x_query.shape[1]),
        "n_features": int(episode.x_context.shape[2]),
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "mean_gate": gate.mean().item(),
    }


def append_evaluation(
    records: list[dict[str, float | int | str]],
    phase: str,
    epoch: int,
    backbone: TabICL,
    hyperspline: HyperSplineTransform,
    episodes: list[Episode],
) -> float:
    metrics = [evaluate_episode(backbone, hyperspline, episode) for episode in episodes]
    mean_loss = float(np.mean([float(metric["loss"]) for metric in metrics]))
    mean_accuracy = float(np.mean([float(metric["accuracy"]) for metric in metrics]))
    for metric in metrics:
        records.append({"phase": phase, "epoch": epoch, **metric})
    print(
        f"[{phase} epoch={epoch}] mean_per_dataset_loss={mean_loss:.6f}, "
        f"mean_per_dataset_accuracy={mean_accuracy:.4f}",
        flush=True,
    )
    return mean_loss


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} evaluation records to {path}", flush=True)


def print_final_summary(records: list[dict[str, float | int | str]], final_epoch: int) -> None:
    initial = {(row["dataset"], row["split_seed"]): row for row in records if row["phase"] == "identity"}
    final = [row for row in records if row["phase"] == "trained" and row["epoch"] == final_epoch]
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in final:
        baseline = initial[(row["dataset"], row["split_seed"])]
        grouped[str(row["dataset"])].append(
            (float(baseline["loss"]) - float(row["loss"]), float(row["accuracy"]) - float(baseline["accuracy"]))
        )
    print("\nShared-HyperSpline overfit summary (same datasets/splits used for training):", flush=True)
    for dataset, deltas in grouped.items():
        print(
            f"  {dataset}: loss_delta={np.mean([delta[0] for delta in deltas]):+.6f}, "
            f"accuracy_delta={np.mean([delta[1] for delta in deltas]):+.4f}, runs={len(deltas)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", default="iris,wine,breast_cancer,digits")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("results/hyperspline_overfit_pilot.csv"))
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Optional trained HyperSpline checkpoint. This is an overfit artifact, not a zero-shot result.",
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.epochs <= 0 or args.log_every <= 0 or args.n_control_points <= 3:
        raise ValueError("--epochs and --log-every must be positive; --n-control-points must exceed 3")
    if not 0 < args.gate_initial_probability < 1:
        raise ValueError("--gate-initial-probability must be between 0 and 1")
    datasets = parse_csv(args.datasets, str)
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choices are {sorted(DATASETS)}")
    seeds = parse_csv(args.seeds, int)

    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    device = torch.device(args.device)
    print(f"Running shared-HyperSpline overfit pilot on device: {device}", flush=True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        torch.cuda.manual_seed_all(args.training_seed)
        print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"IMPORTANT: train and evaluation use the same fixed splits. "
        f"datasets={datasets}, seeds={seeds}, epochs={args.epochs}, lr={args.lr}, "
        f"control_points={args.n_control_points}, gate_init={args.gate_initial_probability}",
        flush=True,
    )

    backbone, _ = load_backbone(args, device)
    episodes = [build_episode(name, seed, args, device) for name in datasets for seed in seeds]
    if any(torch.unique(episode.y_context).numel() > backbone.max_classes for episode in episodes):
        raise ValueError(f"one or more datasets exceed this backbone's {backbone.max_classes}-class limit")

    hyperspline_config = {
        "n_control_points": args.n_control_points,
        "hidden_dim": args.hidden_dim,
        "gate_initial_probability": args.gate_initial_probability,
        "target_aware": False,
    }
    hyperspline = HyperSplineTransform(**hyperspline_config).to(device).train()
    optimizer = torch.optim.Adam(hyperspline.parameters(), lr=args.lr)
    print(
        f"Initialized shared HyperSpline with {sum(parameter.numel() for parameter in hyperspline.parameters()):,} "
        f"trainable parameters across {len(episodes)} fixed episodes.",
        flush=True,
    )

    records: list[dict[str, float | int | str]] = []
    append_evaluation(records, "identity", 0, backbone, hyperspline, episodes)
    rng = np.random.default_rng(args.training_seed)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        ordered_indices = rng.permutation(len(episodes))
        losses = []
        for position, index in enumerate(ordered_indices):
            episode = episodes[int(index)]
            backbone.clear_cache()
            loss, logits, _ = forward_episode(backbone, hyperspline, episode)
            if epoch == 1 and position == 0:
                print(
                    f"First training forward: dataset={episode.dataset}, "
                    f"context={tuple(episode.x_context.shape)}, query={tuple(episode.x_query.shape)}, "
                    f"logits={tuple(logits.shape)}, logits.requires_grad={logits.requires_grad}",
                    flush=True,
                )
            (loss / len(episodes)).backward()
            losses.append(loss.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), max_norm=1.0)
        if epoch == 1:
            print(
                f"First shared-HyperSpline backward: pre_clip_grad_norm={gradient_norm.item():.6g}",
                flush=True,
            )
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[train epoch={epoch}] mean_episode_loss={np.mean(losses):.6f}", flush=True)
            append_evaluation(records, "trained", epoch, backbone, hyperspline, episodes)

    write_csv(args.output_csv, records)
    print_final_summary(records, args.epochs)
    if args.output_checkpoint is not None:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_hyperspline_checkpoint(
            args.output_checkpoint,
            hyperspline,
            hyperspline_config,
            backbone_reference=args.checkpoint_version,
            backbone_hash=backbone_state_dict_hash(backbone),
            step=args.epochs,
        )
        print(f"Saved overfit HyperSpline checkpoint to {args.output_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
