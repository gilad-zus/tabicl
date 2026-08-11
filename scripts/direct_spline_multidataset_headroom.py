"""Direct-spline oracle headroom pilot across bundled classification datasets.

This is not a zero-shot HyperSpline experiment: each DirectSpline is optimized
against that split's query labels.  Its purpose is to measure the upper-bound
benefit available to the spline family before amortizing it with HyperSpline.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import DirectSplineTransform
from tabicl._model.tabicl import TabICL


DatasetLoader = Callable[[], tuple[np.ndarray, np.ndarray]]


def _load_sklearn_dataset(loader: Callable) -> tuple[np.ndarray, np.ndarray]:
    data = loader()
    return np.asarray(data.data, dtype=np.float32), np.asarray(data.target)


DATASETS: dict[str, DatasetLoader] = {
    "iris": lambda: _load_sklearn_dataset(load_iris),
    "wine": lambda: _load_sklearn_dataset(load_wine),
    "breast_cancer": lambda: _load_sklearn_dataset(load_breast_cancer),
    "digits": lambda: _load_sklearn_dataset(load_digits),
}


def parse_csv(value: str, converter: Callable[[str], object]) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return [converter(item) for item in items]


def load_backbone(args: argparse.Namespace, device: torch.device) -> tuple[TabICL, Path]:
    # ``tensor.to(cuda:N)`` is explicit, but some internal/third-party CUDA
    # allocations use PyTorch's process-current device.  Pin it before model
    # construction so a caller's --device cuda:N is never silently split with
    # generic ``cuda`` allocations on GPU 0.
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {device}, but CUDA is unavailable")
        if device.index is not None:
            torch.cuda.set_device(device)
        current = torch.cuda.current_device()
        expected = current if device.index is None else device.index
        if current != expected:
            raise RuntimeError(f"failed to activate requested CUDA device {device}; current device is cuda:{current}")
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
    # TabICL's eval path uses no_grad inference managers.  Train mode selects
    # the differentiable path while parameters remain permanently frozen.
    backbone.to(device).train()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    print(
        f"Loaded frozen differentiable TabICL backbone ({parameter_count:,} parameters; dropout disabled).",
        flush=True,
    )
    return backbone, checkpoint_path


def prepare_split(
    name: str,
    seed: int,
    test_size: float,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = DATASETS[name]()
    _, y = np.unique(y, return_inverse=True)
    print(f"[{name} seed={seed}] loaded dataset: X={x.shape}, classes={np.unique(y).size}", flush=True)

    if max_rows > 0 and x.shape[0] > max_rows:
        x, _, y, _ = train_test_split(
            x, y, train_size=max_rows, random_state=seed, stratify=y
        )
        print(f"[{name} seed={seed}] stratified subsample: X={x.shape}", flush=True)

    x_context, x_query, y_context, y_query = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(
        f"[{name} seed={seed}] split: context={x_context.shape}, query={x_query.shape}",
        flush=True,
    )
    return x_context, x_query, y_context, y_query


def evaluate(
    backbone: TabICL,
    spline: DirectSplineTransform,
    x_context: torch.Tensor,
    x_query: torch.Tensor,
    y_context: torch.Tensor,
    y_query: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
        logits = backbone(transformed, y_context)
        loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
        accuracy = (logits.argmax(dim=-1).flatten() == y_query.flatten()).float().mean()
    return loss.item(), accuracy.item()


def run_split(
    backbone: TabICL,
    name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    backbone.clear_cache()

    x_context_np, x_query_np, y_context_np, y_query_np = prepare_split(
        name, seed, args.test_size, args.max_rows
    )
    if np.unique(y_context_np).size > backbone.max_classes:
        raise ValueError(
            f"{name} has {np.unique(y_context_np).size} classes but this TabICL checkpoint supports "
            f"at most {backbone.max_classes}"
        )

    x_context = torch.as_tensor(x_context_np, dtype=torch.float32, device=device).unsqueeze(0)
    x_query = torch.as_tensor(x_query_np, dtype=torch.float32, device=device).unsqueeze(0)
    y_context = torch.as_tensor(y_context_np, dtype=torch.float32, device=device).unsqueeze(0)
    y_query = torch.as_tensor(y_query_np, dtype=torch.long, device=device)

    spline = DirectSplineTransform(x_context, args.n_control_points).to(device)
    optimizer_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(spline.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    baseline_loss, baseline_accuracy = evaluate(backbone, spline, x_context, x_query, y_context, y_query)
    print(
        f"[{name} seed={seed}] baseline: loss={baseline_loss:.6f}, accuracy={baseline_accuracy:.4f}; "
        f"DirectSpline parameters={sum(parameter.numel() for parameter in spline.parameters())}",
        flush=True,
    )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
        logits = backbone(transformed, y_context)
        loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
        if step == 0:
            print(
                f"[{name} seed={seed}] first forward: transformed={tuple(transformed.shape)}, "
                f"logits={tuple(logits.shape)}, logits.requires_grad={logits.requires_grad}",
                flush=True,
            )
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"[{name} seed={seed}] step={step + 1} loss={loss.item():.6f}", flush=True)

    final_loss, final_accuracy = evaluate(backbone, spline, x_context, x_query, y_context, y_query)
    gate_mean = spline.parameters_for_transform().gate.mean().item()
    if device.type == "cuda":
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    else:
        peak_allocated_gib = 0.0
        peak_reserved_gib = 0.0

    result = {
        "dataset": name,
        "seed": seed,
        "n_context": int(x_context.shape[1]),
        "n_query": int(x_query.shape[1]),
        "n_features": int(x_context.shape[2]),
        "n_classes": int(np.unique(y_context_np).size),
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "loss_delta": baseline_loss - final_loss,
        "relative_loss_improvement": (baseline_loss - final_loss) / max(baseline_loss, 1e-12),
        "baseline_accuracy": baseline_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_delta": final_accuracy - baseline_accuracy,
        "final_gate_mean": gate_mean,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "steps": args.steps,
        "learning_rate": args.lr,
        "n_control_points": args.n_control_points,
    }
    print(
        f"[{name} seed={seed}] final: loss={final_loss:.6f} "
        f"(delta={result['loss_delta']:.6f}, relative={result['relative_loss_improvement']:.2%}), "
        f"accuracy={final_accuracy:.4f} (delta={result['accuracy_delta']:+.4f}), "
        f"gate_mean={gate_mean:.4f}, peak_allocated={peak_allocated_gib:.3f} GiB",
        flush=True,
    )
    return result


def write_results(path: Path, results: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} run records to {path}", flush=True)


def print_summary(results: list[dict[str, float | int | str]]) -> None:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for result in results:
        grouped[str(result["dataset"])].append(result)
    print("\nPer-dataset summary (mean across seeds):", flush=True)
    for name, records in grouped.items():
        relative = np.mean([float(record["relative_loss_improvement"]) for record in records])
        accuracy_delta = np.mean([float(record["accuracy_delta"]) for record in records])
        print(
            f"  {name}: relative_loss_improvement={relative:.2%}, "
            f"accuracy_delta={accuracy_delta:+.4f}, runs={len(records)}",
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
    parser.add_argument("--max-rows", type=int, default=1024, help="0 disables stratified row capping")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-csv", type=Path, default=Path("results/direct_spline_headroom_pilot.csv"))
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.steps <= 0 or args.log_every <= 0 or args.n_control_points <= 3:
        raise ValueError("--steps and --log-every must be positive; --n-control-points must exceed 3")
    datasets = parse_csv(args.datasets, str)
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choices are {sorted(DATASETS)}")
    seeds = parse_csv(args.seeds, int)

    device = torch.device(args.device)
    print(f"Running multi-dataset DirectSpline headroom pilot on device: {device}", flush=True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"Plan: datasets={datasets}, seeds={seeds}, steps={args.steps}, lr={args.lr}, "
        f"control_points={args.n_control_points}, max_rows={args.max_rows}",
        flush=True,
    )

    backbone, _ = load_backbone(args, device)
    results = [run_split(backbone, name, seed, args, device) for name in datasets for seed in seeds]
    write_results(args.output_csv, results)
    print_summary(results)


if __name__ == "__main__":
    main()
