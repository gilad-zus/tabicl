"""Train shared HyperSpline on meta-train datasets and evaluate held-out datasets.

Normal mode uses explicit disjoint meta-train, meta-validation, and meta-test
dataset groups.  ``--leave-one-dataset-out`` instead trains one model per fold
on every remaining dataset and evaluates the held-out dataset without updates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from hyperspline_multidataset_overfit import (
    DATASETS,
    Episode,
    build_episode,
    evaluate_episode,
    forward_episode,
    load_backbone,
    parse_csv,
)
from tabicl._hyperspline import HyperSplineTransform, backbone_state_dict_hash, save_hyperspline_checkpoint


def evaluate_group(backbone, hyperspline, group: str, episodes: list[Episode], epoch: int, records: list) -> float:
    values = []
    for episode in episodes:
        backbone.clear_cache()
        metrics = evaluate_episode(backbone, hyperspline, episode)
        records.append({"phase": group, "epoch": epoch, **metrics})
        values.append(float(metrics["loss"]))
    mean_loss = float(np.mean(values))
    print(f"[{group} epoch={epoch}] mean_per_episode_loss={mean_loss:.6f}", flush=True)
    return mean_loss


def make_episodes(names: list[str], seeds: list[int], args, device: torch.device) -> list[Episode]:
    episodes = [build_episode(name, seed, args, device) for name in names for seed in seeds]
    if any(torch.unique(episode.y_context).numel() > args.backbone_max_classes for episode in episodes):
        raise ValueError("one or more datasets exceed this backbone's class limit")
    return episodes


def train_fold(backbone, train_episodes, validation_episodes, test_episodes, args, fold: str, records: list):
    config = {
        "n_control_points": args.n_control_points,
        "hidden_dim": args.hidden_dim,
        "gate_initial_probability": args.gate_initial_probability,
        "target_aware": False,
    }
    hyperspline = HyperSplineTransform(**config).to(args.device).train()
    optimizer = torch.optim.Adam(hyperspline.parameters(), lr=args.lr)
    print(f"[{fold}] initialized HyperSpline with {sum(p.numel() for p in hyperspline.parameters()):,} trainable parameters", flush=True)
    evaluate_group(backbone, hyperspline, f"{fold}/identity_train", train_episodes, 0, records)
    if validation_episodes:
        evaluate_group(backbone, hyperspline, f"{fold}/identity_validation", validation_episodes, 0, records)
    evaluate_group(backbone, hyperspline, f"{fold}/identity_test", test_episodes, 0, records)

    best_validation = float("inf")
    best_state = None
    rng = np.random.default_rng(args.training_seed)
    for epoch in range(1, args.epochs + 1):
        hyperspline.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in rng.permutation(len(train_episodes)):
            backbone.clear_cache()
            loss, _, _ = forward_episode(backbone, hyperspline, train_episodes[int(index)])
            (loss / len(train_episodes)).backward()
            losses.append(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[{fold} train epoch={epoch}] mean_train_loss={np.mean(losses):.6f}", flush=True)
            hyperspline.eval()
            evaluate_group(backbone, hyperspline, f"{fold}/trained_train", train_episodes, epoch, records)
            validation_loss = evaluate_group(backbone, hyperspline, f"{fold}/validation", validation_episodes, epoch, records) if validation_episodes else float(np.mean(losses))
            evaluate_group(backbone, hyperspline, f"{fold}/test", test_episodes, epoch, records)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in hyperspline.state_dict().items()}

    if best_state is not None:
        hyperspline.load_state_dict(best_state)
        hyperspline.eval()
        evaluate_group(backbone, hyperspline, f"{fold}/selected_train", train_episodes, args.epochs, records)
        if validation_episodes:
            evaluate_group(backbone, hyperspline, f"{fold}/selected_validation", validation_episodes, args.epochs, records)
        evaluate_group(backbone, hyperspline, f"{fold}/selected_test", test_episodes, args.epochs, records)
    return hyperspline, config, best_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--meta-train", default="iris,wine,breast_cancer,banknote_authentication,qsar_biodeg")
    parser.add_argument("--meta-validation", default="blood_transfusion,climate_model_crashes")
    parser.add_argument("--meta-test", default="digits,phoneme,spambase")
    parser.add_argument("--leave-one-dataset-out", default=None, help="Comma-separated datasets; overrides the three meta groups.")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="3,4,5")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("results/hyperspline_meta_generalization.csv"))
    parser.add_argument("--output-checkpoint-dir", type=Path, default=Path("results/hyperspline_meta_checkpoints"))
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or args.epochs <= 0 or args.log_every <= 0 or args.n_control_points <= 3:
        raise ValueError("invalid split or optimization arguments")
    device = torch.device(args.device)
    args.device = device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    backbone, _ = load_backbone(args, device)
    args.backbone_max_classes = backbone.max_classes
    train_seeds, eval_seeds = parse_csv(args.train_seeds, int), parse_csv(args.eval_seeds, int)
    if set(train_seeds).intersection(eval_seeds):
        raise ValueError("--train-seeds and --eval-seeds must be disjoint")

    if args.leave_one_dataset_out:
        lodo = parse_csv(args.leave_one_dataset_out, str)
        unknown = sorted(set(lodo).difference(DATASETS))
        if unknown or len(lodo) < 2:
            raise ValueError(f"LODO requires at least two known datasets; unknown={unknown}")
        folds = [(f"lodo_holdout_{name}", [other for other in lodo if other != name], [], [name]) for name in lodo]
    else:
        groups = [parse_csv(args.meta_train, str), parse_csv(args.meta_validation, str), parse_csv(args.meta_test, str)]
        unknown = sorted(set().union(*map(set, groups)).difference(DATASETS))
        if unknown or set(groups[0]) & set(groups[1]) or set(groups[0]) & set(groups[2]) or set(groups[1]) & set(groups[2]):
            raise ValueError(f"meta groups must be disjoint known datasets; unknown={unknown}")
        folds = [("meta", *groups)]

    records = []
    for fold, train_names, validation_names, test_names in folds:
        print(f"\n[{fold}] train={train_names}; validation={validation_names}; test={test_names}", flush=True)
        train_episodes = make_episodes(train_names, train_seeds, args, device)
        validation_episodes = make_episodes(validation_names, eval_seeds, args, device) if validation_names else []
        test_episodes = make_episodes(test_names, eval_seeds, args, device)
        hyperspline, config, best_validation = train_fold(backbone, train_episodes, validation_episodes, test_episodes, args, fold, records)
        checkpoint_path = args.output_checkpoint_dir / f"{fold}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_hyperspline_checkpoint(checkpoint_path, hyperspline, config, backbone_reference=args.checkpoint_version, backbone_hash=backbone_state_dict_hash(backbone), step=args.epochs)
        print(f"[{fold}] best_validation_loss={best_validation:.6f}; saved {checkpoint_path}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
