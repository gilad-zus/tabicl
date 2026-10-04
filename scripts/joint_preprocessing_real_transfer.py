"""Fixed real-family transfer panel for the synthetic-only joint generator.

``prepare`` only encodes context-fitted data and locks splits. ``report`` requires
all six validation-selected synthetic checkpoints and never updates a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.hyperspline_real_task_bank import load_pmlb_frame
from scripts.hyperspline_real_zero_shot_eval import DatasetSpec, load_dataset_frame
from scripts.joint_preprocessing_synthetic_pilot import (
    ARMS, comparison_summary, frozen_inference, hash_file, json_write,
    load_frozen, manifest_path, read_manifest,
)
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor
from tabicl._hyperspline.statistics import summarize_context
from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical

FAMILY_MANIFEST = Path(__file__).resolve().parents[1] / "docs/experiments/joint_preprocessing_real_transfer_manifest_20260928.json"


def _take_rows(frame, indices):
    return frame.iloc[indices].reset_index(drop=True) if hasattr(frame, "iloc") else frame[indices]


def _load_source(entry: dict[str, Any], cache_dir: Path):
    source, name = entry["source"], entry["name"]
    if source == "pmlb":
        frame, target, _ = load_pmlb_frame(name, cache_dir=cache_dir)
        content_hash = hash_file(cache_dir / f"{name}.tsv.gz")
    elif source in {"sklearn", "openml"}:
        frame, target = load_dataset_frame(DatasetSpec(name, "mixed" if source == "openml" else "numerical_only"))
        content_hash = None
    else:
        raise ValueError(f"unsupported family source {source!r}")
    target = np.asarray(target)
    valid = np.asarray([value is not None and str(value) != "nan" for value in target], dtype=bool)
    frame = _take_rows(frame, np.flatnonzero(valid))
    target = LabelEncoder().fit_transform(target[valid])
    return frame, target, content_hash


def _episode(frame, labels: np.ndarray, *, family: str, seed: int, max_rows: int,
             test_fraction: float) -> dict[str, Any]:
    if len(labels) < 256:
        raise ValueError(f"{family}: fewer than 256 usable rows")
    indices = np.arange(len(labels))
    if len(indices) > max_rows:
        indices, _ = train_test_split(indices, train_size=max_rows, stratify=labels, random_state=seed)
    sampled_labels = LabelEncoder().fit_transform(labels[indices])
    context, query = train_test_split(np.arange(len(indices)), test_size=test_fraction,
                                      stratify=sampled_labels, random_state=seed)
    y_context, y_query = sampled_labels[context], sampled_labels[query]
    classes = np.unique(sampled_labels)
    if not 2 <= len(classes) <= 10 or not np.array_equal(np.unique(y_context), classes) or not np.array_equal(np.unique(y_query), classes):
        raise ValueError(f"{family}: class coverage or count failed for split {seed}")
    x_context_raw = _take_rows(frame, indices[context])
    x_query_raw = _take_rows(frame, indices[query])
    encoder = TransformToNumerical().fit(x_context_raw)
    context_parts = encoder.transform_parts(x_context_raw)
    query_parts = encoder.transform_parts(x_query_raw)
    n_categorical = context_parts.categorical.shape[1]
    n_numerical = context_parts.numerical.shape[1]
    n_encoded = n_categorical + n_numerical
    if not 5 <= n_encoded <= 100 or n_numerical < 1:
        raise ValueError(f"{family}: expected 5-100 encoded features and at least one numerical feature, got {n_encoded}/{n_numerical}")
    x_context = np.concatenate((context_parts.categorical, context_parts.numerical), axis=1).astype(np.float32)
    x_query = np.concatenate((query_parts.categorical, query_parts.numerical), axis=1).astype(np.float32)
    if not np.isfinite(x_context).all() or not np.isfinite(x_query).all():
        raise ValueError(f"{family}: encoded features are nonfinite")
    return dict(family=family, split_seed=seed, n_classes=len(classes),
                context_indices=torch.from_numpy(indices[context].copy()),
                query_indices=torch.from_numpy(indices[query].copy()),
                x_context=torch.from_numpy(x_context).unsqueeze(0),
                x_query=torch.from_numpy(x_query).unsqueeze(0),
                y_context=torch.from_numpy(y_context.astype(np.float32)).unsqueeze(0),
                y_query=torch.from_numpy(y_query.astype(np.int64)),
                numerical_mask=torch.tensor([False] * n_categorical + [True] * n_numerical),
                context_missing=torch.from_numpy(context_parts.numerical_missing.copy()),
                query_missing=torch.from_numpy(query_parts.numerical_missing.copy()))


def prepare(args: argparse.Namespace) -> None:
    source_path = args.family_manifest
    source = json.loads(source_path.read_text(encoding="utf8"))
    if len(source["families"]) != 20 or len({(x["source"], x["name"]) for x in source["families"]}) != 20:
        raise ValueError("the locked real panel must list 20 unique families")
    output = args.output_dir / "real_transfer"
    bank_path = output / "bank.pt"
    if bank_path.exists() or (output / "manifest.json").exists():
        raise FileExistsError("real-transfer bank already exists; use a fresh output root")
    episodes, sources, failures = [], [], []
    for entry in source["families"]:
        family = entry["name"]
        try:
            frame, labels, content_hash = _load_source(entry, args.cache_dir)
            family_episodes = [_episode(frame, labels, family=family, seed=seed,
                                        max_rows=int(source["max_rows"]),
                                        test_fraction=float(source["test_fraction"]))
                               for seed in source["split_seeds"]]
            episodes.extend(family_episodes)
            sources.append(dict(**entry, n_rows=len(labels), raw_content_sha256=content_hash,
                                encoded_features=family_episodes[0]["x_context"].shape[-1]))
            print(f"Prepared {family}: {len(labels)} rows, {family_episodes[0]['x_context'].shape[-1]} encoded features", flush=True)
        except Exception as error:
            failures.append(dict(**entry, error=f"{type(error).__name__}: {error}"))
            print(f"Failed {family}: {type(error).__name__}: {error}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "availability.json", dict(eligible=sources, failures=failures,
                                                   source_manifest_sha256=hash_file(source_path)))
    if failures:
        raise RuntimeError("real panel has unavailable/ineligible families; resolve before freezing or testing checkpoints")
    torch.save(dict(format_version=1, episodes=episodes), bank_path)
    json_write(output / "manifest.json", dict(format_version=1, source_manifest_sha256=hash_file(source_path),
                                               bank_sha256=hash_file(bank_path), families=sources,
                                               n_episodes=len(episodes)))
    print(f"Prepared frozen real-transfer bank: {bank_path}", flush=True)


def _prepared_views(episode: dict[str, Any], estimators: int = 8):
    x_context = episode["x_context"].squeeze(0).numpy()
    x_query = episode["x_query"].squeeze(0).numpy()
    y_context = episode["y_context"].squeeze(0).numpy().astype(int)
    generator = EnsembleGenerator(classification=True, n_estimators=estimators,
                                  norm_methods=["none", "power"], feat_shuffle_method="latin",
                                  class_shuffle_method="shift", random_state=0).fit(x_context, y_context)
    members = generator.transform(x_query, mode="both")
    keep = generator.unique_filter_.features_to_keep_
    numerical = episode["numerical_mask"].numpy()
    numerical_keep = keep[numerical]
    canonical_numerical = np.flatnonzero(numerical[keep])
    if not len(canonical_numerical):
        raise ValueError(f"{episode['family']}: all numerical features were constant in context")
    return generator, members, canonical_numerical, numerical_keep, keep


def episode_logits(backbone, model: JointPreprocessor | str, episode: dict[str, Any],
                   estimators: int = 8) -> tuple[torch.Tensor, int]:
    """Predict from context labels and query features; query labels are never read."""
    generator, members, numerical_positions, numerical_keep, keep = _prepared_views(episode, estimators)
    device = next(backbone.parameters()).device
    context = episode["x_context"][..., keep].to(device)
    query = episode["x_query"][..., keep].to(device)
    y_context = episode["y_context"].to(device)
    xc_num = context[..., numerical_positions]
    xq_num = query[..., numerical_positions]
    missing_context = episode["context_missing"][None, :, numerical_keep].to(device)
    missing_query = episode["query_missing"][None, :, numerical_keep].to(device)
    if isinstance(model, JointPreprocessor):
        with torch.no_grad():
            generated = model.generate(xc_num, y_context, missing_context)
        numeric_slots = [(model.apply(xc_num, generated, slot, missing_context),
                          model.apply(xq_num, generated, slot, missing_query)) for slot in (0, 1)]
    elif model == "identity":
        with torch.no_grad():
            stats = summarize_context(xc_num, missing_context, y_context)
            base_context = ((xc_num - stats.location[:, None]) / stats.scale[:, None]).masked_fill(missing_context, 0)
            base_query = ((xq_num - stats.location[:, None]) / stats.scale[:, None]).masked_fill(missing_query, 0)
        numeric_slots = [(base_context, base_query)] * 2
    elif model == "ordinary":
        numeric_slots = []
    else:
        raise ValueError("model must be a learned arm, identity or ordinary")
    logits = []
    with torch.no_grad(), frozen_inference(backbone):
        for method, (xs, ys) in members.items():
            slot = 0 if method == "none" else 1
            for index, (feature_shuffle, class_shuffle) in enumerate(generator.ensemble_configs_[method]):
                x_view = torch.from_numpy(xs[index:index + 1]).to(device=device, dtype=torch.float32)
                if numeric_slots:
                    c_num, q_num = numeric_slots[slot]
                    merged = torch.cat((c_num, q_num), dim=1)
                    x_view = x_view.clone()
                    inverse = {int(original): position for position, original in enumerate(feature_shuffle)}
                    for column, original in enumerate(numerical_positions):
                        x_view[..., inverse[int(original)]] = merged[..., column]
                backbone.clear_cache()
                raw = backbone(x_view, torch.from_numpy(ys[index:index + 1]).to(device=device, dtype=torch.float32),
                               feature_shuffles=[list(feature_shuffle)], return_logits=True)
                class_index = torch.as_tensor(class_shuffle, device=device, dtype=torch.long)
                logits.append(raw[..., :episode["n_classes"]][..., class_index])
    average = torch.stack(logits).mean(dim=0)
    return average, len(logits)


def evaluate_episode(backbone, model: JointPreprocessor | str, episode: dict[str, Any]) -> dict[str, float]:
    average, _ = episode_logits(backbone, model, episode)
    device = average.device
    labels = episode["y_query"].to(device)
    log_prob = F.log_softmax(average.float() / 0.9, dim=-1).flatten(0, 1)
    nll = float(F.nll_loss(log_prob, labels))
    accuracy = float((log_prob.argmax(dim=-1) == labels).float().mean())
    auc = float("nan")
    if episode["n_classes"] == 2 and torch.unique(labels).numel() == 2:
        auc = float(roc_auc_score(labels.cpu().numpy(), log_prob[:, 1].exp().cpu().numpy()))
    if not np.isfinite(nll):
        raise FloatingPointError(f"{episode['family']} split {episode['split_seed']}: nonfinite NLL")
    return dict(nll=nll, accuracy=accuracy, auc=auc)


def report(args: argparse.Namespace) -> None:
    synthetic_manifest = read_manifest(args.output_dir)
    real_root = args.output_dir / "real_transfer"
    real_manifest = json.loads((real_root / "manifest.json").read_text(encoding="utf8"))
    if real_manifest["source_manifest_sha256"] != hash_file(args.family_manifest) or hash_file(real_root / "bank.pt") != real_manifest["bank_sha256"]:
        raise ValueError("real source manifest or frozen bank hash mismatch")
    selected_paths = {(arm, seed): args.output_dir / "runs" / f"{arm}_seed{seed}" / "selected.pt"
                      for arm in ARMS for seed in (0, 1)}
    if any(not path.is_file() for path in selected_paths.values()):
        raise FileNotFoundError("all six synthetic-selected checkpoints are required before real evaluation")
    device = torch.device(args.device)
    backbone, _, backbone_hash = load_frozen(args, device)
    models = {}
    for (arm, seed), path in selected_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["backbone_hash"] != backbone_hash or payload["manifest_sha256"] != hash_file(manifest_path(args.output_dir)):
            raise ValueError("selected checkpoint provenance mismatch")
        model = JointPreprocessor(arm).to(device)
        model.load_state_dict(payload["model"])
        models[(arm, seed)] = model.eval()
    bank = torch.load(real_root / "bank.pt", map_location="cpu", weights_only=True)
    if bank["format_version"] != 1 or len(bank["episodes"]) != real_manifest["n_episodes"]:
        raise ValueError("unsupported or incomplete real bank")
    rows = []
    for episode in bank["episodes"]:
        reference = {name: evaluate_episode(backbone, name, episode) for name in ("identity", "ordinary")}
        for (arm, seed), model in models.items():
            result = evaluate_episode(backbone, model, episode)
            rows.append(dict(family=episode["family"], split_seed=episode["split_seed"],
                             arm=arm, model_seed=seed,
                             nll=result["nll"], accuracy=result["accuracy"], auc=result["auc"],
                             ordinary_nll=reference["ordinary"]["nll"],
                             ordinary_accuracy=reference["ordinary"]["accuracy"],
                             identity_nll=reference["identity"]["nll"],
                             identity_accuracy=reference["identity"]["accuracy"]))
        print(f"Scored {episode['family']} split {episode['split_seed']}", flush=True)
    output = real_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "real_episodes.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    families = [entry["name"] for entry in json.loads(args.family_manifest.read_text(encoding="utf8"))["families"]]
    task_vectors, summaries = {}, {}
    for arm in ARMS:
        relevant = [row for row in rows if row["arm"] == arm]
        candidate = np.asarray([np.mean([row["nll"] for row in relevant if row["family"] == family]) for family in families])
        ordinary = np.asarray([np.mean([row["ordinary_nll"] for row in relevant if row["family"] == family]) for family in families])
        identity = np.asarray([np.mean([row["identity_nll"] for row in relevant if row["family"] == family]) for family in families])
        task_vectors[arm] = candidate
        summaries[arm] = dict(vs_ordinary=comparison_summary(candidate, ordinary),
                              vs_identity=comparison_summary(candidate, identity))
    pairwise = {f"{left}_vs_{right}": comparison_summary(task_vectors[left], task_vectors[right])
                for left, right in (("joint", "restricted"), ("no_spline", "restricted"), ("joint", "no_spline"))}
    json_write(output / "real_summary.json", dict(arms=summaries, pairwise=pairwise,
                                                   real_bank_sha256=real_manifest["bank_sha256"], families=families))
    print(json.dumps(summaries, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "report"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--family-manifest", type=Path, default=FAMILY_MANIFEST)
        if name == "prepare":
            sub.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
        else:
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    {"prepare": prepare, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
