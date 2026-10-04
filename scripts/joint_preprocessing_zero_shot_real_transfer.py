"""Evaluate the locked fresh/repeated/teacher models on unseen real datasets.

Preparation locks real context/query rows and the existing synthetic-selected
models. Reporting performs inference only, with one resumable cache per split.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from scripts import joint_preprocessing_real_transfer as real
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_zero_shot_comparison as synthetic

ARMS = synthetic.ARMS
TEMPERATURE = 0.9
FLOOR = 1e-4


def model_lock(source: Path) -> dict:
    """Verify and reuse the original choices without consulting real outcomes."""
    lock = synthetic.read(source / "lock.json")
    config = synthetic.read(source / "config.json")
    manifest = synthetic.read(source / "manifest.json")
    fingerprint = lock["fingerprint"]
    if config["fingerprint"] != fingerprint or lock["manifest_sha256"] != pilot.hash_file(source / "manifest.json"):
        raise ValueError("synthetic source lock/config differs")
    expected = synthetic.digest(dict(settings=manifest["experiment"],
        manifest_sha256=lock["manifest_sha256"], backbone_hash=config["backbone_hash"]))
    if expected != fingerprint:
        raise ValueError("synthetic source fingerprint differs")
    complete = synthetic.read(source / "test_report/complete.json")
    if complete["lock_hash"] != pilot.hash_file(source / "lock.json") or complete["fingerprint"] != fingerprint:
        raise ValueError("synthetic test completion differs from source lock")
    arms = {}
    for arm in ARMS:
        choice = lock["arms"][arm]
        done = synthetic.read(source / arm / "complete.json")
        if (choice["checkpoint_sha256"] != pilot.hash_file(source / arm / "selected.pt")
                or choice["complete_sha256"] != pilot.hash_file(source / arm / "complete.json")
                or done["experiment_fingerprint"] != fingerprint
                or done["selected_sha256"] != choice["checkpoint_sha256"]
                or done["selected_step"] != choice["selected_step"]):
            raise ValueError(f"locked {arm} model changed")
        if float(choice["alpha"]) != 0.5:
            raise ValueError("current real protocol requires the existing globally selected alpha=0.5")
        arms[arm] = {key: choice[key] for key in ("alpha", "selected_step", "checkpoint_sha256", "complete_sha256")}
    return dict(source_fingerprint=fingerprint, source_lock_sha256=pilot.hash_file(source / "lock.json"),
        source_config_sha256=pilot.hash_file(source / "config.json"),
        training_revision=config["revision"], backbone_sha256=config["backbone_hash"], arms=arms)


def prepare(args) -> None:
    output = args.output_dir / "real_transfer"
    locked = model_lock(args.source_dir)
    locked.update(family_manifest_sha256=pilot.hash_file(args.family_manifest),
        protocol="synthetic-only model selection; no updates or selection on real datasets",
        temperature=TEMPERATURE, nll_floor=FLOOR, bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=20261004)
    path = output / "model_lock.json"
    if path.exists():
        if synthetic.read(path) != locked:
            raise ValueError("real preparation lock changed")
    else:
        pilot.json_write(path, locked)
    if (output / "manifest.json").exists():
        manifest = synthetic.read(output / "manifest.json")
        if (manifest["source_manifest_sha256"] != locked["family_manifest_sha256"]
                or manifest["bank_sha256"] != pilot.hash_file(output / "bank.pt")):
            raise ValueError("prepared real bank differs")
        print("Real bank is already locked; preparation skipped.", flush=True)
        return
    real.prepare(args)


def metrics(logits, labels, classes: int) -> dict:
    log_prob = F.log_softmax(logits.float() / TEMPERATURE, dim=-1).flatten(0, 1)
    labels = labels.flatten().to(log_prob.device)
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("nonfinite real prediction")
    auc = None
    if classes == 2:
        auc = float(roc_auc_score(labels.cpu().numpy(), log_prob[:, 1].exp().cpu().numpy()))
    return dict(nll=float(F.nll_loss(log_prob, labels)),
        accuracy=float((log_prob.argmax(-1) == labels).float().mean()), auc=auc)


def _timed_prediction(backbone, model, episode, estimators, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits, views = real.episode_logits(backbone, model, episode, estimators)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if views != estimators:
        raise ValueError(f"requested {estimators} views but generated {views}")
    return dict(logits=logits.detach().cpu(), views=views, seconds=seconds)


def family_summary(candidate, baseline, *, samples=10000) -> dict:
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    ratio = (candidate + FLOOR) / (baseline + FLOOR)
    logs, delta = np.log(ratio), candidate - baseline
    rng, boot = np.random.default_rng(20261004), []
    for start in range(0, samples, 256):
        indices = rng.integers(0, len(logs), (min(256, samples-start), len(logs)))
        boot.extend(logs[indices].mean(1))
    gains = 100 * (1-ratio)
    return dict(families=len(candidate), mean_nll=float(candidate.mean()),
        mean_nll_delta=float(delta.mean()), mean_nll_reduction=1-float(candidate.mean()/baseline.mean()),
        geometric_gain=1-float(np.exp(logs.mean())),
        gain_ci95=[1-float(np.exp(np.quantile(boot, q))) for q in (.975, .025)],
        wins=int((delta < -1e-6).sum()), losses=int((delta > 1e-6).sum()),
        ties=int((np.abs(delta) <= 1e-6).sum()),
        gain_percentiles_pct={str(p): float(np.percentile(gains, p)) for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)},
        **{f"harms_over_{p}pct": int((ratio > 1+p/100).sum()) for p in (1, 5, 10)})


def aggregate(rows, families, samples):
    """Average split losses within families before comparing datasets."""
    methods = ("ordinary8", "ordinary16", *[f"{arm}_{kind}" for arm in ARMS for kind in ("learned", "blend")])
    family_rows = []
    vectors = {}
    for method in methods:
        vectors[method] = []
        for family in families:
            group = [r for r in rows if r["family"] == family and r["method"] == method]
            if len(group) != 2 or {r["split_seed"] for r in group} != {0, 1}:
                raise ValueError(f"incomplete split coverage for {family}/{method}")
            record = dict(family=family, method=method,
                **{key: float(np.mean([r[key] for r in group])) for key in ("nll", "accuracy", "seconds")},
                auc=float(np.mean([r["auc"] for r in group])) if group[0]["auc"] is not None else None,
                views=group[0]["views"])
            vectors[method].append(record["nll"])
            family_rows.append(record)
    comparisons = {f"{method}_vs_{baseline}": family_summary(vectors[method], vectors[baseline], samples=samples)
        for method in methods[2:] for baseline in ("ordinary8", "ordinary16")}
    comparisons["ordinary16_vs_ordinary8"] = family_summary(vectors["ordinary16"], vectors["ordinary8"], samples=samples)
    for left, right in (("repeated", "fresh"), ("teacher", "repeated"), ("teacher", "fresh")):
        for kind in ("learned", "blend"):
            comparisons[f"{left}_{kind}_vs_{right}_{kind}"] = family_summary(vectors[f"{left}_{kind}"], vectors[f"{right}_{kind}"], samples=samples)
    return family_rows, comparisons


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args) -> None:
    root = args.output_dir / "real_transfer"
    locked = synthetic.read(root / "model_lock.json")
    current = model_lock(args.source_dir)
    if any(current[key] != locked[key] for key in current):
        raise ValueError("synthetic model choices changed after real preparation")
    manifest = synthetic.read(root / "manifest.json")
    if (pilot.hash_file(args.family_manifest) != locked["family_manifest_sha256"]
            or manifest["source_manifest_sha256"] != locked["family_manifest_sha256"]
            or pilot.hash_file(root / "bank.pt") != manifest["bank_sha256"]
            or args.bootstrap_samples != locked["bootstrap_samples"]):
        raise ValueError("real bank/protocol differs")
    source = synthetic.read(args.family_manifest)
    families = [entry["name"] for entry in source["families"]]
    if source["split_seeds"] != [0, 1] or len(families) != 20 or manifest["n_episodes"] != 40:
        raise ValueError("expected the reserved 20-family, two-split panel")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if backbone_hash != locked["backbone_sha256"]:
        raise ValueError("TabICL backbone changed")
    backbone.eval().requires_grad_(False)
    models = {arm: synthetic.selected_model(args.source_dir, arm, locked["source_fingerprint"], device)[0].eval().requires_grad_(False) for arm in ARMS}
    bank = torch.load(root / "bank.pt", map_location="cpu", weights_only=True)
    episodes = bank["episodes"]
    if bank["format_version"] != 1 or len(episodes) != 40:
        raise ValueError("incomplete real bank")
    if {(e["family"], e["split_seed"]) for e in episodes} != {(f, s) for f in families for s in (0, 1)}:
        raise ValueError("real bank episode identities differ")
    identity = dict(model_lock_sha256=pilot.hash_file(root / "model_lock.json"),
        real_manifest_sha256=pilot.hash_file(root / "manifest.json"),
        bank_sha256=manifest["bank_sha256"],
        code_sha256={path.name: pilot.hash_file(path) for path in (Path(__file__), Path(real.__file__))})
    fingerprint = synthetic.digest(identity)
    destination = root / "report"
    started = destination / "started.json"
    if started.exists() and synthetic.read(started)["fingerprint"] != fingerprint:
        raise ValueError("real reporting already started under different files")
    if not started.exists():
        pilot.json_write(started, dict(fingerprint=fingerprint, **identity))
    completed = destination / "complete.json"
    if completed.exists():
        if synthetic.read(completed)["fingerprint"] != fingerprint:
            raise ValueError("completed report differs")
        print("Real transfer report already complete.", flush=True)
        return
    rows = []
    for index, e in enumerate(episodes):
        cache = destination / "predictions" / f"{e['family']}_split{e['split_seed']}.pt"
        if cache.exists():
            saved = torch.load(cache, map_location="cpu", weights_only=True)
            if saved["fingerprint"] != fingerprint:
                raise ValueError("prediction cache fingerprint differs")
        else:
            try:
                values = {f"ordinary{n}": _timed_prediction(backbone, "ordinary", e, n, device) for n in (8, 16)}
                values.update({f"{arm}_learned": _timed_prediction(backbone, models[arm], e, 8, device) for arm in ARMS})
                for arm in ARMS:
                    learned, ordinary = values[f"{arm}_learned"], values["ordinary8"]
                    alpha = locked["arms"][arm]["alpha"]
                    values[f"{arm}_blend"] = dict(logits=(1-alpha)*ordinary["logits"]+alpha*learned["logits"],
                        views=ordinary["views"]+learned["views"], seconds=ordinary["seconds"]+learned["seconds"])
                saved = dict(fingerprint=fingerprint, predictions=values, labels=e["y_query"])
                pilot.atomic_save(cache, saved)
            except Exception as error:
                pilot.json_write(destination / "failure.json", dict(family=e["family"], split_seed=e["split_seed"], error=f"{type(error).__name__}: {error}"))
                raise
        if not torch.equal(saved["labels"], e["y_query"]):
            raise ValueError("prediction cache labels differ")
        for method, value in saved["predictions"].items():
            row = dict(family=e["family"], split_seed=e["split_seed"], method=method,
                n_context=e["x_context"].shape[1], n_query=e["x_query"].shape[1],
                n_features=e["x_context"].shape[-1], n_numerical=int(e["numerical_mask"].sum()), n_classes=e["n_classes"],
                **metrics(value["logits"], e["y_query"], e["n_classes"]), views=value["views"], seconds=value["seconds"])
            rows.append(row)
        _write_csv(destination / "episodes.csv", rows)
        print(f"Scored {index+1}/40: {e['family']} split {e['split_seed']} ({len(saved['predictions'])} methods)", flush=True)
    family_rows, comparisons = aggregate(rows, families, args.bootstrap_samples)
    _write_csv(destination / "families.csv", family_rows)
    pilot.json_write(completed, dict(fingerprint=fingerprint, families=families, n_episodes=40,
        model_choices=locked["arms"], comparisons=comparisons,
        timing_note="Synchronized single-pass preprocessing plus inference. Shared context encoding excluded. Blend seconds sum ordinary8 and learned8; baseline8 may be reused across arms. First episode includes cold-start effects.",
        uncertainty="Paired family bootstrap over 20 families, after averaging two splits per family; one training seed; historically examined development panel.",
        conditioning="Only labeled context generates parameters; query features are transformed but do not condition the hypernetwork."))
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, default=real.FAMILY_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap sample count must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
