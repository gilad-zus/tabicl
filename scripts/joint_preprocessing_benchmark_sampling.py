"""Paired fixed/resampled direct native-residual fits on TabArena-Lite classifiers."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from scripts import joint_preprocessing_statistics_diagnostic as diagnostic
from tabicl._experiments.direct_spline_openml import load_tabarena_openml_task
from tabicl._sklearn.preprocessing import TransformToNumerical

residual, pilot, previous = diagnostic.residual, diagnostic.pilot, diagnostic.previous
ARMS = ("fixed", "resampled")
# Frozen suite-457 membership from the existing full_D_500_big_gpu_v1 manifest.
TASK_IDS = (363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
    363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629, 363630,
    363631, 363632, 363671, 363672, 363673, 363674, 363675, 363676, 363677,
    363678, 363679, 363681, 363682, 363683, 363684, 363685, 363686, 363689,
    363691, 363693, 363694, 363696, 363697, 363698, 363699, 363700, 363702,
    363704, 363705, 363706, 363707, 363708, 363711, 363712)
CHECKPOINT_HASH = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def settings(args):
    if min(args.steps, args.evaluate_every, args.outer_train_rows, args.query_chunk_rows) < 1:
        raise ValueError("all budgets must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be positive")
    # Reuse the existing dependency inventory without inheriting its data banks.
    inventory_args = argparse.Namespace(steps=args.steps, evaluate_every=args.evaluate_every,
        save_every=args.evaluate_every, lr=args.lr, continuation_seed=0,
        candidate_manifest=residual.DEFAULT_CANDIDATES, expected_revision=args.expected_revision)
    code_hashes = residual.settings(inventory_args)["code_hashes"]
    repository = Path(__file__).resolve().parents[1]
    for source_path in (Path(__file__), Path(diagnostic.__file__),
                        repository / "src/tabicl/_experiments/direct_spline_openml.py",
                        repository / "src/tabicl/_experiments/tabarena_direct_spline_protocol.py"):
        code_hashes[source_path.relative_to(repository).as_posix()] = residual.objective.canonical_hash(source_path)
    # The direct map class lives separately from the original residual generator.
    code_hashes["src/tabicl/_hyperspline/statistics_preprocessing.py"] = residual.objective.canonical_hash(
        repository / "src/tabicl/_hyperspline/statistics_preprocessing.py")
    return dict(version=1, suite_id=457, task_ids=list(TASK_IDS), arms=list(ARMS), model_seed=0,
        split_seed=20261011, outer_split=dict(repeat=0, fold=0, sample=0),
        outer_train_rows=args.outer_train_rows, fitting_fraction=.8, training_context_fraction=.75,
        steps=args.steps, evaluate_every=args.evaluate_every, lr=args.lr, weight_decay=.0001,
        gradient_clip=1., temperature=.9, query_chunk_rows=args.query_chunk_rows,
        max_classes=10, max_features=2000, checkpoint_sha256=CHECKPOINT_HASH,
        selection="minimum external validation NLL including identity; same fitting context for validation/test",
        objective="native16 mean-logit CE; eight unchanged and eight residual-adapted views",
        transformation="same direct affine/spline/neural/rank4 mixing outputs, two slots",
        preprocessing="raw encoding/imputation fitted on fitting pool only; native transforms on episode context",
        test_use="full official test partition, report only after both arms' selections locked",
        code_hashes=code_hashes, expected_revision=args.expected_revision)


def split_positions(labels, context_rows, seed, *, reserve_per_class=1):
    """Partition a pool, reserving enough rows of every class for its queries."""
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    context_quota = diagnostic.bank.bank._allocation(counts, context_rows, minimum=1,
                                                     upper=counts - reserve_per_class)
    random_generator = np.random.default_rng(seed)
    context_positions, query_positions = [], []
    for class_label, context_count in zip(classes, context_quota, strict=True):
        class_positions = random_generator.permutation(np.flatnonzero(labels == class_label))
        context_positions.extend(class_positions[:context_count])
        query_positions.extend(class_positions[context_count:])
    return tuple(torch.tensor(random_generator.permutation(positions), dtype=torch.long)
                 for positions in (context_positions, query_positions))


def encode_partition(encoder, frame, labels, row_ids):
    parts = encoder.transform_parts(frame)
    values = np.concatenate((parts.categorical, parts.numerical), axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite encoded features")
    return dict(values=torch.from_numpy(values).unsqueeze(0), labels=torch.tensor(labels, dtype=torch.long),
        missing=torch.from_numpy(parts.numerical_missing.copy()), row_ids=torch.tensor(row_ids, dtype=torch.long))


def make_entry(task, specification):
    """Use the published outer split, then reserve independent validation rows."""
    if task.problem_type == "regression":
        raise ValueError("regression is outside this classifier diagnostic")
    if not 2 <= task.n_classes <= specification["max_classes"]:
        raise ValueError("class count outside the native classifier diagnostic")
    if task.x_train.shape[1] > specification["max_features"]:
        raise ValueError("feature count exceeds the declared 2000-column budget")
    classes, counts = np.unique(task.y_train, return_counts=True)
    total_rows = min(specification["outer_train_rows"], len(task.y_train))
    # Three rows/class allow one fitting context, one fitting query and one validation query.
    selected_quota = diagnostic.bank.bank._allocation(counts, total_rows, minimum=3)
    random_generator = np.random.default_rng(specification["split_seed"] + task.task_id)
    selected_positions = np.concatenate([
        random_generator.permutation(np.flatnonzero(task.y_train == class_label))[:class_count]
        for class_label, class_count in zip(classes, selected_quota, strict=True)])
    selected_positions = random_generator.permutation(selected_positions)
    fitting_count = int(.8 * total_rows)
    # Validation needs one/class, and fitting needs at least two/class.
    validation_local, fitting_local = split_positions(task.y_train[selected_positions],
        total_rows - fitting_count, specification["split_seed"] + task.task_id + 1,
        reserve_per_class=2)
    fitting_positions = selected_positions[fitting_local.numpy()]
    validation_positions = selected_positions[validation_local.numpy()]
    fitting_frame = task.x_train.iloc[fitting_positions].reset_index(drop=True)
    encoder = TransformToNumerical().fit(fitting_frame)
    fitting = encode_partition(encoder, fitting_frame, task.y_train[fitting_positions], fitting_positions)
    validation = encode_partition(encoder, task.x_train.iloc[validation_positions],
                                  task.y_train[validation_positions], validation_positions)
    # IDs use the concatenated outer-training/outer-test coordinate system.
    test = encode_partition(encoder, task.x_test, task.y_test,
                            np.arange(len(task.y_test)) + len(task.y_train))
    parts = encoder.transform_parts(fitting_frame)
    categorical_count, numerical_count = parts.categorical.shape[1], parts.numerical.shape[1]
    if numerical_count == 0 or not (parts.numerical.std(0) > 0).any():
        raise ValueError("no nonconstant numerical feature for residual adaptation")
    metadata = dict(task_id=task.task_id, dataset_id=task.dataset_id, family=task.dataset_name,
        source_group=f"openml-dataset:{task.dataset_id}", split_seed=specification["split_seed"],
        n_classes=task.n_classes, problem_type=task.problem_type,
        categorical_features=list(range(categorical_count)),
        numerical_mask=torch.tensor([False] * categorical_count + [True] * numerical_count),
        outer_split_hash=task.outer_split_hash, official_train_rows=len(task.y_train),
        official_test_rows=len(task.y_test), encoded_features=categorical_count + numerical_count,
        row_id_coordinates="outer-training positions followed by offset outer-test positions")
    entry = dict(metadata=metadata, fitting=fitting, validation=validation,
                 training_context_rows=int(.75 * fitting_count))
    # Verify that the requested training role split supports every class.
    split_positions(fitting["labels"].numpy(), entry["training_context_rows"], task.task_id)
    return entry, test


def prediction_episode(entry, query_partition):
    fitting = entry["fitting"]
    return dict(entry["metadata"], x_context=fitting["values"], y_context=fitting["labels"].float()[None],
        context_missing=fitting["missing"], context_indices=fitting["row_ids"],
        x_query=query_partition["values"], y_query=query_partition["labels"],
        query_missing=query_partition["missing"], query_indices=query_partition["row_ids"],
        n_context=len(fitting["labels"]), n_query=len(query_partition["labels"]))


def training_episode(entry, arm, step, *, probe_number=None):
    if arm not in ARMS:
        raise ValueError("unknown training arm")
    # A step-derived seed makes row roles reproducible across restart boundaries.
    draw = 0 if arm == "fixed" else step
    if probe_number is not None:
        draw = 100000 + probe_number
    seed = int(np.random.SeedSequence([20261011, entry["metadata"]["task_id"], draw]).generate_state(1)[0])
    context_positions, query_positions = split_positions(entry["fitting"]["labels"].numpy(),
                                                        entry["training_context_rows"], seed)
    fitting_episode = prediction_episode(entry, entry["fitting"])
    return diagnostic.inner_episode(fitting_episode, context_positions, query_positions, "training")


def task_folder(args, task_id):
    return args.output_dir / "tasks" / str(task_id)


def prepare(args):
    specification = settings(args)
    residual.diversity.verify_revision(specification)
    intent_path = args.output_dir / "intent.json"
    if intent_path.exists() and previous.read(intent_path) != specification:
        raise ValueError("preparation settings changed")
    pilot.json_write(intent_path, specification)
    if (args.output_dir / "manifest.json").exists():
        checked_manifest(args)
        return
    accepted, excluded = [], []
    import openml
    for task_id in TASK_IDS:
        folder = task_folder(args, task_id)
        if (folder / "prepared.json").exists():
            record = previous.read(folder / "prepared.json")
        else:
            # Inspect task type before downloading features for out-of-scope regression.
            public_task = openml.tasks.get_task(task_id, download_data=False)
            if "regression" in str(public_task.task_type).lower():
                record = dict(task_id=task_id, status="excluded", reason="regression outside classifier scope")
            else:
                task = load_tabarena_openml_task(task_id, outer_repeat=0, outer_fold=0, outer_sample=0)
                try:
                    entry, test = make_entry(task, specification)
                except ValueError as error:
                    record = dict(task_id=task_id, dataset=task.dataset_name, status="excluded", reason=str(error))
                else:
                    pilot.atomic_save(folder / "fit_validation.pt", entry)
                    pilot.atomic_save(folder / "test.pt", test)
                    metadata = {name: value for name, value in entry["metadata"].items() if name != "numerical_mask"}
                    record = dict(metadata, status="accepted", fitting_rows=len(entry["fitting"]["labels"]),
                        validation_rows=len(entry["validation"]["labels"]),
                        training_context_rows=entry["training_context_rows"],
                        fitting_sha256=pilot.hash_file(folder / "fit_validation.pt"),
                        test_sha256=pilot.hash_file(folder / "test.pt"))
            pilot.json_write(folder / "prepared.json", record)
        (accepted if record["status"] == "accepted" else excluded).append(record)
        print(f"Prepared task {task_id}: {record['status']} {record.get('family', record.get('reason', ''))}", flush=True)
    if len(accepted) < 25:
        raise ValueError(f"only {len(accepted)} eligible benchmark datasets; inspect exclusions before GPU use")
    if len({record["dataset_id"] for record in accepted}) != len(accepted):
        raise ValueError("benchmark has repeated dataset IDs")
    manifest = dict(settings=specification, accepted=accepted, excluded=excluded)
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(args.output_dir / "manifest.json", manifest)
    print(f"Benchmark bank ready: {len(accepted)} datasets, {len(excluded)} exclusions", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if previous.digest({name: value for name, value in manifest.items() if name != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("manifest fingerprint changed")
    if manifest["settings"] != settings(args):
        raise ValueError("experiment settings/code changed")
    residual.diversity.verify_revision(manifest["settings"])
    for record in manifest["accepted"]:
        folder = task_folder(args, record["task_id"])
        for filename, hash_key in (("fit_validation.pt", "fitting_sha256"), ("test.pt", "test_sha256")):
            if pilot.hash_file(folder / filename) != record[hash_key]:
                raise ValueError(f"task {record['task_id']} {filename} changed")
    return manifest


def score_partition(backbone, model, entry, partition, chunk_rows):
    # Each chunk receives exactly the same complete fitting context and fitted native transforms.
    probabilities, labels = [], partition["labels"]
    for query_start in range(0, len(labels), chunk_rows):
        query_stop = min(query_start + chunk_rows, len(labels))
        chunk = dict(values=partition["values"][:, query_start:query_stop],
            labels=labels[query_start:query_stop], missing=partition["missing"][query_start:query_stop],
            row_ids=partition["row_ids"][query_start:query_stop])
        logits = residual.episode_logits(backbone, model, prediction_episode(entry, chunk))
        probabilities.append(torch.log_softmax(logits.float().cpu().flatten(0, 1) / .9, -1))
    log_probabilities = torch.cat(probabilities)
    score = dict(nll=float(torch.nn.functional.nll_loss(log_probabilities, labels)),
                 accuracy=float((log_probabilities.argmax(-1) == labels).float().mean()), rows=len(labels))
    if entry["metadata"]["n_classes"] == 2 and labels.unique().numel() == 2:
        score["auc"] = float(roc_auc_score(labels.numpy(), log_probabilities[:, 1].exp().numpy()))
    if not math.isfinite(score["nll"]):
        raise FloatingPointError("nonfinite score")
    return score


def fit_arm(args, backbone, initial, manifest, entry, arm):
    folder = task_folder(args, entry["metadata"]["task_id"]) / arm
    if (folder / "fit_complete.json").exists():
        completion = previous.read(folder / "fit_complete.json")
        if completion["fingerprint"] != manifest["fingerprint"] or completion["steps"] != args.steps:
            raise ValueError("completed fit differs")
        for choice in ("selected", "final"):
            if pilot.hash_file(folder / f"{choice}.pt") != completion[f"{choice}_sha256"]:
                raise ValueError("locked map changed")
        return completion
    device = next(backbone.parameters()).device
    model = diagnostic.direct_model(initial, prediction_episode(entry, entry["validation"]), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    state_path = folder / "state.pt"
    probe_episodes = [training_episode(entry, "resampled", 0, probe_number=number) for number in range(3)]
    exposure = torch.zeros(len(entry["fitting"]["labels"]), dtype=torch.long)
    baseline_validation = score_partition(backbone, None, entry, entry["validation"], args.query_chunk_rows)["nll"]
    step, selected_step = 0, 0
    baseline_probes = [diagnostic.score_episode(backbone, None, episode)["nll"] for episode in probe_episodes]
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("state exists; use --resume")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("resume fingerprint changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        diagnostic.core.capacity.optimizer_to(optimizer, device)
        step, selected_step = state["step"], state["selected_step"]
        selected_score, selected_model = state["selected_score"], state["selected_model"]
        exposure = state["exposure"]
        for filename in ("training.csv", "validation.csv"):
            diagnostic.core.capacity.trim_csv(folder / filename, step)
    else:
        # Identity is a real native16 fallback and an eligible step-zero selection.
        episode = prediction_episode(entry, entry["validation"])
        torch.testing.assert_close(residual.episode_logits(backbone, model, episode),
                                   residual.episode_logits(backbone, None, episode), rtol=0, atol=0)
        selected_score, selected_model = baseline_validation, pilot.state_cpu(model)

    def save_state():
        pilot.atomic_save(state_path, dict(fingerprint=manifest["fingerprint"], step=step,
            model=pilot.state_cpu(model), optimizer=optimizer.state_dict(), exposure=exposure,
            selected_step=selected_step, selected_score=selected_score, selected_model=selected_model))

    def evaluate():
        nonlocal selected_score, selected_step, selected_model
        validation_nll = score_partition(backbone, model, entry, entry["validation"], args.query_chunk_rows)["nll"]
        probe_nll = float(np.mean([diagnostic.score_episode(backbone, model, episode)["nll"] for episode in probe_episodes]))
        if validation_nll < selected_score:
            selected_score, selected_step, selected_model = validation_nll, step, pilot.state_cpu(model)
        row = dict(step=step, validation_nll=validation_nll, ordinary16_nll=baseline_validation,
            validation_gain_percent=100 * (1 - validation_nll / baseline_validation),
            train_probe_nll=probe_nll, train_probe_baseline=float(np.mean(baseline_probes)),
            train_probe_gain_percent=100 * (1 - probe_nll / np.mean(baseline_probes)),
            selected_step=selected_step, query_exposure_min=int(exposure.min()),
            query_exposure_max=int(exposure.max()), query_exposure_fraction=float((exposure > 0).float().mean()))
        pilot.csv_append(folder / "validation.csv", row)
        pilot.atomic_save(folder / "checkpoints" / f"step_{step:04d}.pt", dict(model=pilot.state_cpu(model), step=step,
            fingerprint=manifest["fingerprint"]))
        save_state()
        print(f"{entry['metadata']['family']} {arm} {step}/{args.steps}: probe={row['train_probe_gain_percent']:+.3f}% val={row['validation_gain_percent']:+.3f}% best@{selected_step}", flush=True)

    if not state_path.exists():
        evaluate()
    stop_step = min(args.steps, args.max_steps) if args.max_steps is not None else args.steps
    for step in range(step + 1, stop_step + 1):
        started = time.perf_counter()
        episode = training_episode(entry, arm, step)
        backbone.train()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = residual.ensemble_backward(backbone, model, episode, scale=1.)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(gradient_norm) or any(parameter.grad is not None for parameter in backbone.parameters()):
            raise FloatingPointError("invalid map gradient or unfrozen backbone")
        optimizer.step()
        fitting_row_lookup = {int(row_id): position for position, row_id in enumerate(entry["fitting"]["row_ids"])}
        exposure[torch.tensor([fitting_row_lookup[int(row_id)] for row_id in episode["query_indices"]])] += 1
        pilot.csv_append(folder / "training.csv", dict(step=step, fitting_nll=float(loss),
            preclip_gradient_norm=gradient_norm, clipped=gradient_norm > 1.,
            context_rows=episode["n_context"], query_rows=episode["n_query"], seconds=time.perf_counter() - started))
        if step == 1 or step % args.evaluate_every == 0 or step == stop_step:
            evaluate()
    if step < args.steps:
        return None
    # Both choices are immutable before the separate test-report stage can load labels.
    for choice, parameters, choice_step in (("selected", selected_model, selected_step),
                                           ("final", pilot.state_cpu(model), step)):
        pilot.atomic_save(folder / f"{choice}.pt", dict(model=parameters, step=choice_step,
            fingerprint=manifest["fingerprint"]))
    completion = dict(fingerprint=manifest["fingerprint"], steps=step, selected_step=selected_step,
        selected_validation_nll=selected_score, ordinary16_validation_nll=baseline_validation,
        selected_sha256=pilot.hash_file(folder / "selected.pt"), final_sha256=pilot.hash_file(folder / "final.pt"),
        test_labels_used_for_optimization=False, test_labels_used_for_selection=False)
    pilot.json_write(folder / "fit_complete.json", completion)
    return completion


def comparison(candidate, baseline):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    relative_gains = 100 * (1 - candidate / np.maximum(baseline, 1e-4))
    random_generator = np.random.default_rng(20261011)
    bootstrap_means = np.array([relative_gains[random_generator.integers(len(candidate), size=len(candidate))].mean()
                                for _ in range(10000)])
    return dict(residual.objective.comparison(candidate, baseline),
        arithmetic_mean_relative_gain_percent=float(relative_gains.mean()),
        median_relative_gain_percent=float(np.median(relative_gains)),
        mean_nll_gain_percent=100 * (1 - float(candidate.mean() / baseline.mean())),
        gain_percentiles={str(percentile): float(np.percentile(relative_gains, percentile))
                          for percentile in (0, 5, 10, 25, 50, 75, 90, 95, 100)},
        bootstrap_mean_relative_gain_95_percent=np.quantile(bootstrap_means, [.025, .975]).tolist())


def report(args, backbone, initial, manifest):
    # Verify every fit and map lock before opening any test bank.
    for record in manifest["accepted"]:
        for arm in ARMS:
            folder = task_folder(args, record["task_id"]) / arm
            completion = previous.read(folder / "fit_complete.json")
            if completion["fingerprint"] != manifest["fingerprint"] or completion["steps"] != args.steps:
                raise ValueError("all benchmark arms must complete before test scoring")
            for choice in ("selected", "final"):
                if pilot.hash_file(folder / f"{choice}.pt") != completion[f"{choice}_sha256"]:
                    raise ValueError("selected/final map changed")
    rows = []
    for record in manifest["accepted"]:
        folder = task_folder(args, record["task_id"])
        result_path = folder / "test_scores.json"
        if result_path.exists():
            scores = previous.read(result_path)
            if scores["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("test report fingerprint changed")
        else:
            entry = torch.load(folder / "fit_validation.pt", map_location="cpu", weights_only=True)
            test = torch.load(folder / "test.pt", map_location="cpu", weights_only=True)
            scores = dict(task_id=record["task_id"], family=record["family"], fingerprint=manifest["fingerprint"],
                ordinary16=score_partition(backbone, None, entry, test, args.query_chunk_rows))
            for arm in ARMS:
                model = diagnostic.direct_model(initial, prediction_episode(entry, entry["validation"]), next(backbone.parameters()).device)
                for choice in ("selected", "final"):
                    state = torch.load(folder / arm / f"{choice}.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(state["model"])
                    scores[f"{arm}_{choice}"] = score_partition(backbone, model, entry, test, args.query_chunk_rows)
            pilot.json_write(result_path, scores)
        rows.append(scores)
        print(f"Test complete: {record['family']} ({record['official_test_rows']} rows)", flush=True)
    baseline = [row["ordinary16"]["nll"] for row in rows]
    summaries = {f"{arm}_{choice}_vs_ordinary16": comparison([row[f"{arm}_{choice}"]["nll"] for row in rows], baseline)
                 for arm in ARMS for choice in ("selected", "final")}
    summaries.update({f"resampled_vs_fixed_{choice}": comparison([row[f"resampled_{choice}"]["nll"] for row in rows],
        [row[f"fixed_{choice}"]["nll"] for row in rows]) for choice in ("selected", "final")})
    pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"],
        datasets=len(rows), results=rows, comparisons=summaries, exclusions=manifest["excluded"],
        qualification="One seed, one official outer fold, capped fitting pool, full test; per-dataset adaptation diagnostic, not zero-shot transfer or official leaderboard replication."))


def run(args):
    manifest = checked_manifest(args)
    device = torch.device(args.device)
    backbone, _, checkpoint_hash = pilot.load_frozen(args, device)
    if checkpoint_hash != CHECKPOINT_HASH:
        raise ValueError("classifier checkpoint changed")
    initial = diagnostic.new_model("statistics").to(device)
    if (args.output_dir / "complete.json").exists():
        if previous.read(args.output_dir / "complete.json")["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("completed report differs")
        print("Benchmark already complete", flush=True)
        return
    for record in manifest["accepted"]:
        entry = torch.load(task_folder(args, record["task_id"]) / "fit_validation.pt", map_location="cpu", weights_only=True)
        for arm in ARMS:
            fit_arm(args, backbone, initial, manifest, entry, arm)
    if args.max_steps is None or args.max_steps >= args.steps:
        report(args, backbone, initial, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--outer-train-rows", type=int, default=750)
    parser.add_argument("--query-chunk-rows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--expected-revision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
