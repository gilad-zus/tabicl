"""One small zero-shot hypernetwork, with eight held-out-row direct-fit references."""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_residual_conditioning as residual
from tabicl._hyperspline.statistics_preprocessing import (
    DirectResidualPreprocessor, StatisticsResidualPreprocessor,
)

bank, core, pilot, previous = residual.bank, residual.core, residual.pilot, residual.previous
ARMS = ("statistics",)
DIRECT_SEED = 20261010


def settings(args):
    specification = residual.settings(args)
    repository = Path(__file__).resolve().parents[1]
    for source_path in (Path(__file__), Path(inspect.getfile(StatisticsResidualPreprocessor)),
                        Path(__file__).with_name("run_joint_preprocessing_statistics_diagnostic.sh")):
        specification["code_hashes"][source_path.relative_to(repository).as_posix()] = residual.objective.canonical_hash(source_path)
    if min(args.direct_datasets, args.direct_steps, args.direct_rows, args.direct_evaluate_every) < 1:
        raise ValueError("direct-reference budgets must be positive")
    if not math.isfinite(args.direct_lr) or args.direct_lr <= 0:
        raise ValueError("direct-reference learning rate must be positive")
    specification.update(
        arms=list(ARMS), initialization="fresh seed-zero heads; statistics-only MLP encoder",
        conditioning=dict(inputs="31 labelled-context column summaries, their table mean, and log row/column/class counts",
            encoder="LayerNorm(65), Linear(65,32), GELU, Linear(32,64), GELU; no attention",
            queries="no query features, labels, masks, or statistics enter the conditioner"),
        paired_rows="each source capped independently; direct-reference query rows excluded from meta-training",
        hypothesis="a small summary conditioner can learn useful native residuals; direct fitting measures real held-out-row headroom",
        transformation_family="unchanged affine, 20-control cubic spline, 8-unit neural residual, rank-4 mixing, two native slots",
        direct_reference=dict(datasets=args.direct_datasets, steps=args.direct_steps, rows=args.direct_rows,
            lr=args.direct_lr, evaluate_every=args.direct_evaluate_every, seed=DIRECT_SEED,
            selection="minimum inner-selection NLL; identity eligible; outer query scored after locking",
            outer_context_fraction=.75, inner_context_fraction=.6, inner_fitting_query_fraction=.2,
            optimizer="AdamW, weight decay .0001, clip 1; no teacher targets or initialization"),
        comparison_qualification="old attention runs differ in row sampling and held-out-row reservations; historical comparison is descriptive",
    )
    return specification


def new_model(arm, embedding_dim=128):
    if arm != "statistics":
        raise ValueError("only the statistics arm is authorized")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return StatisticsResidualPreprocessor()


def checked_manifest(args):
    manifest = residual.checked_manifest(args, protocol=sys.modules[__name__])
    if pilot.hash_file(args.output_dir / "direct_bank_manifest.json") != manifest["direct_bank_manifest_sha256"]:
        raise ValueError("direct-reference manifest changed")
    load_direct_bank(args, manifest)
    return manifest


def setup(args):
    return residual.setup(args, protocol=sys.modules[__name__])


def partition_context(outer_episode, seed):
    """Reserve fitting and selection queries inside context; never use outer queries."""
    labels = outer_episode["y_context"].flatten().numpy().astype(np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if counts.min() < 3:
        raise ValueError("each class needs an inner context, fitting query, and selection query")
    context_count = int(.6 * len(labels))
    fitting_count = int(.2 * len(labels))
    context_quota = bank.bank._allocation(counts, context_count, minimum=1, upper=counts - 2)
    fitting_quota = bank.bank._allocation(counts - context_quota, fitting_count, minimum=1,
                                         upper=counts - context_quota - 1)
    random_generator = np.random.default_rng(seed)
    context_positions, fitting_positions, selection_positions = [], [], []
    for label, context_size, fitting_size in zip(classes, context_quota, fitting_quota, strict=True):
        class_positions = random_generator.permutation(np.flatnonzero(labels == label))
        context_positions.extend(class_positions[:context_size])
        fitting_positions.extend(class_positions[context_size:context_size + fitting_size])
        selection_positions.extend(class_positions[context_size + fitting_size:])
    return tuple(torch.tensor(random_generator.permutation(positions), dtype=torch.long)
                 for positions in (context_positions, fitting_positions, selection_positions))


def inner_episode(outer_episode, context_positions, query_positions, stage):
    """All selected rows come from outer context, including encoding and imputation."""
    episode = {name: outer_episode[name] for name in (
        "family", "source_group", "split_seed", "n_classes", "categorical_features", "numerical_mask",
    )}
    episode.update(
        x_context=outer_episode["x_context"].index_select(1, context_positions),
        x_query=outer_episode["x_context"].index_select(1, query_positions),
        y_context=outer_episode["y_context"].index_select(1, context_positions),
        y_query=outer_episode["y_context"].flatten().index_select(0, query_positions).long(),
        context_missing=outer_episode["context_missing"].index_select(0, context_positions),
        query_missing=outer_episode["context_missing"].index_select(0, query_positions),
        context_indices=outer_episode["context_indices"].index_select(0, context_positions),
        query_indices=outer_episode["context_indices"].index_select(0, query_positions),
        n_context=len(context_positions), n_query=len(query_positions), stage=stage,
    )
    return episode


def prepare_direct_bank(args, manifest):
    lock_path = args.output_dir / "direct_bank_manifest.json"
    if lock_path.exists():
        load_direct_bank(args, manifest)
        return
    sources = bank.load_bank(args.output_dir, manifest, "large_train")
    ordered_sources = sorted(sources, key=lambda source: source["family"])
    source_order = np.random.default_rng(DIRECT_SEED).permutation(len(ordered_sources))
    chosen, rejected = [], []
    for source_position in source_order:
        source = ordered_sources[int(source_position)]
        episode_seed = DIRECT_SEED + 1009 * int(source_position)
        try:
            outer = bank.bank.sample_real_episode(source, args.direct_rows, .75, episode_seed)
            context_positions, fitting_positions, selection_positions = partition_context(outer, episode_seed + 1)
        except ValueError as error:
            rejected.append(dict(family=source["family"], reason=str(error)))
            continue
        chosen.append(dict(outer=outer,
            fitting=inner_episode(outer, context_positions, fitting_positions, "fitting"),
            selection=inner_episode(outer, context_positions, selection_positions, "selection")))
        if len(chosen) == args.direct_datasets:
            break
    if len(chosen) != args.direct_datasets:
        raise ValueError("not enough direct-reference sources with three-way class coverage")
    bank_path = args.output_dir / "banks" / "direct_reference.pt"
    pilot.atomic_save(bank_path, dict(values=chosen))
    pilot.json_write(lock_path, dict(specification_fingerprint=previous.digest(manifest["settings"]),
        path=bank_path.relative_to(args.output_dir).as_posix(), sha256=pilot.hash_file(bank_path),
        families=[entry["outer"]["family"] for entry in chosen], rejected=rejected,
        protocol=manifest["settings"]["direct_reference"],
        exclusions="outer query source rows removed from the shared-model training bank"))
    print(f"Direct-reference sources: {[entry['outer']['family'] for entry in chosen]}", flush=True)


def load_direct_bank(args, manifest):
    lock = previous.read(args.output_dir / "direct_bank_manifest.json")
    bank_path = args.output_dir / lock["path"]
    if lock["specification_fingerprint"] != previous.digest(manifest["settings"]) or lock["protocol"] != settings(args)["direct_reference"]:
        raise ValueError("direct-reference protocol changed")
    if pilot.hash_file(bank_path) != lock["sha256"]:
        raise ValueError("direct-reference bank changed")
    entries = torch.load(bank_path, weights_only=True, map_location="cpu")["values"]
    if [entry["outer"]["family"] for entry in entries] != lock["families"]:
        raise ValueError("direct-reference source ordering changed")
    return entries


def manifest_extras(args, manifest):
    prepare_direct_bank(args, manifest)
    lock = previous.read(args.output_dir / "direct_bank_manifest.json")
    return dict(direct_bank_manifest_sha256=pilot.hash_file(args.output_dir / "direct_bank_manifest.json"),
                direct_bank_sha256=lock["sha256"])


def prepare(args):
    residual.prepare(args, protocol=sys.modules[__name__])
    checked_manifest(args)


def load_source_banks(args, manifest):
    sources = bank.load_bank(args.output_dir, manifest, "large_train")
    reserved_rows = {entry["outer"]["family"]: set(entry["outer"]["query_indices"].tolist())
                     for entry in load_direct_bank(args, manifest)}
    filtered_sources = []
    for source in sources:
        if source["family"] not in reserved_rows:
            filtered_sources.append(source)
            continue
        # Original source-row IDs survive filtering, enabling exact leakage checks.
        retained_positions = np.array([position for position, source_row in enumerate(source["source_indices"])
                                       if int(source_row) not in reserved_rows[source["family"]]], dtype=np.int64)
        filtered = dict(source)
        filtered["columns"] = bank.bank.pack_frame(bank.bank.unpack_frame(source["columns"], retained_positions))
        filtered["labels"] = np.asarray(source["labels"])[retained_positions].tolist()
        filtered["source_indices"] = np.asarray(source["source_indices"])[retained_positions].tolist()
        filtered_sources.append(filtered)
    return dict(large=filtered_sources)


def training_batch(arm, step, source_banks):
    if arm != "statistics":
        raise ValueError("unknown statistics arm")
    requested_rows, context_fraction = core.shape_for_step(0, step)
    sources = residual.diversity.scheduled_sources(source_banks["large"], step)
    episodes = []
    for source_position, source in enumerate(sources):
        episode_seed = int(np.random.SeedSequence([20261007, 0, step, source_position]).generate_state(1)[0])
        episode = bank.bank.sample_real_episode(source, requested_rows, context_fraction, episode_seed)
        episode.update(domain="real", source_seed=episode_seed,
            task_id=17_000_000_000 + (step - 1) * 4 + source_position,
            scheduled_rows=requested_rows, paired_row_cap=episode["actual_rows"])
        episodes.append(episode)
    return episodes


def preflight(args):
    residual.preflight(args, protocol=sys.modules[__name__])


def train(args):
    residual.train(args, protocol=sys.modules[__name__])


def direct_model(initial, outer_episode, device):
    numerical_mask = outer_episode["numerical_mask"]
    return DirectResidualPreprocessor(initial,
        outer_episode["x_context"][..., numerical_mask].to(device),
        outer_episode["y_context"].to(device),
        outer_episode["context_missing"][None].to(device)).to(device)


def score_episode(backbone, model, episode):
    logits = residual.episode_logits(backbone, model, episode)
    return previous.score(logits, episode["y_query"].to(logits.device))


def direct(args):
    backbone, initial, manifest, device = setup(args)
    if device.type == "cuda":
        audit = previous.read(args.output_dir / "preflight" / "complete.json")
        if audit["fingerprint"] != manifest["fingerprint"] or not audit["both_passed"]:
            raise ValueError("GPU preflight must pass first")
    for source_position, entry in enumerate(load_direct_bank(args, manifest)):
        folder = args.output_dir / "direct" / f"source_{source_position:02d}"
        if (folder / "complete.json").exists():
            completion = previous.read(folder / "complete.json")
            if completion["fingerprint"] != manifest["fingerprint"] or pilot.hash_file(folder / "selected.pt") != completion["selected_sha256"]:
                raise ValueError("completed direct reference changed")
            continue
        model = direct_model(initial, entry["outer"], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.direct_lr, weight_decay=1e-4)
        state_path = folder / "state.pt"
        step, selected_step = 0, 0
        if state_path.exists():
            if not args.resume:
                raise FileExistsError("direct state exists; pass --resume")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            if state["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("direct resume fingerprint changed")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            core.capacity.optimizer_to(optimizer, device)
            step, selected_step = state["step"], state["selected_step"]
            selected_score, selected_model = state["selected_score"], state["selected_model"]
            for log_name in ("training.csv", "selection.csv"):
                core.capacity.trim_csv(folder / log_name, step)
        else:
            selected_score = score_episode(backbone, model, entry["selection"])["nll"]
            selected_model = pilot.state_cpu(model)
            baseline_logits = residual.episode_logits(backbone, None, entry["selection"])
            torch.testing.assert_close(residual.episode_logits(backbone, model, entry["selection"]),
                                       baseline_logits, rtol=0, atol=0)
            pilot.csv_append(folder / "selection.csv", dict(step=0, nll=selected_score))

        def save_state():
            pilot.atomic_save(state_path, dict(fingerprint=manifest["fingerprint"], step=step,
                selected_step=selected_step, selected_score=selected_score, selected_model=selected_model,
                model=pilot.state_cpu(model), optimizer=optimizer.state_dict()))

        save_state()
        for step in range(step + 1, args.direct_steps + 1):
            started = time.perf_counter()
            backbone.train()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            fitting_loss = residual.ensemble_backward(backbone, model, entry["fitting"], scale=1.)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if not math.isfinite(gradient_norm) or any(parameter.grad is not None for parameter in backbone.parameters()):
                raise FloatingPointError("invalid direct-reference gradient or unfrozen backbone")
            optimizer.step()
            pilot.csv_append(folder / "training.csv", dict(step=step, fitting_nll=float(fitting_loss),
                preclip_gradient_norm=gradient_norm, seconds=time.perf_counter() - started))
            if step % args.direct_evaluate_every == 0 or step == args.direct_steps:
                selection_score = score_episode(backbone, model, entry["selection"])["nll"]
                pilot.csv_append(folder / "selection.csv", dict(step=step, nll=selection_score))
                if selection_score < selected_score:
                    selected_score, selected_step, selected_model = selection_score, step, pilot.state_cpu(model)
                save_state()
                print(f"Direct {entry['outer']['family']}: {step}/{args.direct_steps}, fit={float(fitting_loss):.6f}, selection={selection_score:.6f}, best@{selected_step}", flush=True)
        # Lock the selected weights before any outer-query labels are scored.
        selected_path = folder / "selected.pt"
        pilot.atomic_save(selected_path, dict(fingerprint=manifest["fingerprint"],
            model=selected_model, step=selected_step))
        model.load_state_dict(selected_model)
        selected_outer = score_episode(backbone, model, entry["outer"])
        baseline_outer = score_episode(backbone, None, entry["outer"])
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True)["model"])
        final_outer = score_episode(backbone, model, entry["outer"])
        pilot.json_write(folder / "complete.json", dict(fingerprint=manifest["fingerprint"],
            family=entry["outer"]["family"], steps=step, selected_step=selected_step,
            selected_selection_nll=selected_score, selected_sha256=pilot.hash_file(selected_path),
            selected_outer=selected_outer, final_outer=final_outer, ordinary16_outer=baseline_outer,
            fitting_rows=entry["fitting"]["n_query"], selection_rows=entry["selection"]["n_query"],
            held_out_rows=entry["outer"]["n_query"],
            outer_query_used_for_optimization=False, outer_query_used_for_selection=False))


def report(args):
    backbone, initial, manifest, device = setup(args)
    run_folder = core.run_dir(args.output_dir, "statistics", 0)
    run = previous.read(run_folder / "complete.json")
    expected_fingerprint = core.run_fingerprint(manifest["fingerprint"], "statistics", 0)
    if run["fingerprint"] != expected_fingerprint or run["steps"] != args.steps:
        raise ValueError("full shared training budget required")
    episode_rows = core.capacity.read_rows(run_folder / "evaluation_episodes.csv")
    summaries = {}
    for choice, step in (("initial", 0), ("selected", run["selected_step"]), ("final", args.steps)):
        for panel in residual.PANELS:
            rows = [row for row in episode_rows if int(row["step"]) == step and row["panel"] == panel]
            names, candidate = residual.grouped(rows, "blend_nll", manifest["banks"][panel]["count"])
            _, baseline = residual.grouped(rows, "ordinary16_nll")
            summaries[f"{choice}/{panel}"] = dict(names=names,
                versus_ordinary16=residual.objective.comparison(candidate, baseline),
                mean_nll_reduction=1 - float(candidate.mean() / baseline.mean()))
    validation_steps = sorted({int(row["step"]) for row in episode_rows if row["panel"] == "real_validation"})
    passing_steps = []
    for step in validation_steps:
        rows = [row for row in episode_rows if int(row["step"]) == step and row["panel"] == "real_validation"]
        _, candidate = residual.grouped(rows, "blend_nll")
        _, baseline = residual.grouped(rows, "ordinary16_nll")
        if residual.passing(candidate, baseline):
            passing_steps.append(step)
    selected_position = validation_steps.index(run["selected_step"])
    adjacent_steps = validation_steps[max(0, selected_position - 1):selected_position] + validation_steps[selected_position + 1:selected_position + 2]
    direct_entries = load_direct_bank(args, manifest)
    reference_scores = [previous.read(args.output_dir / "direct" / f"source_{position:02d}" / "complete.json")
                        for position in range(len(direct_entries))]
    baseline_nll = np.array([reference["ordinary16_outer"]["nll"] for reference in reference_scores])
    direct_nll = np.array([reference["selected_outer"]["nll"] for reference in reference_scores])
    source_comparisons = {}
    for choice in ("selected", "final"):
        checkpoint_path = run_folder / f"{choice}.pt"
        if pilot.hash_file(checkpoint_path) != run[f"{choice}_sha256"]:
            raise ValueError("shared checkpoint changed")
        model = copy.deepcopy(initial)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"])
        shared_scores = [score_episode(backbone, model, entry["outer"]) for entry in direct_entries]
        shared_nll = np.array([score["nll"] for score in shared_scores])
        source_comparisons[choice] = dict(shared_scores=shared_scores,
            shared_versus_ordinary16=residual.objective.comparison(shared_nll, baseline_nll),
            direct_versus_ordinary16=residual.objective.comparison(direct_nll, baseline_nll),
            shared_versus_direct=residual.objective.comparison(shared_nll, direct_nll))
    result = dict(fingerprint=manifest["fingerprint"], run=run, summaries=summaries,
        passing_steps=passing_steps,
        worth_fresh_confirmation=run["selected_step"] > 0 and run["selected_step"] in passing_steps
            and any(step in passing_steps for step in adjacent_steps),
        direct_references=reference_scores, source_comparisons=source_comparisons,
        qualification="One seed; eight training-source held-out-row diagnostics and 25 separate development validation sources. Direct fits are adaptation references, not zero-shot deployment or an upper bound. Historical attention comparisons also change row scheduling.")
    report_path = args.output_dir / "complete.json"
    if report_path.exists() and previous.read(report_path) != result:
        raise ValueError("locked final report changed")
    pilot.json_write(report_path, result)
    print(f"Statistics diagnostic complete: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "direct", "train", "report", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-bank-dir", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, default=residual.DEFAULT_CANDIDATES)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS, default="statistics")
    parser.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--evaluate-every", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--direct-datasets", type=int, default=8)
    parser.add_argument("--direct-steps", type=int, default=250)
    parser.add_argument("--direct-rows", type=int, default=1024)
    parser.add_argument("--direct-evaluate-every", type=int, default=25)
    parser.add_argument("--direct-lr", type=float, default=.001)
    parser.add_argument("--preflight-family", default="Credit_Risk_Modeling")
    parser.add_argument("--preflight-split-seed", type=int, default=0)
    parser.add_argument("--expected-revision")
    parser.add_argument("--source-fingerprint", default="7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.command != "pipeline":
        globals()[args.command](args)
        return
    if (args.output_dir / "complete.json").exists():
        report(args)
        return
    args.resume = True
    preflight(args)
    direct(args)
    train(args)
    if args.max_steps is None or args.max_steps >= args.steps:
        report(args)


if __name__ == "__main__":
    main()
