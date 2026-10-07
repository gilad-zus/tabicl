import argparse
import copy
import json
import random

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_dataset_diversity as runner
from scripts import joint_preprocessing_dataset_diversity_bank as bank
from scripts import joint_preprocessing_dataset_diversity_catalog as catalog
from tests.test_joint_preprocessing_real_meta_continuation import (
    Backbone, assert_nested_equal, initial_model, read_rows, real_episode, source_family,
)


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def args_for(root, **changes):
    values = dict(output_dir=root, candidate_manifest=runner.DEFAULT_CANDIDATES,
        cache_dir=root / "cache", arm="small", continuation_seed=0, device="cpu",
        checkpoint=None, steps=2, evaluate_every=1, save_every=1, lr=.0003,
        resume=False, max_steps=None)
    values.update(changes)
    return argparse.Namespace(**values)


def records(count):
    return [dict(family=f"source{i}", descriptors=dict(classes=2 if i % 3 else 3,
        features=6 if i % 2 else 25, categorical_fraction=0. if i % 4 else .5)) for i in range(count)]


def test_balanced_allocation_is_nested_disjoint_and_deterministic():
    values = records(185)
    counts = dict(small=40, large=160, validation=25)
    a, b = bank.allocate(values, counts, 20261007), bank.allocate(list(reversed(values)), counts, 20261007)
    for role in a:
        assert {r["family"] for r in a[role]} == {r["family"] for r in b[role]}
        assert len(a[role]) == counts[role]
    small, large, val = [{r["family"] for r in a[k]} for k in ("small", "large", "validation")]
    assert small < large and not large & val
    assert {bank.stratum(r["descriptors"]) for r in a["small"]} == {bank.stratum(r["descriptors"]) for r in values}


def test_catalog_collapses_known_shared_sources_and_ignores_unreviewed_uploads():
    old = {"candidates": {"train": [], "validation": [], "test": []}}
    def entry(did, name):
        q = dict(NumberOfClasses=2, NumberOfFeatures=7, NumberOfInstances=300, NumberOfNumericFeatures=6)
        return dict(did=did, name=name, quality=[dict(name=k, value=v) for k, v in q.items()])
    snapshot = dict(data=dict(dataset=[entry(49, "heart-c"), entry(51, "heart-h"), entry(999999, "Synthetic-Copy")]))
    result = catalog.build_catalog(snapshot, old)
    assert len(result["candidates"]) == 2
    assert len({r["source_group"] for r in result["candidates"]}) == 1


def test_catalog_holds_adult_derivatives_and_aircraft_targets_in_source_groups():
    old = {"candidates": {"train": [dict(source="openml", data_id=1590,
        name="adult", source_group="uci_adult_census")], "validation": [], "test": []}}
    q = dict(NumberOfClasses=2, NumberOfFeatures=7, NumberOfInstances=300, NumberOfNumericFeatures=6)
    rows = [dict(did=did, name=name, quality=[dict(name=k, value=v) for k, v in q.items()])
            for did, name in [(1037, "ada_prior"), (41156, "ada"), (734, "ailerons"),
                              (846, "elevators"), (819, "delta_elevators")]]
    entries = catalog.build_catalog(dict(data=dict(dataset=rows)), old)["candidates"]
    groups = {e["name"]: e["source_group"] for e in entries}
    assert groups["adult"] == groups["ada_prior"] == groups["ada"]
    assert groups["ailerons"] == groups["elevators"] == groups["delta_elevators"]
    assert groups["adult"] != groups["ailerons"]


def test_training_schedule_balances_visits_and_pairs_shapes():
    source_banks = dict(small=[source_family(f"f{i}", rows=256 + i) for i in range(40)],
                        large=[source_family(f"f{i}", rows=256 + i) for i in range(160)])
    counts = {arm: {} for arm in runner.ARMS}
    for step in range(1, 161):
        for arm in runner.ARMS:
            for f in runner.scheduled_sources(source_banks[arm], step):
                counts[arm][f["family"]] = counts[arm].get(f["family"], 0) + 1
    assert set(counts["small"].values()) == {16}
    assert set(counts["large"].values()) == {4}
    for step in (1, 10, 81, 1024):
        a, b = [runner.training_batch(arm, step, source_banks) for arm in runner.ARMS]
        for x, y in zip(a, b, strict=True):
            assert (x["source_seed"], x["task_id"], x["n_context"], x["n_query"], x["paired_row_cap"]) == (
                y["source_seed"], y["task_id"], y["n_context"], y["n_query"], y["paired_row_cap"])
            assert not set(x["context_indices"].tolist()) & set(x["query_indices"].tolist())


def test_fresh_initialization_is_independent_of_global_rng_and_preserves_it():
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    a = runner.new_model()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    torch.rand(7)
    b = runner.new_model()
    assert_nested_equal(a.state_dict(), b.state_dict())
    assert torch.count_nonzero(a.affine_head.weight) == 0


def test_fresh_default_network_has_finite_learning_gradients_after_initial_update():
    model, backbone, episode = runner.new_model(), Backbone(), real_episode()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = runner.objective.ensemble_backward(backbone, model, episode, 1.)
        norms = runner.core.diagnostic.gradient_norms(model)
        assert np.isfinite(float(value)) and all(np.isfinite(v) for v in norms.values())
        assert sum(norms.values()) > 0
        if step:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        optimizer.step()
    assert all(p.grad is None for p in backbone.parameters())


@pytest.fixture
def fixture(monkeypatch):
    sources = dict(small=[source_family(f"source{i}") for i in range(4)],
                   large=[source_family(f"source{i}") for i in range(8)])
    values = {f"{k}_train": v for k, v in sources.items()}
    values.update({p: [{k: v for k, v in real_episode().items()
                       if k not in ("task_id", "source_seed")}]
                   for p in runner.PANELS})
    model = initial_model()
    manifest = dict(fingerprint="diversity-fixture", initial_sha256="same-initial",
        settings={}, banks={p: dict(count=len(v)) for p, v in values.items()})
    loaded = []
    monkeypatch.setattr(runner, "setup", lambda args: (Backbone(), model, manifest, torch.device("cpu")))
    monkeypatch.setattr(runner, "checked_manifest", lambda args: manifest)
    def load(root, manifest, panel):
        loaded.append(panel)
        return values[panel]
    monkeypatch.setattr(bank, "load_bank", load)
    monkeypatch.setattr(runner.core, "shape_for_step", lambda seed, step: (20, .5))
    return model, manifest, loaded


def train_fixture(args):
    runner.pilot.json_write(args.output_dir / "backbone_lock.json", dict(sha256="fixture"))
    runner.train(args)


def test_preflight_checks_inference_and_gradients_without_training(tmp_path, fixture):
    initial = copy.deepcopy(fixture[0].state_dict())
    episode = real_episode()
    root = tmp_path / "preflight"
    runner.preflight(args_for(root, preflight_family=episode['family'],
                              preflight_split_seed=episode['split_seed']))
    assert fixture[2] == ['real_validation']
    assert_nested_equal(initial, fixture[0].state_dict())
    result = runner.previous.read(root / 'preflight' / 'complete.json')
    assert result['default_amp_finite'] and result['no_optimizer_update']
    audit = runner.previous.read(root / 'preflight' / 'execution_audit.json')
    assert audit['task_id'] is None
    assert (audit['family'], audit['split_seed']) == (episode['family'], episode['split_seed'])
    assert not (root / 'runs').exists()


@pytest.mark.parametrize("arm", runner.ARMS)
def test_resume_preserves_weights_optimizer_rng_and_logs(tmp_path, fixture, arm):
    full, partial = tmp_path / "full", tmp_path / "partial"
    original = copy.deepcopy(fixture[0].state_dict())
    train_fixture(args_for(full, arm=arm))
    train_fixture(args_for(partial, arm=arm, max_steps=1))
    folder = runner.core.run_dir(partial, arm, 0)
    with (folder / "training.csv").open("a") as f:
        f.write("99,0,0\n")
    random.random()
    np.random.rand(3)
    torch.rand(3)
    train_fixture(args_for(partial, arm=arm, resume=True))
    a, b = [torch.load(runner.core.run_dir(root, arm, 0) / "state.pt", weights_only=True) for root in (full, partial)]
    for k in ("model", "optimizer", "best_model", "best_step", "best_score", "rng", "other_rng", "clipped"):
        assert_nested_equal(a[k], b[k])
    assert len(read_rows(folder / "presentations.csv")) == 8
    assert len(read_rows(folder / "learning.csv")) == 9
    assert [int(r["step"]) for r in read_rows(folder / "training.csv")] == [1, 2]
    assert_nested_equal(fixture[0].state_dict(), original)


def test_report_requires_both_runs_and_locks_completed_weights(tmp_path, fixture):
    root = tmp_path / "run"
    train_fixture(args_for(root, arm="small"))
    with pytest.raises(FileNotFoundError):
        runner.report(args_for(root))
    train_fixture(args_for(root, arm="large"))
    runner.report(args_for(root))
    report = runner.previous.read(root / "complete.json")
    assert set(report["large_versus_small"]) == {f"{choice}/{panel}" for choice in ("selected", "final") for panel in runner.PANELS}
    assert set(fixture[2]) == {"small_train", "large_train", *runner.PANELS}
    runner.report(args_for(root))
    with pytest.raises(ValueError, match="locked"):
        runner.train(args_for(root, resume=True))
    (runner.core.run_dir(root, "large", 0) / "selected.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.report(args_for(root))


def test_selection_uses_arithmetic_mean_nll(tmp_path, fixture):
    train_fixture(args_for(tmp_path))
    folder = runner.core.run_dir(tmp_path, "small", 0)
    rows = [r for r in read_rows(folder / "learning.csv") if r["panel"] == "real_validation"]
    done = runner.previous.read(folder / "complete.json")
    selected = min(rows, key=lambda r: float(r["mean_blend_nll"]))
    assert done["selected_step"] == int(selected["step"])
    assert done["selected_score"] == float(selected["mean_blend_nll"])


def test_duplicate_evaluations_and_nonzero_seed_are_rejected(tmp_path):
    rows = [dict(family="a", split_seed=0, blend_nll=.2)]
    with pytest.raises(ValueError, match="duplicated"):
        runner.grouped(rows * 2, "blend_nll", 2)
    with pytest.raises(ValueError, match="seed zero"):
        runner.settings(args_for(tmp_path, continuation_seed=1))
    with pytest.raises(ValueError, match="test-bank"):
        bank.load_bank(tmp_path, {}, "real_test")


def test_success_threshold_requires_broad_wins_and_mean_benefit():
    baseline = np.ones(25)
    assert runner.passing(baseline * .99, baseline)
    candidate = baseline.copy()
    candidate[:10] *= .8
    assert not runner.passing(candidate, baseline)
    assert not runner.passing(baseline * .999, baseline)


def test_bank_failure_cannot_create_ready_manifest(tmp_path, monkeypatch):
    source = dict(format_version=1, target_counts=dict(large=4, small=2, validation=1), candidates=[],
        minimum_rows=256, max_source_rows=16384, seed=20261007)
    candidate = tmp_path / "candidates.json"
    candidate.write_text(json.dumps(source))
    with pytest.raises(RuntimeError, match="no GPU run"):
        bank.prepare(tmp_path / "bank", candidate, tmp_path / "cache")
    assert not (tmp_path / "bank/banks_manifest.json").exists()


def test_prepare_and_manifest_reject_changed_code_candidates_or_initial_weights(tmp_path, monkeypatch):
    banks = dict(counts=dict(small=40, large=160, validation=25), banks={})
    def prepare(root, candidate, cache):
        runner.pilot.json_write(root / "banks_manifest.json", banks)
        return banks
    monkeypatch.setattr(bank, "prepare", prepare)
    root = tmp_path / "run"
    args = args_for(root)
    runner.prepare(args)
    runner.prepare(args)
    assert runner.checked_manifest(args)["no_test_bank"]
    with pytest.raises(ValueError, match="changed"):
        runner.prepare(args_for(root, steps=3))
    (root / "initial.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="initial weights changed"):
        runner.checked_manifest(args)


def test_bank_prepares_balanced_nested_panels_and_skips_copies_before_allocation(tmp_path, monkeypatch):
    entries = [dict(source="sklearn", name=f"table{i}", source_group=f"group{i}") for i in range(6)]
    source = dict(format_version=1, target_counts=dict(large=4, small=2, validation=1), candidates=entries,
        minimum_rows=256, max_source_rows=16384, seed=20261007, split_seeds=[0, 1], scope="fixture")
    candidate = tmp_path / "candidates.json"
    candidate.write_text(json.dumps(source))
    def load(entry, cache):
        raw = source_family(rows=256)
        frame = bank.bank.unpack_frame(raw["columns"])
        index = int(entry["name"][-1])
        # First two entries are undeclared copies; the remaining four independent.
        frame["column_0"] += max(0, index - 1)
        return frame, np.array(raw["labels"]), None, "collected observations"
    monkeypatch.setattr(bank, "load_candidate", load)
    root = tmp_path / "bank"
    manifest = bank.prepare(root, candidate, tmp_path / "cache")
    assert len(manifest["selection"]["large"]) == 4
    small = bank.load_bank(root, manifest, "small_train")
    large = bank.load_bank(root, manifest, "large_train")
    validation = bank.load_bank(root, manifest, "real_validation")
    assert {f["family"] for f in small} < {f["family"] for f in large}
    assert not {f["family"] for f in large} & {e["family"] for e in validation}
    assert len(validation) == 2
    assert len(runner.previous.read(root / "availability.json")["skipped"]) == 1
    bank.prepare(root, candidate, tmp_path / "cache")
    # Recuration must rebuild allocations, while preserving the prior raw cache.
    cache_hashes = {p.name: runner.pilot.hash_file(p) for p in (root / "source_cache").glob("*.pt")}
    for e in entries:
        e.update(source_group="curated_" + e["source_group"], provenance="reviewed provenance")
    candidate.write_text(json.dumps(source))
    def no_download(*args):
        raise AssertionError("cached sources must not be downloaded again")
    monkeypatch.setattr(bank, "load_candidate", no_download)
    rebuilt = tmp_path / "recurated"
    new = bank.prepare(rebuilt, candidate, tmp_path / "cache", reuse_source_cache=root / "source_cache")
    assert len(new["selection"]["large"]) == 4
    assert runner.previous.read(rebuilt / "availability.json")["reused_cached_sources"] == 6
    assert all(r["source_group"].startswith("curated_") for r in new["selection"]["large"])
    assert {p.name: runner.pilot.hash_file(p) for p in (root / "source_cache").glob("*.pt")} == cache_hashes
    assert not (rebuilt / "source_cache").exists()
    # Return the candidate metadata to the original lock before its tamper check.
    for e in entries:
        e["source_group"] = e["source_group"].removeprefix("curated_")
        e.pop("provenance")
    candidate.write_text(json.dumps(source))
    first = manifest["banks"]["small_train"]
    (root / first["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen bank changed"):
        bank.prepare(root, candidate, tmp_path / "cache")


def test_cache_recuration_never_relaxes_data_loading_identity():
    original = dict(source="openml", data_id=1037, name="ada_prior",
                    source_group="old", provenance="old", target="class")
    curated = dict(original, source_group="uci_adult_census", provenance="reviewed")
    assert bank.cached_entry_identity(original) == bank.cached_entry_identity(curated)
    for field, value in [("data_id", 1590), ("name", "adult"), ("target", "other"), ("source", "pmlb")]:
        assert bank.cached_entry_identity(original) != bank.cached_entry_identity(dict(curated, **{field: value}))


def test_reuse_frozen_bank_preserves_data_and_excludes_old_model_state(tmp_path):
    source = tmp_path / "previous"
    target = tmp_path / "repaired"
    candidate = tmp_path / "candidates.json"
    candidate.write_text('{"fixed": true}')
    panels = {}
    for panel in ("small_train", "large_train", "real_probe", "large_only_probe", "real_validation"):
        path = source / "banks" / f"{panel}.pt"
        runner.pilot.atomic_save(path, dict(values=[real_episode()]))
        panels[panel] = dict(path=f"banks/{panel}.pt", count=1, sha256=runner.pilot.hash_file(path))
    manifest = dict(candidate_sha256=runner.pilot.hash_file(candidate), banks=panels)
    runner.pilot.json_write(source / "banks_manifest.json", manifest)
    runner.pilot.json_write(source / "availability.json", dict(accepted=["unchanged"]))
    runner.pilot.atomic_save(source / "references" / "real_probe.pt", dict(old=True))
    runner.pilot.atomic_save(source / "runs" / "small_seed0" / "state.pt", dict(old=True))
    source_hashes = {p.relative_to(source).as_posix(): runner.pilot.hash_file(p)
                     for p in source.rglob('*') if p.is_file()}
    bank.reuse_frozen_banks(target, candidate, source)
    bank.reuse_frozen_banks(target, candidate, source)
    assert runner.previous.read(target / "banks_manifest.json") == manifest
    assert not (target / "references").exists() and not (target / "runs").exists()
    assert {p.relative_to(source).as_posix(): runner.pilot.hash_file(p)
            for p in source.rglob('*') if p.is_file()} == source_hashes
    for entry in panels.values():
        assert runner.pilot.hash_file(target / entry['path']) == entry['sha256']
    candidate.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="candidates changed"):
        bank.reuse_frozen_banks(tmp_path / "changed", candidate, source)
    candidate.write_text('{"fixed": true}')
    (source / panels['large_train']['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match="bank hash changed"):
        bank.reuse_frozen_banks(tmp_path / "corrupted", candidate, source)
    assert not (tmp_path / "corrupted" / "banks_manifest.json").exists()


def test_reused_cache_cannot_bypass_new_synthetic_description_exclusion(tmp_path, monkeypatch):
    import hashlib
    entries = [dict(source="sklearn", name=f"cached{i}", source_group=f"source{i}") for i in range(2)]
    source = dict(target_counts=dict(large=1, small=1, validation=1), candidates=entries,
        minimum_rows=256, max_source_rows=16384, seed=20261007, split_seeds=[0, 1], scope="fixture")
    candidate = tmp_path / "candidates.json"
    candidate.write_text(json.dumps(source))
    cache = tmp_path / "old_cache"
    for i, entry in enumerate(entries):
        raw = source_family(rows=256)
        key = hashlib.sha256(bank.bank._entry_identity(entry).encode()).hexdigest()[:20]
        runner.pilot.atomic_save(cache / f"{key}.pt", dict(entry=entry, columns=raw["columns"],
            labels=raw["labels"], raw_hash=None,
            description="A synthetic version inspired by observations" if i == 0 else "real measurements"))
    def no_download(*args):
        raise AssertionError("all sources are already cached")
    monkeypatch.setattr(bank, "load_candidate", no_download)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no GPU run"):
        bank.prepare(root, candidate, tmp_path / "cache", reuse_source_cache=cache)
    failures = runner.previous.read(root / "availability.json")["failures"]
    assert len(failures) == 1 and "generated/simulated" in failures[0]["reason"]


def test_curated_renamed_copies_share_groups_and_synthetic_names_are_excluded():
    aliases = catalog.alias_map({"candidates": {}})
    for names in [("satimage", "Satellite"), ("spam", "spambase"),
                  ("Credit_Risk_Modeling", "dataset_credit_risk_file_2"),
                  ("credit_risk_china", "dataset_china"), ("credit", "Give-Me-Some-Credit-Sampled"),
                  ("CreditCardSubset", "Credit_Card_Fraud_Classification")]:
        assert len({aliases[bank.bank.normalized_name(n)] for n in names}) == 1
    assert catalog.EXCLUDED_NAMES == bank.KNOWN_GENERATED


def test_generated_source_description_exclusion_is_explicit():
    assert bank.GENERATED_DESCRIPTION.search("This synthetic dataset was generated for classification.")
    assert bank.GENERATED_DESCRIPTION.search("simulated data from Monte Carlo simulation")
    assert bank.GENERATED_DESCRIPTION.search("This dataset is a synthetic version inspired by a real dataset.")
    assert bank.GENERATED_DESCRIPTION.search("All content is synthetic.")
    assert not bank.GENERATED_DESCRIPTION.search("Measurements from hospitals; features include temperature.")


def test_numeric_array_coverage_matches_dataframe():
    raw = source_family(rows=256, mixed=False)
    frame = bank.bank.unpack_frame(raw["columns"])
    episode = bank.bank.sample_real_episode(raw, 128, .7, 0)
    a = bank.descriptors(frame, np.array(raw["labels"]), episode)
    b = bank.descriptors(frame.to_numpy(), np.array(raw["labels"]), episode)
    assert a == b


def test_unfinished_preparation_repair_archives_intent_but_refuses_frozen_data(tmp_path, monkeypatch):
    root = tmp_path / "run"
    runner.pilot.json_write(root / "preparation.json", dict(settings="previous version"))
    class StopBeforeLoading(Exception):
        pass
    def stop(*args):
        raise StopBeforeLoading
    monkeypatch.setattr(bank, "prepare", stop)
    args = args_for(root, repair_unfinished_preparation=True)
    with pytest.raises(StopBeforeLoading):
        runner.prepare(args)
    assert len(list((root / "preparation_revisions").glob("*.json"))) == 1
    runner.pilot.json_write(root / "banks_manifest.json", dict(frozen=True))
    with pytest.raises(ValueError, match="cannot repair frozen"):
        runner.prepare(args_for(root, steps=3, repair_unfinished_preparation=True))


def test_revision_pin_rejects_wrong_head_or_dirty_dependency(monkeypatch):
    import hashlib
    import subprocess
    spec = dict(expected_revision="approved", code_hashes={"runner.py": hashlib.sha256(b"a\nb\n").hexdigest()})
    def call(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="approved\n" if "rev-parse" in argv else b"a\r\nb\r\n")
    monkeypatch.setattr(runner.subprocess, "run", call)
    runner.verify_revision(spec)
    spec["code_hashes"]["runner.py"] = "changed"
    with pytest.raises(ValueError, match="uncommitted"):
        runner.verify_revision(spec)
    spec["expected_revision"] = "different"
    with pytest.raises(ValueError, match="Git revision"):
        runner.verify_revision(spec)
