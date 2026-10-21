import copy
import json

import pytest
import torch

from scripts import joint_preprocessing_frozen_transfer as runner
from tests.test_joint_preprocessing_real_meta_continuation import source_family
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def prepared_parent(tmp_path, monkeypatch):
    options = runner.parse_args(["pipeline", "--output-dir", str(tmp_path / "evaluation"),
        "--source-dir", str(tmp_path / "parent"), "--device", "cpu", "--query-chunk-rows", "7"])
    entries = []
    for family, panel, rows in (("training", "train", 96), ("validation", "validation", 96),
                                ("outside_a", "evaluation", 101), ("outside_b", "evaluation", 111)):
        source = source_family(family, rows=rows, mixed=False)
        cache_path = options.source_dir / "raw" / f"{family}.pt"
        runner.base.pilot.atomic_save(cache_path, source)
        entries.append(dict(family=family, source_group=family, panel=panel, full_rows=rows,
            cache_path=str(cache_path), cache_sha256=runner.base.pilot.hash_file(cache_path),
            exposure="catalog_only_source" if family == "outside_b" else "previously_declared_source"))
    runner.base.pilot.atomic_save(options.source_dir / "sources.pt", dict(entries=entries))
    counts = dict(train=1, validation=1, evaluation=2)
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", counts)
    parent_manifest = dict(settings=dict(code_hashes=runner.specification(options)["code_hashes"]),
        counts=counts, backbone=dict(path="fixture.pt", sha256="fixture"),
        files={"sources.pt": runner.base.pilot.hash_file(options.source_dir / "sources.pt")})
    parent_manifest["fingerprint"] = runner.base.previous.digest(parent_manifest)
    monkeypatch.setattr(runner, "PARENT_FINGERPRINT", parent_manifest["fingerprint"])
    runner.base.pilot.json_write(options.source_dir / "manifest.json", parent_manifest)
    completion = dict(fingerprint=runner.PARENT_FINGERPRINT, completed_updates_per_arm=256,
                      absolute_step=576, reached_target=True)
    for filename in ("training_complete.json", "complete.json"):
        runner.base.pilot.json_write(options.source_dir / filename, completion)
    model = runner.base.prior.new_model()
    with torch.no_grad():
        model.affine_head.weight.fill_(.003)
        model.spline_head.weight.fill_(.002)
    model_state = runner.base.pilot.state_cpu(model)
    runner.base.pilot.atomic_save(options.source_dir / "control.pt", dict(model=model_state, step=576))
    runner.base.pilot.atomic_save(options.source_dir / "state.pt", dict(fingerprint=runner.PARENT_FINGERPRINT,
        completed_updates=256, models=dict(control=model_state)))
    monkeypatch.setattr(runner.base.pilot, "load_frozen", lambda *arguments:
                        (tiny(), "fixture.pt", "fixture"))
    return options, entries


def test_selects_only_outside_sources_and_rejects_source_overlap(tmp_path, monkeypatch):
    _, entries = prepared_parent(tmp_path, monkeypatch)
    assert [entry["family"] for entry in runner.evaluation_entries(entries)] == ["outside_a", "outside_b"]
    overlapping = copy.deepcopy(entries)
    overlapping[-1]["source_group"] = "training"
    with pytest.raises(ValueError, match="overlap"):
        runner.evaluation_entries(overlapping)
    with pytest.raises(ValueError, match="allocation"):
        runner.evaluation_entries(entries[:-1])


def test_export_must_match_durable_control_and_completed_endpoint(tmp_path, monkeypatch):
    options, _ = prepared_parent(tmp_path, monkeypatch)
    runner.prepare(options)
    checkpoint = torch.load(options.source_dir / "control.pt", weights_only=True)
    checkpoint["model"]["affine_head.weight"].add_(.1)
    runner.base.pilot.atomic_save(options.source_dir / "control.pt", checkpoint)
    with pytest.raises(ValueError, match="durable paired state"):
        runner.prepare(options)


def test_manifest_rejects_modified_checkpoint_bytes(tmp_path, monkeypatch):
    options, _ = prepared_parent(tmp_path, monkeypatch)
    runner.prepare(options)
    with (options.source_dir / "control.pt").open("ab") as handle:
        handle.write(b"changed bytes")
    with pytest.raises(ValueError, match="inputs, checkpoint or settings changed"):
        runner.prepare(options)


def test_actual_tiny_backbone_scores_full_queries_without_training(tmp_path, monkeypatch):
    options, _ = prepared_parent(tmp_path, monkeypatch)
    before = runner.base.pilot.hash_file(options.source_dir / "control.pt")
    def unexpected_optimizer(*arguments, **keywords):
        raise AssertionError("frozen inference must not create an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", unexpected_optimizer)
    assert runner.pipeline(options)
    report = json.loads((options.output_dir / "complete.json").read_text())
    assert report["datasets"] == 2 and report["episodes"] == 4 and report["training_updates"] == 0
    assert {record["family"] for record in report["records"]} == {"outside_a", "outside_b"}
    assert [record["query_rows"] for record in report["records"]] == [21, 21, 23, 23]
    assert all(record["context_rows"] + record["query_rows"] in (101, 111) for record in report["records"])
    assert before == runner.base.pilot.hash_file(options.source_dir / "control.pt")
    for progress_path in (options.output_dir / "predictions").glob("*/progress.pt"):
        saved = torch.load(progress_path, weights_only=True)
        assert saved["query_offset"] in (21, 23)
        assert set(saved["predictions"]) == {"native16", "candidate"}
    monkeypatch.setattr(runner.base.pilot, "load_frozen", lambda *arguments:
                        pytest.fail("completed evaluation must skip model loading"))
    assert runner.pipeline(options)


def test_resume_skips_completed_predictions_and_reproduces_full_report(tmp_path, monkeypatch):
    options, _ = prepared_parent(tmp_path, monkeypatch)
    original_prediction = runner.base.prediction_record
    calls = []
    def interrupted_prediction(*arguments):
        calls.append(arguments[3]["family"])
        return original_prediction(*arguments) if len(calls) == 1 else None
    monkeypatch.setattr(runner.base, "prediction_record", interrupted_prediction)
    assert not runner.pipeline(options)
    assert len(list((options.output_dir / "predictions").glob("*/complete.json"))) == 1
    resumed_calls = []
    def resumed_prediction(*arguments):
        resumed_calls.append(arguments[3]["family"])
        return original_prediction(*arguments)
    monkeypatch.setattr(runner.base, "prediction_record", resumed_prediction)
    assert runner.pipeline(options)
    assert len(resumed_calls) == 3
    resumed = json.loads((options.output_dir / "complete.json").read_text())
    # Replaying the same frozen experiment in a new folder must preserve row hashes and scores.
    options.output_dir = tmp_path / "uninterrupted"
    assert runner.pipeline(options)
    uninterrupted = json.loads((options.output_dir / "complete.json").read_text())
    assert resumed["records"] == uninterrupted["records"]
    assert resumed["summaries"] == uninterrupted["summaries"]


def test_historical_relation_is_preserved_in_exposure_reporting(tmp_path, monkeypatch):
    options, _ = prepared_parent(tmp_path, monkeypatch)
    manifest, entries, _ = runner.prepare(options)
    entries[-1]["historical_relation"] = "historically related source"
    monkeypatch.setattr(runner, "prepare", lambda *arguments: (manifest, entries,
        torch.load(options.source_dir / "control.pt", weights_only=True)))
    assert runner.pipeline(options)
    report = json.loads((options.output_dir / "complete.json").read_text())
    assert report["summaries"]["previously_declared_source"]["datasets"] == 2
    assert "catalog_only_source" not in report["summaries"]


def test_summary_keeps_equal_dataset_weight_and_largest_contributor_sensitivity():
    records = [dict(family=family, exposure="previously_declared_source",
                    scores=dict(native16=dict(nll=baseline), candidate=dict(nll=candidate)))
               for family, baseline, candidate in (("a", .5, .4), ("a", .5, .6), ("b", 1., .8))]
    result = runner.summaries(records)
    assert result["overall"]["mean_nll"] == pytest.approx(.65)
    assert result["overall"]["datasets"] == 2
    assert result["without_largest_contributor"]["removed_family"] == "b"
    assert result["without_largest_contributor"]["mean_nll_reduction"] == pytest.approx(0.)
    assert result["overall"]["local_nll_elo_equivalent"] > 0
