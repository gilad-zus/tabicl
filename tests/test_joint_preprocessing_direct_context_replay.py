import copy

import pytest
import torch

from scripts import joint_preprocessing_direct_context_replay as replay
from tests.test_joint_preprocessing_statistics_diagnostic import diagnostic_entry
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def one_thread():
    previous_count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_count)


def test_original_context_keeps_all_outer_queries_and_saved_inner_context():
    entry = diagnostic_entry()
    episode = replay.query_episode(entry, "original")
    for field in ("x_context", "y_context", "context_missing", "context_indices"):
        assert torch.equal(episode[field], entry["fitting"][field])
    for field in ("x_query", "y_query", "query_missing", "query_indices"):
        assert torch.equal(episode[field], entry["outer"][field])
    assert episode["n_context"] < entry["outer"]["n_context"]
    assert episode["n_query"] == entry["outer"]["n_query"]


@pytest.mark.parametrize("violation", ["context", "fitting_query", "different_context"])
def test_query_leakage_and_mismatched_inner_context_rejected(violation):
    entry = diagnostic_entry()
    if violation == "context":
        entry["outer"]["context_indices"][0] = entry["outer"]["query_indices"][0]
    elif violation == "fitting_query":
        entry["fitting"]["query_indices"][0] = entry["outer"]["query_indices"][0]
    else:
        entry["selection"]["context_indices"] = entry["selection"]["context_indices"].clone()
        entry["selection"]["context_indices"][0] += 1
    with pytest.raises(ValueError):
        replay.query_episode(entry, "original")


def test_frozen_maps_replay_native_scores_with_real_tiny_backbone():
    entry, backbone = diagnostic_entry(), tiny()
    initial = replay.source.new_model("statistics")
    model = replay.source.direct_model(initial, entry["outer"], torch.device("cpu"))
    selected = dict(fingerprint="fixture", step=0, model=copy.deepcopy(model.state_dict()))
    # A nonidentity final map exercises column alignment and changing preprocessing.
    with torch.no_grad():
        model.raw_outputs["affine_head"].fill_(.03)
    final = dict(fingerprint="fixture", step=2, model=copy.deepcopy(model.state_dict()))
    ordinary = replay.source.score_episode(backbone, None, entry["outer"])
    reference = dict(fingerprint="fixture", family=entry["outer"]["family"],
        selected_step=0, steps=2, selected_outer=ordinary, ordinary16_outer=ordinary,
        selected_selection_nll=replay.source.score_episode(backbone, None, entry["selection"])["nll"],
        final_outer=replay.source.score_episode(backbone, model, entry["outer"]))
    result = replay.evaluate_entry(backbone, initial, entry, selected, final, reference)
    for context in result["contexts"].values():
        assert context["selected"] == context["ordinary16"]
    assert result["maximum_full_context_nll_replay_difference"] == 0.
    assert all(parameter.grad is None for parameter in backbone.parameters())
    broken_reference = dict(reference, ordinary16_outer=dict(ordinary, nll=ordinary["nll"] + .01))
    with pytest.raises(ValueError, match="did not replay"):
        replay.evaluate_entry(backbone, initial, entry, selected, final, broken_reference)


def test_replay_tolerance_rejects_drift_in_small_losses():
    with pytest.raises(ValueError, match="did not replay"):
        replay.check_score(dict(nll=.01, accuracy=.9), dict(nll=.01001, accuracy=.9))


def test_input_lock_rejects_modified_saved_map(tmp_path):
    source_dir = tmp_path / "source"
    entry = diagnostic_entry()
    settings = dict(code_hashes={}, direct_reference=dict(datasets=1))
    for filename in ("banks_manifest.json", "backbone_lock.json"):
        replay.pilot.json_write(source_dir / filename, {})
    replay.pilot.atomic_save(source_dir / "initial_statistics.pt", dict(model={}))
    replay.pilot.atomic_save(source_dir / "banks/direct_reference.pt", dict(values=[entry]))
    lock = dict(specification_fingerprint=replay.previous.digest(settings), protocol=settings["direct_reference"],
        path="banks/direct_reference.pt", sha256=replay.pilot.hash_file(source_dir / "banks/direct_reference.pt"),
        families=[entry["outer"]["family"]])
    replay.pilot.json_write(source_dir / "direct_bank_manifest.json", lock)
    manifest = dict(settings=settings, backbone={},
        banks_manifest_sha256=replay.pilot.hash_file(source_dir / "banks_manifest.json"),
        direct_bank_manifest_sha256=replay.pilot.hash_file(source_dir / "direct_bank_manifest.json"),
        initial_sha256=dict(statistics=replay.pilot.hash_file(source_dir / "initial_statistics.pt")))
    manifest["fingerprint"] = replay.previous.digest(manifest)
    replay.pilot.json_write(source_dir / "manifest.json", manifest)
    replay.pilot.json_write(source_dir / "complete.json", dict(fingerprint=manifest["fingerprint"]))
    replay.pilot.json_write(source_dir / "preflight/complete.json", dict(fingerprint=manifest["fingerprint"], both_passed=True))
    folder = source_dir / "direct/source_00"
    replay.pilot.atomic_save(folder / "selected.pt", dict(model={}))
    replay.pilot.atomic_save(folder / "state.pt", dict(model={}))
    replay.pilot.json_write(folder / "complete.json", dict(fingerprint=manifest["fingerprint"],
        family=entry["outer"]["family"], selected_sha256=replay.pilot.hash_file(folder / "selected.pt"),
        outer_query_used_for_optimization=False, outer_query_used_for_selection=False))
    replay.checked_inputs(source_dir, manifest["fingerprint"])
    replay.pilot.atomic_save(folder / "selected.pt", dict(model={"changed": torch.ones(1)}))
    with pytest.raises(ValueError, match="selected map changed"):
        replay.checked_inputs(source_dir, manifest["fingerprint"])
