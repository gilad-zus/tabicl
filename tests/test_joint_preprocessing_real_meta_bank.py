import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import joint_preprocessing_real_meta_bank as bank


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def family(seed=0, rare=True):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(5)})
    frame["category"] = pd.Categorical(np.resize(["a", "b", "c"], len(frame)))
    frame.loc[::13, "x0"] = np.nan
    labels = np.resize(np.arange(3), len(frame))
    if rare:
        labels[-2:] = 3
    return dict(family=f"fixture{seed}", source_group=f"source{seed}",
                columns=bank.pack_frame(frame), labels=labels.tolist(),
                source_indices=list(range(len(frame))))


@pytest.mark.parametrize("rows", [128, 256, 512, 1024])
@pytest.mark.parametrize("fraction", [.5, .7, .85])
def test_rare_classes_are_preserved_and_requested_caps_reported(rows, fraction):
    raw = family()
    episode = bank.sample_real_episode(raw, rows, fraction, 4)
    assert episode["requested_rows"] == rows
    assert episode["actual_rows"] == min(rows, 300)
    assert episode["n_context"] + episode["n_query"] == episode["actual_rows"]
    assert not set(episode["context_indices"].tolist()) & set(episode["query_indices"].tolist())
    assert set(episode["y_context"].flatten().tolist()) == {0, 1, 2, 3}
    assert set(episode["y_query"].tolist()) == {0, 1, 2, 3}
    for key in ("x_context", "x_query", "y_context", "y_query", "context_indices", "query_indices"):
        torch.testing.assert_close(episode[key], bank.sample_real_episode(raw, rows, fraction, 4)[key], rtol=0, atol=0)
    assert torch.isfinite(episode["x_context"]).all() and torch.isfinite(episode["x_query"]).all()
    assert len(episode["categorical_features"]) == 1
    assert int(episode["numerical_mask"].sum()) == 5


def test_encoding_and_imputation_fit_context_only():
    raw = family(rare=False)
    first = bank.sample_real_episode(raw, 256, .7, 9)
    changed = copy.deepcopy(raw)
    query_rows = set(first["query_indices"].tolist())
    # Query-only new categories cannot change the context encoding; new unseen
    # values must receive the existing encoder's unknown token.
    changed["columns"][-1]["values"] = ["new_query_category" if i in query_rows else value
                                         for i, value in enumerate(changed["columns"][-1]["values"])]
    changed["columns"][0]["values"] = [1e7 if i in query_rows else value
                                        for i, value in enumerate(changed["columns"][0]["values"])]
    second = bank.sample_real_episode(changed, 256, .7, 9)
    torch.testing.assert_close(first["x_context"], second["x_context"], rtol=0, atol=0)
    assert torch.all(second["x_query"][..., 0] == -1)


def test_content_hash_ignores_row_column_order_and_detects_target_changes():
    frame = bank.unpack_frame(family()["columns"])
    labels = np.asarray(family()["labels"])
    perm = np.random.default_rng(3).permutation(len(labels))
    original = bank.content_fingerprints(frame, labels)
    moved = bank.content_fingerprints(frame.iloc[perm, ::-1], labels[perm])
    assert original["inputs_sha256"] == moved["inputs_sha256"]
    assert original["table_sha256"] == moved["table_sha256"]
    relabeled = bank.content_fingerprints(frame, (labels + 1) % 4)
    assert original["inputs_sha256"] == relabeled["inputs_sha256"]
    assert original["table_sha256"] != relabeled["table_sha256"]
    with pytest.raises(ValueError, match="unsafe duplicate"):
        bank._reject_duplicate(relabeled, "new", [dict(family="old", fingerprints=original)])


def manifest_fixture():
    entry = lambda name: dict(source="pmlb", name=name, source_group=f"group_{name}")
    return dict(format_version=1, target_counts=dict(train=2, validation=1, test=1),
                split_seeds=[0, 1], probe_seed=24, source_sample_seed=42,
                context_fraction=.7, max_episode_rows=256, max_source_rows=256,
                eligibility=dict(min_rows=256, min_classes=2, max_classes=10),
                candidates=dict(train=[entry("a"), entry("b")], validation=[entry("c")], test=[entry("d")]))


def test_source_aliases_and_historical_test_members_are_rejected():
    manifest = manifest_fixture()
    manifest["candidates"]["test"][0]["aliases"] = ["a"]
    with pytest.raises(ValueError, match="source alias"):
        bank._validate_candidates(manifest)
    manifest = manifest_fixture()
    manifest["historically_inspected_names"] = ["d"]
    with pytest.raises(ValueError, match="historical"):
        bank._validate_candidates(manifest)
    manifest = manifest_fixture()
    manifest["candidates"]["test"][0]["source_group"] = "group_a"
    with pytest.raises(ValueError, match="source group"):
        bank._validate_candidates(manifest)


def test_real_banks_are_separate_safe_resume_and_locked(tmp_path, monkeypatch):
    manifest = manifest_fixture()
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(manifest), encoding="utf8")
    calls = []

    def fake_source(entry, cache_dir):
        calls.append(entry["name"])
        raw = family(ord(entry["name"]))
        return bank.unpack_frame(raw["columns"]), np.asarray(raw["labels"]), None

    monkeypatch.setattr(bank, "_load_candidate", fake_source)
    output = tmp_path / "output"
    frozen = bank.prepare_real_bank(output, source, tmp_path / "cache")
    assert set(frozen["banks"]) == {"real_train", "real_probe", "real_validation", "real_test"}
    assert len(bank.load_real_bank(output, "train")) == 2
    assert len(bank.load_real_bank(output, "probe")) == 2
    assert len(bank.load_real_bank(output, "validation")) == 2
    assert len(bank.load_real_bank(output, "test")) == 2
    for name, metadata in frozen["banks"].items():
        payload = torch.load(output / metadata["path"], weights_only=True)
        assert set(payload) == {"format_version", "families" if name == "real_train" else "episodes"}
    assert bank.prepare_real_bank(output, source, tmp_path / "cache") == frozen
    assert calls == ["a", "b", "c", "d"]
    availability = json.loads((output / "availability.json").read_text())
    assert "scores" not in availability
    altered = copy.deepcopy(manifest)
    altered["probe_seed"] = 90
    source.write_text(json.dumps(altered), encoding="utf8")
    with pytest.raises(ValueError, match="manifest changed"):
        bank.prepare_real_bank(output, source, tmp_path / "cache")
    metadata = frozen["banks"]["real_train"]
    with (output / metadata["path"]).open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        bank.load_real_bank(output, "train")


def test_insufficient_pool_fails_and_cached_preparation_resumes(tmp_path, monkeypatch):
    manifest = manifest_fixture()
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(manifest), encoding="utf8")
    failure = {"enabled": True}
    calls = []

    def fake_source(entry, cache_dir):
        calls.append(entry["name"])
        if entry["name"] == "c" and failure["enabled"]:
            raise RuntimeError("temporary source unavailable")
        raw = family(ord(entry["name"]))
        return bank.unpack_frame(raw["columns"]), np.asarray(raw["labels"]), None

    monkeypatch.setattr(bank, "_load_candidate", fake_source)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="1 eligible independent|0/1"):
        bank.prepare_real_bank(output, source, tmp_path / "cache")
    assert not (output / "real_manifest.json").exists()
    assert not (output / "banks" / "real_test.pt").exists()
    failure["enabled"] = False
    bank.prepare_real_bank(output, source, tmp_path / "cache")
    assert calls.count("a") == calls.count("b") == 1


def test_checked_in_candidates_have_disjoint_aliases_and_fresh_test():
    manifest = json.loads(bank.DEFAULT_CANDIDATES.read_text(encoding="utf8"))
    bank._validate_candidates(manifest)
    for panel, target in manifest["target_counts"].items():
        assert len({entry["source_group"] for entry in manifest["candidates"][panel]}) >= target
