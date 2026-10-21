import copy
import dataclasses
import json

import pytest
import torch

from scripts import joint_preprocessing_class_profiles as runner
from tabicl._hyperspline.class_profile_preprocessing import ClassProfileContextEncoder
from tabicl._hyperspline.statistics import summarize_context
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, source_family
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def parent_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = runner.base.prior.new_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(models=dict(control=runner.base.pilot.state_cpu(model)),
                optimizers=dict(control=optimizer.state_dict()), coverage={})


def sample_context():
    context_values = torch.tensor([[[-9., 1.], [-1., float('nan')], [1., 3.], [9., 4.],
                                    [-9., 5.], [-1., 6.], [1., 7.], [9., 8.]]])
    context_labels = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]])
    context_missing = ~torch.isfinite(context_values)
    statistics = summarize_context(context_values, context_missing, context_labels)
    return context_values, context_labels, statistics, context_missing


def test_profiles_keep_probabilities_class_frequency_and_missing_category():
    context_values, labels, statistics, missing = sample_context()
    conditional = ClassProfileContextEncoder(True)
    marginal = ClassProfileContextEncoder(False)
    profiles, frequencies = conditional.profiles(context_values, labels, statistics, missing)[0]
    control_profiles, _ = marginal.profiles(context_values, labels, statistics, missing)[0]
    assert profiles.shape == (2, 2, 20)
    torch.testing.assert_close(profiles[..., :19].sum(-1), torch.ones(2, 2))
    torch.testing.assert_close(profiles[..., 19], torch.full((2, 2), .5))
    assert profiles[0, 1, 18] == .25 and profiles[1, 1, 18] == 0
    assert control_profiles[0, 1, 18] == .125
    torch.testing.assert_close(control_profiles[0], control_profiles[1])
    assert not torch.equal(profiles[0], profiles[1])
    torch.testing.assert_close((profiles[..., :19] * frequencies[:, None, None]).sum(0), control_profiles[0, :, :19])


def test_tail_bins_and_all_missing_columns_are_finite():
    values = torch.tensor([[[-10., float('nan')], [0., float('nan')], [10., float('nan')]]])
    labels = torch.tensor([[0, 0, 0]])
    missing = ~torch.isfinite(values)
    statistics = summarize_context(values, missing, labels)
    # Fix the reference scale so the example explicitly exercises both tail bins.
    statistics.scale.fill_(1.)
    encoder = ClassProfileContextEncoder(True)
    profiles, _ = encoder.profiles(values, labels, statistics, missing)[0]
    assert profiles[0, 0, 0] == pytest.approx(1 / 3)
    assert profiles[0, 0, 17] == pytest.approx(1 / 3)
    assert profiles[0, 1, 18] == 1
    assert torch.isfinite(encoder(values, labels, statistics, missing)).all()


def test_nonzero_profile_encoder_is_class_invariant_and_column_equivariant():
    values, labels, statistics, missing = sample_context()
    encoder = ClassProfileContextEncoder(True)
    with torch.no_grad():
        encoder.profile_projection.weight.fill_(.03)
    expected = encoder(values, labels, statistics, missing)
    renamed_labels = 11 - 7 * labels
    renamed_statistics = summarize_context(values, missing, renamed_labels)
    torch.testing.assert_close(encoder(values, renamed_labels, renamed_statistics, missing), expected)
    column_positions = torch.tensor([1, 0])
    permuted_values = values.index_select(-1, column_positions)
    permuted_missing = missing.index_select(-1, column_positions)
    permuted_statistics = summarize_context(permuted_values, permuted_missing, labels)
    torch.testing.assert_close(encoder(permuted_values, labels, permuted_statistics, permuted_missing),
                               expected.index_select(1, column_positions))


def test_initial_outputs_and_old_optimizer_moments_match_exactly():
    parent = parent_state()
    untouched = copy.deepcopy(parent)
    models, optimizers = runner.load_arms(parent, 'cpu')
    starting = runner.base.prior.new_model()
    starting.load_state_dict(parent['models']['control'])
    values, labels, _, missing = sample_context()
    expected_parameters = starting.generate(values, labels, missing)
    parent_names = [name for name, _ in starting.named_parameters()]
    saved_ids = parent['optimizers']['control']['param_groups'][0]['params']
    for arm in runner.ARMS:
        actual = models[arm].generate(values, labels, missing)
        for field in dataclasses.fields(actual):
            expected_value = getattr(expected_parameters, field.name)
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(expected_value, getattr(actual, field.name))
        named_parameters = dict(models[arm].named_parameters())
        for name, parameter_id in zip(parent_names, saved_ids, strict=True):
            assert_nested_equal(optimizers[arm].state[named_parameters[name]],
                                parent['optimizers']['control']['state'][parameter_id])
        for name, parameter in named_parameters.items():
            if name not in parent_names:
                assert parameter not in optimizers[arm].state
    assert_nested_equal(models[runner.ARMS[0]].state_dict(), models[runner.ARMS[1]].state_dict())
    assert_nested_equal(parent, untouched)


def fixture_run(tmp_path):
    entries = []
    for family, panel, rows in [('training', 'train', 96), ('validation', 'validation', 101),
                                ('outside_a', 'evaluation', 106), ('outside_b', 'evaluation', 111)]:
        source = source_family(family, rows=rows, mixed=False)
        cache_path = tmp_path / 'raw' / f'{family}.pt'
        runner.base.pilot.atomic_save(cache_path, source)
        entry = dict(family=family, panel=panel, source_group=family, full_rows=rows,
            cache_path=str(cache_path), cache_sha256=runner.base.pilot.hash_file(cache_path),
            exposure='previously_declared_source')
        if panel == 'train':
            entry.update(fitting_positions=torch.arange(88), probe_query_positions=torch.arange(88, 96))
        entries.append(entry)
    parent = parent_state()
    parent['coverage'] = {'training': {role: torch.zeros(96, dtype=torch.bool) for role in ('context', 'query')}}
    options = runner.parse_args(['pipeline', '--output-dir', str(tmp_path / 'run'), '--source-dir', str(tmp_path / 'parent'),
        '--device', 'cpu', '--additional-updates', '2', '--resume'])
    manifest = dict(fingerprint='fixture', source_lock=dict(backbone=dict(path='fixture.pt', sha256='fixture')))
    return options, entries, parent, manifest


def test_real_backprop_and_atomic_resume_reproduce_pair(tmp_path):
    options, entries, parent, manifest = fixture_run(tmp_path)
    initial_models, _ = runner.load_arms(parent, 'cpu')
    initial_mlp = copy.deepcopy(initial_models[runner.ARMS[1]].encoder.class_profile_mlp.state_dict())
    for folder in (tmp_path / 'whole', tmp_path / 'resumed'):
        options.output_dir = folder
        models, optimizers = runner.load_arms(parent, 'cpu')
        options.additional_updates = 2 if folder.name == 'whole' else 1
        runner.train(options, tiny(), models, optimizers, entries, manifest, float('inf'), parent)
        if folder.name == 'resumed':
            runner.base.append_jsonl(folder / 'episodes.jsonl', dict(additional_update=2, arm='incomplete'))
            options.additional_updates = 2
            models, optimizers = runner.load_arms(parent, 'cpu')
            runner.train(options, tiny(), models, optimizers, entries, manifest, float('inf'), parent)
    whole = torch.load(tmp_path / 'whole/state.pt', weights_only=True)
    resumed = torch.load(tmp_path / 'resumed/state.pt', weights_only=True)
    assert_nested_equal(whole, resumed)
    assert not resumed['coverage']['training']['context'][88:].any()
    assert not resumed['coverage']['training']['query'][88:].any()
    episodes = [json.loads(line) for line in (tmp_path / 'resumed/episodes.jsonl').read_text().splitlines()]
    assert len(episodes) == 16
    for marginal, conditional in zip(episodes[::2], episodes[1::2], strict=True):
        assert all(marginal[key] == conditional[key] for key in ('context_hash', 'query_hash', 'native_nll', 'step', 'family'))
        assert marginal['context_rows'] <= 4096 and marginal['query_rows'] <= 512
    assert episodes[0]['profile_branch_rms'] == 0
    assert episodes[-1]['profile_branch_rms'] > 0
    assert any(not torch.equal(initial_mlp[name], resumed['models'][runner.ARMS[1]][f'encoder.class_profile_mlp.{name}'])
               for name in initial_mlp)
    updates = [json.loads(line) for line in (tmp_path / 'resumed/updates.jsonl').read_text().splitlines()]
    assert all(record['profile_gradient_norm'] > 0 and record['parameter_update_norm'] > 0 for record in updates)


def test_evaluation_pipeline_preserves_full_queries_and_resume_endpoint(tmp_path, monkeypatch):
    options, entries, parent, manifest = fixture_run(tmp_path)
    options.additional_updates = 1
    checkpoint = dict(model=parent['models']['control'], step=576)
    monkeypatch.setattr(runner, 'prepare', lambda *arguments: (manifest, entries, checkpoint, parent))
    monkeypatch.setattr(runner.base.pilot, 'load_frozen', lambda *arguments: (tiny(), 'fixture.pt', 'fixture'))
    runner.pipeline(options)
    report = json.loads((options.output_dir / 'complete.json').read_text())
    assert report['completed_updates_per_arm'] == 1
    expected_episodes = dict(training_replay=4, reserved_training=1, validation_context4096=1, validation=3, evaluation=4)
    for stage, expected_count in expected_episodes.items():
        panel = json.loads((options.output_dir / stage / 'complete.json').read_text())
        assert len(panel['records']) == expected_count
        assert set(panel['summaries']) == {*runner.ARMS, 'starting'}
    reserved = json.loads((options.output_dir / 'reserved_training/complete.json').read_text())
    assert reserved['records'][0]['query_rows'] == 8
    outside = json.loads((options.output_dir / 'evaluation/complete.json').read_text())
    assert [record['query_rows'] for record in outside['records']] == [22, 22, 23, 23]
    assert outside['outside_details'][runner.ARMS[1]]['overall']['datasets'] == 2
    checkpoint_hashes = {name: runner.base.pilot.hash_file(options.output_dir / f'{name}.pt') for name in (*runner.ARMS, 'starting')}
    (options.output_dir / 'complete.json').unlink()
    def unexpected_training(*arguments):
        pytest.fail('evaluation resume must not train')
    monkeypatch.setattr(runner, 'train', unexpected_training)
    monkeypatch.setattr(runner.base, 'episode_for', lambda *arguments, **keywords: pytest.fail('completed panels must skip preprocessing'))
    runner.pipeline(options)
    assert checkpoint_hashes == {name: runner.base.pilot.hash_file(options.output_dir / f'{name}.pt') for name in checkpoint_hashes}


def test_budget_stop_keeps_both_arms_at_zero_updates(tmp_path):
    options, entries, parent, manifest = fixture_run(tmp_path)
    models, optimizers = runner.load_arms(parent, 'cpu')
    assert runner.train(options, tiny(), models, optimizers, entries, manifest, 0., parent) == 0
    saved = torch.load(options.output_dir / 'state.pt', weights_only=True)
    assert saved['completed_updates'] == 0
    assert_nested_equal(saved['models'][runner.ARMS[0]], saved['models'][runner.ARMS[1]])
    assert runner.panel_visits(options, entries, 'training_replay') == []


def test_prepare_pins_parent_state_and_rejects_later_changes(tmp_path, monkeypatch):
    options, entries, parent, _ = fixture_run(tmp_path)
    runner.base.pilot.atomic_save(options.source_dir / 'sources.pt', dict(entries=entries))
    parent_manifest = dict(settings=dict(code_hashes=runner.specification(options)['code_hashes']),
        counts=dict(train=1, validation=1, evaluation=2), backbone=dict(path='fixture.pt', sha256='fixture'),
        files={'sources.pt': runner.base.pilot.hash_file(options.source_dir / 'sources.pt')})
    parent_manifest['fingerprint'] = runner.base.previous.digest(parent_manifest)
    monkeypatch.setattr(runner.frozen, 'PARENT_FINGERPRINT', parent_manifest['fingerprint'])
    monkeypatch.setattr(runner.frozen, 'EXPECTED_COUNTS', dict(train=1, validation=1, evaluation=2))
    runner.base.pilot.json_write(options.source_dir / 'manifest.json', parent_manifest)
    parent.update(fingerprint=parent_manifest['fingerprint'], completed_updates=256)
    runner.base.pilot.atomic_save(options.source_dir / 'state.pt', parent)
    monkeypatch.setattr(runner, 'PARENT_STATE_SHA256', runner.base.pilot.hash_file(options.source_dir / 'state.pt'))
    runner.base.pilot.atomic_save(options.source_dir / 'control.pt', dict(model=parent['models']['control'], step=576))
    for filename in ('complete.json', 'training_complete.json'):
        runner.base.pilot.json_write(options.source_dir / filename, dict(fingerprint=parent_manifest['fingerprint'],
            completed_updates_per_arm=256, reached_target=True, absolute_step=576))
    manifest, _, _, _ = runner.prepare(options)
    assert 'src/tabicl/_hyperspline/class_profile_preprocessing.py' in manifest['settings']['code_hashes']
    with (options.source_dir / 'state.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='audited completed bytes'):
        runner.prepare(options)
