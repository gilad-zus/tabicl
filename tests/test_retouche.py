"""Mathematical and training-lifecycle checks for the paper reconstruction."""
from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import retouche_benchmark as runner
from tabicl._experiments.retouche import RetoucheAdapter, coslog4, epoch_episodes


def test_adapter_matches_stacked_cross_formula_and_zero_gate_identity():
    torch.manual_seed(1)
    adapter = RetoucheAdapter(8).eval()
    features = torch.randn(19,8)
    representation = features
    for layer in adapter.layers:
        interactions = representation @ layer.input_projection.weight.T @ layer.output_projection.weight.T + layer.cross_bias
        correction = features * interactions
        correction = (correction-layer.normalization.running_mean) / torch.sqrt(layer.normalization.running_var+layer.normalization.eps)
        correction = correction*layer.normalization.weight+layer.normalization.bias
        representation = representation+correction
    torch.testing.assert_close(adapter(features),features+adapter.gate*(representation-features))
    with torch.no_grad():
        adapter.gate.zero_()
    assert torch.equal(adapter(features),features)


def test_optimizer_groups_and_all_branches_receive_gradients():
    torch.manual_seed(2)
    adapter = RetoucheAdapter(12)
    optimizer = adapter.optimizer()
    groups = {group["group_name"]:group for group in optimizer.param_groups}
    assert groups["gate"]["lr"] == .015 and groups["gate"]["weight_decay"] == 0
    assert groups["matrices"]["weight_decay"] == .003
    assert groups["biases_and_normalizations"]["weight_decay"] == 0
    assert optimizer.defaults["betas"] == (.9,.97)
    features = torch.randn(31,12)
    (adapter(features)*torch.randn_like(features)).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())
    assert all(layer.input_projection.weight.grad.norm()>0 and layer.output_projection.weight.grad.norm()>0 for layer in adapter.layers)
    assert adapter.gate.grad.norm()>0


def test_projection_dimension_and_query_batch_invariance():
    torch.manual_seed(3)
    adapter = RetoucheAdapter(12,feature_cap=5).eval()
    features = torch.randn(21,12)
    assert adapter(features).shape == (21,5)
    torch.testing.assert_close(adapter(features),torch.cat((adapter(features[:7]),adapter(features[7:]))))
    assert isinstance(adapter.projection,nn.Linear)
    torch.testing.assert_close(adapter.projection.weight@adapter.projection.weight.T,torch.eye(5),atol=1e-6,rtol=1e-6)


def test_full_pool_episode_coverage_disjointness_and_reproducibility():
    labels = np.array([0]*97+[1]*4+[2]*3)
    first = list(epoch_episodes(labels,context_rows=20,query_rows=7,seed=1))
    repeated = list(epoch_episodes(labels,context_rows=20,query_rows=7,seed=1))
    for (context_positions,query_positions),(repeated_context,repeated_query) in zip(first,repeated):
        assert not np.intersect1d(context_positions,query_positions).size
        assert set(labels[context_positions]) == {0,1,2}
        assert 1<=len(query_positions)<=7 and len(context_positions)<=20
        np.testing.assert_array_equal(context_positions,repeated_context)
        np.testing.assert_array_equal(query_positions,repeated_query)
    assert set(np.concatenate([np.concatenate(episode) for episode in first])) == set(range(len(labels)))
    all_query_positions = set()
    for epoch_number in range(100):
        for context_positions,query_positions in epoch_episodes(labels,context_rows=20,query_rows=7,seed=epoch_number):
            all_query_positions.update(query_positions.tolist())
    assert all_query_positions == set(range(len(labels)))


def test_schedule_matches_primary_coslog4_definition():
    assert coslog4(0)==0
    assert coslog4(1)==pytest.approx(0,abs=1e-12)
    for cycle_number in range(4):
        peak_progress=(2**(cycle_number+.5)-1)/15
        assert coslog4(peak_progress)==pytest.approx(1)
    with pytest.raises(ValueError):
        coslog4(1.1)


def test_saved_metrics_distinguish_probability_and_decision_quality():
    labels=np.array([0,0,1,1])
    probabilities=np.array([[.9,.1],[.6,.4],[.4,.6],[.1,.9]])
    scores=runner.metrics(labels,probabilities)
    assert scores["accuracy"]==scores["balanced_accuracy"]==scores["macro_f1"]==scores["auc"]==1
    assert scores["multiclass_brier"]==pytest.approx(2*scores["binary_brier"])
    with pytest.raises(FloatingPointError):
        runner.metrics(labels,probabilities*np.nan)


def test_real_tabicl_training_path_is_differentiable():
    from tabicl._model.tabicl import TabICL
    torch.set_num_threads(1)
    backbone=TabICL(max_classes=2,embed_dim=16,col_num_blocks=1,col_nhead=2,col_num_inds=4,
        row_num_blocks=1,row_nhead=2,row_num_cls=1,icl_num_blocks=1,icl_nhead=2,
        zero_init=False,dropout=0.)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    runner.native._set_frozen_autograd_routing(backbone)
    adapter=RetoucheAdapter(4)
    features=torch.randn(20,4)
    labels=torch.arange(20)%2
    loss=runner.forward_loss(backbone,adapter,features[:14],labels[:14],features[14:],labels[14:],2)
    loss.backward()
    assert torch.isfinite(loss) and adapter.gate.grad.norm()>0
    assert all(parameter.grad is None for parameter in backbone.parameters())
    probabilities=runner.predict(backbone,adapter,features[:14].numpy(),labels[:14].numpy(),
        features[14:].numpy(),2,4)
    assert probabilities.shape==(6,2) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1),1,atol=1e-6)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder=nn.Parameter(torch.zeros(()),requires_grad=False)

    def clear_cache(self):
        pass

    def forward(self,features,context_labels,**kwargs):
        context_count=context_labels.shape[1]
        context_features=features[:,:context_count]
        query_features=features[:,context_count:]
        centers=torch.stack([context_features[0,context_labels[0]==label].mean(0) for label in (0,1)])
        return -((query_features[:,:,None]-centers)**2).sum(-1)


class TinyEstimator:
    def predict_proba(self,frame):
        positive=1/(1+np.exp(-frame.iloc[:,0].to_numpy()))
        return np.stack((1-positive,positive),axis=1)


def tiny_task():
    random_generator=np.random.default_rng(9)
    features=random_generator.normal(size=(80,4)).astype(np.float32)
    labels=(features[:,0]>0).astype(np.int64)
    return Namespace(task_id=1,dataset_id=1,dataset_name="tiny",problem_type="binary",n_classes=2,
        x_train=pd.DataFrame(features),y_train=labels,x_test=pd.DataFrame(features[:5]),y_test=labels[:5])


def tiny_args(output_dir):
    return Namespace(output_dir=output_dir,epochs=2,patience=2,context_rows=12,query_rows=6,
        prediction_context_rows=0,query_chunk_rows=8,checkpoint=None,device="cpu")


def test_fold_resume_matches_uninterrupted_training_and_test_lock(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    task=tiny_task()
    fitting_positions=np.arange(60)
    validation_positions=np.arange(60,80)
    monkeypatch.setattr(runner,"native_estimator",lambda *args: TinyEstimator())
    manifest={"fingerprint":"test"}
    uninterrupted=tiny_args(tmp_path/"uninterrupted")
    assert runner.fit_fold(uninterrupted,task,fitting_positions,validation_positions,0,TinyBackbone(),manifest,float("inf"))
    resumed=tiny_args(tmp_path/"resumed")
    actual_clock=runner.time.monotonic
    clock_calls=iter([0,0,0,2])
    monkeypatch.setattr(runner.time,"monotonic",lambda:next(clock_calls,2))
    assert not runner.fit_fold(resumed,task,fitting_positions,validation_positions,0,TinyBackbone(),manifest,1)
    assert not (resumed.output_dir/"tasks/1/fold_0/lock.json").exists()
    assert not list(resumed.output_dir.rglob("*test*"))
    monkeypatch.setattr(runner.time,"monotonic",actual_clock)
    assert runner.fit_fold(resumed,task,fitting_positions,validation_positions,0,TinyBackbone(),manifest,float("inf"))
    first=torch.load(uninterrupted.output_dir/"tasks/1/fold_0/selected.pt",weights_only=True)
    second=torch.load(resumed.output_dir/"tasks/1/fold_0/selected.pt",weights_only=True)
    for parameter_name in first["adapter"]:
        torch.testing.assert_close(first["adapter"][parameter_name],second["adapter"][parameter_name],atol=0,rtol=0)


def test_rejected_adapter_test_prediction_equals_native_fallback(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    task=tiny_task()
    fitting_positions=np.arange(60)
    validation_positions=np.arange(60,80)
    args=tiny_args(tmp_path)
    monkeypatch.setattr(runner,"native_estimator",lambda *arguments: TinyEstimator())
    manifest={"fingerprint":"test"}
    runner.fit_fold(args,task,fitting_positions,validation_positions,0,TinyBackbone(),manifest,float("inf"))
    lock_path=tmp_path/"tasks/1/fold_0/lock.json"
    lock=json.loads(lock_path.read_text())
    lock["guard"]["use_adapted"]=False
    runner.write_json(lock_path,lock)
    assert runner.score_task(args,task,[(fitting_positions,validation_positions)],TinyBackbone(),manifest,float("inf"))
    predictions=np.load(tmp_path/"tasks/1/predictions.npz")
    np.testing.assert_array_equal(predictions["guarded"],predictions["native_bagged"])
