"""Seed-zero, paper-based Retouche default on TabArena-Lite classification.

Eight preserved inner folds, full fitting pools, metric-based identity guard,
and recoverable training state. This is a documented reconstruction; the paper
does not publish enough implementation detail for a bit-identical reproduction.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    brier_score_loss, f1_score, log_loss, roc_auc_score)
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from tabicl import TabICLClassifier
from tabicl._experiments import direct_spline_openml as native
from tabicl._experiments.direct_spline_protocol import FoldPreprocessor, choose_identity_guard, deployment_error
from tabicl._experiments.retouche import RetoucheAdapter, coslog4, epoch_episodes


# Frozen suite-457 classification membership, including categorical-only tasks.
CLASSIFICATION_TASKS = (363613,363614,363616,363618,363619,363620,363621,363623,
    363624,363626,363627,363628,363629,363630,363632,363671,363673,363674,
    363676,363677,363679,363681,363682,363683,363684,363685,363689,363691,
    363694,363696,363699,363700,363702,363704,363706,363707,363711,363712)
CLASSIFIER_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def save_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def save_predictions(path, **arrays):
    temporary_path=path.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(temporary_path,path)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_csv(path, field, maximum):
    if not path.exists():
        return
    with path.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        field_names=reader.fieldnames
        records=[record for record in reader if int(record[field])<=maximum]
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def metrics(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise FloatingPointError("predictions must be finite nonnegative probabilities")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    class_confusion=confusion_matrix(labels,predictions,labels=np.arange(probabilities.shape[1]))
    class_counts=class_confusion.sum(axis=1)
    # Balanced accuracy averages recall over classes actually present in this query.
    balanced_accuracy=float(np.mean(class_confusion.diagonal()[class_counts>0]/class_counts[class_counts>0]))
    scores = dict(rows=len(labels), nll=float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=balanced_accuracy,
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)))
    one_hot_labels = np.eye(probabilities.shape[1])[labels]
    scores["multiclass_brier"] = float(np.mean(np.sum((probabilities - one_hot_labels)**2, axis=1)))
    if probabilities.shape[1] == 2 and len(np.unique(labels)) == 2:
        scores.update(auc=float(roc_auc_score(labels, probabilities[:,1])),
            average_precision=float(average_precision_score(labels, probabilities[:,1])),
            binary_brier=float(brier_score_loss(labels, probabilities[:,1])))
    return scores


def protocol(args):
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repository, text=True).strip()
    if args.expected_revision and revision != args.expected_revision:
        raise ValueError("checkout revision differs from the declared experiment")
    # Pin the adapter, runner, data/guard utilities and classification runtime.
    dependency_paths = [Path(__file__), repository/"src/tabicl/_experiments/retouche.py",
        repository/"src/tabicl/_experiments/direct_spline_openml.py",
        repository/"src/tabicl/_experiments/direct_spline_protocol.py",
        repository/"src/tabicl/_experiments/tabarena_direct_spline_protocol.py"]
    dependency_paths += [path for directory in ("_model","_sklearn")
        for path in (repository/"src/tabicl"/directory).glob("*.py")
        if path.name not in ("quantile_dist.py", "regressor.py")]
    code_hashes = {path.relative_to(repository).as_posix():hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
                   for path in dependency_paths}
    return dict(version=1, revision=revision, code_hashes=code_hashes,
        task_ids=list(args.task_ids or CLASSIFICATION_TASKS), outer_split=[0,0,0], seed=0, folds=8,
        epochs=args.epochs, patience=args.patience, context_rows=args.context_rows,
        query_rows=args.query_rows, prediction_context_rows=args.prediction_context_rows,
        query_chunk_rows=args.query_chunk_rows, checkpoint_sha256=native._file_sha256(args.checkpoint),
        defaults=dict(layers=2,rank_ratio=.25,batch_norm=True,alpha=.02,lr=.005,
            weight_decay=.003,gate_lr_factor=3,beta2=.97,label_smoothing=.15,clip=2,schedule="coslog4"),
        reconstruction_choices=dict(small_normal_std=.01,batch_norm="correction before residual",
            gate="direct parameter, clamp [0,1] after updates",rank_floor=1,
            batching="shuffled complete fitting-pool pass in mini-tables, fresh class anchors each epoch",
            precision="FP32 differentiable training with activation checkpointing; public inference precision",
            selection="minimum deployment-metric validation error; 0.5% guard against native raw-input fold baseline",
            feature_guard="fixed standardized input clamp [-100,100] after fold-local scaling",
            scope="38 classification tasks; regression deferred; no HPO"))


def checked_manifest(args):
    specification = protocol(args)
    if specification["checkpoint_sha256"] != CLASSIFIER_SHA256:
        raise ValueError("unexpected TabICLv2 classifier checkpoint")
    content = json.dumps(specification, sort_keys=True).encode()
    manifest = dict(specification=specification, fingerprint=hashlib.sha256(content).hexdigest())
    path = args.output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("resume settings or code changed")
    write_json(path, manifest)
    return manifest


def tensor_features(features, device):
    return torch.as_tensor(features, dtype=torch.float32, device=device)


def predict(backbone, adapter, context_features, context_labels, query_features, n_classes, query_chunk_rows):
    """Inference uses fixed support and frozen adapter BatchNorm statistics."""
    device = next(backbone.parameters()).device
    adapter.eval()
    backbone.eval()
    predictions = []
    try:
        with torch.no_grad():
            transformed_context = adapter(tensor_features(context_features,device))
            for start in range(0,len(query_features),query_chunk_rows):
                transformed_query = adapter(tensor_features(query_features[start:start+query_chunk_rows],device))
                backbone.clear_cache()
                logits = backbone(torch.cat((transformed_context,transformed_query))[None],
                    tensor_features(context_labels,device)[None], return_logits=True)
                predictions.append((logits[0,:,:n_classes]/.9).softmax(-1).float().cpu().numpy())
    finally:
        native._set_frozen_autograd_routing(backbone)
    return np.concatenate(predictions)


def native_estimator(args, fit_frame, fit_labels):
    # This raw-input estimator is the paper's actual identity fallback.
    estimator = TabICLClassifier(n_estimators=8, norm_methods=["none","power"],
        feat_shuffle_method="latin", class_shuffle_method="shift", outlier_threshold=4.,
        softmax_temperature=.9, average_logits=True, support_many_classes=True,
        batch_size=8, kv_cache=False, numerical_preprocessing="existing", random_state=0,
        model_path=str(args.checkpoint), device=args.device, verbose=False)
    estimator.fit(fit_frame, fit_labels)
    return estimator


def forward_loss(backbone, adapter, context_features, context_labels, query_features, query_labels, n_classes):
    adapter.train()
    transformed_features = adapter(torch.cat((context_features,query_features)))
    def forward_model(features):
        backbone.clear_cache()
        return backbone(features[None],context_labels[None])[0,:,:n_classes]
    logits = checkpoint(forward_model, transformed_features, use_reentrant=False)
    return functional.cross_entropy(logits/.9,query_labels.long(),label_smoothing=.15)


def fit_fold(args, task, fitting_positions, validation_positions, fold_number, backbone, manifest, deadline):
    folder = args.output_dir / "tasks" / str(task.task_id) / f"fold_{fold_number}"
    folder.mkdir(parents=True, exist_ok=True)
    lock_path, state_path = folder/"lock.json", folder/"state.pt"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("fold lock fingerprint mismatch")
        return True
    fitting_frame = task.x_train.iloc[fitting_positions].reset_index(drop=True)
    validation_frame = task.x_train.iloc[validation_positions].reset_index(drop=True)
    fitting_labels, validation_labels = task.y_train[fitting_positions], task.y_train[validation_positions]
    preprocessor = FoldPreprocessor.fit(fitting_frame)
    # Outlying standardized inputs previously overflowed frozen-model AMP.
    fitting_features = np.clip(preprocessor.transform(fitting_frame),-100,100)
    validation_features = np.clip(preprocessor.transform(validation_frame),-100,100)
    device = next(backbone.parameters()).device
    torch.manual_seed(native._seed(0,task.task_id,fold_number))
    adapter = RetoucheAdapter(fitting_features.shape[1]).to(device)
    initial_adapter = copy.deepcopy(adapter).eval()
    optimizer = adapter.optimizer()
    support_positions = native.sample_prediction_context(fitting_labels,problem_type=task.problem_type,
        max_context_rows=args.prediction_context_rows or len(fitting_labels),
        rng=np.random.default_rng(native._seed(0,task.task_id,fold_number,42)))
    context_features, context_labels = fitting_features[support_positions], fitting_labels[support_positions]
    probe_context_positions,probe_query_positions=next(epoch_episodes(fitting_labels,
        context_rows=args.context_rows,query_rows=args.query_rows,seed=native._seed(0,task.task_id,fold_number,777)))
    epoch_number, batch_number, updates, stale_epochs = 0, 0, 0, 0
    best_error, best_epoch, best_state = math.inf, 0, None
    query_exposures = np.zeros(len(fitting_labels), dtype=np.int64)
    if state_path.exists():
        state = torch.load(state_path,map_location="cpu",weights_only=False)
        if state["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("training state fingerprint mismatch")
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for name,value in optimizer_state.items():
                if isinstance(value,torch.Tensor):
                    optimizer_state[name] = value.to(device)
        epoch_number,batch_number,updates,stale_epochs = (state[name] for name in ("epoch","batch","updates","stale"))
        best_error,best_epoch,best_state = (state[name] for name in ("best_error","best_epoch","best_state"))
        query_exposures = state["query_exposures"]
        # A crash can leave log rows newer than the last atomic training state.
        truncate_csv(folder/"training.csv","update",updates)
        truncate_csv(folder/"validation.csv","epoch",epoch_number)
    else:
        truncate_csv(folder/"training.csv","update",0)
        truncate_csv(folder/"validation.csv","epoch",0)
    baseline_path = folder/"baseline_validation.npz"
    if baseline_path.exists():
        baseline_validation = np.load(baseline_path)["probabilities"]
    else:
        estimator = native_estimator(args,fitting_frame,fitting_labels)
        baseline_validation = estimator.predict_proba(validation_frame.copy())
        save_predictions(baseline_path,probabilities=baseline_validation,labels=validation_labels)
        del estimator
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_error = deployment_error(task.problem_type,validation_labels,baseline_validation,n_classes=task.n_classes)
    write_json(folder/"data.json",dict(fingerprint=manifest["fingerprint"],fitting_positions=fitting_positions.tolist(),
        validation_positions=validation_positions.tolist(),support_positions=support_positions.tolist(),
        fitting_rows=len(fitting_labels),validation_rows=len(validation_labels),feature_count=fitting_features.shape[1],
        trainable_parameters=sum(parameter.numel() for parameter in adapter.parameters()),
        preprocessing=preprocessor.__dict__,baseline_validation_metrics=metrics(validation_labels,baseline_validation)))
    def checkpoint_state():
        save_state(state_path,dict(fingerprint=manifest["fingerprint"],adapter=native._cpu_state_dict(adapter),
            optimizer=optimizer.state_dict(),epoch=epoch_number,batch=batch_number,updates=updates,stale=stale_epochs,
            best_error=best_error,best_epoch=best_epoch,best_state=best_state,query_exposures=query_exposures))
    print(f"FIT {task.dataset_name} fold={fold_number} fit={len(fitting_labels)} val={len(validation_labels)} support={len(support_positions)} features={fitting_features.shape[1]} resume_epoch={epoch_number}",flush=True)
    while epoch_number < args.epochs and stale_epochs < args.patience:
        episodes = list(epoch_episodes(fitting_labels,context_rows=args.context_rows,query_rows=args.query_rows,
            seed=native._seed(0,task.task_id,fold_number,epoch_number)))
        for episode_number in range(batch_number,len(episodes)):
            if time.monotonic() > deadline:
                checkpoint_state()
                print("Allocation budget reached; state saved for --resume",flush=True)
                return False
            context_positions, query_positions = episodes[episode_number]
            progress = (epoch_number+(episode_number+.5)/len(episodes))/args.epochs
            rate_factor = coslog4(progress)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = (.015 if parameter_group["group_name"] == "gate" else .005)*rate_factor
            optimizer.zero_grad(set_to_none=True)
            loss = forward_loss(backbone,adapter,
                tensor_features(fitting_features[context_positions],device),tensor_features(fitting_labels[context_positions],device),
                tensor_features(fitting_features[query_positions],device),tensor_features(fitting_labels[query_positions],device),task.n_classes)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Retouche objective")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                adapter.gate.clamp_(0,1)
            query_exposures[query_positions] += 1
            updates += 1
            batch_number = episode_number+1
            append_csv(folder/"training.csv",dict(epoch=epoch_number+1,batch=batch_number,update=updates,
                loss=float(loss.detach()),learning_rate=.005*rate_factor,gate_learning_rate=.015*rate_factor,
                gradient_norm=float(gradient_norm),gate_mean=float(adapter.gate.detach().mean()),
                context_rows=len(context_positions),query_rows=len(query_positions),
                query_coverage=float((query_exposures>0).mean())))
            if updates % 10 == 0:
                checkpoint_state()
        candidate = predict(backbone,adapter,context_features,context_labels,validation_features,task.n_classes,args.query_chunk_rows)
        candidate_error = deployment_error(task.problem_type,validation_labels,candidate,n_classes=task.n_classes)
        epoch_number += 1
        batch_number = 0
        if candidate_error < best_error:
            best_error,best_epoch,best_state,stale_epochs = candidate_error,epoch_number,native._cpu_state_dict(adapter),0
        else:
            stale_epochs += 1
        candidate_metrics = metrics(validation_labels,candidate)
        probe_prediction=predict(backbone,adapter,fitting_features[probe_context_positions],
            fitting_labels[probe_context_positions],fitting_features[probe_query_positions],task.n_classes,args.query_chunk_rows)
        probe_metrics=metrics(fitting_labels[probe_query_positions],probe_prediction)
        append_csv(folder/"validation.csv",dict(epoch=epoch_number,update=updates,deployment_error=candidate_error,
            baseline_error=baseline_error,best_error=best_error,best_epoch=best_epoch,stale_epochs=stale_epochs,
            fixed_training_probe_nll=probe_metrics["nll"],fixed_training_probe_accuracy=probe_metrics["accuracy"],
            **candidate_metrics))
        checkpoint_state()
        print(f"VAL {task.dataset_name} fold={fold_number} epoch={epoch_number}/{args.epochs} error={candidate_error:.6g} base={baseline_error:.6g} best@{best_epoch} accuracy={candidate_metrics['accuracy']:.5f} queries_seen={(query_exposures>0).mean():.3f}",flush=True)
    adapter.load_state_dict(best_state)
    selected_validation = predict(backbone,adapter,context_features,context_labels,validation_features,task.n_classes,args.query_chunk_rows)
    guard = choose_identity_guard(identity_error=baseline_error,adapted_error=best_error,required_relative_improvement=.005)
    guard_record = guard.__dict__.copy()
    # A perfect baseline has undefined relative improvement and keeps identity.
    if not math.isfinite(guard_record["relative_improvement"]):
        guard_record["relative_improvement"] = None
    save_state(folder/"selected.pt",dict(adapter=native._cpu_state_dict(adapter),initial_adapter=native._cpu_state_dict(initial_adapter),
        fingerprint=manifest["fingerprint"],epoch=best_epoch))
    save_predictions(folder/"selected_validation.npz",probabilities=selected_validation,labels=validation_labels)
    write_json(lock_path,dict(fingerprint=manifest["fingerprint"],task_id=task.task_id,fold=fold_number,
        selected_epoch=best_epoch,executed_epochs=epoch_number,updates=updates,guard=guard_record,
        selected_sha256=native._file_sha256(folder/"selected.pt"),
        query_coverage=float((query_exposures>0).mean()),test_labels_used_for_training=False,test_labels_used_for_selection=False))
    return True


def score_task(args, task, splits, backbone, manifest, deadline):
    task_folder = args.output_dir/"tasks"/str(task.task_id)
    completion_path = task_folder/"complete.json"
    if completion_path.exists():
        return True
    # Every fold's guard/checkpoint is locked before any outer-test metric is read.
    locks = [json.loads((task_folder/f"fold_{number}"/"lock.json").read_text()) for number in range(len(splits))]
    for fold_number,lock in enumerate(locks):
        if lock["fingerprint"] != manifest["fingerprint"] or native._file_sha256(task_folder/f"fold_{fold_number}"/"selected.pt") != lock["selected_sha256"]:
            raise ValueError("selected model lock changed")
    all_predictions = {name:[] for name in ("native_bagged","standardized_identity","adapted","guarded")}
    for fold_number,(fitting_positions,validation_positions) in enumerate(splits):
        if time.monotonic() > deadline:
            return False
        folder = task_folder/f"fold_{fold_number}"
        predictions_path = folder/"test_predictions.npz"
        if predictions_path.exists():
            predictions = dict(np.load(predictions_path))
        else:
            fitting_frame = task.x_train.iloc[fitting_positions].reset_index(drop=True)
            fitting_labels = task.y_train[fitting_positions]
            metadata = json.loads((folder/"data.json").read_text())
            preprocessing_record=metadata["preprocessing"].copy()
            for column_field in ("columns","numerical_columns","categorical_columns"):
                preprocessing_record[column_field]=tuple(preprocessing_record[column_field])
            preprocessor = FoldPreprocessor(**preprocessing_record)
            fitting_features = np.clip(preprocessor.transform(fitting_frame),-100,100)
            test_features = np.clip(preprocessor.transform(task.x_test),-100,100)
            device = next(backbone.parameters()).device
            adapter = RetoucheAdapter(fitting_features.shape[1]).to(device)
            selected = torch.load(folder/"selected.pt",map_location="cpu",weights_only=True)
            adapter.load_state_dict(selected["adapter"])
            support_positions = np.array(metadata["support_positions"])
            adapted_prediction = predict(backbone,adapter,fitting_features[support_positions],fitting_labels[support_positions],test_features,task.n_classes,args.query_chunk_rows)
            adapter.load_state_dict(selected["initial_adapter"])
            # Zeroing the gate yields the exact fixed standardized/projection control.
            with torch.no_grad():
                adapter.gate.zero_()
            identity_prediction = predict(backbone,adapter,fitting_features[support_positions],fitting_labels[support_positions],test_features,task.n_classes,args.query_chunk_rows)
            estimator = native_estimator(args,fitting_frame,fitting_labels)
            native_prediction = estimator.predict_proba(task.x_test.copy())
            del estimator
            predictions = dict(native_bagged=native_prediction,standardized_identity=identity_prediction,
                adapted=adapted_prediction,guarded=adapted_prediction if locks[fold_number]["guard"]["use_adapted"] else native_prediction)
            save_predictions(predictions_path,**predictions)
        for method in all_predictions:
            all_predictions[method].append(predictions[method])
    full_baseline_path = task_folder/"native_full_predictions.npz"
    if full_baseline_path.exists():
        native_full = np.load(full_baseline_path)["probabilities"]
    else:
        estimator = native_estimator(args,task.x_train,task.y_train)
        native_full = estimator.predict_proba(task.x_test.copy())
        save_predictions(full_baseline_path,probabilities=native_full)
        del estimator
    combined = {method:np.mean(predictions,axis=0) for method,predictions in all_predictions.items()}
    combined["native_full"] = native_full
    save_predictions(task_folder/"predictions.npz",labels=task.y_test,**combined)
    write_json(completion_path,dict(fingerprint=manifest["fingerprint"],task_id=task.task_id,dataset=task.dataset_name,
        problem_type=task.problem_type,folds=len(splits),adapted_folds=sum(lock["guard"]["use_adapted"] for lock in locks),
        scores={method:metrics(task.y_test,predictions) for method,predictions in combined.items()}))
    print(f"TEST COMPLETE {task.dataset_name}: {completion_path}",flush=True)
    return True


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.context_rows < 10 or args.query_rows < 1 or args.prediction_context_rows < 0:
        raise ValueError("invalid training or context budget")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("nonempty output directory requires --resume")
    manifest = checked_manifest(args)
    deadline = time.monotonic()+args.max_runtime_minutes*60
    device = torch.device(args.device)
    backbone,_,_ = native.load_frozen_backbone(problem_type="binary",device=device,
        classifier_checkpoint=args.checkpoint,regressor_checkpoint=None)
    preflight_path=args.output_dir/"preflight.json"
    if not preflight_path.exists():
        # Check the actual checkpoint's differentiable path before expensive fits.
        preflight_records=[]
        for feature_count in (8,500):
            torch.manual_seed(0)
            adapter=RetoucheAdapter(feature_count).to(device)
            features=torch.randn(80,feature_count,device=device)
            labels=torch.arange(80,device=device)%2
            loss=forward_loss(backbone,adapter,features[:64],labels[:64],features[64:],labels[64:],2)
            loss.backward()
            parameter_gradients=[parameter.grad for parameter in adapter.parameters()]
            if not torch.isfinite(loss) or any(gradient is None or not torch.isfinite(gradient).all() for gradient in parameter_gradients):
                raise FloatingPointError("Retouche checkpoint gradient preflight failed")
            gradient_norm=float(torch.stack([gradient.norm() for gradient in parameter_gradients]).norm())
            if gradient_norm<=0 or any(parameter.grad is not None for parameter in backbone.parameters()):
                raise RuntimeError("preflight requires nonzero adapter gradients and frozen backbone")
            preflight_records.append(dict(feature_count=feature_count,loss=float(loss.detach()),gradient_norm=gradient_norm))
            del adapter,features,loss
        write_json(preflight_path,dict(fingerprint=manifest["fingerprint"],checks=preflight_records))
        print("CHECKPOINT PREFLIGHT PASSED",preflight_records,flush=True)
    for task_id in manifest["specification"]["task_ids"]:
        if time.monotonic() > deadline:
            break
        task = native.load_tabarena_openml_task(task_id,outer_repeat=0,outer_fold=0,outer_sample=0)
        if task.problem_type == "regression" or task.n_classes > 10:
            raise ValueError("declared classification panel has an unsupported task")
        splits = list(native._bag_splits(task,requested_bags=8,seed=0))
        task_folder = args.output_dir/"tasks"/str(task_id)
        import pandas as pd
        data_hash=hashlib.sha256(pd.util.hash_pandas_object(task.x_train,index=True).values.tobytes()
            +np.asarray(task.y_train,dtype=np.int64).tobytes()
            +pd.util.hash_pandas_object(task.x_test,index=True).values.tobytes()).hexdigest()
        task_record=dict(task_id=task_id,dataset_id=task.dataset_id,dataset=task.dataset_name,data_sha256=data_hash,
            outer_split_hash=task.outer_split_hash,official_train_rows=len(task.y_train),official_test_rows=len(task.y_test),
            n_classes=task.n_classes,effective_folds=len(splits),fingerprint=manifest["fingerprint"])
        task_path=task_folder/"task.json"
        if task_path.exists() and json.loads(task_path.read_text())!=task_record:
            raise ValueError("task data or official split changed")
        write_json(task_path,task_record)
        for fold_number,(fitting_positions,validation_positions) in enumerate(splits):
            if not fit_fold(args,task,fitting_positions,validation_positions,fold_number,backbone,manifest,deadline):
                return
        if not score_task(args,task,splits,backbone,manifest,deadline):
            return
    completed = [json.loads(path.read_text()) for path in sorted(args.output_dir.glob("tasks/*/complete.json"))]
    write_json(args.output_dir/"progress.json",dict(fingerprint=manifest["fingerprint"],completed_datasets=len(completed),
        requested_datasets=len(manifest["specification"]["task_ids"]),results=completed))
    if len(completed) == len(manifest["specification"]["task_ids"]):
        write_json(args.output_dir/"complete.json",dict(fingerprint=manifest["fingerprint"],results=completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--context-rows",type=int,default=1024)
    parser.add_argument("--query-rows",type=int,default=256)
    parser.add_argument("--prediction-context-rows",type=int,default=0)
    parser.add_argument("--query-chunk-rows",type=int,default=256)
    parser.add_argument("--max-runtime-minutes",type=float,default=465)
    parser.add_argument("--expected-revision")
    parser.add_argument("--task-ids",type=int,nargs="+")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    try:
        run(args)
    finally:
        # Partial allocations still expose useful progress without opening test labels.
        if (args.output_dir/"manifest.json").exists():
            write_json(args.output_dir/"status.json",dict(
                locked_folds=len(list(args.output_dir.glob("tasks/*/fold_*/lock.json"))),
                completed_datasets=len(list(args.output_dir.glob("tasks/*/complete.json"))),
                all_complete=(args.output_dir/"complete.json").exists()))


if __name__ == "__main__":
    main()
