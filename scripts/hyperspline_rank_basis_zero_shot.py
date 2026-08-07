"""Dataset-disjoint, end-to-end rank-basis HyperSpline experiment.

The spline basis is fitted only on DirectSpline teachers from meta-train
datasets.  The conditioner itself never imitates teacher parameters: it is
trained through frozen-TabICL query cross entropy.  Shape and normalization
have independent bounded gates and, in the joint arm, independent encoders.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tabicl._hyperspline import HyperSplineTransform, summarize_context

try:
    from scripts.direct_spline_function_basis import TeacherBag, fit_pca
    from scripts.direct_spline_multidataset_headroom import load_backbone
except ModuleNotFoundError:  # pragma: no cover
    from direct_spline_function_basis import TeacherBag, fit_pca
    from direct_spline_multidataset_headroom import load_backbone


class FactorizedRankBasisSpline(nn.Module):
    """A monotone rank-basis spline with separately gated output factors."""

    def __init__(self, mean: torch.Tensor, components: torch.Tensor, *, branch: str,
                 hidden_dim: int, coefficient_bound: float, location_bound: float,
                 log_scale_bound: float, target_aware: bool, raw_context: bool,
                 gate_initial_probability: float) -> None:
        super().__init__()
        if branch not in {"shape", "normalization", "joint"}:
            raise ValueError(f"unknown branch: {branch}")
        self.branch = branch
        self.rank = components.shape[0]
        self.coefficient_bound = coefficient_bound
        self.location_bound = location_bound
        self.log_scale_bound = log_scale_bound
        self.register_buffer("grid", torch.linspace(-4.0, 4.0, mean.numel()))
        self.register_buffer("mean_curve", mean)
        self.register_buffer("components", components)

        def encoder() -> HyperSplineTransform:
            return HyperSplineTransform(
                n_control_points=self.rank + 1,
                hidden_dim=hidden_dim,
                target_aware=target_aware,
                raw_context_residual=raw_context,
                raw_context_num_heads=4,
                generate_location=True,
                generate_scale=True,
                gate_initial_probability=gate_initial_probability,
                raw_context_gate_initial_probability=gate_initial_probability,
            )

        self.shape_encoder = encoder() if branch in {"shape", "joint"} else None
        self.normalization_encoder = encoder() if branch in {"normalization", "joint"} else None

    @staticmethod
    def _labels(y: torch.Tensor, enabled: bool) -> torch.Tensor | None:
        return y.float().unsqueeze(0) if enabled else None

    def _raw(self, encoder: HyperSplineTransform, x: torch.Tensor, y: torch.Tensor, stats):
        return encoder.generate_raw(stats, x_context=x, y_context=self._labels(y, encoder.target_aware))[0]

    def parameters(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        enabled = bool((self.shape_encoder or self.normalization_encoder).target_aware)
        stats = summarize_context(x, y_context=self._labels(y, enabled))
        batch, _, columns = x.shape
        identity = self.grid.view(1, 1, -1).expand(batch, columns, -1)
        values = identity
        shape_gate = torch.zeros(batch, columns, device=x.device)
        coefficients = torch.zeros(batch, columns, self.rank, device=x.device)
        if self.shape_encoder is not None:
            raw = self._raw(self.shape_encoder, x, y, stats)
            coefficients = self.coefficient_bound * torch.tanh(raw[..., :self.rank])
            candidate = identity + self.mean_curve.view(1, 1, -1) + torch.matmul(coefficients, self.components)
            increments = torch.diff(candidate, dim=-1)
            increments = increments + F.softplus(-increments / 1e-4) * 1e-4
            candidate = torch.cat((candidate[..., :1], candidate[..., :1] + increments.cumsum(-1)), -1)
            shape_gate = torch.sigmoid(raw[..., self.rank])
            values = identity + shape_gate.unsqueeze(-1) * (candidate - identity)

        location = stats.location
        scale = stats.scale.clamp_min(1e-6)
        normalization_gate = torch.zeros_like(shape_gate)
        location_delta = torch.zeros_like(shape_gate)
        log_scale_delta = torch.zeros_like(shape_gate)
        if self.normalization_encoder is not None:
            raw = self._raw(self.normalization_encoder, x, y, stats)
            normalization_gate = torch.sigmoid(raw[..., self.rank])
            location_delta = normalization_gate * self.location_bound * torch.tanh(raw[..., self.rank + 1])
            log_scale_delta = normalization_gate * self.log_scale_bound * torch.tanh(raw[..., self.rank + 2])
            location = stats.location + stats.scale * location_delta
            scale = stats.scale * torch.exp(log_scale_delta)
        return {"values": values, "location": location, "scale": scale.clamp_min(1e-6),
                "shape_gate": shape_gate, "normalization_gate": normalization_gate,
                "coefficients": coefficients, "location_delta": location_delta,
                "log_scale_delta": log_scale_delta}

    def transform(self, x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
        z = (x.float() - p["location"].unsqueeze(1)) / p["scale"].unsqueeze(1)
        clipped = z.clamp(float(self.grid[0]), float(self.grid[-1]))
        position = (clipped - self.grid[0]) / (self.grid[-1] - self.grid[0]) * (self.grid.numel() - 1)
        left = position.floor().long().clamp(0, self.grid.numel() - 2)
        fraction = (position - left).to(x.dtype)
        table = p["values"].unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        low = table.gather(3, left.unsqueeze(-1)).squeeze(-1)
        high = table.gather(3, (left + 1).unsqueeze(-1)).squeeze(-1)
        return low + fraction * (high - low) + z - clipped

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor):
        p = self.parameters(context_x, context_y)
        return torch.cat((self.transform(context_x, p), self.transform(query_x, p)), 1), p

    def trust_region(self, p: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = (p["values"] - self.grid.view(1, 1, -1)).square().mean()
        normalization = p["location_delta"].square().mean() + p["log_scale_delta"].square().mean()
        return shape + normalization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--train-dataset", action="append", required=True)
    p.add_argument("--validation-dataset", action="append", default=[])
    p.add_argument("--test-dataset", action="append", required=True)
    p.add_argument("--branch", choices=("shape", "normalization", "joint"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8); p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--coefficient-bound", type=float, default=1.5)
    p.add_argument("--location-bound", type=float, default=1.0); p.add_argument("--log-scale-bound", type=float, default=1.0)
    p.add_argument("--gate-initial-probability", type=float, default=0.01)
    p.add_argument("--regularization", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--validate-every", type=int, default=250); p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--patience-validations", type=int, default=20)
    p.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None); p.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    return p.parse_args()


def tensors(bag: TeacherBag, split: str, device: torch.device):
    cx = bag.support_x.to(device)
    cy = bag.support_y.to(device).long().squeeze(0)
    x = getattr(bag, f"{split}_x"); y = getattr(bag, f"{split}_y")
    return cx, cy, torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0), torch.as_tensor(y, dtype=torch.long, device=device)


def predictions(backbone, model, bag, split, device):
    cx, cy, qx, qy = tensors(bag, split, device)
    backbone.clear_cache()
    transformed, params = model(cx, cy, qx)
    return backbone(transformed, cy.float().unsqueeze(0)), qy, params


def metrics(prob: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    p = prob.flatten(0, 1).clamp_min(1e-8); target = y.flatten(); pred = p.argmax(-1)
    classes = torch.unique(target)
    recalls = torch.stack([(pred[target == c] == c).float().mean() for c in classes])
    confidence, _ = p.max(-1); correct = pred.eq(target).float()
    ece = p.new_zeros(())
    for low in torch.linspace(0, .9, 10, device=p.device):
        mask = (confidence >= low) & (confidence < low + .1)
        if mask.any(): ece += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    one_hot = F.one_hot(target, p.shape[-1]).float()
    return {"loss": float(F.nll_loss(p.log(), target)), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(recalls.mean()), "brier": float((p - one_hot).square().sum(-1).mean()),
            "ece": float(ece), "mean_confidence": float(confidence.mean())}


def parameter_metrics(p: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for name in ("shape_gate", "normalization_gate"):
        value = p[name].detach()
        result.update({f"{name}_mean": float(value.mean()), f"{name}_min": float(value.min()), f"{name}_max": float(value.max())})
    for name in ("coefficients", "location_delta", "log_scale_delta"):
        value = p[name].detach()
        result[f"{name}_rms"] = float(value.square().mean().sqrt())
        result[f"{name}_abs_max"] = float(value.abs().max())
    return result


@torch.no_grad()
def evaluate(backbone, model, bags, split, device, stage, step):
    rows=[]
    for bag in bags:
        logits, y, params = predictions(backbone, model, bag, split, device)
        prob = logits.softmax(-1)
        rows.append({"stage":stage,"step":step,"dataset":bag.dataset,"outer_seed":bag.seed,"bag":bag.bag,
                     **metrics(prob,y),**parameter_metrics(params)})
    return rows


def mean_loss(rows): return float(np.mean([row["loss"] for row in rows]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main() -> None:
    args=parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    payload=torch.load(args.teacher_cache.resolve(),map_location="cpu",weights_only=False)
    bags:list[TeacherBag]=payload["bags"]; available={b.dataset for b in bags}
    train,validation,test=set(args.train_dataset),set(args.validation_dataset),set(args.test_dataset)
    if not validation:
        validation={sorted(train)[-1]}; train-=validation
    if train & validation or train & test or validation & test: raise ValueError("dataset splits must be disjoint")
    missing=(train|validation|test)-available
    if missing: raise ValueError(f"datasets absent from teacher cache: {sorted(missing)}")
    train_bags=[b for b in bags if b.dataset in train]; val_bags=[b for b in bags if b.dataset in validation]; test_bags=[b for b in bags if b.dataset in test]
    curves=torch.cat([b.curves for b in train_bags]); mean,components,explained=fit_pca(curves)
    if args.rank > components.shape[0]: raise ValueError(f"rank {args.rank} unavailable; maximum is {components.shape[0]}")
    basis_hash=hashlib.sha256(torch.cat((mean.flatten(),components[:args.rank].flatten())).numpy().tobytes()).hexdigest()
    manifest={"status":"running","args":vars(args)|{"teacher_cache":str(args.teacher_cache),"output_dir":str(out)},
              "train_datasets":sorted(train),"validation_datasets":sorted(validation),"test_datasets":sorted(test),
              "train_bags":len(train_bags),"validation_bags":len(val_bags),"test_bags":len(test_bags),
              "basis_sha256":basis_hash,"basis_explained_variance":np.asarray(explained[:args.rank]).tolist()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf8")
    device=torch.device(args.device); backbone,_=load_backbone(args,device)
    model=FactorizedRankBasisSpline(mean.to(device),components[:args.rank].to(device),branch=args.branch,
        hidden_dim=args.hidden_dim,coefficient_bound=args.coefficient_bound,location_bound=args.location_bound,
        log_scale_bound=args.log_scale_bound,target_aware=args.target_aware,raw_context=args.raw_context,
        gate_initial_probability=args.gate_initial_probability).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed); training_rows=[]; evaluation_rows=[]
    identity_val=evaluate(backbone,model,val_bags,"guard",device,"identity_validation",0)
    identity_test=evaluate(backbone,model,test_bags,"test",device,"identity_test",0)
    evaluation_rows.extend(identity_val); evaluation_rows.extend(identity_test)
    best_loss=mean_loss(identity_val); best_step=0; best_state=copy.deepcopy(model.state_dict()); stale=0
    started=time.time()
    for step in range(1,args.steps+1):
        bag=train_bags[int(rng.integers(len(train_bags)))]
        logits,y,params=predictions(backbone,model,bag,"guard",device)
        task_loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        penalty=model.trust_region(params); objective=task_loss+args.regularization*penalty
        optimizer.zero_grad(set_to_none=True); objective.backward()
        pre_clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip)); optimizer.step()
        if step==1 or step%args.log_every==0:
            row={"step":step,"dataset":bag.dataset,"task_loss":float(task_loss),"objective":float(objective),
                 "trust_region":float(penalty),"weighted_regularization":float(args.regularization*penalty),
                 "pre_clip_gradient_norm":pre_clip,"elapsed_seconds":time.time()-started,**parameter_metrics(params)}
            if device.type=="cuda": row.update({"cuda_allocated_mb":torch.cuda.memory_allocated(device)/2**20,
                "cuda_reserved_mb":torch.cuda.memory_reserved(device)/2**20,"cuda_peak_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20})
            training_rows.append(row); print("[train] "+" ".join(f"{k}={v:.6g}" if isinstance(v,float) else f"{k}={v}" for k,v in row.items() if k in {"step","dataset","task_loss","trust_region","pre_clip_gradient_norm","shape_gate_mean","normalization_gate_mean","cuda_allocated_mb"}),flush=True)
            write_csv(out/"training.csv",training_rows)
        if step%args.validate_every==0 or step==args.steps:
            current=evaluate(backbone,model,val_bags,"guard",device,"validation",step); evaluation_rows.extend(current)
            loss=mean_loss(current); improved=loss < best_loss
            if improved: best_loss,best_step,best_state,stale=loss,step,copy.deepcopy(model.state_dict()),0
            else: stale+=1
            print(f"[validation] step={step} mean_loss={loss:.6f} identity={mean_loss(identity_val):.6f} best={best_loss:.6f}@{best_step} improved={improved}",flush=True)
            write_csv(out/"evaluations.csv",evaluation_rows)
            torch.save({"state_dict":model.state_dict(),"best_state_dict":best_state,"optimizer":optimizer.state_dict(),"step":step,"best_step":best_step,"best_validation_loss":best_loss,"manifest":manifest},out/"last.pt")
            if args.patience_validations and stale>=args.patience_validations: print("Early stopping: validation patience exhausted.",flush=True); break
    model.load_state_dict(best_state)
    selected_val=evaluate(backbone,model,val_bags,"guard",device,"selected_validation",best_step)
    selected_test=evaluate(backbone,model,test_bags,"test",device,"selected_test",best_step)
    evaluation_rows.extend(selected_val); evaluation_rows.extend(selected_test); write_csv(out/"evaluations.csv",evaluation_rows)
    identity_by={(r["dataset"],r["outer_seed"],r["bag"]):r for r in identity_test}
    paired=[]
    for row in selected_test:
        base=identity_by[(row["dataset"],row["outer_seed"],row["bag"])]
        paired.append({"dataset":row["dataset"],"outer_seed":row["outer_seed"],"bag":row["bag"],
            "loss_delta":row["loss"]-base["loss"],"accuracy_delta":row["accuracy"]-base["accuracy"],
            "balanced_accuracy_delta":row["balanced_accuracy"]-base["balanced_accuracy"],"brier_delta":row["brier"]-base["brier"],"ece_delta":row["ece"]-base["ece"]})
    write_csv(out/"paired_test.csv",paired)
    per_dataset=[]
    for dataset in sorted(test):
        group=[r for r in paired if r["dataset"]==dataset]
        per_dataset.append({"dataset":dataset,"n":len(group),**{key:float(np.mean([r[key] for r in group])) for key in ("loss_delta","accuracy_delta","balanced_accuracy_delta","brier_delta","ece_delta")}})
    write_csv(out/"per_dataset.csv",per_dataset)
    summary={"branch":args.branch,"best_step":best_step,"identity_validation_loss":mean_loss(identity_val),"best_validation_loss":best_loss,
        "validation_loss_delta":best_loss-mean_loss(identity_val),"test_bags":len(paired),
        "macro_test_loss_delta":float(np.mean([r["loss_delta"] for r in paired])),"macro_test_accuracy_delta":float(np.mean([r["accuracy_delta"] for r in paired])),
        "test_loss_win_fraction":float(np.mean([r["loss_delta"]<0 for r in paired])),"dataset_loss_win_fraction":float(np.mean([r["loss_delta"]<0 for r in per_dataset])),
        "per_dataset":per_dataset}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    torch.save({"state_dict":best_state,"mean":mean,"components":components[:args.rank],"summary":summary,"manifest":manifest},out/"best.pt")
    manifest["status"]="complete"; manifest["summary"]=summary
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__": main()
