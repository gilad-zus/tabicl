"""Matched synthetic/real/mixed continuation of one frozen-backbone preprocessor.

Preparation freezes source weights and disjoint family banks. Training accesses
only source families and validation; final testing requires all six runs locked.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_real_transfer as real
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_zero_shot_comparison as previous
from scripts import joint_preprocessing_zero_shot_real_transfer as transfer
from scripts import hyperspline_synthetic_train as synthetic
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor

ARMS, SEEDS = ("synthetic", "real", "mixed"), (0, 1)
GENERATOR = dict(prior_type="mix_scm", min_features=5, max_features=100,
                 max_classes=10, prior_n_jobs=1,
                 synthetic_observation_mode="coverage_expanded", train_seed=401001)
VAL_SEED, TEST_SEED = 411001, 421001
TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET = 12_000_000_000, 13_000_000_000, 14_000_000_000
LOGS = ("training.csv", "presentations.csv", "evaluation.csv", "evaluation_episodes.csv")
DEFAULT_CANDIDATES = Path(__file__).resolve().parents[1] / "docs/experiments/joint_preprocessing_real_meta_candidates_20261004.json"


def run_dir(root, arm, seed):
    return root / "runs" / f"{arm}_seed{seed}"


def settings(args):
    if min(args.steps, args.evaluate_every, args.save_every, args.synthetic_validation_tasks,
           args.synthetic_test_tasks, args.synthetic_probe_tasks, args.bootstrap_samples) < 1:
        raise ValueError("experiment budgets must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.synthetic_probe_tasks % 4:
        raise ValueError("invalid learning rate or probe budget")
    paths = [Path(__file__), Path(pilot.__file__), Path(real.__file__),
             Path(previous.__file__), Path(diagnostic.__file__), Path(synthetic.__file__),
             Path(inspect.getfile(JointPreprocessor)),
             Path(__file__).with_name("joint_preprocessing_real_meta_bank.py"),
             Path(__file__).with_name("joint_preprocessing_real_meta_report.py")]
    result = dict(format_version=1, steps=args.steps, evaluate_every=args.evaluate_every,
                save_every=args.save_every, lr=args.lr, tasks_per_update=4,
                weight_decay=1e-4, gradient_clip=1., optimizer_resets="step zero only",
                arms=ARMS, continuation_seeds=SEEDS, alpha=.5, temperature=.9,
                synthetic_validation_tasks=args.synthetic_validation_tasks,
                synthetic_test_tasks=args.synthetic_test_tasks,
                synthetic_probe_tasks=args.synthetic_probe_tasks,
                bootstrap_samples=args.bootstrap_samples, generator=GENERATOR,
                lengths=pilot.LENGTHS, fractions=pilot.FRACTIONS,
                code_hashes={p.name: pilot.hash_file(p) for p in paths})
    return json.loads(json.dumps(result))


def fresh_seed(seed, step):
    return (401001 + seed * 100003 + step * 1_000_003) % 2**32


def seed_audit(spec):
    streams = [{fresh_seed(s, i) for i in range(1, spec["steps"]+1)} for s in SEEDS]
    val = {VAL_SEED + 10_000_019*i for i in range(1, 13)}
    test = {TEST_SEED + 10_000_019*i for i in range(1, 13)}
    old = {(base + i*1_000_003) % 2**32 for base, count in ((161001,10000),(201001,10240)) for i in range(1,count+1)}
    old |= {base+10_000_019*i for base in (171001,172001,181001,182001,211001,221001) for i in range(1,13)}
    groups = streams+[val,test]
    if any(g & old for g in groups) or any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError("continuation seed schedules overlap historical/current banks")
    if any(len(g) != spec["steps"] for g in streams):
        raise ValueError("training stream contains duplicate seeds")
    return dict(training_seed_counts=[len(g) for g in streams],
                historical_seeds_checked=len(old), validation_seeds=sorted(val), test_seeds=sorted(test))


def shape_for_step(seed, step):
    shapes = [(n,f) for n in pilot.LENGTHS for f in pilot.FRACTIONS]
    order = np.random.default_rng(401001+seed*100003).permutation(len(shapes))
    return shapes[int(order[(step-1)%len(shapes)])]


def from_synthetic(e):
    return dict(domain="synthetic", family=f"synthetic_{e.task_id}", split_seed=0,
                task_id=e.task_id, source_seed=e.source_seed, n_classes=e.n_classes,
                x_context=e.x_context.cpu(), x_query=e.x_query.cpu(),
                y_context=e.y_context.cpu(), y_query=e.y_query.cpu(),
                numerical_mask=torch.ones(e.x_context.shape[-1], dtype=torch.bool),
                context_missing=torch.zeros(e.x_context.shape[1:], dtype=torch.bool),
                query_missing=torch.zeros(e.x_query.shape[1:], dtype=torch.bool))


def to_synthetic(e):
    return synthetic.SyntheticEpisode(e["task_id"], e["source_seed"], e["x_context"],
        e["x_query"], e["y_context"], e["y_query"], e["n_classes"], "coverage_expanded")


def fresh_batch(seed, step, spec, *, n_rows=None, fraction=None):
    length, f = shape_for_step(seed, step)
    args = argparse.Namespace(**spec["generator"], sequence_length=n_rows or length,
                              context_fraction=fraction or f)
    raw = synthetic.generate_episodes(args, 4, source_seed=fresh_seed(seed,step),
        task_offset=TRAIN_OFFSET+seed*1_000_000+(step-1)*4, device=torch.device("cpu"))
    return [from_synthetic(e) for e in raw]


def prepare(args):
    from scripts import joint_preprocessing_real_meta_bank as bank
    root = args.output_dir
    spec = settings(args)
    source = transfer.model_lock(args.source_dir)
    if source["arms"]["repeated"]["selected_step"] != 5120:
        raise ValueError("expected the prespecified repeated checkpoint at step 5120")
    intent = dict(settings=spec, source_lock=source,
                  candidate_sha256=pilot.hash_file(args.candidate_manifest), seed_audit=seed_audit(spec))
    intent["fingerprint"] = previous.digest(intent)
    path = root / "preparation.json"
    if path.exists():
        if previous.read(path) != intent:
            raise ValueError("preparation settings/source/candidates changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("use a new empty result root")
        pilot.json_write(path, intent)
    bank.prepare_real_bank(root, args.candidate_manifest, args.cache_dir)
    real_manifest = previous.read(root / "real_manifest.json")
    banks = {}
    for panel, count, seed, offset in (("synthetic_validation",args.synthetic_validation_tasks,VAL_SEED,VAL_OFFSET),
        ("synthetic_test",args.synthetic_test_tasks,TEST_SEED,TEST_OFFSET)):
        path = root / "banks" / f"{panel}.pt"
        meta = path.with_suffix(".json")
        if meta.exists():
            info = previous.read(meta)
            if pilot.hash_file(path) != info["sha256"] or info["count"] != count:
                raise ValueError("synthetic bank changed")
        else:
            generated = synthetic.generate_scheduled_episodes(argparse.Namespace(**GENERATOR),count,
                source_seed=seed,task_offset=offset,device=torch.device("cpu"),
                sequence_lengths=pilot.LENGTHS,context_fractions=pilot.FRACTIONS,
                observation_mode="coverage_expanded")
            synthetic.validate_episode_classes(generated,10)
            values = [from_synthetic(e) for e in generated]
            pilot.atomic_save(path,dict(format_version=1,episodes=values))
            info = dict(path=path.relative_to(root).as_posix(), count=count, sha256=pilot.hash_file(path),
                        content_hashes=[previous.episode_hash(e) for e in generated])
            pilot.json_write(meta,info)
        banks[panel] = info
    path = root / "banks/synthetic_probe.pt"
    meta = path.with_suffix(".json")
    if not meta.exists():
        # These are actual first fresh-stream tasks, not independent test draws.
        families = load_panel(root,dict(banks=real_manifest["banks"]),"real_train")
        values = [e for step in range(1,args.synthetic_probe_tasks//4+1)
                  for e in training_batch("synthetic",0,step,families,spec)]
        pilot.atomic_save(path,dict(format_version=1,episodes=values))
        pilot.json_write(meta,dict(path=path.relative_to(root).as_posix(),count=len(values),sha256=pilot.hash_file(path),
            content_hashes=[previous.episode_hash(to_synthetic(e)) for e in values]))
    info = previous.read(meta)
    if info["count"] != args.synthetic_probe_tasks or pilot.hash_file(path) != info["sha256"]:
        raise ValueError("synthetic probe changed")
    banks["synthetic_probe"] = info
    seen = set()
    for info in banks.values():
        if seen & set(info["content_hashes"]) or len(set(info["content_hashes"])) != info["count"]:
            raise ValueError("duplicate synthetic bank content")
        seen.update(info["content_hashes"])
    banks.update(real_manifest["banks"])
    manifest = dict(format_version=1, settings=spec, source_lock=source,
        preparation_sha256=pilot.hash_file(root/"preparation.json"),
        real_manifest_sha256=pilot.hash_file(root/"real_manifest.json"), banks=banks)
    manifest["fingerprint"] = previous.digest(manifest)
    path = root / "manifest.json"
    if path.exists() and previous.read(path) != manifest:
        raise ValueError("frozen experiment manifest differs")
    pilot.json_write(path,manifest)
    print(f"Prepared continuation banks and source lock: {root}",flush=True)


def load_panel(root, manifest, panel):
    info = manifest["banks"][panel]
    path = root/info["path"]
    if pilot.hash_file(path) != info["sha256"]:
        raise ValueError(f"{panel} bank hash changed")
    saved = torch.load(path,map_location="cpu",weights_only=True)
    values = saved["families"] if panel == "real_train" else saved["episodes"]
    if len(values) != info["count"]:
        raise ValueError(f"{panel} bank count changed")
    return values


def setup(args):
    manifest = previous.read(args.output_dir/"manifest.json")
    if manifest["settings"] != settings(args):
        raise ValueError("settings/code differ from prepared experiment")
    check = {k:v for k,v in manifest.items() if k != "fingerprint"}
    if previous.digest(check) != manifest["fingerprint"]:
        raise ValueError("manifest fingerprint changed")
    if transfer.model_lock(args.source_dir) != manifest["source_lock"]:
        raise ValueError("synthetic source weights/choices changed")
    device = torch.device(args.device)
    backbone,_,backbone_hash = pilot.load_frozen(args,device)
    if backbone_hash != manifest["source_lock"]["backbone_sha256"] or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("backbone must match frozen source")
    backbone.train()
    for module in backbone.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p = 0.
    initial,_ = previous.selected_model(args.source_dir,"repeated",manifest["source_lock"]["source_fingerprint"],device)
    return backbone,initial,manifest,manifest["fingerprint"],device


def run_fingerprint(fp,arm,seed):
    return previous.digest(dict(experiment=fp,arm=arm,continuation_seed=seed))


def rng_state():
    algorithm,values,position,has_gauss,cached = np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(algorithm=algorithm,values=values.tolist(),
        position=position,has_gauss=has_gauss,cached=cached))


def restore_rng(saved):
    random.setstate(saved["python"])
    n = saved["numpy"]
    np.random.set_state((n["algorithm"],np.asarray(n["values"],dtype=np.uint32),
                         n["position"],n["has_gauss"],n["cached"]))


def numeric_context(e,device):
    _,_,positions,numerical_keep,keep = real._prepared_views(e,8)
    xc = e["x_context"][...,keep][...,positions].to(device)
    xq = e["x_query"][...,keep][...,positions].to(device)
    yc = e["y_context"].to(device)
    mc = e["context_missing"][None,:,numerical_keep].to(device)
    mq = e["query_missing"][None,:,numerical_keep].to(device)
    return xc,xq,yc,mc,mq


def surrogate_logits(backbone,model,e):
    """Raw single-view logits; frozen weights retain input autograd in train mode."""
    device = next(backbone.parameters()).device
    if e.get("domain") == "synthetic":
        episode = pilot.on_device(pilot.filtered_episode(to_synthetic(e)),device)
        return pilot.forward_views(backbone,model,episode,view_index=diagnostic.training_view(episode))
    generator,members,positions,numerical_keep,keep = real._prepared_views(e,8)
    views = [(0 if method == "none" else 1,xs,ys,index,features,classes)
             for method,(xs,ys) in members.items()
             for index,(features,classes) in enumerate(generator.ensemble_configs_[method])]
    view_seed = e.get("task_id",e.get("source_seed",e["split_seed"]))
    slot,xs,ys,index,features,classes = views[random.Random(view_seed).randrange(len(views))]
    xc,xq,yc,mc,mq = numeric_context(e,device)
    if model is None:
        from tabicl._hyperspline.statistics import summarize_context
        with torch.no_grad():
            stats = summarize_context(xc,mc,yc)
        c = ((xc-stats.location[:,None])/stats.scale[:,None]).masked_fill(mc,0)
        q = ((xq-stats.location[:,None])/stats.scale[:,None]).masked_fill(mq,0)
    else:
        p = model.generate(xc,yc,mc)
        c,q = model.apply(xc,p,slot,mc),model.apply(xq,p,slot,mq)
    transformed = torch.from_numpy(xs[index:index+1]).to(device=device,dtype=torch.float32).clone()
    inverse = {int(original):position for position,original in enumerate(features)}
    columns = torch.tensor([inverse[int(original)] for original in positions],device=device)
    transformed = transformed.index_copy(-1,columns,torch.cat((c,q),1))
    backbone.clear_cache()
    labels = torch.from_numpy(ys[index:index+1]).to(device=device,dtype=torch.float32)
    raw = backbone(transformed,labels)
    return raw[...,:e["n_classes"]][...,torch.as_tensor(classes,device=device,dtype=torch.long)]


def surrogate_nll(backbone,model,e):
    logits = surrogate_logits(backbone,model,e)
    return F.cross_entropy(logits.flatten(0,1),e["y_query"].flatten().to(logits.device))


def panel_logits(backbone,model,e,estimators=8):
    if e.get("domain") == "synthetic" and isinstance(model,JointPreprocessor):
        if estimators != 8:
            raise ValueError("learned ensemble has eight views")
        source = pilot.on_device(to_synthetic(e),next(backbone.parameters()).device)
        with torch.no_grad(),pilot.frozen_inference(backbone):
            return pilot.forward_views(backbone,model,source),8
    return real.episode_logits(backbone,model,e,estimators)


def transform_diagnostics(model,e,device):
    xc,xq,yc,mc,mq = numeric_context(e,device)
    with torch.no_grad():
        p = model.generate(xc,yc,mc)
        valid = ~torch.cat((mc,mq),1)
        x = torch.cat((xc,xq),1)
        z = ((x-p.location[:,None])/p.scale[:,None]).masked_fill(~valid,0)
        result = {}
        for slot in (0,1):
            output = model.apply(x,p,slot,~valid)
            a = p.log_scale[:,slot,None].exp()*z+p.shift[:,slot,None]
            hidden = torch.tanh(a[...,None]*p.neural_first_weight[:,slot,None]+p.neural_first_bias[:,slot,None])
            residual = (hidden*p.neural_last_weight[:,slot,None]).sum(-1)+p.neural_last_bias[:,slot,None]
            result[f"slot{slot}_correction_rms"] = float((output-z)[valid].square().mean().sqrt())
            result[f"slot{slot}_neural_saturation"] = float((torch.tanh(residual)[valid].abs()>.95).float().mean())
            result[f"slot{slot}_spline_gate"] = float(p.spline_gate[:,slot].mean())
            result[f"slot{slot}_neural_gate"] = float(p.neural_gate[:,slot].mean())
            result[f"slot{slot}_mixing_norm"] = float(p.mixing[:,slot].norm(dim=(-2,-1)).mean())
    return result


def training_batch(arm,seed,step,families,spec):
    from scripts import joint_preprocessing_real_meta_bank as bank
    length,fraction = shape_for_step(seed,step)
    # Cap the entire batch to its sampled real source's available rows. Synthetic
    # controls use the same cap schedule so size changes are not a data-source effect.
    def family_at(draw):
        epoch,index = divmod(draw,len(families))
        order = np.random.default_rng(431001+seed*100003+epoch).permutation(len(families))
        return families[int(order[index])]
    assigned = [family_at((step-1)*4+i) for i in range(4)]
    sizes = [min(length,len(f["labels"])) for f in assigned]
    n_rows = min(sizes)
    batch = fresh_batch(seed,step,spec,n_rows=n_rows,fraction=fraction) if arm != "real" else [None]*4
    positions = list(range(4)) if arm == "real" else ([0,1] if step%2 else [2,3]) if arm == "mixed" else []
    for i in positions:
        sample_seed = int(np.random.SeedSequence([441001,seed,step,i]).generate_state(1)[0])
        e = bank.sample_real_episode(assigned[i],n_rows,fraction,sample_seed)
        e.update(domain="real",source_seed=sample_seed,task_id=15_000_000_000+seed*1_000_000+(step-1)*4+i)
        batch[i] = e
    return batch


def references(backbone,initial,episodes,root,panel,fp):
    path = root/"references"/f"{panel}.pt"
    ids = [(e["family"],e["split_seed"]) for e in episodes]
    cache = torch.load(path,map_location="cpu",weights_only=True) if path.exists() else dict(fingerprint=fp,ids=ids,values={})
    if cache["fingerprint"] != fp or cache["ids"] != ids:
        raise ValueError("reference cache changed")
    for i,e in enumerate(episodes):
        if i in cache["values"]:
            continue
        ordinary,v8 = panel_logits(backbone,"ordinary",e,8)
        control,v16 = panel_logits(backbone,"ordinary",e,16)
        learned,vl = panel_logits(backbone,initial,e,8)
        if (v8,v16,vl) != (8,16,8):
            raise ValueError("baseline view counts differ")
        with torch.no_grad():
            single = float(surrogate_nll(backbone,initial,e))
        cache["values"][i] = dict(ordinary=ordinary.cpu(),control=control.cpu(),
            initial_single_nll=single,initial_learned_nll=previous.score(learned,e["y_query"].to(learned.device))["nll"],
            initial_blend_nll=previous.score(.5*(ordinary+learned),e["y_query"].to(learned.device))["nll"])
        if i%20 == 19:
            pilot.atomic_save(path,cache)
            print(f"Cached {panel} references {i+1}/{len(episodes)}",flush=True)
    pilot.atomic_save(path,cache)
    return cache["values"]


def evaluate(backbone,model,banks,refs,root,step,device):
    model.eval()
    summaries = {}
    for panel,episodes in banks.items():
        start = time.perf_counter()
        rows = []
        for i,e in enumerate(episodes):
            learned,views = panel_logits(backbone,model,e,8)
            ref = refs[panel][i]
            ordinary,control = ref["ordinary"].to(learned.device),ref["control"].to(learned.device)
            labels = e["y_query"].to(learned.device)
            scores = {name:previous.score(logits,labels) for name,logits in
                (("learned",learned),("blend",.5*(ordinary+learned)),("ordinary8",ordinary),("ordinary16",control))}
            with torch.no_grad():
                single = float(surrogate_nll(backbone,model,e))
            diagnostic_keys = [f"slot{slot}_{key}" for slot in (0,1) for key in
                ("correction_rms","neural_saturation","spline_gate","neural_gate","mixing_norm")]
            diagnostics = (transform_diagnostics(model,e,device) if panel == "real_probe"
                           else dict.fromkeys(diagnostic_keys))
            row = dict(step=step,panel=panel,family=e["family"],split_seed=e["split_seed"],
                single_nll=single,initial_single_nll=ref["initial_single_nll"],
                initial_learned_nll=ref["initial_learned_nll"],initial_blend_nll=ref["initial_blend_nll"],
                learned_views=views,blend_views=views+8,
                **{f"{name}_{k}":v for name,s in scores.items() for k,v in s.items()},**diagnostics)
            if not all(math.isfinite(float(v)) for k,v in row.items() if isinstance(v,(int,float))):
                raise FloatingPointError("nonfinite diagnostic score")
            rows.append(row)
            pilot.csv_append(root/"evaluation_episodes.csv",row)
        families = list(dict.fromkeys(r["family"] for r in rows))
        grouped = {key:np.array([np.mean([r[key] for r in rows if r["family"]==f]) for f in families])
                   for key in ("blend_nll","learned_nll","ordinary8_nll","ordinary16_nll","single_nll","initial_single_nll")}
        ratios = (grouped["blend_nll"]+1e-4)/(grouped["ordinary16_nll"]+1e-4)
        delta = grouped["blend_nll"]-grouped["ordinary16_nll"]
        summary = dict(step=step,panel=panel,families=len(families),episodes=len(rows),
            seconds=time.perf_counter()-start,validation_score=float(np.log(ratios).mean()),
            geometric_gain=1-float(np.exp(np.log(ratios).mean())),median_gain=float(np.median(1-ratios)),
            mean_nll=float(grouped["blend_nll"].mean()),mean_single_nll=float(grouped["single_nll"].mean()),
            single_reduction_from_initial=1-float(grouped["single_nll"].mean()/grouped["initial_single_nll"].mean()),
            wins=int((delta < -1e-6).sum()),losses=int((delta > 1e-6).sum()),
            harms_over_1pct=int((ratios>1.01).sum()),harms_over_5pct=int((ratios>1.05).sum()))
        summaries[panel] = summary
        pilot.csv_append(root/"evaluation.csv",summary)
        print(f"step={step} {panel} blend_gain={100*summary['geometric_gain']:.3f}% W/L={summary['wins']}/{summary['losses']} "
              f"single_change={100*summary['single_reduction_from_initial']:.3f}%",flush=True)
    return summaries


def train(args):
    root = args.output_dir
    if (root/"lock.json").exists() or (root/"test_report/started.json").exists():
        raise ValueError("all choices are locked; further training forbidden")
    backbone,initial,manifest,fp,device = setup(args)
    if args.arm not in ARMS or args.continuation_seed not in SEEDS:
        raise ValueError("unknown continuation arm/seed")
    folder = run_dir(root,args.arm,args.continuation_seed)
    runfp = run_fingerprint(fp,args.arm,args.continuation_seed)
    if (folder/"complete.json").exists():
        if previous.read(folder/"complete.json")["fingerprint"] != runfp:
            raise ValueError("completed run fingerprint changed")
        return
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError("run exists; pass --resume")
    torch.manual_seed(451001+args.continuation_seed)
    np.random.seed(451001+args.continuation_seed)
    random.seed(451001+args.continuation_seed)
    model = copy.deepcopy(initial)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    families = load_panel(root,manifest,"real_train")
    panels = ("real_probe","real_validation","synthetic_probe","synthetic_validation")
    banks = {p:load_panel(root,manifest,p) for p in panels}
    refs = {p:references(backbone,initial,e,folder,p,runfp) for p,e in banks.items()}
    state_path = folder/"state.pt"
    step = best_step = clipped = 0
    if state_path.exists():
        state = torch.load(state_path,map_location="cpu",weights_only=True)
        if state["fingerprint"] != runfp:
            raise ValueError("resume fingerprint changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        capacity.optimizer_to(optimizer,device)
        step,best_step,clipped = state["step"],state["best_step"],state["clipped"]
        best_score,best_model,last = state["best_score"],state["best_model"],state["last"]
        torch.set_rng_state(state["rng"])
        restore_rng(state["other_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for name in LOGS:
            capacity.trim_csv(folder/name,step)
    else:
        last = evaluate(backbone,model,banks,refs,folder,0,device)
        best_score,best_model = last["real_validation"]["validation_score"],pilot.state_cpu(model)
    pilot.json_write(folder/"config.json",dict(experiment_fingerprint=fp,fingerprint=runfp,arm=args.arm,
        continuation_seed=args.continuation_seed,source_step=5120,settings=manifest["settings"],
        source_checkpoint_sha256=manifest["source_lock"]["arms"]["repeated"]["checkpoint_sha256"],
        probe_note="Real probe uses seen source families with fixed rows; not unseen-family evidence. Synthetic probe comprises the first seed-zero fresh-stream tasks.",
        revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()))

    def save():
        pilot.atomic_save(state_path,dict(fingerprint=runfp,step=step,best_step=best_step,clipped=clipped,
            best_score=best_score,best_model=best_model,last=last,model=pilot.state_cpu(model),
            optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
            other_rng=rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []))

    save()
    stop = args.steps if args.max_steps is None else min(args.steps,args.max_steps)
    if stop < step:
        raise ValueError("requested stop precedes durable state")
    for step in range(step+1,stop+1):
        started = time.perf_counter()
        batch = training_batch(args.arm,args.continuation_seed,step,families,manifest["settings"])
        model.train()
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position,e in enumerate(batch):
            loss = surrogate_nll(backbone,model,e)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training CE")
            (loss/4).backward()
            losses.append(float(loss.detach()))
            pilot.csv_append(folder/"presentations.csv",dict(step=step,position=position,domain=e.get("domain","real"),
                family=e["family"],source_seed=e["source_seed"],task_id=e["task_id"],
                n_context=e["x_context"].shape[1],n_query=e["x_query"].shape[1],
                n_features=e["x_context"].shape[-1],n_classes=e["n_classes"]))
        norms = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(norm) or not all(math.isfinite(x) for x in norms.values()):
            raise FloatingPointError("nonfinite gradient")
        clipped += int(norm>1)
        optimizer.step()
        pilot.csv_append(folder/"training.csv",dict(step=step,seconds=time.perf_counter()-started,
            objective=float(np.mean(losses)),real_episodes=sum(e.get("domain")=="real" for e in batch),
            synthetic_episodes=sum(e.get("domain")=="synthetic" for e in batch),lr=args.lr,
            preclip_gradient_norm=norm,clip_factor=min(1.,1/(norm+1e-6)),clipped_fraction=clipped/step,
            **norms))
        evaluated = step%args.evaluate_every==0 or step==args.steps
        if evaluated:
            last = evaluate(backbone,model,banks,refs,folder,step,device)
            score = last["real_validation"]["validation_score"]
            if score < best_score:
                best_score,best_step,best_model = score,step,pilot.state_cpu(model)
        if evaluated or step%args.save_every==0 or step==stop:
            save()
            print(f"arm={args.arm} seed={args.continuation_seed} step={step}/{args.steps} CE={np.mean(losses):.6f} "
                  f"gradient={norm:.4f} clipped={clipped/step:.3f} best={best_score:.6f}@{best_step}",flush=True)
    if step != args.steps:
        return
    for name,values,saved_step in (("selected",best_model,best_step),("final",pilot.state_cpu(model),step)):
        pilot.atomic_save(folder/f"{name}.pt",dict(fingerprint=runfp,model=values,step=saved_step))
    pilot.json_write(folder/"complete.json",dict(experiment_fingerprint=fp,fingerprint=runfp,
        arm=args.arm,continuation_seed=args.continuation_seed,steps=step,selected_step=best_step,
        selected_score=best_score,selected_sha256=pilot.hash_file(folder/"selected.pt"),
        final_sha256=pilot.hash_file(folder/"final.pt"),final=last,
        recorded_training_seconds=sum(float(r["seconds"]) for r in capacity.read_rows(folder/"training.csv")),
        recorded_evaluation_seconds=sum(float(r["seconds"]) for r in capacity.read_rows(folder/"evaluation.csv"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","train","lock","test","pipeline"))
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--candidate-manifest",type=Path,default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache-dir",type=Path,default=Path("results/pmlb_cache"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--arm",choices=ARMS,default="synthetic")
    parser.add_argument("--continuation-seed",type=int,choices=SEEDS,default=0)
    parser.add_argument("--steps",type=int,default=4096)
    parser.add_argument("--evaluate-every",type=int,default=512)
    parser.add_argument("--save-every",type=int,default=50)
    parser.add_argument("--lr",type=float,default=.0003)
    parser.add_argument("--synthetic-validation-tasks",type=int,default=128)
    parser.add_argument("--synthetic-test-tasks",type=int,default=1024)
    parser.add_argument("--synthetic-probe-tasks",type=int,default=48)
    parser.add_argument("--bootstrap-samples",type=int,default=10000)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-steps",type=int)
    args = parser.parse_args()
    if args.command in ("prepare","train"):
        globals()[args.command](args)
        return
    from scripts import joint_preprocessing_real_meta_report as report
    if args.command == "lock":
        report.lock(args)
    elif args.command == "test":
        report.test(args)
    else:
        if not (args.output_dir/"lock.json").exists():
            for arm in ARMS:
                for seed in SEEDS:
                    args.arm,args.continuation_seed,args.resume = arm,seed,True
                    train(args)
                    if args.max_steps is not None and args.max_steps<args.steps:
                        return
        report.lock(args)
        report.test(args)


if __name__ == "__main__":
    main()
