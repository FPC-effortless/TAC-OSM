"""Frozen ME111 five-seed typed-role runner; import never loads data or fits.

The native state graph and full ME110 weights are inherited; scales change.
Only explicit fit mode creates optimizers or performs updates. FIT monitors are
ancestor-exposed internal diagnostics, and never select checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import me109_residual_training as old
from scripts import me110_sampling as sampling
from scripts.me111_objective import losses
from scripts.validate_research_parity import check_manifest
from tac_osm.typed_role_learning import uncertainty_masks
from tac_osm.conditional_belief_learning import TypedRoleAdamW, routed_gradients
from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld
from scripts import me110_typed_training as previous
from tac_osm.parameter_budget import (budgeted_optimizer, checked_optimizer_step,
    save_training_checkpoint, validate_runtime_evidence, live_inventory)

data, me101 = old.data, old.me101
sha, write, frozen = old.sha, old.write, old.frozen
fit_data, tensor_batch = old.fit_data, old.tensor_batch
blocks, parameter_groups, ROLE_BLOCKS = old.blocks, old.parameter_groups, old.ROLE_BLOCKS
OUT = ROOT / 'research/replay-bundles/ME111'
DOC = ROOT / 'research/mathematical-engineering/me111'
REG = DOC / 'FAIR_COMPARISON.json'
PARENT, PARENT_DOC, PARENT_SEEDS = previous.OUT, previous.DOC, previous.SEEDS
SEEDS = tuple(range(111001, 111006))
ARCH = 'conditional_belief'
ARMS = (ARCH,)
STEPS, BATCH, CONTEXTS = 1536, 16, (8, 64, 256)
CHECKPOINTS = (512, 1024, 1536)
SOURCE_FILES = list(dict.fromkeys(previous.SOURCE_FILES + [
    'scripts/me111_conditional_training.py', 'scripts/me110_sampling.py',
    'scripts/me111_objective.py', 'src/tac_osm/conditional_belief_learning.py',
    'src/tac_osm/variants/conditional_belief_sensor_world.py', 'scripts/me111_tools.py',
    'scripts/me111_graph_reference.py',
    'src/tac_osm/inward_role_guard.py', 'scripts/me110_tools.py',
    'src/tac_osm/research_anticheat.py','src/tac_osm/research_observatory.py']))
SOURCES = SOURCE_FILES


def parent_artifacts(seed):
    parent_seed=seed-1000
    path=PARENT/previous.ARCH/f'seed-{parent_seed}'/'final.pt'
    row=next(r for r in json.loads((PARENT/'FIT_INDEX.json').read_text())['models'] if r['seed']==parent_seed)
    if sha(path)!=row['checkpoint_sha256']:raise RuntimeError('ME110 parent identity changed')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['seed']!=parent_seed or cp['steps']!=4096:raise RuntimeError('wrong ME110 parent')
    return path,cp,dict(checkpoint_sha256=sha(path),fit_index_sha256=sha(PARENT/'FIT_INDEX.json'))


def initialize(d,seed):
    torch.manual_seed(seed)
    model=ConditionalBeliefSensorWorld(torch.tensor(old.gap.floor_rates(d)['rates'],dtype=torch.float32))
    with torch.no_grad():
        model.target_mean.copy_(torch.from_numpy(d['mean'][:6]));model.target_scale.copy_(torch.from_numpy(d['std'][:6]))
    return model


def load_parent(model,cp):
    model.load_state_dict(cp['model'],strict=True)
    return []


def initial_preservation(model,cp,x):
    ref=old.initialize(fit_data(),111900);ref.load_state_dict(cp['model'],strict=True)
    with torch.no_grad():
        before,bs=ref(*x[:6]);after,fs=model(*x[:6])
        assert torch.equal(ref.raw_quantiles(before)[...,1],model.raw_quantiles(after)[...,1])
        assert all(torch.equal(v,getattr(fs,k)) for k,v in vars(bs).items() if isinstance(v,torch.Tensor))
    return dict(raw_median_bit_exact=True,native_tensor_state_bit_exact=True,
                scale_contract_intentionally_changed=True,old_mean_side_scale=float((before.left_scale+before.right_scale).mean()),
                new_mean_side_scale=float((after.left_scale+after.right_scale).mean()))


def raw_scale(d):
    return torch.tensor(np.nanstd(d['raw'][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)


def source_hashes():
    return {p: sha(ROOT / p) for p in SOURCE_FILES}


def registration_check():
    registration = json.loads(REG.read_text())
    check_manifest(registration, 'preflight')
    if registration.get('study_type') != 'single_model_training' or [
        (m.get('architecture'), m.get('seed')) for m in registration.get('models', [])
    ] != [(ARCH, seed) for seed in SEEDS]:
        raise RuntimeError('ME111 requires exactly the five fixed conditional_belief seed arms')
    return registration


def inventory(model):
    counts, rows = live_inventory(model)
    if counts != dict(trainable_parameters=104663, deployed_parameters=104663) or len(rows) != 45 or not all(r['requires_grad'] for r in rows):
        raise RuntimeError('ME111 requires actual104663 trainable/deployed parameters in45 tensors')
    return counts, rows


def profile(model, x, rawscale):
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                               with_flops=True, record_shapes=True) as prof:
        forecast, _, trace = model(*x[:6], return_trace=True)
        point, auxiliary, parts = losses(model, forecast, trace, x, rawscale)
        routed_gradients(model, point, auxiliary, parts['reconstruction_nll'])
    events = prof.key_averages(group_by_input_shape=True)
    ops = sorted((e.key, e.count, str(e.input_shapes)) for e in events if e.key.startswith('aten::'))
    return dict(dense_flops=sum(e.flops for e in events), aten_signature=old.digest(ops),
                aten_calls=sum(e.count for e in events if e.key.startswith('aten::')),
                flops_scope='partial actual forward and routed two-VJP; optimizer and unsupported operations excluded')


def auxiliary_preservation(model, x):
    """Actual-source functional uncertainty-only displacement; no optimizer step."""
    masks = uncertainty_masks(model)
    candidate = {name: torch.where(mask, p.detach() - .001*p.grad, p.detach())
                 for (name,p),mask in zip(model.named_parameters(),masks)}
    with torch.no_grad():
        before, _ = model(*x[:6])
        after, _ = torch.func.functional_call(model, candidate, tuple(x[:6]))
        median_before = model.raw_quantiles(before)[..., 1]
        median_after = model.raw_quantiles(after)[..., 1]
    if not torch.equal(median_before, median_after):
        raise RuntimeError('uncertainty-only functional candidate changed actual-source raw median')
    changed = sum(int(torch.count_nonzero(candidate[n] != p)) for n,p in model.named_parameters())
    if not changed: raise RuntimeError('auxiliary preservation probe has no changed coordinates')
    return dict(raw_median_bit_exact=True, changed_uncertainty_coordinates=changed,
                candidate_only=True, no_optimizer_step=True, step_size=.001)


def prepare():
    names = ('FIT_PLANS.npz', 'PREFIT_PROFILES.json', 'PREFIT_DESIGN_RECEIPT.json')
    if OUT.exists() or any((DOC / name).exists() for name in names):
        raise RuntimeError('ME111 artifacts exist; no overwrite or retry')
    registration_check()
    d = fit_data(); monitor = sampling.make_monitor(d)
    plans = {str(seed): sampling.make_plan(d, seed, monitor, nsteps=STEPS) for seed in SEEDS}
    plan_checks = {key: sampling.validate_plan(value,d,monitor,nsteps=STEPS) for key,value in plans.items()}
    rawscale = raw_scale(d); parent_hashes = {}; audits = []; profiles = {}; tool_audits=[]
    for seed in SEEDS:
        ppath, cp, hashes = parent_artifacts(seed); parent_hashes[str(seed)] = hashes
        model = initialize(d, seed); dropped = load_parent(model, cp)
        counts, rows = inventory(model)
        x = tensor_batch(d, plans[str(seed)][0], 256, np.random.default_rng(seed + 10000))
        preservation = initial_preservation(model, cp, x)
        model.zero_grad(set_to_none=True)
        forecast, _, trace = model(*x[:6], return_trace=True)
        point, auxiliary, parts = losses(model, forecast, trace, x, rawscale)
        if not torch.isfinite(point) or not torch.isfinite(auxiliary): raise RuntimeError('nonfinite prefit objective')
        pgrads, routing = routed_gradients(model, point, auxiliary, parts['reconstruction_nll'])
        _, block_indices = blocks(model)
        audits.append(dict(seed=seed,parent_seed=seed-1000,parent_checkpoint_sha256=sha(ppath),
            dropped_parent_tensors=dropped,initial_forecast_preservation=preservation,
            auxiliary_only_actual_source_preservation=auxiliary_preservation(model,x),
            live_inventory=rows, **counts, parameter_tensors=len(rows), routing=routing,
            parts={k:float(v.detach()) for k,v in parts.items()},
            point_gradient=old.gradient_norms(pgrads,block_indices),
            routed_gradient=old.gradient_norms([p.grad for p in model.parameters()],block_indices),
            no_optimizer_step=True))
        from scripts.me111_tools import examine
        tool_audits.append(examine(d,seed,plans[str(seed)][0],rawscale))
        if seed == SEEDS[0]:
            for length in CONTEXTS:
                sample=tensor_batch(d,plans[str(seed)][0],length,np.random.default_rng(seed+10000))
                profiles[str(length)]=profile(model,sample,rawscale)
    np.savez_compressed(DOC / 'FIT_PLANS.npz', monitor=monitor, **plans)
    write(DOC/'TOOL_PREFIT.json',dict(scope='actual FIT-only scoped probes; full integrity INCONCLUSIVE',audits=tool_audits,no_optimizer_steps=True))
    write(DOC / 'PREFIT_PROFILES.json',dict(profiles=profiles,no_optimizer_updates=True))
    write(DOC / 'PREFIT_DESIGN_RECEIPT.json',dict(parent_hashes=parent_hashes,
        plan_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in plans.items()},
        monitor_sha256=hashlib.sha256(monitor.tobytes()).hexdigest(),monitor_scope=sampling.MONITOR_SCOPE,
        plan_checks=plan_checks,audits=audits,profiles=profiles,trainable_parameters=104663,
        deployed_parameters=104663,parameter_tensors=45,no_optimizer_updates=True))


def verify_prefit():
    registration = registration_check()
    for key,name in (('fit_plans_sha256','FIT_PLANS.npz'),
                     ('prefit_design_sha256','PREFIT_DESIGN_RECEIPT.json'),
                     ('prefit_profiles_sha256','PREFIT_PROFILES.json'),('tool_prefit_sha256','TOOL_PREFIT.json')):
        if registration.get(key) != sha(DOC/name): raise RuntimeError(f'ME111 registered {key} changed')
    if registration.get('source_files') != source_hashes(): raise RuntimeError('ME111 source closure changed')
    return registration


def frozen_inputs():
    return [REG,DOC/'TOOL_PREFIT.json',DOC/'FIT_PLANS.npz',DOC/'PREFIT_DESIGN_RECEIPT.json',DOC/'PREFIT_PROFILES.json',
            PARENT/'FIT_INDEX.json',PARENT/'RESULTS.json',PARENT_DOC/'FAIR_COMPARISON.json',
            *[ROOT/p for p in SOURCE_FILES],
            *[PARENT/previous.ARCH/f'seed-{s}'/name for s in PARENT_SEEDS
              for name in ('final.pt','FIT_RECEIPT.json','live-budget.json')]]


def monitor_model(model,d,monitor,folder,step):
    """Natural-input context256, ordered64 gap then256 disjoint general."""
    started_cpu,started_wall=time.process_time(),time.monotonic()
    was_training=model.training;model.eval();arrays={'origins':monitor};results={}
    with torch.no_grad():
        for name,origins in (('gap',monitor[:64]),('general',monitor[64:])):
            qs=[];ys=[];valid=[];nll_sum=0.;nll_count=0
            for i in range(0,len(origins),32):
                x=[torch.from_numpy(v) for v in data.batch(d,origins[i:i+32],256)]
                forecast,_=model(*x[:6]);qs.append(model.raw_quantiles(forecast).numpy())
                ys.append(x[6].numpy());valid.append(x[7].numpy())
                count=int(x[7].sum());nll_sum+=float(model.nll(forecast,x[6],x[7]))*count;nll_count+=count
            q,y,observed=map(np.concatenate,(qs,ys,valid))
            metrics=me101.metrics(q,y,observed);metrics['raw_density_nll']=nll_sum/nll_count
            results[name]=dict(count=len(origins),**metrics)
            arrays.update({name+'_quantiles':q,name+'_truth':y,name+'_observed':observed})
    model.train(was_training)
    path=folder/f'monitor-{step}.npz';np.savez_compressed(path,**arrays)
    row=dict(step=step,context=256,scope=sampling.MONITOR_SCOPE,cohorts=results,
             arrays_sha256=sha(path),cpu_seconds=time.process_time()-started_cpu,
             wall_seconds=time.monotonic()-started_wall,no_checkpoint_selection=True)
    write(folder/f'MONITOR-{step}.json',row)
    return row


def fit_one(seed):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if seed not in SEEDS: raise RuntimeError('unregistered ME111 seed')
    verify_prefit();frozen(frozen_inputs())
    d=fit_data();ppath,parent,parent_hashes=parent_artifacts(seed)
    design=json.loads((DOC/'PREFIT_DESIGN_RECEIPT.json').read_text())
    if parent_hashes!=design['parent_hashes'][str(seed)]:raise RuntimeError('parent changed since prefit')
    with np.load(DOC/'FIT_PLANS.npz') as archive:
        plan=archive[str(seed)];monitor=archive['monitor']
    if not np.array_equal(monitor,sampling.make_monitor(d)):raise RuntimeError('fixed monitor changed')
    sampling.validate_plan(plan,d,monitor,nsteps=STEPS)
    if hashlib.sha256(plan.tobytes()).hexdigest()!=design['plan_sha256'][str(seed)]:raise RuntimeError('plan identity changed')
    model=initialize(d,seed);dropped=load_parent(model,parent);inventory(model)
    named,block_indices=blocks(model);folder=OUT/ARCH/f'seed-{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    optimizer=budgeted_optimizer(model,TypedRoleAdamW,parameters=parameter_groups(model),
        blocks=list(block_indices.values()),lr=3e-5,weight_decay=.001,betas=(.9,.999),eps=1e-8,
        budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture=ARCH)
    if optimizer.state:raise RuntimeError('fresh optimizer has inherited moments')
    random.seed(seed);rng=np.random.default_rng(seed+10000);rawscale=raw_scale(d)
    input_hash=hashlib.sha256();history=[];first_grad=None;guard_counts={};checkpoint_hashes={}
    started_cpu,started_wall=time.process_time(),time.monotonic();fit_cpu=0.;fit_wall=0.
    monitors=[monitor_model(model,d,monitor,folder,0)]
    for step,origins in enumerate(plan,1):
        step_cpu,step_wall=time.process_time(),time.monotonic()
        x=tensor_batch(d,origins,CONTEXTS[(step-1)%3],rng)
        for v in x[:6]:input_hash.update(v.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        forecast,_,trace=model(*x[:6],return_trace=True)
        point,auxiliary,parts=losses(model,forecast,trace,x,rawscale)
        if not torch.isfinite(point) or not torch.isfinite(auxiliary):raise RuntimeError('nonfinite objective; no retry')
        point_grads,routing=routed_gradients(model,point,auxiliary,parts['reconstruction_nll'])
        if step==1:
            first_grad=dict(parameters={n:dict(norm=float(p.grad.norm()),nonzero=bool(torch.count_nonzero(p.grad))) for n,p in named},
                            point=old.gradient_norms(point_grads,block_indices),routing=routing)
        optimizer.set_point_gradients(point_grads)
        guard_start=time.monotonic();checked_optimizer_step(optimizer)
        guard_counts['guard_seconds']=guard_counts.get('guard_seconds',0.)+time.monotonic()-guard_start
        receipt=optimizer.last_guard_receipt
        if receipt is None or len(receipt)!=13:raise RuntimeError('missing13-block guard receipt')
        for block,record in zip(ROLE_BLOCKS,receipt):
            upper=float(record['certified_upper_dot'])
            if not np.isfinite(upper) or upper>0:raise RuntimeError(f'point guard failed: {block}')
            for key,value in (('fallbacks',int(record['fallback'])),('projections',int(record['alpha']>0)),('checks',1)):
                counter=block+'.'+key;guard_counts[counter]=guard_counts.get(counter,0)+value
            counter=block+'.max_certified_upper_dot';guard_counts[counter]=max(guard_counts.get(counter,float('-inf')),upper)
        history.append(dict(step=step,context=CONTEXTS[(step-1)%3],
            parts={k:float(v.detach()) for k,v in parts.items()},routing=routing,
            reconstruction_labels=int(x[9].sum()),future_labels=int(x[7].sum()),
            missing_current_by_channel=(~x[1][:,-1,:6]).sum(0).tolist(),
            guard={name:record for name,record in zip(ROLE_BLOCKS,receipt)}))
        fit_cpu+=time.process_time()-step_cpu;fit_wall+=time.monotonic()-step_wall
        if step%512==0:print(json.dumps(dict(seed=seed,step=step,point=float(point.detach()),cpu_seconds=fit_cpu)),flush=True)
        if step in CHECKPOINTS:
            inventory(model)
            filename='final.pt' if step==STEPS else f'checkpoint-{step}.pt'
            path=folder/filename
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),architecture=ARCH,
                seed=seed,steps=step,plan=plan,monitor=monitor,parent_checkpoint_sha256=sha(ppath),
                dropped_parent_tensors=dropped,history=history,python_rng=random.getstate(),
                augmentation_rng_initial_seed=seed+10000,augmentation_rng_final=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),source_sha256=data.SHA,
                source_files=source_hashes()),path)
            checkpoint_hashes[str(step)]=sha(path)
            monitors.append(monitor_model(model,d,monitor,folder,step))
    if len(optimizer.state)!=45:raise RuntimeError('optimizer must retain all45 tensor states')
    profiles=json.loads((DOC/'PREFIT_PROFILES.json').read_text())['profiles']
    counts={context:sum(CONTEXTS[i%3]==context for i in range(STEPS)) for context in CONTEXTS}
    tokens=BATCH*sum(context*count for context,count in counts.items())
    if tokens!=2686976:raise RuntimeError('registered token count changed')
    write(folder/'TRAINING_HISTORY.json',history)
    row=dict(architecture=ARCH,seed=seed,trainable_parameters=104663,deployed_parameters=104663,
        actual_steps=STEPS,actual_examples=STEPS*BATCH,actual_tokens=tokens,
        checkpoint_sha256=sha(folder/'final.pt'),checkpoint_hashes=checkpoint_hashes,
        parent_checkpoint_sha256=sha(ppath),dropped_parent_tensors=dropped,
        actual_sample_order_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
        augmented_inputs_sha256=input_hash.hexdigest(),optimizer_state_tensors=len(optimizer.state),
        parameter_tensors=45,first_gradient_receipt=first_grad,guard_aggregate=guard_counts,
        optimizer_cpu_seconds=fit_cpu,fit_pipeline_cpu_seconds_excluding_monitors_and_checkpoint_io=fit_cpu,
        monitor_cpu_seconds=sum(m['cpu_seconds'] for m in monitors),
        total_cpu_seconds=time.process_time()-started_cpu,wall_seconds=time.monotonic()-started_wall,
        fit_pipeline_wall_seconds=fit_wall,monitor_scope=sampling.MONITOR_SCOPE,
        monitor_steps=[m['step'] for m in monitors],monitor_metrics=monitors,
        training_history_sha256=sha(folder/'TRAINING_HISTORY.json'),
        peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        training_flops=sum(int(profiles[str(c)]['dense_flops'])*n for c,n in counts.items()),
        flops_scope='partial actual forward/routed two-VJP profiler estimate; optimizer and unsupported operations excluded')
    write(folder/'FIT_RECEIPT.json',row)
    return row


def fit():
    if OUT.exists():raise RuntimeError('ME111 output exists; no overwrite or retry')
    verify_prefit();frozen(frozen_inputs());OUT.mkdir(parents=True,exist_ok=False)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(fit_one,SEEDS))
    write(OUT/'FIT_INDEX.json',dict(models=rows,source_files=source_hashes()))


def budget_audit():
    verify_prefit();index=json.loads((OUT/'FIT_INDEX.json').read_text())
    if len(index['models'])!=5 or [(r.get('architecture'),r.get('seed')) for r in index['models']] != [(ARCH,s) for s in SEEDS]:
        raise RuntimeError('ME111 FIT_INDEX requires all five original models')
    if index['source_files']!=source_hashes():raise RuntimeError('ME111 source closure changed after fitting')
    frozen([*frozen_inputs(),OUT/'FIT_INDEX.json',
        *[p for seed in SEEDS for p in (OUT/ARCH/f'seed-{seed}').glob('*')]])
    for row in index['models']:
        folder=OUT/ARCH/f"seed-{row['seed']}"
        if json.loads((folder/'FIT_RECEIPT.json').read_text())!=row:raise RuntimeError('FIT_INDEX/receipt mismatch')
        if row['actual_steps']!=STEPS or row['actual_examples']!=24576 or row['actual_tokens']!=2686976:
            raise RuntimeError('registered exposure changed')
        for step in CHECKPOINTS:
            path=folder/('final.pt' if step==STEPS else f'checkpoint-{step}.pt')
            if row['checkpoint_hashes'][str(step)]!=sha(path):raise RuntimeError('checkpoint identity changed')
        if row['training_history_sha256']!=sha(folder/'TRAINING_HISTORY.json'):raise RuntimeError('history changed')
        for record in row['monitor_metrics']:
            if record['arrays_sha256']!=sha(folder/f"monitor-{record['step']}.npz"):raise RuntimeError('monitor identity changed')
        validate_runtime_evidence(dict(row,_prefit_registration_sha256=sha(REG),_expected_optimizer_steps=STEPS,
            runtime_evidence=dict(receipt_path=str(folder/'live-budget.json'),receipt_sha256=sha(folder/'live-budget.json'),
                checkpoint_path=str(folder/'final.pt'),checkpoint_sha256=sha(folder/'final.pt'))))
    return index


def score(replay=False):
    index = budget_audit()
    d, fit = data.load("report"), fit_data()
    forecasts_path = ROOT / "research/replay-bundles/ME101/FORECASTS.npz"
    old_results_path = ROOT / "research/replay-bundles/ME101/RESULTS.json"
    parent_results_path = ROOT / "research/replay-bundles/ME103/RESULTS.json"
    arrays0 = np.load(forecasts_path)
    frozen([forecasts_path, old_results_path, parent_results_path])
    arrays = {name + "_" + suffix: arrays0[name + "_" + suffix]
              for name in ("gap", "general") for suffix in ("origins", "truth", "observed", "locf", "ridge")}
    cohorts = {"gap": d["gap"], "general": d["general"]}
    for name, origins in cohorts.items():
        if not np.array_equal(np.asarray(origins), arrays[name + "_origins"]):
            raise RuntimeError(f"ME111 {name} REPORT origins differ from the frozen original evaluation cohort")
    results = {name: dict(count=len(origins), models=[]) for name, origins in cohorts.items()}
    model_rows = []
    rawscale = torch.tensor(np.nanstd(fit["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    with torch.no_grad():
        for row in index["models"]:
            seed = row["seed"]
            cp = torch.load(OUT / ARCH / f"seed-{seed}" / "final.pt", map_location="cpu", weights_only=False)
            model = initialize(fit, seed); model.load_state_dict(cp["model"], strict=True); model.eval()
            scored = {}
            for name, origins in cohorts.items():
                qs, nll_sum, nll_count = [], 0., 0
                for i in range(0, len(origins), 32):
                    x = [torch.from_numpy(v) for v in data.batch(d, origins[i:i+32], 256)]
                    forecast, _ = model(*x[:6]); qs.append(model.raw_quantiles(forecast).numpy())
                    if not np.array_equal(x[6].numpy(),arrays[name+'_truth'][i:i+32]) or not np.array_equal(x[7].numpy(),arrays[name+'_observed'][i:i+32]):
                        raise RuntimeError('fresh source truth/mask differs from original scoring arrays')
                    nll_sum += float(model.nll(forecast, x[6], x[7])) * int(x[7].sum()); nll_count += int(x[7].sum())
                q = np.concatenate(qs); arrays[f"{name}_conditional_belief_{seed}"] = q
                metrics = me101.metrics(q, arrays[f"{name}_truth"], arrays[f"{name}_observed"])
                metrics["raw_density_nll"] = nll_sum / nll_count; scored[name] = metrics
                results[name]["models"].append(dict(architecture=ARCH, seed=seed, **metrics))
            model_rows.append(dict(architecture=ARCH, seed=seed, scores=scored))
    old = json.loads(old_results_path.read_text()); parent = json.loads(parent_results_path.read_text())
    old_gap = {r["seed"]: r for r in old["cohorts"]["gap"]["models"] if r.get("context") == 256}
    parent_gap = {r["seed"]: r for r in parent["cohorts"]["gap"]["models"] if r["recipe"] == "strong_reconstruction"}
    parent_daily = {r["seed"]: r for r in parent["cohorts"]["general"]["models"] if r["recipe"] == "strong_reconstruction"}
    daily_locf = float(np.abs(arrays["general_locf"][:, 0] - arrays["general_truth"][:, 0]).mean())
    gates = []
    for row in model_rows:
        seed = row["seed"]; oldrow, pg, pd = old_gap[seed-10000], parent_gap[seed-9000], parent_daily[seed-9000]
        g, n = row["scores"]["gap"], row["scores"]["general"]
        checks = dict(coverage=.90 <= g["coverage_by_channel"][0] <= .98,
            gap_point=g["pm_mae"] <= min(1.05*oldrow["pm_mae"],1.05*pg["pm_mae"]),
            daily_point=n["pm_mae"] <= min(1.05*daily_locf,1.05*pd["pm_mae"]),
            gap_nll=g["raw_density_nll"] <= min(6.234504261016846,1.05*pg["raw_density_nll"]),
            interval_score=g["interval_score_by_channel"][0] <= min(1.05*oldrow["interval_score_by_channel"][0],.90*pg["interval_score_by_channel"][0]))
        prior=next(r for r in json.loads((PARENT/'RESULTS.json').read_text())['models'] if r['seed']==seed-1000)
        joint=dict(all_channel_coverage=all(.85<=v<=.98 for cohort in (g,n) for v in cohort['coverage_by_channel']),
            all_channel_point=all(v<=1.05*ref for name,cohort in [('gap',g),('general',n)] for v,ref in zip(cohort['mae_by_channel'],prior['scores'][name]['mae_by_channel'])),
            parent_interval_improvement=g['interval_score_by_channel'][0]<=.90*prior['scores']['gap']['interval_score_by_channel'][0])
        gates.append(dict(architecture=ARCH, seed=seed, gates=checks, full_gate=all(checks.values()),
            joint_checks=joint,joint_gate=all(checks.values()) and all(joint.values())))
    result = dict(status="EXPOSED2015_ME111_CONDITIONAL_BELIEF_TRAINING", models=model_rows, cohorts=results, model_gates=gates,
        all_seed_gate=all(r["full_gate"] for r in gates), all_seed_joint_gate=all(r["joint_gate"] for r in gates), independent_qualification=False, protected_test_scored=False)
    if replay:
        prior = np.load(OUT / "FORECASTS.npz")
        for k, v in arrays.items(): np.testing.assert_allclose(v, prior[k], rtol=1e-5, atol=1e-6)
        expected = json.loads((OUT / "RESULTS.json").read_text())
        if result["model_gates"] != expected["model_gates"] or result["all_seed_gate"] != expected["all_seed_gate"]:
            raise RuntimeError("ME111 replay gate booleans differ from original score")
        _compare_numeric(result, expected)
    else:
        if (OUT / "RESULTS.json").exists() or (DOC / "POSTFIT.json").exists(): raise RuntimeError("ME111 scores already exist")
        np.savez_compressed(OUT / "FORECASTS.npz", **arrays); write(OUT / "RESULTS.json", result)
        post = json.loads(REG.read_text())
        for model_record in post["models"]:
            row = next(r for r in index["models"]
                       if r["architecture"] == model_record["architecture"] and r["seed"] == model_record["seed"])
            model_record["measured"] = row
        write(DOC / "POSTFIT.json", post); check_manifest(post, "final")


def _compare_numeric(actual, expected, path="result"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys(): raise RuntimeError(f"replay key mismatch at {path}")
        for key in actual: _compare_numeric(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected): raise RuntimeError(f"replay length mismatch at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)): _compare_numeric(a, b, f"{path}[{i}]")
    elif isinstance(actual, (int, float, np.number)) and isinstance(expected, (int, float, np.number)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=path)
    elif actual != expected: raise RuntimeError(f"replay mismatch at {path}")


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "fit", "score", "replay"))
    mode = parser.parse_args().mode
    {"prepare": prepare, "fit": fit, "score": score, "replay": lambda: score(True)}[mode]()


if __name__ == "__main__": main()
