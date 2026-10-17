"""Execute the fixed DEV-055 matrix, save checkpoints and every trajectory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.learned_stack import (
    LearnedStack, PhysicalWorld, FittedFilter, collect, independent_transitions, COSTS)

PROTOCOL = ROOT/'research/dev055/protocol.json'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def fit(model, train, validation, seed, config, smoke=False):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'])
    rng = torch.Generator().manual_seed(seed+101)
    curves, gradients = [], {}
    started = time.monotonic()
    updates = 3 if smoke else config['updates']
    def record(update):
        model.eval()
        with torch.no_grad():
            values = {name: float(F.binary_cross_entropy_with_logits(model(ds[0], ds[1]), ds[2]))
                      for name, ds in [('train', train), ('validation', validation)]}
        curves.append({'update': update, **values})
        model.train()
    record(0)
    model.completed_updates = 0
    for step in range(1, updates+1):
        ids = torch.randint(len(train[0]), (config['batch'],), generator=rng)
        x, a, y = [v[ids] for v in train]
        loss = F.binary_cross_entropy_with_logits(model(x, a), y)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if step == 1:
            for group in ('encoder', 'memory', 'router', 'workspace', 'compute', 'query', 'head'):
                gradients[group] = sum(float(p.grad.abs().sum()) for p in getattr(model, group).parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); model.completed_updates = step
        if step % config['log_every'] == 0 or step == updates: record(step)
    model.eval()
    ref = train[0][:1, 0]
    state = model.initial(ref)
    return {'curves': curves, 'first_step_gradient_l1': gradients, 'updates': updates,
            'seconds': time.monotonic()-started, 'trainable_parameters': model.trainable_count,
            'stored_parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'retained_state_bytes': sum(s.numel()*s.element_size() for s in state if s is not None)
                + (2*3*4*4 if model.arm != 'gru_matched' else 0)}


@torch.no_grad()
def evaluate(policy, spec, family, transitions, independent, smoke=False, neutral_memory=False):
    seed, count = spec
    if smoke: seed, count = seed+10_000_000, 3
    worlds = [PhysicalWorld(seed, i, family) for i in range(count)]
    rows = [{'seed': seed, 'index': i, 'family': family, 'cost': 0., 'fault_exposure': 0,
             'acquisition_cost': 0., 'actions': [], 'work': {'routing_candidates': 0, 'retained': 0, 'operator_calls': 0}} for i in range(count)]
    filters = None
    neural = isinstance(policy, LearnedStack)
    if neural:
        state = policy.initial(torch.zeros(count, 10))
    elif policy != 'oracle':
        filters = [FittedFilter(independent if policy == 'misspecified_independent' else transitions,
                                policy == 'misspecified_independent') for _ in worlds]
    started = time.monotonic()
    for tick in range(20):
        events = [w.last for w in worlds]
        if neural:
            public = torch.tensor([e.tensor() for e in events], dtype=torch.float32)
            state = policy.update(state, public)
        elif filters:
            for filt, event in zip(filters, events): filt.update(event, initial=(tick == 0))
        if tick < 12:
            actions = [int(np.random.default_rng(np.random.SeedSequence([seed, i, tick, 441])).integers(5)) for i in range(count)]
        else:
            if neural:
                inference_state = policy.initial(public) if neutral_memory else state
                expanded = tuple(s[:,None].expand(-1,5,*s.shape[1:]).reshape(-1,*s.shape[1:]) if s is not None else None for s in inference_state)
                probabilities = policy.predict(expanded, torch.arange(5).repeat(count), torch.full((count*5,), tick)).sigmoid().view(count,5,2).numpy()
                # Attribute actual dispatched samples, including uneven halting.
                per_world_calls = [0]*count
                if policy.arm != 'no_casm':
                    for trace in policy.compute.last_trace:
                        for _, samples in trace.dispatched:
                            for sample in samples: per_world_calls[sample//5] += 1
                for index, row in enumerate(rows):
                    row['work']['routing_candidates'] += 15
                    row['work']['retained'] += 15 if policy.arm == 'uniform_route' else 10
                    row['work']['operator_calls'] += per_world_calls[index]
            elif policy == 'oracle': probabilities = np.array([[w.expected(a) for a in range(5)] for w in worlds])
            else: probabilities = np.array([f.predict() for f in filters])
            actions = (COSTS[None,:]+4*probabilities.sum(-1)).argmin(-1).tolist()
        for row, world, action in zip(rows, worlds, actions):
            _, cost, label = world.step(action)
            row['cost'] += cost
            row['fault_exposure'] += sum(label['after'])
            row['acquisition_cost'] += cost if tick < 12 else 0
            row['actions'].append({'action': action, 'cost': cost, 'after': label['after'],
                                   'public': world.last.tensor()})
    elapsed = time.monotonic()-started
    return {'rows': rows, 'seconds': elapsed, 'cost': float(np.mean([r['cost'] for r in rows])),
            'fault_exposure': float(np.mean([r['fault_exposure'] for r in rows]))}


def bootstrap(costs, reference, rng, draws=2000, alpha=.05):
    # Arrays [initialization, independent world cluster]; same bootstrap world
    # indices for each initialization; family-stratified clustering.
    delta = costs-reference
    values = []
    half = delta.shape[1]//2
    for _ in range(draws):
        ids = np.concatenate([rng.integers(0, half, half), rng.integers(half, 2*half, half)])
        seeds = rng.integers(0, delta.shape[0], delta.shape[0])
        values.append(delta[seeds][:,ids].mean())
    return {'mean': float(delta.mean()), 'interval': np.quantile(values, [alpha/2,1-alpha/2]).tolist()}


def summarize(out, protocol, reports, controls):
    families = ('coupling', 'drift')
    def array(arm, metric='cost'):
        if arm in controls:
            a = np.array([[r[metric] for f in families for r in controls[arm][f]['rows']]])
            return np.repeat(a, len(protocol['model_seeds']), axis=0)
        return np.array([[r[metric] for f in families for r in reports[str(seed)][arm]['evaluation'][f]['rows']]
                         for seed in protocol['model_seeds']])
    if any(reports[str(seed)][arm].get('status') != 'COMPLETE' for seed in protocol['model_seeds'] for arm in protocol['arms']):
        return {'status':'INCOMPLETE', 'decision': 'No scientific acceptance; failed cells preserved.'}
    rng = np.random.default_rng(75501)
    result = {'status':'COMPLETE_UNAUDITED', 'costs':{}, 'contrasts':{}}
    for arm in protocol['arms']+protocol['controls']:
        result['costs'][arm] = {'total_stream_cost':float(array(arm).mean()),
                              'fault_exposure':float(array(arm, 'fault_exposure').mean())}
    for arm in ['fitted_hmm', 'gru_matched', 'mtsk_reset', 'uniform_route', 'no_casm']:
        result['contrasts'][arm] = bootstrap(array('mtsk'), array(arm), rng, alpha=.05 if arm == 'fitted_hmm' else .05/4)
    headroom = bootstrap(array('fitted_hmm')/2, array('oracle')/2, rng)
    feasible = all((array('oracle','fault_exposure')[s, h:h+array('oracle').shape[1]//2].sum() <=
                    array('fitted_hmm','fault_exposure')[s,h:h+array('oracle').shape[1]//2].sum())
                   for s in range(len(protocol['model_seeds'])) for h in (0,array('oracle').shape[1]//2))
    result['oracle_headroom_per_entity'] = headroom
    result['oracle_reliability_feasible'] = bool(feasible)
    result['adequate_headroom'] = bool(feasible and headroom['mean'] >= .1 and headroom['interval'][0]>0)
    reliable = all(array('mtsk','fault_exposure')[s,h:h+array('mtsk').shape[1]//2].sum() <=
                   array(ref,'fault_exposure')[s,h:h+array('mtsk').shape[1]//2].sum()
                   for ref in ('gru_matched','fitted_hmm') for s in range(len(protocol['model_seeds']))
                   for h in (0,array('mtsk').shape[1]//2))
    result['reliability_guard'] = bool(reliable)
    accepted = reliable and result['adequate_headroom'] and all(result['contrasts'][a]['interval'][1]<0 for a in ('gru_matched','fitted_hmm'))
    result['decision'] = 'DEVELOPMENT_CANDIDATE_REQUIRES_REPLICATION' if accepted else 'NO_SUPERIORITY_ESTABLISHED'
    return result


def run(output, smoke=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    p = json.loads(PROTOCOL.read_text())
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    source = git('rev-parse','HEAD')
    if not smoke and git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('commit source before full run')
    started = time.monotonic()
    datasets, labels = {}, {}
    for name in ('train','validation'):
        seed, count = p['splits'][name]
        if smoke: seed, count = seed+10_000_000, 8
        datasets[name], labels[name] = collect(seed,count)
    transitions = FittedFilter.fit(labels['train'], datasets['train'][1])
    independent = independent_transitions(labels['train'], datasets['train'][1])
    np.savez(out/'filters.npz', transitions=transitions, independent=independent)
    controls = {}
    for arm in p['controls']:
        controls[arm] = {f: evaluate(arm,p['splits'][f],f,transitions,independent,smoke) for f in ('coupling','drift')}
    write(out/'controls.json', controls)
    reports = {}
    for seed in p['model_seeds'][:1] if smoke else p['model_seeds']:
        reports[str(seed)] = {}
        for arm in p['arms']:
            print('fit',seed,arm,flush=True)
            model = LearnedStack(seed,arm)
            try:
                training = fit(model,datasets['train'],datasets['validation'],seed,p['training'],smoke)
                checkpoint = out/f'{seed}-{arm}.pt'; torch.save(model.state_dict(),checkpoint)
                evaluation = {f:evaluate(model,p['splits'][f],f,transitions,independent,smoke) for f in ('coupling','drift')}
                reports[str(seed)][arm] = {'status':'COMPLETE','training':training,'evaluation':evaluation,
                                          'checkpoint_sha256':digest(checkpoint)}
                print('done',seed,arm,training['seconds'], {f:e['cost'] for f,e in evaluation.items()},flush=True)
            except Exception:
                torch.save(model.state_dict(),out/f'{seed}-{arm}-FAILED.pt')
                reports[str(seed)][arm] = {'status':'FAILED','error':traceback.format_exc(),'updates':getattr(model,'completed_updates',0)}
                print(reports[str(seed)][arm]['error'],flush=True)
            write(out/'reports.json',reports)
    result = {'status':'SMOKE_ONLY'} if smoke else summarize(out,p,reports,controls)
    result.update({'source':source,'protocol_sha256':digest(PROTOCOL),'smoke':smoke,
                   'elapsed_seconds':time.monotonic()-started,'torch_version':torch.__version__,
                   'numpy_version':np.__version__,'python_version':sys.version})
    write(out/'summary.json',result)
    write(out/'manifest.json',{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True); parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); run(args.output,args.smoke)
