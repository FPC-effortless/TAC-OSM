"""Matched training and complete evaluation for protection DEV-005."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from .memory_diagnostics import seed_interval
from .noisy_memory import parameter_hash
from .protected_memory import ProtectionModel
from .protection_benchmark import euclidean,fingerprint,make_pair,seed_for,tensors

ID = 'RETENTION-PROTECTION-DEV-005'
ROOT = Path(__file__).resolve().parents[3]


def protocol() -> dict:
    return json.loads((ROOT/'development_protocols'/f'{ID}.json').read_text())


def matched_models(seed: int) -> dict[str,ProtectionModel]:
    torch.manual_seed(seed)
    source = ProtectionModel()
    models = {}
    for arm in protocol()['arms']:
        model = copy.deepcopy(source)
        model.mode = model.memory.mode = arm
        for p in model.memory.protection.parameters():
            p.requires_grad_(arm=='learned_protection')
        models[arm] = model
    if len({parameter_hash(m) for m in models.values()})!=1:
        raise AssertionError('unequal starting tensors')
    return models


def training_batch(seed: int,step: int):
    cfg = protocol()['training']
    h,d = cfg['history_lengths'][step%2],cfg['intervening_writes'][(step//2)%2]
    fraction,noise = cfg['hard_fractions'][(step//4)%2],cfg['noise_levels'][(step//8)%2]
    episodes = []
    for j in range(cfg['batch_size']//2):
        task = ('retention','acquisition')[j%2]
        pair = make_pair(seed_for(cfg['namespace'],seed,step,j),h,d,fraction,noise,task)
        episodes.extend((pair.left,pair.right))
    return episodes


def train(seed: int,*,smoke: bool=False):
    cfg = protocol()['training']
    steps = 2 if smoke else cfg['steps']
    models = matched_models(seed)
    records = {arm:{'initial_hash':parameter_hash(model),'losses':[],
                    'gradient_seen':{'key':False,'write':False,'read':False,'operator':False},
                    'gate_gradient_seen':False} for arm,model in models.items()}
    optimizers = {arm:torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                       lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                  for arm,model in models.items()}
    fingerprints = []
    for step in range(steps):
        episodes = training_batch(seed,step)
        a,v,q,c = tensors(episodes)
        labels = torch.tensor([e.target for e in episodes])
        fingerprints.extend(fingerprint(e) for e in episodes)
        for arm,model in models.items():
            model.train()
            state = model.encode(a,v,trace=False)
            logits = model.predict(state,q,c)
            loss = F.cross_entropy(logits,labels)
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = {'key':model.memory.key.weight,'write':model.memory.write.weight,
                          'read':model.read_selector[0].weight,'operator':model.compute.selector[0].weight}
            for name,p in parameters.items():
                g = p.grad
                records[arm]['gradient_seen'][name] |= bool(g is not None and torch.isfinite(g).all() and g.abs().sum()>0)
            g = model.memory.protection[-1].weight.grad
            records[arm]['gate_gradient_seen'] |= bool(g is not None and torch.isfinite(g).all() and g.abs().sum()>0)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),cfg['gradient_clip'])
            optimizer.step()
            records[arm]['losses'].append(float(loss.detach()))
    for arm,model in models.items():
        model.eval()
        record = records[arm]
        record.update({'final_hash':parameter_hash(model),'steps':steps,
                       'total_parameters':sum(p.numel() for p in model.parameters()),
                       'active_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'recurrent_state_scalars':768})
        if record['initial_hash']==record['final_hash'] or not all(record['gradient_seen'].values()):
            raise AssertionError('missing real training')
        if record['gate_gradient_seen']!=(arm=='learned_protection'):
            raise AssertionError('gate intervention gradient mismatch')
    return models,{'seed':seed,'steps':steps,'arms':records,'train_fingerprints':fingerprints}


@torch.no_grad()
def measure(model: ProtectionModel,seed: int,h: int,d: int,fraction: float,noise: float,task: str) -> dict:
    if model.training:
        raise ValueError('evaluation requires eval mode')
    pair = make_pair(seed,h,d,fraction,noise,task)
    episodes = [pair.left,pair.right]
    a,v,q,c = tensors(episodes)
    state = model.encode(a,v)
    gate_trace = [list(x) for x in model.last_gate_trace]
    native = model.predict(state,q,c).argmax(-1).tolist()
    dispatch = [asdict(t) for t in model.compute.last_trace]
    swapped = model.predict(state.flip(0),q,c).argmax(-1).tolist()
    reset = model.predict(torch.zeros_like(state),q,c).argmax(-1).tolist()
    earlier = model.encode(a[:,:h],v[:,:h])
    no_write = model.predict(earlier,q,c).argmax(-1).tolist()
    classical = [euclidean(e) for e in episodes]
    return {'seed':seed,'H':h,'D':d,'fraction':fraction,'noise':noise,'task':task,
            'fingerprints':[fingerprint(e) for e in episodes],
            'target':[e.target for e in episodes],'oracle_stream_slot':pair.left.target_stream_slot,
            'target_age':h+d-1-pair.left.target_stream_slot,
            'predictions':{'native':native,'swapped_history':swapped,'reset':reset,
                           'no_intervening_writes_oracle':no_write,'dense_euclidean':[x[0] for x in classical]},
            'classical_indices':[x[1] for x in classical],
            'paired_state_equal':bool(torch.equal(state[0],state[1])),
            'state_contrast_norm':float(torch.linalg.vector_norm(state[0]-state[1])),
            'gate_trace_mean_min_max':gate_trace,'gate_history_mean':statistics.fmean(x[0] for x in gate_trace[:h]),
            'gate_new_writes_mean':statistics.fmean(x[0] for x in gate_trace[h:]),
            'write_calls':h+d,'dispatch':dispatch,'state_scalars':state[0].numel()}


def evaluate(model: ProtectionModel,seed: int,*,smoke: bool=False) -> list[dict]:
    cfg = protocol()['evaluation']
    hs,ds,fs,ns,tasks,count = ([32],[4],[0.0],[.08],cfg['tasks'],1) if smoke else (
        cfg['history_lengths'],cfg['intervening_writes'],cfg['hard_fractions'],cfg['noise_levels'],
        cfg['tasks'],cfg['pairs_per_cell'])
    return [measure(model,seed_for(cfg['namespace'],seed,h,d,f,n,task,j),h,d,f,n,task)
            for h in hs for d in ds for f in fs for n in ns for task in tasks for j in range(count)]


def accuracy(rows: list[dict],name: str='native') -> float:
    return sum(y==t for r in rows for y,t in zip(r['predictions'][name],r['target']))/(2*len(rows))


def summarize(records: list[dict]) -> dict:
    arms = protocol()['arms']
    result = {}
    for arm in arms:
        rows = [r for record in records for r in record['rows'][arm]]
        tasks = {task:accuracy([r for r in rows if r['task']==task]) for task in ('retention','acquisition')}
        result[arm] = {'overall':accuracy(rows),**tasks,'pairs':len(rows),
                       'controls':{name:accuracy(rows,name) for name in rows[0]['predictions'] if name!='native'},
                       'gate_history_mean':statistics.fmean(r['gate_history_mean'] for r in rows),
                       'gate_new_writes_mean':statistics.fmean(r['gate_new_writes_mean'] for r in rows),
                       'by_history':{str(h):{task:accuracy([r for r in rows if r['H']==h and r['task']==task])
                                             for task in ('retention','acquisition')}
                                      for h in sorted({r['H'] for r in rows})},
                       'by_interference':{str(f):{task:accuracy([r for r in rows if r['fraction']==f and r['task']==task])
                                                  for task in ('retention','acquisition')}
                                          for f in sorted({r['fraction'] for r in rows})}}
    gaps = {}
    for control in ('always_update','fixed_protection'):
        gaps[control] = {}
        for task in ('overall','retention','acquisition'):
            values = []
            for record in records:
                a,b = record['rows']['learned_protection'],record['rows'][control]
                if task!='overall':
                    a,b = [r for r in a if r['task']==task],[r for r in b if r['task']==task]
                values.append(accuracy(a)-accuracy(b))
            gaps[control][task] = seed_interval(values)
    return {'arms':result,'paired_seed_gaps':gaps,'development_only':True}
