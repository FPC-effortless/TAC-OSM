"""Frozen-model diagnostics; oracle arms are explicitly evaluator-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tac_osm.integration_002_benchmark import make_pair
from .conditional_compute import selected_weights
from .memory_experiment import episode_fingerprint, followup, tensors
from .noisy_memory import NoisyMemoryModel, parameter_hash

ID = 'MEMORY-DIAGNOSTICS-DEV-004'
ROOT = Path(__file__).resolve().parents[3]


def protocol() -> dict:
    return json.loads((ROOT/'development_protocols'/f'{ID}.json').read_text())


def seed_for(namespace: str, *parts: object) -> int:
    text = json.dumps([ID,namespace,*parts],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')


def raw_reads(model: NoisyMemoryModel, state: Tensor, address: Tensor) -> Tensor:
    return torch.einsum('bevk,bek->bev',state,model.memory.keys(address)).flatten(1)


class ReadProbe(nn.Module):
    def __init__(self, inputs: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,2))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def train_probe(model: NoisyMemoryModel, model_seed: int, *, smoke: bool = False) -> tuple[ReadProbe,dict]:
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('probe training requires explicitly frozen parent parameters')
    config = protocol()['probe']
    count,steps = (8,2) if smoke else (config['train_pairs'],config['steps'])
    before = parameter_hash(model)
    torch.manual_seed(model_seed+4000)
    probe = ReadProbe(model.config.experts*model.config.value_dim,config['hidden_dim'])
    initial = parameter_hash(probe)
    features,labels,fingerprints = [],[],[]
    with torch.no_grad():
        for i in range(count):
            pair = make_pair(seed_for(config['namespace'],model_seed,i),config['train_history'],
                             config['train_delay'],(0.0,1.0)[i%2],(.08,.25)[(i//2)%2])
            episodes = (pair.left,pair.right)
            a,v,q,c = tensors(episodes)
            state = model.encode(a,v)
            features.append(raw_reads(model,state,q))
            labels.extend(e.target ^ e.current_bit for e in episodes)
            fingerprints.extend(episode_fingerprint(e) for e in episodes)
    x,y = torch.cat(features),torch.tensor(labels)
    optimizer = torch.optim.AdamW(probe.parameters(),lr=config['lr'])
    rng = torch.Generator().manual_seed(seed_for('probe-batches',model_seed))
    losses,gradient = [],False
    for _ in range(steps):
        indices = torch.randint(len(y),(config['batch_size'],),generator=rng)
        loss = F.cross_entropy(probe(x[indices]),y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = probe.net[0].weight.grad
        gradient |= bool(torch.isfinite(g).all() and g.abs().sum()>0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if parameter_hash(model) != before:
        raise AssertionError('probe altered parent memory model')
    return probe.eval(),{'model_seed':model_seed,'steps':steps,'pairs':count,
                         'initial_hash':initial,'final_hash':parameter_hash(probe),
                         'nonzero_finite_gradient':gradient,'parent_hash':before,
                         'losses':losses,'train_fingerprints':fingerprints}


def route(model: NoisyMemoryModel, state: Tensor, address: Tensor, current: Tensor,
          *, uniform: bool = False) -> tuple[Tensor,Tensor]:
    q = model.query(torch.cat((address,F.one_hot(current,2).to(address.dtype)),-1))
    reads = raw_reads(model,state,address).view(-1,model.config.experts,model.config.value_dim)
    views = model.read_projection(reads)
    if uniform:
        return views.mean(1),q
    logits = model.read_selector(torch.cat((views,q[:,None].expand_as(views)),-1)).squeeze(-1)
    indices,weights = selected_weights(logits,model.config.read_top_k,False)
    selected = views.gather(1,indices[...,None].expand(-1,-1,views.shape[-1]))
    return (selected*weights.gather(1,indices)[...,None]).sum(1),q


@torch.no_grad()
def retrieval_row(model: NoisyMemoryModel, probe: ReadProbe, seed: int,
                  h: int,d: int,fraction: float,noise: float) -> dict:
    if model.training or probe.training:
        raise ValueError('diagnostics require evaluation mode')
    pair = make_pair(seed,h,d,fraction,noise)
    episodes = (pair.left,pair.right)
    a,v,q,c = tensors(episodes)
    state = model.encode(a,v)
    index = pair.left.target_slot
    exact = a[:,index]
    only = model.encode(a[:,index:index+1],v[:,index:index+1])
    order = [i for i in range(h+d) if i != index]+[index]
    last = model.encode(a[:,order],v[:,order])
    reads = raw_reads(model,state,q)
    native = model.predict(state,q,c).argmax(-1)
    routed,query = route(model,state,q,c)
    uniform,uniform_query = route(model,state,q,c,uniform=True)
    computed,_ = model.compute(uniform,uniform_query)
    past = probe(reads).argmax(-1)
    predictions = {
        'native':native.tolist(),
        'exact_address_oracle':model.predict(state,exact,c).argmax(-1).tolist(),
        'uniform_read':model.answer(torch.cat((computed,uniform_query),-1)).argmax(-1).tolist(),
        'no_compute':model.answer(torch.cat((routed,query),-1)).argmax(-1).tolist(),
        'target_only_oracle':model.predict(only,q,c).argmax(-1).tolist(),
        'target_last_oracle':model.predict(last,q,c).argmax(-1).tolist(),
        'read_probe':(past ^ c).tolist(),
        'exact_read_probe':(probe(raw_reads(model,state,exact)).argmax(-1)^c).tolist(),
        'target_only_probe':(probe(raw_reads(model,only,q)).argmax(-1)^c).tolist(),
        'target_last_probe':(probe(raw_reads(model,last,q)).argmax(-1)^c).tolist(),
        'reset':model.predict(torch.zeros_like(state),q,c).argmax(-1).tolist(),
    }
    return {'seed':seed,'H':h,'D':d,'fraction':fraction,'noise':noise,
            'fingerprints':[episode_fingerprint(e) for e in episodes],
            'target':[e.target for e in episodes],'predictions':predictions,
            'target_position':index,'target_age':h+d-1-index,
            'paired_state_equal':bool(torch.equal(state[0],state[1])),
            'state_difference_norm':float(torch.linalg.vector_norm(state[0]-state[1])),
            'read_difference_norm':float(torch.linalg.vector_norm(reads[0]-reads[1])),
            'state_elements':state[0].numel()}


def competing_writes(seed: int,address: Tensor,delay: int,fraction: float) -> tuple[Tensor,Tensor]:
    if delay < 0 or fraction not in (0.0,1.0):
        raise ValueError('invalid delayed-carry intervention')
    rng = random.Random(seed)
    rows,values = [],[]
    base = address[0].tolist()
    for _ in range(delay):
        rows.append([x+rng.gauss(0,.12) for x in base] if fraction else
                    [rng.gauss(0,1) for _ in base])
        values.append(rng.randrange(2))
    a = torch.tensor(rows,dtype=torch.float32).reshape(1,delay,12).expand(2,-1,-1)
    v = torch.tensor(values,dtype=torch.long).reshape(1,delay).expand(2,-1)
    return a,v


@torch.no_grad()
def carry_row(model: NoisyMemoryModel, seed: int, delay: int,fraction: float) -> dict:
    f = followup(seed)
    # A matched nonempty background, identical in both partners. The commit
    # starts from the same state, so reset/no-commit have an exact 50% ceiling.
    rng = random.Random(seed_for('carry-background',seed))
    a = torch.tensor([[[rng.gauss(0,1) for _ in range(12)] for _ in range(32)]],dtype=torch.float32).expand(2,-1,-1)
    v = torch.tensor([[rng.randrange(2) for _ in range(32)]]).expand(2,-1)
    before = model.encode(a,v)
    committed,reliability = model.verified_write(before,f['address'],f['value'],f['evidence'])
    writes,payload = competing_writes(seed_for('carry-competition',seed),f['address'],delay,fraction)
    carried,no_commit = committed,before
    if delay:
        carried = model.encode(writes,payload,state=carried)
        no_commit = model.encode(writes,payload,state=no_commit)
    predictions = {name:model.predict(state,f['query'],f['current']).argmax(-1).tolist()
                   for name,state in [('verified',carried),('no_commit',no_commit),
                                      ('reset',torch.zeros_like(carried)),('swapped',carried.flip(0))]}
    return {'seed':seed,'delay':delay,'fraction':fraction,'target':f['target'].tolist(),
            'predictions':predictions,'reliability':reliability.sigmoid().tolist(),
            'paired_state_equal':bool(torch.equal(carried[0],carried[1])),
            'state_difference_norm':float(torch.linalg.vector_norm(carried[0]-carried[1])),
            'write_calls':32+1+delay}


@torch.no_grad()
def verifier_cases(model: NoisyMemoryModel, model_seed: int, *, calibration: bool) -> list[dict]:
    cfg = protocol()['verifier']
    namespace = cfg['calibration_namespace'] if calibration else cfg['evaluation_namespace']
    count = cfg['calibration_samples_per_level'] if calibration else cfg['samples_per_level']
    rows = []
    for evidence in cfg['evidence_levels']:
        rng = random.Random(seed_for(namespace,model_seed,evidence))
        address = torch.tensor([[rng.gauss(0,1) for _ in range(12)] for _ in range(count)],dtype=torch.float32)
        value = torch.tensor([rng.randrange(2) for _ in range(count)])
        x = torch.cat((address,F.one_hot(value,2).float(),torch.full((count,1),evidence)),-1)
        p = model.verifier(x).sigmoid().squeeze(-1).tolist()
        for i,prob in enumerate(p):
            rows.append({'seed':seed_for(namespace,model_seed,evidence,i),'evidence':evidence,
                         'valid':evidence>0,'probability':prob})
    return rows


def verifier_errors(rows: list[dict],threshold: float) -> dict:
    positive = [x for x in rows if x['valid']]
    negative = [x for x in rows if not x['valid']]
    if not positive or not negative:
        raise ValueError('verifier evaluation needs both classes')
    false_accept = sum(x['probability']>=threshold for x in negative)/len(negative)
    false_reject = sum(x['probability']<threshold for x in positive)/len(positive)
    return {'false_accept_rate':false_accept,'false_reject_rate':false_reject,
            'weighted_error':5*false_accept+false_reject}


def calibrate(rows: list[dict]) -> tuple[float,list[dict]]:
    grid = protocol()['verifier']['calibration_threshold_grid']
    table = [{'threshold':t,**verifier_errors(rows,t)} for t in grid]
    best = min(table,key=lambda x:(x['weighted_error'],-x['threshold']))
    return best['threshold'],table


def evaluate(model: NoisyMemoryModel, probe: ReadProbe,model_seed: int,diagnostic_seed: int,
             *, smoke: bool = False) -> dict:
    p = protocol()
    hs,ds,fs,ns,pairs = ([32],[1],[0.0],[.08],1) if smoke else (
        p['history_lengths'],p['intervening_writes'],p['hard_fractions'],p['noise_levels'],p['pairs_per_cell'])
    retrieval = [retrieval_row(model,probe,seed_for('diagnostic-eval',diagnostic_seed,h,d,f,n,j),h,d,f,n)
                 for h in hs for d in ds for f in fs for n in ns for j in range(pairs)]
    delays,fractions,count = ([0,1],[0.0],1) if smoke else (
        p['carry_delays'],p['carry_hard_fractions'],p['carry_pairs_per_cell'])
    carry = [carry_row(model,seed_for('diagnostic-carry-eval',diagnostic_seed,d,f,j),d,f)
             for d in delays for f in fractions for j in range(count)]
    calibration = verifier_cases(model,model_seed,calibration=True)
    threshold,table = calibrate(calibration)
    evaluation = verifier_cases(model,model_seed,calibration=False)
    if {x['seed'] for x in calibration}&{x['seed'] for x in evaluation}:
        raise AssertionError('calibration/evaluation overlap')
    return {'model_seed':model_seed,'diagnostic_seed':diagnostic_seed,
            'retrieval':retrieval,'carry':carry,'verifier':{
                'calibration_rows':calibration,'calibration_table':table,'selected_threshold':threshold,
                'evaluation_rows':evaluation,
                'native_errors':verifier_errors(evaluation,p['verifier']['native_threshold']),
                'calibrated_errors':verifier_errors(evaluation,threshold),
                'sign_rule_errors':{'false_accept_rate':0.0,'false_reject_rate':0.0}}}


def accuracy(rows: list[dict],arm: str) -> float:
    return sum(y==t for row in rows for y,t in zip(row['predictions'][arm],row['target']))/(2*len(rows))


def seed_interval(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) != 3:
        return {'mean':mean,'exploratory_t95':None,'n_seeds':len(values)}
    critical = .95*math.sqrt(2/(1-.95**2))  # exact t(.975, df=2)
    half = critical*statistics.stdev(values)/math.sqrt(3)
    return {'mean':mean,'exploratory_t95':[mean-half,mean+half],'n_seeds':3}


def summary(records: list[dict]) -> dict:
    retrieval = [row for record in records for row in record['retrieval']]
    carry = [row for record in records for row in record['carry']]
    arms = retrieval[0]['predictions']
    by_h = {str(h):{'accuracy':{arm:accuracy([r for r in retrieval if r['H']==h],arm) for arm in arms},
                    'equal_state_fraction':statistics.fmean(r['paired_state_equal'] for r in retrieval if r['H']==h)}
            for h in sorted({r['H'] for r in retrieval})}
    by_delay = {f'{d}/{f}':{arm:accuracy([r for r in carry if r['delay']==d and r['fraction']==f],arm)
                           for arm in ('verified','no_commit','reset','swapped')}
                for d,f in sorted({(r['delay'],r['fraction']) for r in carry})}
    by_age = {}
    for low,high in ((0,15),(16,63),(64,127),(128,543)):
        subset = [r for r in retrieval if low <= r['target_age'] <= high]
        if subset:
            by_age[f'{low}-{high}'] = {'pairs':len(subset),
                'native_accuracy':accuracy(subset,'native'),'probe_accuracy':accuracy(subset,'read_probe'),
                'equal_state_fraction':statistics.fmean(r['paired_state_equal'] for r in subset)}
    return {'retrieval_pairs':len(retrieval),'carry_pairs':len(carry),
            'retrieval':{arm:accuracy(retrieval,arm) for arm in arms},'by_history':by_h,'by_target_age':by_age,
            'carry_by_delay_hard_fraction':by_delay,
            'native_seed_interval':seed_interval([accuracy(r['retrieval'],'native') for r in records]),
            'probe_minus_native_seed_interval':seed_interval([
                accuracy(r['retrieval'],'read_probe')-accuracy(r['retrieval'],'native') for r in records]),
            'verifier':{kind:{metric:statistics.fmean(r['verifier'][kind][metric] for r in records)
                              for metric in ('false_accept_rate','false_reject_rate')}
                        for kind in ('native_errors','calibrated_errors','sign_rule_errors')},
            'development_only':True}
