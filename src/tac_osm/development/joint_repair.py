"""DEV-054: learned regime beliefs with an identical explicit joint planner.

Only the posterior encoder is learned. Physics, catalogue, addressing and joint
Bayes updates remain explicit. This is an MTSK lane, not full TAC-OSM.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .adaptive_repair import (AdaptiveModel, HARD, EASY, rng_for, coupon,
                             baseline_forecasts, expected_nll, rollout_probe)
from .full_plm_reference import FullPLMConfig, SwiLAMTSK

NEURAL = ('mtsk', 'gru_matched', 'gru128', 'gru_hmm', 'mtsk_reset')
BASELINES = ('stationary_beta', 'hmm', 'oracle')
LEVELS = (0, 8, 32)


def collect(spec, smoke=False):
    rows = []
    for index in range(min(3, spec['worlds']) if smoke else spec['worlds']):
        seed = spec['seed']
        rng = rng_for('dev054-path', seed, index)
        kind = spec['kind']
        markov = kind == 'switching' or (kind not in ('switching', 'drift') and rng.random() < .5)
        family = 'markov' if markov else rng.choice(('sine', 'triangle'))
        test = kind in ('switching', 'drift')
        hazard = rng.choice((.04, .12, .24) if test else (.02, .08, .16, .32))
        period = rng.choice((10, 24, 48) if test else (8, 12, 20, 36, 64))
        phase, z = rng.random(), rng.randrange(2)
        targets = []
        for t in range(33):
            if markov:
                if t and rng.random() < hazard: z = 1-z
                amount = float(z)
            elif family == 'sine': amount = .5+.5*math.sin(2*math.pi*(phase+t/period))
            else: amount = 1-abs(2*((phase+t/period) % 1)-1)
            targets.append((HARD+amount*(EASY-HARD)).tolist())
        observations = [coupon(seed, index, t, targets[t]) for t in range(32)]
        tokens = [[float(o['attempt'] == a) for a in (1, 2, 3)]+[float(o['success']), 1.] for o in observations]
        rows.append({'id': f'dev054:{seed}:{index}', 'seed': seed, 'index': index,
                     'family': family, 'hazard': hazard, 'period': period,
                     'targets': targets, 'tokens': tokens, 'observations': observations})
    return rows


class JointModel(AdaptiveModel):
    def __init__(self, seed, arm):
        nn.Module.__init__(self)
        if arm not in NEURAL: raise ValueError(arm)
        torch.manual_seed(seed)
        self.arm = arm
        if arm.startswith('gru'):
            self.hidden = {'gru_matched': 15, 'gru128': 128, 'gru_hmm': 14}[arm]
            self.encoder = nn.Identity()
            self.kernel = nn.GRUCell(7 if arm == 'gru_hmm' else 5, self.hidden)
            k = 8 if arm == 'gru_hmm' else 6
            self.head = nn.Linear(self.hidden, 2) if arm == 'gru128' else nn.Sequential(nn.Linear(self.hidden, k), nn.Tanh(), nn.Linear(k, 2))
        else:
            cfg = FullPLMConfig(d_model=8, num_heads=2, num_experts=3, fast_experts=1, medium_experts=1, slow_experts=1)
            self.encoder = nn.Sequential(nn.Linear(5, 8), nn.Tanh())
            self.kernel = SwiLAMTSK(cfg)
            self.head = nn.Linear(8, 2)


def data(rows, params, hazard, hybrid=False):
    x = np.array([r['tokens'] for r in rows])
    if hybrid:
        _, w = baseline_forecasts(rows, 'hmm_plugin', params, {'hazard': hazard})
        x = np.concatenate((x, w[:, 1:]), axis=2)
    amount = (np.array([r['targets'] for r in rows])[:, :, 0]-HARD[0])/(EASY[0]-HARD[0])
    y = np.stack((1-amount, amount), axis=-1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def soft_ce(logits, y):
    return -(y*F.log_softmax(logits, -1)).sum(-1).mean()


def fit(model, train, validation, seed, updates, log_updates):
    x, y = train
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+554)
    curve = []
    start = time.monotonic()
    def record(step):
        with torch.no_grad():
            curve.append({'update': step, 'train_ce': float(soft_ce(model.sequence(x), y)),
                          'validation_ce': float(soft_ce(model.sequence(validation[0]), validation[1]))})
    record(0)
    model.completed_updates = 0
    for step in range(1, updates+1):
        ids = torch.randint(len(x), (32,), generator=rng)
        loss = soft_ce(model.sequence(x[ids]), y[ids])
        if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite update {step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); model.completed_updates = step
        if step in log_updates or step == updates: record(step)
    model.eval()
    prior = next((r for r in curve if r['update'] == 768), None)
    relative = (prior['validation_ce']-curve[-1]['validation_ce'])/prior['validation_ce'] if prior else None
    return {'updates': updates, 'curve': curve, 'training_seconds': time.monotonic()-start,
            'plateau_relative_improvement': relative,
            'plateau': abs(relative) <= .01 if relative is not None else None,
            'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'retained_state_bytes': model.state_scalars*4+(8 if model.arm == 'gru_hmm' else 0),
            'hybrid_extra_state_note': 'HMM posterior: two FP32 scalars; constant endpoints and hazard additional shared parameters' if model.arm == 'gru_hmm' else None}


def weights(model, dataset, temperature=1.):
    with torch.no_grad(): return torch.softmax(model.sequence(dataset[0])/temperature, -1).numpy()


def calibrate(model, dataset, rows, params, temperatures):
    trials = []
    for t in temperatures:
        w = weights(model, dataset, t)
        trials.append({'temperature': t, 'nll': expected_nll(w@np.asarray(params['endpoints']), [r['targets'] for r in rows])})
    return min(trials, key=lambda r: r['nll'])['temperature'], trials


def forecasts(rows, arm, params, setting=None):
    setting = setting or {}
    if arm == 'oracle':
        return np.array([r['targets'] for r in rows]), np.full((len(rows), 33, 2), .5)
    return baseline_forecasts(rows, 'hmm_plugin' if arm == 'hmm' else arm, params, setting)


def calibration_bins(p, q):
    p, q = np.asarray(p).ravel(), np.asarray(q).ravel()
    bins = []
    for i in range(10):
        mask = (np.minimum((p*10).astype(int), 9) == i)
        bins.append({'low': i/10, 'high': (i+1)/10, 'count': int(mask.sum()),
                     'prediction': float(p[mask].mean()) if mask.any() else None,
                     'target_probability': float(q[mask].mean()) if mask.any() else None})
    ece = sum(b['count']*abs(b['prediction']-b['target_probability']) for b in bins if b['count'])/len(p)
    return {'bins': bins, 'expected_ece': ece, 'expected_brier': float(((p-q)**2+q*(1-q)).mean()),
            'nll': expected_nll(p, q), 'oracle_entropy': expected_nll(q, q),
            'excess_nll': expected_nll(p, q)-expected_nll(q, q),
            'note': 'Expected calibration against simulator probabilities, not independent Bernoulli samples; descriptive decision/exposure weighting.'}


def evaluate(rows, arm, params, qs, ws):
    results = []
    for i, stream in enumerate(rows):
        for n in LEVELS:
            endpoints = params['endpoints'] if arm not in ('stationary_beta', 'oracle') else [qs[i, n].tolist()]*2
            result = rollout_probe(stream, n, qs[i, n], 'hmm_mixture_dp', {'endpoints': endpoints}, ws[i, n])
            result['arm'] = arm
            result['initial_joint_weights'] = ws[i, n].tolist()
            result['planner_endpoints'] = endpoints
            results.append(result)
    p = qs[:, LEVELS, :]
    q = np.array([r['targets'] for r in rows])[:, LEVELS, :]
    return {'rows': results, 'calibration': calibration_bins(p, q)}
