"""DEV-053B: equal public observations, actual MTSK vs GRU/Bayesian adaptation.

Calibration fixtures deliberately reveal *public final verifier* success of a
known-fault staged attempt. No latent rates/regimes enter controller inputs.
This tests dynamics identification, not learned addressing, CDL or CASM.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import math
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .learned_belief_world import make_world, posterior, bellman
from .full_plm_reference import FullPLMConfig, SwiLAMTSK

HARD = np.array([.25, .35, .45])
EASY = np.array([.9, .85, .95])
LEVELS = (0, 4, 8, 16, 32)
NEURAL = ('gru32', 'gru128', 'mtsk', 'mtsk_frozen_kernel', 'mtsk_reset')
BASELINES = ('oracle', 'prior', 'stationary_beta', 'discounted_beta', 'hmm_plugin', 'hmm_mixture_dp')
ARMS = BASELINES+NEURAL


def rng_for(*parts):
    return random.Random(':'.join(map(str, ('dev053b-v1',)+parts)))


def rate_path(seed, index, kind, length=32):
    rng = rng_for('path', seed, index)
    family = kind if kind in ('markov', 'drift') else ('markov' if rng.random() < .5 else 'triangle')
    phase = rng.random()
    z = int(phase >= .5)
    period = rng.choice((12, 20)) if family == 'drift' else rng.choice((20, 28, 36))
    rates = []
    for step in range(length+1):
        if family == 'markov':
            if step and rng.random() < .08:
                z = 1-z
            amount = float(z)
        elif family == 'triangle':
            amount = 1-abs(2*((phase+step/period) % 1)-1)
        else:
            amount = .5+.5*math.sin(2*math.pi*(phase+step/period))
        rates.append((HARD+amount*(EASY-HARD)).tolist())
    return rates, family


def native_episode(seed, index, level, rates, kind='probe', attempt=None):
    """Harness-only generator. Preserve pinned runtime, replace authored effects."""
    from investigation_world.operational.models import StateAssertion
    ep, _, _ = make_world(seed, index*64+level)
    identity = 'TAC-DEV053B-'+hashlib.sha256(f'{kind}:{seed}:{index}:{level}'.encode()).hexdigest()[:24]
    ep.episode_id = ep.world_id = ep.task.task_id = ep.task.world_id = ep.oracle.task_id = identity
    rng = rng_for('native', kind, seed, index, level)
    initial = {'plant.slot': -1, 'plant.collateral': 0}
    effects = [e for e in ep.oracle.action_effects if e.action_name != 'service']
    for entity in (0, 1):
        fault = rng.randrange(2) if kind == 'probe' else int(entity == 0)
        initial[f'e{entity}.fault'] = fault
        initial[f'e{entity}.attempt'] = attempt-1 if kind == 'coupon' and entity == 0 else 0
        ep.records[entity].fields = {'entity': entity, 'reported_fault': rng.randrange(2)}
        template = next(e for e in ep.oracle.action_effects if e.action_name == 'service' and e.required_parameters == {'entity': entity, 'attempt': 1})
        for a, q in enumerate(rates, 1):
            success = rng.random() < q
            if kind == 'coupon' and entity == 0:
                next_fault = int(not success)  # Staged independent known-fault fixture.
            else:
                fault *= int(not success)
                next_fault = fault
            effects.append(template.model_copy(update={
                'required_parameters': {'entity': entity, 'attempt': a},
                'required_state': [StateAssertion(object_id='plant', field_name='slot', expected_value=entity),
                                   StateAssertion(object_id=f'e{entity}', field_name='attempt', expected_value=a-1)],
                'set_state': {f'e{entity}.fault': next_fault, f'e{entity}.attempt': a, 'plant.slot': -1}}))
    ep.oracle = ep.oracle.model_copy(update={'initial_state': initial, 'action_effects': effects,
                                             'required_action_counts': {}, 'max_cost': 100, 'max_tool_calls': 24})
    ep.task.metadata = {'max_service_attempts': 3, 'max_inspections': 4, 'fixture': kind == 'coupon'}
    if kind == 'coupon':
        ep.task.metadata.update({'public_known_initial_fault': 1, 'public_staged_attempt': attempt,
                                 'public_second_entity_certified_healthy': True})
    return ep


def coupon(seed, index, step, rates):
    from investigation_world.operational.runtime import OperationalRuntime
    from investigation_world.operational.models import EpisodeSubmission
    attempt = rng_for('attempt', seed, index, step).randrange(1, 4)
    episode = native_episode(seed, index, step, rates, 'coupon', attempt)
    runtime = OperationalRuntime(episode)
    calls = []
    for name, args in [('isolate', {'entity': 0}), ('service', {'entity': 0, 'attempt': attempt})]:
        receipt = dict(runtime.act(name, **args))
        assert receipt['accepted']
        calls.append({'name': name, 'parameters': args, 'receipt': receipt})
    verified = runtime.submit(EpisodeSubmission(conclusion='DEV-053B public calibration fixture'))
    # This is observable final verification, not a hidden-runtime fault read.
    success = int(verified.state == 1)
    return {'attempt': attempt, 'success': success, 'calls': calls,
            'verification': verified.model_dump(mode='json'), 'native_spent': runtime.budget.spent,
            'id': episode.episode_id}


def collect_streams(spec, smoke=False):
    worlds = min(3, spec['worlds']) if smoke else spec['worlds']
    rows = []
    for index in range(worlds):
        rates, family = rate_path(spec['seed'], index, spec['kind'])
        observations = [coupon(spec['seed'], index, t, rates[t]) for t in range(32)]
        tokens = [[float(r['attempt'] == a) for a in (1, 2, 3)]+[float(r['success']), 1.] for r in observations]
        rows.append({'seed': spec['seed'], 'index': index, 'id': f"dev053b-stream:{spec['seed']}:{index}",
                     'family': family, 'tokens': tokens, 'targets': rates, 'observations': observations})
    return rows


def tensors(streams):
    return (torch.tensor([r['tokens'] for r in streams], dtype=torch.float32),
            torch.tensor([r['targets'] for r in streams], dtype=torch.float32))


class AdaptiveModel(nn.Module):
    def __init__(self, seed, arm):
        super().__init__()
        if arm not in NEURAL:
            raise ValueError(arm)
        torch.manual_seed(seed)
        self.arm = arm
        self.hidden = 32 if arm == 'gru32' else 128
        if arm.startswith('gru'):
            self.encoder = nn.Identity()
            self.kernel = nn.GRUCell(5, self.hidden)
            self.head = nn.Linear(self.hidden, 3)
        else:
            cfg = FullPLMConfig(d_model=8, num_heads=2, num_experts=3,
                                fast_experts=1, medium_experts=1, slow_experts=1)
            self.encoder = nn.Sequential(nn.Linear(5, 8), nn.Tanh())
            self.kernel = SwiLAMTSK(cfg)
            self.head = nn.Linear(8, 3)
            if arm == 'mtsk_frozen_kernel':
                self.kernel.requires_grad_(False)

    def initial(self, like):
        b = like.shape[0]
        if self.arm.startswith('gru'):
            return like.new_zeros(b, self.hidden)
        return (self.kernel.initial_state(b, like.device, like.dtype), None, like.new_zeros(b, 8))

    def observe(self, state, token):
        if self.arm.startswith('gru'):
            return self.kernel(token, state)
        memory, resp, _ = state
        if self.arm == 'mtsk_reset':
            memory, resp = torch.zeros_like(memory), None
        read, memory, _, resp = self.kernel._step(self.encoder(token), memory, resp)
        return memory, resp, read

    def predict(self, state):
        return self.head(state if self.arm.startswith('gru') else state[2])

    def sequence(self, tokens):
        state = self.initial(tokens[:, 0])
        outputs = [self.predict(state)]
        for t in range(tokens.shape[1]):
            state = self.observe(state, tokens[:, t])
            outputs.append(self.predict(state))
        return torch.stack(outputs, 1)

    @property
    def state_scalars(self):
        return self.hidden if self.arm.startswith('gru') else 96+24+8


def fit_model(model, data, seed, updates=256):
    x, y = data
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+553)
    curve = []
    start = time.monotonic()
    initial = {k: v.clone() for k, v in model.kernel.state_dict().items()}
    model.completed_updates = 0
    for step in range(updates):
        ids = torch.randint(len(x), (32,), generator=rng)
        logits = model.sequence(x[ids])
        loss = F.binary_cross_entropy_with_logits(logits, y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss update{step+1}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        model.completed_updates = step+1
        if step == 0 or step == updates-1:
            curve.append({'update': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
    unchanged = all(torch.equal(v, model.kernel.state_dict()[k]) for k, v in initial.items())
    assert unchanged == (model.arm == 'mtsk_frozen_kernel')
    model.eval()
    return {'curve': curve, 'updates': updates, 'training_seconds': time.monotonic()-start,
            'kernel_unchanged': unchanged, 'state_bytes': model.state_scalars*4,
            'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters())}


def expected_nll(probabilities, targets):
    p = np.clip(np.asarray(probabilities), 1e-7, 1-1e-7)
    q = np.asarray(targets)
    return float((-q*np.log(p)-(1-q)*np.log1p(-p)).mean())


def fit_parameters(streams):
    targets = np.array([r['targets'] for r in streams])
    markov = np.array([r['targets'] for r in streams if r['family'] == 'markov'])
    z = markov[:, :, 0] > .5
    return {'prior': targets.mean((0, 1)).tolist(),
            'endpoints': [markov[~z].mean(0).tolist(), markov[z].mean(0).tolist()],
            'training_hazard': float((z[:, 1:] != z[:, :-1]).mean())}


def baseline_forecasts(streams, arm, params, setting):
    outputs, weights_all = [], []
    prior = np.asarray(params['prior'])
    endpoints = np.asarray(params['endpoints'])
    for row in streams:
        a = prior*setting.get('strength', 2.)
        b = (1-prior)*setting.get('strength', 2.)
        evidence_a, evidence_b = np.zeros(3), np.zeros(3)
        weights = np.array([.5, .5])
        qs = [endpoints.mean(0) if arm.startswith('hmm') else prior]
        ws = [weights.copy()]
        for token in row['tokens']:
            attempt, success = int(np.argmax(token[:3])), token[3]
            if arm.startswith('hmm'):
                weights *= endpoints[:, attempt] if success else 1-endpoints[:, attempt]
                weights /= weights.sum()
                h = setting.get('hazard', params['training_hazard'])
                weights = weights*(1-h)+weights[::-1]*h
                prediction = weights@endpoints
            elif arm == 'prior':
                prediction = prior
            else:
                if arm == 'discounted_beta':
                    evidence_a *= setting['discount']; evidence_b *= setting['discount']
                evidence_a[attempt] += success
                evidence_b[attempt] += 1-success
                prediction = (a+evidence_a)/(a+b+evidence_a+evidence_b)
            qs.append(prediction.copy()); ws.append(weights.copy())
        outputs.append(qs); weights_all.append(ws)
    return np.array(outputs), np.array(weights_all)


def calibrate_baseline(streams, arm, params, proto):
    targets = [r['targets'] for r in streams]
    if arm == 'stationary_beta':
        choices = [{'strength': v} for v in proto['calibration']['beta_prior_strengths']]
    elif arm == 'discounted_beta':
        choices = [{'strength': 2., 'discount': v} for v in proto['calibration']['discount_factors']]
    elif arm.startswith('hmm'):
        choices = [{'hazard': v} for v in proto['calibration']['hmm_hazards']]
    else:
        choices = [{}]
    trials = [{'setting': setting, 'nll': expected_nll(baseline_forecasts(streams, arm, params, setting)[0], targets)} for setting in choices]
    chosen = min(trials, key=lambda r: r['nll'])
    return chosen['setting'], trials


def calibrate_model(model, streams, proto):
    x, y = tensors(streams)
    with torch.no_grad():
        logits = model.sequence(x)
    trials = [{'temperature': t, 'nll': float(F.binary_cross_entropy_with_logits(logits/t, y))}
              for t in proto['calibration']['neural_temperatures']]
    return min(trials, key=lambda r: r['nll'])['temperature'], trials


def mixture_update(joint, command, rates, reading=None):
    joint = np.asarray(joint).reshape(2, 2)
    if command == 'service':
        updated = joint.copy()
        updated[:, 0] += joint[:, 1]*rates
        updated[:, 1] *= 1-rates
        return updated
    likelihood = np.array([.15, .85]) if reading else np.array([.85, .15])
    updated = joint*likelihood[None]
    return updated/updated.sum()


def mixture_values(joint, attempts, senses, endpoints):
    """Exact entity-local joint latent-regime/fault finite-horizon DP."""
    rates = np.asarray(endpoints)
    @lru_cache(maxsize=None)
    def recurse(key, a, s):
        j = np.array(key).reshape(2, 2)
        p = float(j[:, 1].sum())
        choices = {'done': 4*p}
        if a < 3:
            nxt = mixture_update(j, 'service', rates[:, a])
            choices['service'] = 1.2+recurse(tuple(np.round(nxt.ravel(), 8)), a+1, s)[0]
        if s < 4:
            py = .15+.7*p
            pos = mixture_update(j, 'inspect', None, 1)
            neg = mixture_update(j, 'inspect', None, 0)
            choices['inspect'] = .1+py*recurse(tuple(np.round(pos.ravel(), 8)), a, s+1)[0]+(1-py)*recurse(tuple(np.round(neg.ravel(), 8)), a, s+1)[0]
        command = min(choices, key=lambda k: (choices[k], {'inspect': 0, 'service': 1, 'done': 2}[k]))
        return choices[command], command
    j = np.asarray(joint)
    p = float(j[:, 1].sum())
    values = {'done': 4*p}
    if attempts < 3:
        nextj = mixture_update(j, 'service', rates[:, attempts])
        values['service'] = 1.2+recurse(tuple(np.round(nextj.ravel(), 8)), attempts+1, senses)[0]
    if senses < 4:
        py = .15+.7*p
        pos = mixture_update(j, 'inspect', None, 1)
        neg = mixture_update(j, 'inspect', None, 0)
        values['inspect'] = .1+py*recurse(tuple(np.round(pos.ravel(), 8)), attempts, senses+1)[0]+(1-py)*recurse(tuple(np.round(neg.ravel(), 8)), attempts, senses+1)[0]
    return values


class NativeProbe:
    """Harness only. Controllers below use probabilities, counters and receipts."""
    def __init__(self, row, level):
        from investigation_world.operational.runtime import OperationalRuntime
        self.seed, self.index, self.level = row['seed'], row['index'], level
        self.episode = native_episode(self.seed, self.index, level, row['targets'][level])
        self.runtime = OperationalRuntime(self.episode)
        self.reads = [0, 0]
        self.calls = []

    def act(self, name, entity, attempt=None):
        args = {'entity': entity}
        if attempt is not None: args['attempt'] = attempt
        receipt = dict(self.runtime.act(name, **args))
        if name == 'inspect':
            true = int(self.runtime.state[f'e{entity}.fault'])
            rng = random.Random(f'dev053b:sensor:{self.seed}:{self.index}:{self.level}:{entity}:{self.reads[entity]}')
            receipt['fault'] = true if rng.random() < .85 else 1-true
            self.reads[entity] += 1
        self.calls.append({'name': name, 'parameters': args, 'receipt': receipt})
        return receipt

    def finish(self):
        from investigation_world.operational.models import EpisodeSubmission
        failures = sum(int(self.runtime.state[f'e{e}.fault']) for e in (0, 1))
        violations = int(self.runtime.state['plant.slot'] != -1)+int(self.runtime.state['plant.collateral'] != 0)
        verification = self.runtime.submit(EpisodeSubmission(conclusion='DEV-053B maintenance probe'))
        return {'seed': self.seed, 'index': self.index, 'level': self.level, 'world': self.episode.episode_id,
                'cost': self.runtime.budget.spent/10+4*failures,
                'total_cost_including_observations': self.runtime.budget.spent/10+4*failures+1.2*self.level,
                'failures': failures, 'complete': int(verification.state == 1),
                'native_spent': self.runtime.budget.spent, 'blocked': sum(e.blocked for e in self.runtime.events),
                'violations': violations, 'calls': self.calls, 'verification': verification.model_dump(mode='json')}


def rollout_probe(row, level, forecasts, arm, params, weights=None):
    world = NativeProbe(row, level)
    rates = np.asarray(forecasts, dtype=float)
    assert rates.shape == (3,) and np.all((rates >= 0) & (rates <= 1))
    zweights = np.asarray(weights) if weights is not None else np.array([.5, .5])
    decisions = []
    start = time.monotonic()
    # No hidden runtime state or targets are consulted by the controller loop.
    for entity in (0, 1):
        p, attempts, senses = .5, 0, 0
        joint = np.stack((zweights*.5, zweights*.5), axis=1)
        for _ in range(8):
            if arm == 'hmm_mixture_dp':
                values = mixture_values(joint, attempts, senses, params['endpoints'])
                p = float(joint[:, 1].sum())
            else:
                values = {'done': 4*p}
                if attempts < 3:
                    values['service'] = 1.2+bellman(p*(1-rates[attempts]), attempts+1, senses, (.85, tuple(rates)))[0]
                if senses < 4:
                    py = .15+.7*p
                    values['inspect'] = .1+py*bellman(posterior(p, 1, .85), attempts, senses+1, (.85, tuple(rates)))[0]+(1-py)*bellman(posterior(p, 0, .85), attempts, senses+1, (.85, tuple(rates)))[0]
            ordered = sorted(values, key=lambda k: (values[k], {'inspect': 0, 'service': 1, 'done': 2}[k]))
            command = ordered[0]
            decisions.append({'entity': entity, 'command': command, 'p_fault': p, 'attempts': attempts,
                              'senses': senses, 'values': values,
                              'top2_gap': values[ordered[1]]-values[ordered[0]] if len(ordered) > 1 else None})
            if command == 'done':
                if arm == 'hmm_mixture_dp': zweights = joint.sum(1)
                break
            if command == 'inspect':
                receipt = world.act('inspect', entity)
                p = posterior(p, receipt['fault'], .85)
                if arm == 'hmm_mixture_dp': joint = mixture_update(joint, 'inspect', None, receipt['fault'])
                senses += 1
            else:
                assert attempts < 3
                assert world.act('isolate', entity)['accepted']
                assert world.act('service', entity, attempts+1)['accepted']
                if arm == 'hmm_mixture_dp': joint = mixture_update(joint, 'service', np.asarray(params['endpoints'])[:, attempts])
                p *= 1-rates[attempts]
                attempts += 1
        else:
            raise AssertionError('bounded native action horizon exceeded')
    elapsed = time.monotonic()-start
    result = world.finish()
    result.update({'arm': arm, 'forecast_rates': rates.tolist(), 'decisions': decisions,
                   'decision_and_native_seconds': elapsed,
                   'expected_forecast_nll': expected_nll(rates, row['targets'][level])})
    return result


def evaluate(streams, predictions, arm, params, weights=None):
    rows = []
    for i, stream in enumerate(streams):
        for n in LEVELS:
            rows.append(rollout_probe(stream, n, predictions[i, n], arm, params, weights[i, n] if weights is not None else None))
    return {'rows': rows, 'summary': summarize(rows)}


def summarize(rows):
    return {str(n): {'worlds': len(group), 'cost_per_entity': sum(r['cost'] for r in group)/(2*len(group)),
                     'total_cost_per_entity': sum(r['total_cost_including_observations'] for r in group)/(2*len(group)),
                     'residual_faults': sum(r['failures'] for r in group),
                     'complete_worlds': sum(r['complete'] for r in group),
                     'blocked': sum(r['blocked'] for r in group), 'violations': sum(r['violations'] for r in group),
                     'forecast_expected_nll': float(np.mean([r['expected_forecast_nll'] for r in group])),
                     'native_calls': sum(len(r['calls']) for r in group),
                     'decision_and_native_seconds': sum(r['decision_and_native_seconds'] for r in group)}
            for n in LEVELS if (group := [r for r in rows if r['level'] == n])}
