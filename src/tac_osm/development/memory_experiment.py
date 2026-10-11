"""Development protocol and deterministic streaming experiment.

The first query uses the existing noisy-alias benchmark. A separate follow-up
tests a post-action write carried into a later query. That observation's value
is visible at write time only; it is never given to the later query. Its
continuous reliability evidence is independent of the target bit. This is a
synthetic sensor-validity verifier, not learned verification of general tasks.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from tac_osm.integration_002_benchmark import Episode, make_pair, similarity_prediction, visible_inputs
from tac_osm.integration_002_learned_dev import bounded_cache
from .noisy_memory import MemoryConfig, NoisyMemoryModel, parameter_hash

EXPERIMENT_ID = "INTEGRATED-MEMORY-DEV-003"
TRAIN_SEEDS = (43, 59, 71)
EVAL_SEEDS = (131, 149, 167)
TRAIN_STEPS = 180
TRAIN_BATCH = 8
EVAL_PAIRS = 2
H_LEVELS = (32, 128, 512)
D_LEVELS = (1, 16)
FRACTIONS = (0.0, 1.0)
NOISES = (0.08, 0.25)


def data_seed(namespace: str, *parts: object) -> int:
    raw = json.dumps([EXPERIMENT_ID, namespace, *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def episode_fingerprint(ep: Episode) -> str:
    return hashlib.sha256(json.dumps(asdict(ep), sort_keys=True).encode()).hexdigest()


def tensors(episodes: Sequence[Episode], *, include_writes: bool = True) -> tuple[Tensor, ...]:
    views = [visible_inputs(ep) for ep in episodes]
    rows = [v['history'] + (v['writes'] if include_writes else []) for v in views]
    return (
        torch.tensor([[r['address'] for r in x] for x in rows], dtype=torch.float32),
        torch.tensor([[r['value'] for r in x] for x in rows], dtype=torch.long),
        torch.tensor([v['query_address'] for v in views], dtype=torch.float32),
        torch.tensor([v['current_bit'] for v in views], dtype=torch.long),
    )


def followup(seed: int, count: int = 2) -> dict[str, Tensor]:
    """Same later present, opposite post-action past; no query payload leak."""
    if count < 2 or count % 2:
        raise ValueError("follow-up requires complete pairs")
    rng = random.Random(seed)
    address, query, value, bits, evidence, targets = [], [], [], [], [], []
    for _ in range(count//2):
        latent = [rng.gauss(0, 1) for _ in range(12)]
        write_address = [v+rng.gauss(0, .08) for v in latent]
        query_address = [v+rng.gauss(0, .08) for v in latent]
        current = rng.randrange(2)
        valid_evidence = rng.gauss(1, .2)
        for bit in (0, 1):
            address.append(write_address)
            query.append(query_address)
            value.append(bit)
            bits.append(current)
            evidence.append(valid_evidence)
            targets.append(bit ^ current)
    return {
        'address': torch.tensor(address, dtype=torch.float32),
        'value': torch.tensor(value),
        'query': torch.tensor(query, dtype=torch.float32),
        'current': torch.tensor(bits),
        'evidence': torch.tensor(evidence, dtype=torch.float32),
        'target': torch.tensor(targets),  # loss/evaluator only
    }


def train_model(seed: int, *, steps: int = TRAIN_STEPS) -> tuple[NoisyMemoryModel, dict]:
    if not 1 <= steps <= TRAIN_STEPS:
        raise ValueError("development step budget out of range")
    torch.manual_seed(seed)
    model = NoisyMemoryModel()
    initial = parameter_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    losses, fingerprints = [], set()
    names = ('key', 'erase', 'write', 'read_selector', 'operator_selector', 'verifier')
    gradient_seen = {n: False for n in names}
    future_write_gradient = False
    for step in range(steps):
        episodes = []
        for j in range(TRAIN_BATCH//2):
            p = make_pair(data_seed('train', seed, step, j), 32, (1,16)[step%2],
                          FRACTIONS[(step+j)%2], NOISES[(step//2+j)%2])
            episodes.extend((p.left, p.right))
        fingerprints.update(episode_fingerprint(e) for e in episodes)
        addresses, values, query, current = tensors(episodes)
        state = model.encode(addresses, values)
        logits = model.predict(state, query, current)
        labels = torch.tensor([e.target for e in episodes])
        observation = followup(data_seed('train-followup', seed, step), TRAIN_BATCH)
        # Both partners start with exactly the same pre-commit memory. This
        # makes the later contrast isolate the new write, not older history.
        base_state = state[::2].repeat_interleave(2, dim=0).detach()
        carried, _ = model.verified_write(base_state, observation['address'], observation['value'],
                                         observation['evidence'])
        future_logits = model.predict(carried, observation['query'], observation['current'])
        future_loss = F.cross_entropy(future_logits, observation['target'])
        # Verify actual credit from the future decision, independent of the
        # separate verifier supervision objective.
        future_grad = torch.autograd.grad(future_loss, model.memory.write.weight,
                                         retain_graph=True)[0]
        future_write_gradient |= bool(torch.isfinite(future_grad).all() and future_grad.abs().sum() > 0)
        # Validity evidence is generated independently of bit correctness.
        evidence = observation['evidence'].clone()
        validity = torch.arange(TRAIN_BATCH) % 2
        evidence = torch.where(validity.bool(), evidence, -evidence)
        _, reliability_logits = model.verified_write(base_state, observation['address'], observation['value'], evidence)
        verifier_loss = F.binary_cross_entropy_with_logits(reliability_logits, validity.float())
        loss = F.cross_entropy(logits, labels) + future_loss + .2*verifier_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        tracked = (model.memory.key.weight, model.memory.erase.weight,
                   model.memory.write.weight, model.read_selector[0].weight,
                   model.compute.selector[0].weight, model.verifier[0].weight)
        for name, parameter in zip(names, tracked):
            grad = parameter.grad
            gradient_seen[name] |= bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model.eval(), {
        'seed': seed, 'steps': steps, 'initial_hash': initial, 'final_hash': parameter_hash(model),
        'nonzero_finite_task_gradients': gradient_seen,
        'future_decision_write_gradient': future_write_gradient,
        'loss_first': losses[0], 'loss_last': losses[-1], 'loss_trajectory': losses,
        'train_fingerprints': sorted(fingerprints),
    }


def compute_trace(model: NoisyMemoryModel) -> list[dict]:
    return [asdict(x) for x in model.compute.last_trace]


@torch.no_grad()
def measure_pair(model: NoisyMemoryModel, seed: int, h: int, d: int,
                 fraction: float, noise: float) -> dict:
    if model.training:
        raise ValueError("evaluation requires model.eval()")
    pair = make_pair(seed, h, d, fraction, noise)
    episodes = (pair.left, pair.right)
    addresses, values, query, current = tensors(episodes)
    state = model.encode(addresses, values)
    writes = [asdict(x) for x in model.last_writes]
    learned = model.predict(state, query, current).argmax(-1).tolist()
    dispatch = compute_trace(model)
    swapped = model.predict(state.flip(0), query, current).argmax(-1).tolist()
    reset_state = torch.zeros_like(state)
    reset = model.predict(reset_state, query, current).argmax(-1).tolist()
    nowrite_address, nowrite_value, _, _ = tensors(episodes, include_writes=False)
    no_write_state = model.encode(nowrite_address, nowrite_value)
    no_write = model.predict(no_write_state, query, current).argmax(-1).tolist()
    euclidean = [similarity_prediction(e, metric='euclidean') for e in episodes]
    cache_results = []
    for ep in episodes:
        cache = bounded_cache(ep, include_writes=True)
        entries = cache.entries
        index = min(range(len(entries)), key=lambda i:sum(
            (a-b)**2 for a,b in zip(ep.query_address, entries[i]['address'])))
        cache_results.append(entries[index]['value'] ^ ep.current_bit)
    observation = followup(data_seed('eval-followup', seed))
    base_state = state[:1].expand_as(state)
    carried, reliability = model.verified_write(base_state, observation['address'], observation['value'],
                                                observation['evidence'])
    commit_events = [asdict(x) for x in model.last_writes]
    follow_logits = model.predict(carried, observation['query'], observation['current'])
    follow_prediction = follow_logits.argmax(-1).tolist()
    follow_dispatch = compute_trace(model)
    follow_no_commit = model.predict(base_state, observation['query'], observation['current']).argmax(-1).tolist()
    follow_swap = model.predict(carried.flip(0), observation['query'], observation['current']).argmax(-1).tolist()
    follow_reset = model.predict(reset_state, observation['query'], observation['current']).argmax(-1).tolist()
    rejected, bad_reliability = model.verified_write(base_state, observation['address'], observation['value'],
                                                    -observation['evidence'])
    reject_events = [asdict(x) for x in model.last_writes]
    return {
        'seed': seed, 'H': h, 'D': d, 'hard_fraction': fraction, 'noise': noise,
        'fingerprints': [episode_fingerprint(e) for e in episodes],
        'target': [e.target for e in episodes],
        'predictions': {'learned':learned, 'swapped':swapped, 'reset':reset,
                        'no_intervening_write':no_write, 'euclidean':[x[0] for x in euclidean],
                        'nonlearned_cache':cache_results},
        'euclidean_indices': [x[1] for x in euclidean], 'oracle_slots':[e.target_slot for e in episodes],
        'followup': {'target':observation['target'].tolist(), 'prediction':follow_prediction,
                     'no_commit':follow_no_commit, 'swapped':follow_swap, 'reset':follow_reset,
                     'reliability_probabilities':reliability.sigmoid().tolist(),
                     'negative_probabilities':bad_reliability.sigmoid().tolist(),
                     'rejected_state_identity':bool(torch.equal(base_state, rejected))},
        'work': {'state_elements':state[0].numel(), 'history_writes':writes,
                 'commit':commit_events, 'reject':reject_events,
                 'query_experts':model.config.experts, 'dispatch':dispatch,
                 'followup_dispatch':follow_dispatch, 'dense_candidates':h+d},
    }


def evaluate(model: NoisyMemoryModel, seed: int, *, pairs: int = EVAL_PAIRS) -> list[dict]:
    if pairs < 1:
        raise ValueError("must evaluate complete pairs")
    return [measure_pair(model, data_seed('eval',seed,h,d,f,n,j), h,d,f,n)
            for h in H_LEVELS for d in D_LEVELS for f in FRACTIONS
            for n in NOISES for j in range(pairs)]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty evaluation")
    arms = rows[0]['predictions'].keys()
    total = 2*len(rows)
    accuracy = {arm:sum(int(y == t) for r in rows for y,t in zip(r['predictions'][arm],r['target']))/total
                for arm in arms}
    follow_accuracy = {arm:sum(int(y==t) for r in rows for y,t in
                      zip(r['followup'][arm],r['followup']['target']))/total
                       for arm in ('prediction','no_commit','swapped','reset')}
    return {'paired_rows':len(rows), 'episodes':total, 'accuracy':accuracy,
            'followup_accuracy':follow_accuracy,
            'euclidean_recall_at_1':sum(int(i==t) for r in rows for i,t in
                                       zip(r['euclidean_indices'],r['oracle_slots']))/total,
            'state_elements':sorted({r['work']['state_elements'] for r in rows}),
            'all_negative_writes_rejected':all(r['followup']['rejected_state_identity'] for r in rows),
            'operator_sample_calls':sum(sum(len(ids) for _,ids in depth['dispatched'])
                                        for r in rows for depth in r['work']['dispatch']),
            'no_positive_research_claim':True}
