"""DEV-029: public explicit-state baseline and validation-only study boundary.

This is a classical task-informed table, NOT learned entity discovery. The
small prediction head is learned; the table does not call an evaluator oracle.
The existing eight-entity interfaces remain deliberately unchanged.
"""
from __future__ import annotations

import itertools
import torch
from torch import nn

ARMS = ('native', 'masked_local', 'schema41', 'gru768', 'delta768', 'event_table')
LEARNING_RATES = (.0005, .002)
CURRICULA = ('fixed512', '128_then512')
BUDGETS = (216, 648, 2160)
SPLITS = ('train', 'validation', 'test')


def domain(split):
    if split not in SPLITS:
        raise ValueError('unknown split')
    return 'dev029-v1-' + split


def history_at(curriculum, step):
    if curriculum not in CURRICULA or not 0 <= step < BUDGETS[-1]:
        raise ValueError('unregistered curriculum or step')
    return 128 if curriculum == '128_then512' and step < 720 else 512


def select_recipes(records):
    """Fail closed until EVERY arm/recipe/seed/budget has validation evidence.

    A smoke run or the single-recipe pilot cannot unlock test selection.
    Selection uses target-bit validation accuracy, then all-query CE, then
    fixed registration order. Test evidence is explicitly rejected.
    """
    expected = set(itertools.product(ARMS, LEARNING_RATES, CURRICULA, (311, 419, 557), BUDGETS))
    indexed = {}
    for r in records:
        if r['split'] != 'validation' or r.get('smoke'):
            raise ValueError('selection accepts full validation records only')
        key = (r['arm'], r['learning_rate'], r['curriculum'], r['seed'], r['updates'])
        if key not in expected or key in indexed:
            raise ValueError('unregistered or duplicate trial')
        if not (0 <= r['primary_accuracy'] <= 1 and 0 <= r['cross_entropy'] < float('inf')):
            raise ValueError('invalid validation metric')
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError('incomplete tuning matrix: test remains locked')
    chosen = {}
    for arm in ARMS:
        candidates = []
        for lr, curriculum in itertools.product(LEARNING_RATES, CURRICULA):
            rows = [indexed[arm, lr, curriculum, seed, 2160] for seed in (311, 419, 557)]
            candidates.append((sum(r['primary_accuracy'] for r in rows) / 3,
                               -sum(r['cross_entropy'] for r in rows) / 3, lr, curriculum))
        # Stable registration order wins exact ties.
        winner = max(enumerate(candidates), key=lambda p: (p[1][0], p[1][1], -p[0]))[1]
        chosen[arm] = {'learning_rate': winner[2], 'curriculum': winner[3], 'updates': 2160}
    return chosen


class EventTablePredictor(nn.Module):
    """5 values/entity + time: current, generation, due, value, queued generation.

    Tensor adapter consumes only the existing lossless public observation
    encoding. Values/time are integer-exact in its registered range. The head
    sees raw sufficient statistics and action/query features, never future bits.
    Known cancellation/addressing rules give this classical baseline extra
    task knowledge; its performance must not be called learned structure.
    """
    arm = 'event_table'
    state_scalars = 41

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.head = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 2))

    def encode(self, events, entities, values):
        state = events.new_zeros((events.shape[0], 41))
        slots = state[:, :40].reshape(-1, 8, 5)
        slots[:, :, 0] = -1
        slots[:, :, 2:5] = -1
        for t in range(events.shape[1]):
            state = self.update(state, events[:, t], entities[:, t], values[:, t])
        return state

    def update(self, state, features, entities, values):
        state = state.clone()
        slots = state[:, :40].reshape(-1, 8, 5)
        tick = (features[:, 12] * 2048).round()
        if bool((tick < state[:, 40]).any()):
            raise ValueError('public observations must be chronological')
        mature = (slots[:, :, 2] >= 0) & (slots[:, :, 2] <= tick[:, None])
        eligible = mature & (slots[:, :, 1] == slots[:, :, 4])
        slots[:, :, 0] = torch.where(eligible, slots[:, :, 3], slots[:, :, 0])
        slots[:, :, 2] = torch.where(mature, -1., slots[:, :, 2])
        for b in range(len(state)):
            e = int(entities[b])
            if not 0 <= e < 8:
                raise ValueError('eight-entity interface required')
            kind = int(features[b, 8:12].argmax())
            if kind in (0, 1):
                slots[b, e, 0] = values[b]
            if kind == 1:
                slots[b, e, 1] += 1
                slots[b, e, 2] = -1  # invalidated schedules no longer block reissue
            if kind == 2:
                if slots[b, e, 2] >= 0:
                    raise ValueError('multiple live schedules exceed registered table contract')
                slots[b, e, 2] = (features[b, 13] * 2048).round()
                slots[b, e, 3] = values[b]
                slots[b, e, 4] = slots[b, e, 1]
        state[:, 40] = tick
        return state

    def decode(self, state, queries):
        slots = state[:, :40].reshape(-1, 8, 5)
        scale = state.new_tensor([1., 2048., 2048., 1., 2048.])
        table = (slots / scale)[:, :, None, :].expand(-1, -1, 3, -1)
        clock = (state[:, 40] / 2048)[:, None, None, None].expand(-1, 8, 3, 1)
        # Per-entity head shares weights; no oracle predictions or entity one-hot.
        return self.head(torch.cat((table, clock, queries[:, :, :, 8:14]), dim=-1))
