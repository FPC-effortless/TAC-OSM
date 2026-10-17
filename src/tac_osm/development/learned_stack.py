"""DEV-055: learned outcome model; evaluator-only synthetic physical world.

CDL-style routing and CASM neural operators reuse existing local substrates.
This is not original CDL teacher distillation or symbolic program discovery.
Only PublicEvent crosses the runtime model boundary; training labels are
provided separately. No analytic fault transition enters the neural model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .full_plm_reference import FullPLMConfig, SwiLAMTSK, LearnedCDL, LearnedStructuralWorkspace
from .conditional_compute import ConditionalExecutor

COSTS = np.array([0., .6, .6, .15, .15])
ARMS = ("mtsk", "gru_matched", "mtsk_reset", "uniform_route", "no_casm")


@dataclass(frozen=True)
class PublicEvent:
    action: int
    sensors: tuple[int, int]
    exact: tuple[int, int]
    tick: int

    def tensor(self):
        return [float(self.action == a) for a in range(5)] + list(self.sensors) + list(self.exact) + [self.tick / 20.]


class PhysicalWorld:
    """Evaluator capability. Do not pass to a policy or predictor."""
    def __init__(self, seed, index, family="coupling"):
        self.seed, self.index, self.family = seed, index, family
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        self.bits = (self.rng.random(2) < .65).astype(int)
        self.regime = int(self.rng.integers(2))
        self.tick = 0
        self.phase = float(self.rng.uniform(0, 2*math.pi))
        self.last = self.observe(0)

    def rates(self):
        z = float(self.regime) if self.family != "drift" else .5+.5*math.sin(self.phase+self.tick/3.)
        return .9-.65*z, .02+.43*z

    def expected(self, action):
        p = self.bits.astype(float).copy()
        if action in (1, 2):
            entity = action-1
            repair, collateral = self.rates()
            p[entity] *= 1-repair
            p[1-entity] += (1-p[1-entity])*collateral
        return p

    def observe(self, action):
        # Paired noise indexed by world tick and channel, independent of policy.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.index, self.tick, 991]))
        sensors = np.where(rng.random(2) < .8, self.bits, 1-self.bits)
        exact = np.zeros(2, dtype=int)
        if action in (3, 4):
            entity = action-3
            sensors[entity], exact[entity] = self.bits[entity], 1
        return PublicEvent(action, tuple(map(int, sensors)), tuple(map(int, exact)), self.tick)

    def step(self, action):
        if type(action) is not int or action not in range(5):
            raise ValueError("invalid public action")
        old_regime, before = self.regime, self.bits.copy()
        # All candidate actions share the same fixed physical random draw.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.index, self.tick, 771]))
        self.bits = (rng.random(2) < self.expected(action)).astype(int)
        physical_cost = float(COSTS[action] + 4*self.bits.sum())
        if rng.random() < .1:
            self.regime = 1-self.regime
        self.tick += 1
        self.last = self.observe(action)
        return self.last, physical_cost, {"before": before.tolist(), "after": self.bits.tolist(), "regime": old_regime}


def collect(seed, worlds, length=12, family="train"):
    events, actions, targets, labels = [], [], [], []
    for index in range(worlds):
        # Training includes discrete coupled regimes; drift is genuinely held out.
        world = PhysicalWorld(seed, index, family)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 331]))
        es, ac, ys, ls = [], [], [], []
        for _ in range(length):
            es.append(world.last.tensor())
            action = int(rng.integers(5))
            _, _, label = world.step(action)
            ac.append(action); ys.append(label["after"]); ls.append(label)
        events.append(es); actions.append(ac); targets.append(ys); labels.append(ls)
    return (torch.tensor(events, dtype=torch.float32), torch.tensor(actions),
            torch.tensor(targets, dtype=torch.float32)), labels


class LearnedStack(nn.Module):
    def __init__(self, seed=0, arm="mtsk", gru_hidden=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown arm")
        torch.manual_seed(seed)
        self.arm = arm
        self.cfg = FullPLMConfig(d_model=8, num_heads=2, num_experts=3,
                                 fast_experts=1, medium_experts=1, slow_experts=1,
                                 cdl_top_k=2, casm_depth=2)
        self.encoder = nn.Sequential(nn.Linear(10, 8), nn.Tanh())
        self.query = nn.Sequential(nn.Linear(6, 8), nn.Tanh())
        if arm == "gru_matched":
            self.hidden = gru_hidden or matched_hidden()
            self.memory = nn.GRUCell(8, self.hidden)
            self.views = nn.Linear(self.hidden, 24)
        else:
            self.memory = SwiLAMTSK(self.cfg)
            # These existing optional paths are not called in this study.
            for module in (self.memory.query_prior, self.memory.out_proj, self.memory.write_proj):
                for p in module.parameters():
                    p.requires_grad_(False)
        self.router = LearnedCDL(self.cfg)
        self.workspace = LearnedStructuralWorkspace(self.cfg)
        self.compute = ConditionalExecutor(8, operators=3, top_k=1, depth=2, halt_threshold=.95)
        self.head = nn.Linear(16, 2)
        if arm == "uniform_route":
            for p in self.router.parameters(): p.requires_grad_(False)
        if arm == "no_casm":
            for p in self.compute.parameters(): p.requires_grad_(False)
        self.last_work = {}

    def initial(self, reference):
        if self.arm == "gru_matched":
            return reference.new_zeros(reference.shape[0], self.hidden), None
        return self.memory.initial_state(reference.shape[0], reference.device, reference.dtype), None

    def update(self, state, public):
        if public.ndim != 2 or public.shape[-1] != 10 or not torch.isfinite(public).all():
            raise ValueError("finite public event [B,10] required")
        x = self.encoder(public)
        if self.arm == "gru_matched":
            return self.memory(x, state[0]), None
        if self.arm == "mtsk_reset": state = self.initial(public)
        _, memory, _, responsibilities = self.memory._step(x, state[0], state[1])
        return memory, responsibilities

    def predict(self, state, action, tick):
        query = self.query(torch.cat((F.one_hot(action, 5).float(), tick[:, None]/20.), -1))
        if self.arm == "gru_matched":
            views = self.views(state[0]).view(-1, 3, 8)
        else:
            key = self.memory.q_proj(query).view(-1, 2, 4)
            views = torch.einsum("bhjde,bhe->bhjd", state[0], key).permute(0, 2, 1, 3).reshape(-1, 3, 8)
        if self.arm == "uniform_route":
            selected, weights = views, views.new_full(views.shape[:2], 1/3)
        else:
            selected, weights, _ = self.router(query, views)
        graph, q, _ = self.workspace(query, selected, weights)
        current = (graph*weights[..., None]).sum(1)+q
        calls = 0
        if self.arm != "no_casm":
            current, _ = self.compute(current, query)
            calls = sum(t.operator_sample_calls for t in self.compute.last_trace)
        self.last_work = {"routing_candidates": len(query)*3,
                          "retained": len(query)*selected.shape[1], "operator_calls": calls}
        return self.head(torch.cat((current, query), -1))

    def forward(self, events, actions):
        state = self.initial(events[:, 0])
        states, resp = [], []
        for t in range(events.shape[1]):
            state = self.update(state, events[:, t])
            states.append(state[0])
            if state[1] is not None: resp.append(state[1])
        shape = states[0].shape[1:]
        stacked = torch.stack(states, 1).reshape(-1, *shape)
        r = torch.stack(resp, 1).flatten(0, 1) if resp else None
        return self.predict((stacked, r), actions.flatten(), events[:, :, -1].flatten()*20).view(*actions.shape, 2)

    @property
    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@lru_cache(maxsize=1)
def matched_hidden():
    # Count actual used trainable parameters, without dummy padding.
    with torch.random.fork_rng():
        target = LearnedStack(0, "mtsk").trainable_count
        counts = [(abs(LearnedStack(0, "gru_matched", h).trainable_count-target), h) for h in range(1, 49)]
    error, hidden = min(counts)
    if error/target > .02: raise ValueError("no GRU within registered 2% parameter budget")
    return hidden


class FittedFilter:
    """Train-fitted HMM or deliberately misspecified independent dynamics."""
    def __init__(self, transitions, independent=False):
        self.transitions = transitions
        self.independent = independent
        self.belief = np.tile(np.array([.35**2, .35*.65, .65*.35, .65**2])/2, (2, 1))

    @staticmethod
    def fit(labels, actions):
        counts = np.full((2, 5, 4, 4), .5)
        for i, row in enumerate(labels):
            for t, entry in enumerate(row):
                before = entry["before"][0]*2+entry["before"][1]
                after = entry["after"][0]*2+entry["after"][1]
                counts[entry["regime"], int(actions[i,t]), before, after] += 1
        return counts/counts.sum(-1, keepdims=True)

    def update(self, event, initial=False):
        if not initial:
            prediction = np.einsum("zs,zsn->zn", self.belief, self.transitions[:, event.action])
            self.belief = .9*prediction+.1*prediction[::-1]
        states = np.array([[s//2, s%2] for s in range(4)])
        likelihood = np.ones(4)
        for e in range(2):
            accuracy = 1. if event.exact[e] else .8
            likelihood *= np.where(states[:, e] == event.sensors[e], accuracy, 1-accuracy)
        self.belief *= likelihood
        self.belief /= self.belief.sum()
        if self.independent:
            # Projection removes both latent regime and cross-entity dependence.
            p = self.belief.sum(0) @ states
            marginal = np.prod(np.where(states, p, 1-p), axis=1)
            self.belief = np.tile(marginal/2, (2, 1))

    def predict(self):
        states = np.array([[s//2, s%2] for s in range(4)])
        return np.stack([np.einsum("zs,zsn->n", self.belief, self.transitions[:, a]) @ states for a in range(5)])


def independent_transitions(labels, actions):
    """Fit factorized rates, without the regime or other entity in the input."""
    counts = np.full((2, 5, 2, 2), .5)
    for i, row in enumerate(labels):
        for t, entry in enumerate(row):
            for e in range(2): counts[e, int(actions[i,t]), entry["before"][e], entry["after"][e]] += 1
    rates = counts/counts.sum(-1, keepdims=True)
    result = np.empty((2, 5, 4, 4))
    for a in range(5):
        for s in range(4):
            for n in range(4): result[:,a,s,n] = rates[0,a,s//2,n//2]*rates[1,a,s%2,n%2]
    return result
