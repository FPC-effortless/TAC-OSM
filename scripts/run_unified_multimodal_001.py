#!/usr/bin/env python3
"""Registered unified native-model benchmark for language, image and audio.

The benchmark is synthetic by design. It asks whether one causal predictive
substrate can learn held-out sequence/image/audio transitions and reuse only
verified experience. It is not a natural-world capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tac_osm.contract import load_contract

TRAIN_WORLD_SEED = 17001
VAL_WORLD_SEED = 17002
TEST_WORLD_SEED = 17003
MODALITIES = ("language", "image", "audio")
VOCAB = 48
SEQ = 5
N_OPS = 4
LATENT = 32
SLOTS = 4
SLOT_DIM = 8
TRAIN_STEPS = 420
BATCH_SIZE = 18
SEEDS = (0, 1, 2, 3, 4)
VERIFY_LATENT_MSE = 0.25
ANTI_COLLAPSE_FLOOR = 1e-3


@dataclass(frozen=True)
class Example:
    modality: str
    state_id: int
    x: object
    action: int
    y: object
    semantic: Tuple[int, int, int]


class World:
    def __init__(self, seed: int, n: int, split: str):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.split = split
        self.residue = {"train": 2, "val": 1, "test": 0}[split]

    def semantic(self, _i: int) -> Tuple[int, int, int]:
        # Domain holdout is transition-invariant: actions never modify c.
        # Train/validation/test therefore have disjoint target-state domains,
        # not merely disjoint source examples.
        c_values = {"train": (0, 1), "val": (2,), "test": (3,)}[self.split]
        a = int(self.rng.integers(0, 8))
        b = int(self.rng.integers(0, 8))
        c = int(self.rng.choice(c_values))
        return a, b, c

    @staticmethod
    def transition(s, op):
        a, b, c = s
        if op == 0:
            a = (a + 1) % 8
        elif op == 1:
            b = (b + 1) % 8
        elif op == 2:
            a = (a + 2) % 8
        elif op == 3:
            a = (a + 1) % 8
            b = (b + 2) % 8
        else:
            raise ValueError(op)
        return a, b, c

    def render(self, modality, s):
        a, b, c = s
        if modality == "language":
            return np.array(
                [1 + a, 12 + b, 24 + c, 32 + ((a + b) % 8), 40 + ((b + c) % 8)],
                dtype=np.int64,
            )
        if modality == "image":
            # Injective synthetic rendering: independent spatial codes for a,
            # b and c. No modulo folding is allowed because that would make
            # different latent states observationally identical.
            img = np.zeros((8, 8), dtype=np.float32)
            img[0, a] = 1.0
            img[b, 0] += 0.6
            img[7, 1 + c] += 0.25
            img[2 + (a + b) % 5, 7] += 0.15
            return img.reshape(-1)
        t = np.arange(64, dtype=np.float32)
        # Two independent harmonics encode a and b; a third tone encodes c.
        # Frequencies are separated so the state map is injective at 64 samples.
        f_a = 2 + a
        f_b = 18 + b
        f_c = 28 + c
        return (
            0.55 * np.sin(2 * np.pi * f_a * t / 64)
            + 0.30 * np.sin(2 * np.pi * f_b * t / 64 + 0.17)
            + 0.10 * np.sin(2 * np.pi * f_c * t / 64 + 0.31)
        ).astype(np.float32)

    def make(self):
        rows = []
        for i in range(self.n):
            s = self.semantic(i)
            op = int(self.rng.integers(N_OPS))
            ns = self.transition(s, op)
            for modality in MODALITIES:
                rows.append(
                    Example(
                        modality,
                        i,
                        self.render(modality, s),
                        op,
                        self.render(modality, ns),
                        s,
                    )
                )
        return rows


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_audit(train_rows, val_rows, test_rows):
    semantic_sets = [
        {r.semantic for r in rows}
        for rows in (train_rows, val_rows, test_rows)
    ]
    disjoint = (
        semantic_sets[0].isdisjoint(semantic_sets[1])
        and semantic_sets[0].isdisjoint(semantic_sets[2])
        and semantic_sets[1].isdisjoint(semantic_sets[2])
    )
    if not disjoint:
        raise RuntimeError("train/validation/test semantic leakage detected")
    domains = [{r.semantic[2] for r in rows} for rows in (train_rows, val_rows, test_rows)]
    if not (
        domains[0].isdisjoint(domains[1])
        and domains[0].isdisjoint(domains[2])
        and domains[1].isdisjoint(domains[2])
    ):
        raise RuntimeError("train/validation/test domain leakage detected")
    return {
        "train_semantics": len(semantic_sets[0]),
        "validation_semantics": len(semantic_sets[1]),
        "test_semantics": len(semantic_sets[2]),
        "domain_sets": [sorted(d) for d in domains],
        "pairwise_disjoint": True,
        "target_domain_disjoint": True,
    }


def identifiability_audit():
    world = World(TRAIN_WORLD_SEED, 1, "train")
    states = [(a, b, c) for a in range(8) for b in range(8) for c in range(4)]
    result = {}
    for modality in MODALITIES:
        observations = [
            tuple(world.render(modality, s).tolist()) for s in states
        ]
        if len(set(observations)) != len(observations):
            raise RuntimeError(f"{modality} observation map is not identifiable")
        transition_keys = {}
        for s in states:
            x = tuple(world.render(modality, s).tolist())
            for action in range(N_OPS):
                key = (x, action)
                y = tuple(world.render(modality, world.transition(s, action)).tolist())
                prior = transition_keys.get(key)
                if prior is not None and prior != y:
                    raise RuntimeError(
                        f"{modality} transition is not identifiable from observation+action"
                    )
                transition_keys[key] = y
        result[modality] = {
            "state_observation_injective": True,
            "action_conditioned_transition_identifiable": True,
        }
    return result


class Encoder(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == "language":
            self.emb = nn.Embedding(VOCAB, LATENT)
            self.net = nn.Sequential(
                nn.Linear(LATENT, LATENT),
                nn.GELU(),
                nn.Linear(LATENT, LATENT),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(64, 96),
                nn.GELU(),
                nn.Linear(96, LATENT),
            )

    def forward(self, x):
        if self.modality == "language":
            return self.net(self.emb(x).mean(1))
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        self.head = (
            nn.Linear(LATENT, SEQ * VOCAB)
            if modality == "language"
            else nn.Sequential(
                nn.Linear(LATENT, 96),
                nn.GELU(),
                nn.Linear(96, 64),
            )
        )

    def forward(self, z):
        return self.head(z)


class UnifiedBrain(nn.Module):
    """Shared predictive structural latent with modality-specific observations."""

    def __init__(
        self,
        use_structure=True,
        use_persistence=True,
        use_prediction=True,
        shared_transition=True,
    ):
        super().__init__()
        self.use_structure = use_structure
        self.use_persistence = use_persistence
        self.use_prediction = use_prediction
        self.shared_transition = shared_transition

        self.enc = nn.ModuleDict({m: Encoder(m) for m in MODALITIES})
        self.dec = nn.ModuleDict({m: Decoder(m) for m in MODALITIES})

        self.mod_gate = nn.Linear(LATENT + 3, LATENT)

        self.slot_proj = nn.Linear(LATENT, SLOTS * SLOT_DIM)
        self.slot_score = nn.Linear(LATENT, SLOTS)
        self.struct_out = nn.Linear(SLOT_DIM, LATENT)

        self.action_emb = nn.Embedding(N_OPS, LATENT)
        self.op_delta = nn.Parameter(torch.randn(N_OPS, LATENT) * 0.02)

        if shared_transition:
            self.transition = nn.Sequential(
                nn.Linear(2 * LATENT, LATENT),
                nn.GELU(),
                nn.Linear(LATENT, LATENT),
            )
        else:
            self.transition = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(2 * LATENT, LATENT),
                        nn.GELU(),
                        nn.Linear(LATENT, LATENT),
                    )
                    for m in MODALITIES
                }
            )

        self.conf_head = nn.Sequential(
            nn.Linear(2 * LATENT, LATENT),
            nn.GELU(),
            nn.Linear(LATENT, 1),
        )

    def encode(self, modality, x):
        z = self.enc[modality](x)
        idx = MODALITIES.index(modality)
        one_hot = F.one_hot(
            torch.full((z.shape[0],), idx, dtype=torch.long, device=z.device),
            3,
        ).float()
        return self.mod_gate(torch.cat([z, one_hot], -1))

    def discover(self, z):
        if not self.use_structure:
            return z, torch.full(
                (z.shape[0], SLOTS),
                1.0 / SLOTS,
                device=z.device,
            )
        slots = self.slot_proj(z).view(-1, SLOTS, SLOT_DIM)
        weights = F.softmax(self.slot_score(z), -1)
        return self.struct_out(
            (slots * weights.unsqueeze(-1)).sum(1)
        ), weights

    def retrieve(self, z, memory, k=2):
        if not self.use_persistence or not memory:
            return torch.zeros_like(z)
        mem = torch.stack(memory).to(z.device)
        idx = (
            F.normalize(z, -1) @ F.normalize(mem, -1).T
        ).topk(min(k, len(memory)), -1).indices
        return mem[idx].mean(1)

    def predict(self, z, action, memory, modality, proposal_bias=None):
        base, _ = self.discover(z)
        if self.use_persistence and memory:
            base = base + 0.15 * self.retrieve(z, memory)
        action_latent = self.action_emb(action) + self.op_delta[action]
        if proposal_bias is not None:
            action_latent = action_latent + proposal_bias
        if not self.use_prediction:
            return base
        transition = (
            self.transition
            if self.shared_transition
            else self.transition[modality]
        )
        return transition(torch.cat([base, action_latent], -1))

    def confidence(self, predicted, target):
        return torch.sigmoid(
            self.conf_head(torch.cat([predicted, target], -1))
        ).squeeze(-1)


class DurableMemory:
    """Persistent experience store with verified-only writes."""

    def __init__(self, max_items=256):
        self.items = []
        self.operator_delta = {}
        self.max_items = max_items
        self.attempted_writes = 0
        self.committed_writes = 0

    def propose_after_verification(self, z_before, action, z_after, verified):
        self.attempted_writes += 1
        if not verified:
            return False
        self.items.append(z_after.detach().cpu())
        self.items = self.items[-self.max_items :]
        delta = (z_after - z_before).detach().cpu()
        previous = self.operator_delta.get(action)
        self.operator_delta[action] = (
            delta if previous is None else 0.9 * previous + 0.1 * delta
        )
        self.committed_writes += 1
        return True

    def tensors(self):
        return list(self.items)

    @property
    def verified_write_rate(self):
        return self.committed_writes / max(1, self.attempted_writes)


def batch_tensors(rows):
    out = {}
    for modality in MODALITIES:
        subset = [r for r in rows if r.modality == modality]
        if not subset:
            continue
        xs = [r.x for r in subset]
        ys = [r.y for r in subset]
        acts = [r.action for r in subset]
        if modality == "language":
            out[modality] = (
                torch.tensor(np.stack(xs), dtype=torch.long),
                torch.tensor(np.stack(ys), dtype=torch.long),
                torch.tensor(acts, dtype=torch.long),
            )
        else:
            out[modality] = (
                torch.tensor(np.stack(xs), dtype=torch.float32),
                torch.tensor(np.stack(ys), dtype=torch.float32),
                torch.tensor(acts, dtype=torch.long),
            )
    return out


def anti_collapse_report(model, rows):
    model.eval()
    data = batch_tensors(rows)
    report = {}
    with torch.no_grad():
        for modality, (x, _y, _a) in data.items():
            z = model.encode(modality, x)
            std = float(z.std(0).mean())
            report[modality] = {
                "mean_latent_std": std,
                "collapsed": bool(std < ANTI_COLLAPSE_FLOOR),
            }
    if any(v["collapsed"] for v in report.values()):
        raise RuntimeError("anti-collapse gate failed")
    return report


def checkpoint_hash(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


@torch.no_grad()
def target_normalizers(rows):
    out = {}
    data = batch_tensors(rows)
    for modality, (_x, y, _a) in data.items():
        if modality == "language":
            counts = torch.bincount(y.view(-1), minlength=VOCAB)
            out[modality] = float(counts.max()) / float(y.numel())
        else:
            mean_target = y.mean(0, keepdim=True)
            out[modality] = float(F.mse_loss(mean_target.expand_as(y), y))
    return out


@torch.no_grad()
def evaluate(model, rows, memory, include_memory=True, normalizers=None):
    model.eval()
    data = batch_tensors(rows)
    metrics = {}
    all_z = []
    for modality, (x, y, action) in data.items():
        z = model.encode(modality, x)
        z_target = model.encode(modality, y)
        mem = memory.tensors() if include_memory else []
        predicted = model.predict(z, action, mem, modality)
        output = model.dec[modality](predicted)

        if modality == "language":
            prediction = output.view(-1, SEQ, VOCAB).argmax(-1)
            accuracy = float((prediction == y).float().mean())
            loss = float(
                F.cross_entropy(
                    output.view(-1, VOCAB),
                    y.view(-1),
                )
            )
            baseline_accuracy = float((normalizers or {}).get(modality, 1.0 / VOCAB))
            normalized = (
                (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
                if baseline_accuracy < 1.0
                else 0.0
            )
            metrics[modality] = {
                "token_accuracy": accuracy,
                "baseline_accuracy_train_only": baseline_accuracy,
                "normalized_score": normalized,
                "loss": loss,
            }
        else:
            mse = float(F.mse_loss(output, y))
            baseline_mse = float((normalizers or {}).get(modality, 0.0))
            normalized = 1.0 - mse / baseline_mse if baseline_mse > 0.0 else 0.0
            metrics[modality] = {
                "mse": mse,
                "baseline_mse_train_only": baseline_mse,
                "normalized_score": normalized,
            }

        action_predictions = torch.stack(
            [
                model.predict(
                    z,
                    torch.full_like(action, k),
                    mem,
                    modality,
                )
                for k in range(N_OPS)
            ],
            1,
        )
        route_distance = (
            (action_predictions - z_target.unsqueeze(1)) ** 2
        ).mean(-1)
        choice = route_distance.argmin(1)
        metrics[modality]["route_accuracy"] = float(
            (choice == action).float().mean()
        )
        all_z.append(F.normalize(z, -1))

    if len(all_z) == 3:
        perm = torch.randperm(all_z[1].shape[0])
        paired = (
            (all_z[0] - all_z[1]).pow(2).mean()
            + (all_z[0] - all_z[2]).pow(2).mean()
        ) / 2
        shuffled = (all_z[0] - all_z[1][perm]).pow(2).mean()
        metrics["cross_modal"] = {
            "paired_mse": float(paired),
            "shuffled_mse": float(shuffled),
            "separation": float(shuffled - paired),
        }
    return metrics


def train_one(seed, disable, steps, batch_size, shared_transition=True):
    seed_all(seed)

    train_rows = World(TRAIN_WORLD_SEED, 96, "train").make()
    val_rows = World(VAL_WORLD_SEED, 48, "val").make()
    test_rows = World(TEST_WORLD_SEED, 48, "test").make()

    model = UnifiedBrain(
        use_structure="structure" not in disable,
        use_persistence="persistence" not in disable,
        use_prediction="prediction" not in disable,
        shared_transition=shared_transition,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )

    memory = DurableMemory(max_items=64)
    rng = np.random.default_rng(seed + 500)
    state_count = 96

    for _step in range(steps):
        indices = rng.integers(0, state_count, batch_size)
        batch = [train_rows[int(i) * 3 + j] for i in indices for j in range(3)]
        data = batch_tensors(batch)
        optimizer.zero_grad()
        total = torch.zeros((), requires_grad=True)

        for modality, (x, y, action) in data.items():
            z = model.encode(modality, x)
            z_target = model.encode(modality, y)
            predicted = model.predict(
                z,
                action,
                memory.tensors(),
                modality,
            )
            output = model.dec[modality](predicted)

            if modality == "language":
                reconstruction = F.cross_entropy(
                    output.view(-1, VOCAB),
                    y.view(-1),
                )
            else:
                reconstruction = 8.0 * F.mse_loss(output, y)

            prediction_loss = F.mse_loss(predicted, z_target)

            # Anti-collapse regularizer: keep each latent dimension active
            # without using held-out semantics or future test information.
            feature_std = predicted.std(0)
            variance_penalty = F.relu(ANTI_COLLAPSE_FLOOR - feature_std).mean()

            total = (
                total
                + reconstruction
                + 0.35 * prediction_loss
                + 0.05 * variance_penalty
            )

        if "alignment" not in disable:
            zs = [
                F.normalize(model.encode(m, data[m][0]), -1)
                for m in MODALITIES
                if m in data
            ]
            if len(zs) == 3:
                total = total + 0.08 * (
                    (zs[0] - zs[1]).pow(2).mean()
                    + (zs[0] - zs[2]).pow(2).mean()
                )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Atomic temporal boundary: all modality predictions at this step use
        # the same pre-step persistent state. Future targets are committed only
        # after every modality has been predicted and verified.
        with torch.no_grad():
            pending_writes = []
            memory_before_batch = memory.tensors()
            for modality, (x, y, action) in data.items():
                z = model.encode(modality, x)
                z_target = model.encode(modality, y)
                predicted = model.predict(
                    z,
                    action,
                    memory_before_batch,
                    modality,
                )
                for i in range(z.shape[0]):
                    latent_mse = float(
                        ((predicted[i] - z_target[i]) ** 2).mean()
                    )
                    verified = latent_mse <= VERIFY_LATENT_MSE
                    pending_writes.append(
                        (z[i], int(action[i]), z_target[i], verified)
                    )

            for z_i, action_i, z_target_i, verified_i in pending_writes:
                memory.propose_after_verification(
                    z_i,
                    action_i,
                    z_target_i,
                    verified_i,
                )

    collapse = anti_collapse_report(model, train_rows)
    frozen_hash = checkpoint_hash(model)
    normalizers = target_normalizers(train_rows)
    val = evaluate(model, val_rows, memory, normalizers=normalizers)
    test = evaluate(model, test_rows, memory, normalizers=normalizers)

    # Two-stage selection: cheap proposal first, exact transition reranking second.
    routing = {m: {"proposal_recall": 0.0, "conditional_route": 0.0, "goal_conditioned": True} for m in MODALITIES}
    for modality in MODALITIES:
        rows = [r for r in test_rows if r.modality == modality]
        x, _y, action = batch_tensors(rows)[modality]
        z = model.encode(modality, x)
        z_goal = model.encode(
            modality,
            batch_tensors(rows)[modality][1],
        )
        delta = z_goal - z
        prototypes = model.action_emb.weight + model.op_delta
        cheap_scores = F.normalize(delta, -1) @ F.normalize(prototypes, -1).T
        proposed = cheap_scores.topk(min(2, N_OPS), -1).indices
        proposal_hit = (proposed == action.unsqueeze(1)).any(1)
        routing[modality]["proposal_recall"] = float(proposal_hit.float().mean())

        good = 0
        denominator = int(proposal_hit.sum())
        mem = memory.tensors()
        for i in torch.where(proposal_hit)[0].tolist():
            candidates = proposed[i]
            predictions = torch.stack(
                [
                    model.predict(
                        z[i : i + 1],
                        torch.tensor([int(k)]),
                        mem,
                        modality,
                    )
                    for k in candidates
                ],
                1,
            ).squeeze(0)
            distance = ((predictions - z_goal[i]) ** 2).mean(-1)
            good += int(int(candidates[int(distance.argmin())]) == int(action[i]))

        routing[modality]["conditional_route"] = good / denominator if denominator else 0.0

    for modality in MODALITIES:
        test[modality].update(routing[modality])

    return {
        "seed": seed,
        "ablation": list(disable) or ["none"],
        "shared_transition": shared_transition,
        "validation": val,
        "test": test,
        "memory_items": len(memory.items),
        "operator_library": len(memory.operator_delta),
        "verified_write_rate": memory.verified_write_rate,
        "latent_collapse": collapse,
        "checkpoint_sha256": frozen_hash,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    if not args.confirmatory:
        raise SystemExit(
            "Use --confirmatory for the registered runner; implementation pilots are separate."
        )

    seeds = tuple(int(x) for x in args.seeds.split(","))
    contract = load_contract("TACOSM-UNIFIED-MULTIMODAL-001")
    contract.require_levels((96,))
    contract.require_seeds(seeds)
    contract.require_steps(args.steps)
    contract.require_eval_steps(48)
    contract.require_arms(["full", "no_persistence", "no_predictive_transition", "no_cross_modal_alignment", "no_structure_discovery"])
    contract.primary_endpoint()

    train_probe = World(TRAIN_WORLD_SEED, 24, "train").make()
    val_probe = World(VAL_WORLD_SEED, 12, "val").make()
    test_probe = World(TEST_WORLD_SEED, 12, "test").make()

    split = split_audit(train_probe, val_probe, test_probe)
    identifiability = identifiability_audit()

    arms = [
        (),
        ("persistence",),
        ("prediction",),
        ("alignment",),
        ("structure",),
    ]

    results = [
        train_one(seed, arm, args.steps, BATCH_SIZE, shared_transition=True)
        for arm in arms
        for seed in seeds
    ]

    control = []
    train_baselines = target_normalizers(World(TRAIN_WORLD_SEED, 96, "train").make())
    for seed in seeds:
        seed_all(seed)
        rows = World(TEST_WORLD_SEED, 48, "test").make()
        untrained = UnifiedBrain(shared_transition=True)
        control.append(
            {
                "seed": seed,
                "test": evaluate(
                    untrained,
                    rows,
                    DurableMemory(max_items=1),
                    include_memory=False,
                    normalizers=train_baselines,
                ),
            }
        )

    def modality_metric(row, modality):
        return float(row["test"][modality]["normalized_score"])

    trained_full = [
        r for r in results if r["ablation"] == ["none"]
    ]

    trained_means = {
        modality: float(
            np.mean([modality_metric(r, modality) for r in trained_full])
        )
        for modality in MODALITIES
    }
    control_means = {
        modality: float(np.mean([modality_metric(r, modality) for r in control]))
        for modality in MODALITIES
    }

    margins = {
        modality: trained_means[modality] - control_means[modality]
        for modality in MODALITIES
    }

    thresholds = {
        "language": 0.05,
        "image": 0.03,
        "audio": 0.03,
    }
    primary_pass = {
        modality: margins[modality] >= thresholds[modality]
        for modality in MODALITIES
    }

    grouped = {}
    for arm in arms:
        name = "full" if not arm else "no_" + arm[0]
        rr = [
            r for r in results
            if r["ablation"] == (list(arm) if arm else ["none"])
        ]
        grouped[name] = {"n": len(rr)}
        for modality in MODALITIES:
            grouped[name][modality] = float(
                np.mean([modality_metric(r, modality) for r in rr])
            )

    payload = {
        "protocol": {
            "experiment_id": "TACOSM-UNIFIED-MULTIMODAL-001",
            "contract_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("contracts/TACOSM-UNIFIED-MULTIMODAL-001.json").read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "train_world_seed": TRAIN_WORLD_SEED,
            "validation_world_seed": VAL_WORLD_SEED,
            "test_world_seed": TEST_WORLD_SEED,
            "semantic_split": "residue classes 2/1/0 over 5",
            "seeds": list(seeds),
            "steps": args.steps,
            "batch_size": BATCH_SIZE,
            "modalities": list(MODALITIES),
            "confirmatory": args.confirmatory,
            "verify_latent_mse_threshold": VERIFY_LATENT_MSE,
            "anti_collapse_floor": ANTI_COLLAPSE_FLOOR,
            "image_audio_normalization": "1 - model_MSE / train_only_constant_baseline_MSE",
        },
        "preconditions": {
            "split_audit": split,
            "identifiability": identifiability,
        },
        "controls": {
            "untrained_mean": control_means,
        },
        "summary": {
            "trained_full_means": trained_means,
            "margins_vs_untrained": margins,
            "primary_pass_by_modality": primary_pass,
            "all_modalities_pass": all(primary_pass.values()),
            "ablation_means": grouped,
        },
        "results": results,
    }

    out = Path("artifacts/TACOSM-UNIFIED-MULTIMODAL-001.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
