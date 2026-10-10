"""Development-only noisy-alias benchmark for prospective INTEGRATION-002.

This is not preregistration and does not measure learned models. It removes the
exact shared key string that made Integration-001 solvable by an ordinary
dictionary, and tests whether classical observable-only matching remains
trivially perfect. Latent index and uncorrupted vectors are oracle-side ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

DIMS = 12
HISTORY_SIZES = (32, 128, 512)
DELAYS = (1, 4, 16, 32)
HARD_FRACTIONS = (0.0, 0.25, 0.5, 1.0)
NOISE_LEVELS = (0.08, 0.25)


@dataclass(frozen=True)
class Observation:
    address: tuple[float, ...]
    value: int


@dataclass(frozen=True)
class Episode:
    history: tuple[Observation, ...]
    writes: tuple[Observation, ...]
    query_address: tuple[float, ...]
    current_bit: int
    target_slot: int  # Oracle evaluator only; excluded from model_inputs()
    target: int       # Oracle evaluator only; excluded from model_inputs()


@dataclass(frozen=True)
class Pair:
    left: Episode
    right: Episode


def _random_vec(rng: random.Random) -> tuple[float, ...]:
    return tuple(rng.gauss(0.0, 1.0) for _ in range(DIMS))


def _perturb(rng: random.Random, v: Sequence[float], scale: float) -> tuple[float, ...]:
    return tuple(round(x + rng.gauss(0.0, scale), 8) for x in v)


def visible_inputs(ep: Episode) -> dict:
    return {
        "history": [{"address": x.address, "value": x.value} for x in ep.history],
        "writes": [{"address": x.address, "value": x.value} for x in ep.writes],
        "query_address": ep.query_address,
        "current_bit": ep.current_bit,
    }


def make_pair(seed: int, history: int, delay: int, hard_fraction: float,
              noise: float) -> Pair:
    if history not in HISTORY_SIZES or delay not in DELAYS:
        raise ValueError("unregistered development grid size")
    if hard_fraction not in HARD_FRACTIONS or noise not in NOISE_LEVELS:
        raise ValueError("unregistered development corruption channel")
    rng = random.Random(seed)
    target_slot = rng.randrange(history)
    current_bit = rng.randrange(2)
    latent = [_random_vec(rng) for _ in range(history)]
    target_latent = latent[target_slot]
    # History and read query get independently sampled, never exactly shared,
    # noisy views of the same hidden latent key.
    vals = [rng.randrange(2) for _ in range(history)]
    vis_history = tuple(Observation(_perturb(rng, z, noise), v)
                        for z,v in zip(latent, vals))
    flipped = tuple(Observation(x.address, (1-x.value) if i == target_slot else x.value)
                    for i,x in enumerate(vis_history))
    query = _perturb(rng, target_latent, noise)
    n_hard = round(delay * hard_fraction)
    writes = []
    for i in range(delay):
        latent_write = (
            _perturb(rng, target_latent, 0.12)
            if i < n_hard else _random_vec(rng)
        )
        writes.append(Observation(_perturb(rng, latent_write, noise),
                                  rng.randrange(2)))
    writes = tuple(writes)
    left = Episode(vis_history, writes, query, current_bit, target_slot,
                   vis_history[target_slot].value ^ current_bit)
    right = Episode(flipped, writes, query, current_bit, target_slot,
                    flipped[target_slot].value ^ current_bit)
    pair = Pair(left, right)
    validate_pair(pair, history, delay)
    return pair


def validate_pair(pair: Pair, history: int, delay: int) -> None:
    a,b = pair.left, pair.right
    if not (len(a.history) == len(b.history) == history
            and len(a.writes) == len(b.writes) == delay):
        raise AssertionError("invalid shape")
    if (a.target_slot != b.target_slot or a.target == b.target
            or a.query_address != b.query_address or a.current_bit != b.current_bit
            or a.writes != b.writes):
        raise AssertionError("causal pairing invalid")
    if sum(x != y for x,y in zip(a.history, b.history)) != 1:
        raise AssertionError("paired histories differ outside target")
    if any(a.query_address == x.address for x in a.history + a.writes):
        raise AssertionError("query leaks identical stored address")
    for ep in (a,b):
        v = visible_inputs(ep)
        if set(v) != {"history","writes","query_address","current_bit"}:
            raise AssertionError("model-visible privileged marker")
        if any("target" in x or "slot" in x or "latent" in x for x in v):
            raise AssertionError("oracle leakage")
        if ep.target != (ep.history[ep.target_slot].value ^ ep.current_bit):
            raise AssertionError("target cannot be realized by oracle")


def similarity_prediction(ep: Episode, *, metric: str) -> tuple[int, int]:
    """Return prediction and history target rank using ONLY visible tensors.

    The metric never receives ep.target_slot or ep.target. The evaluator may
    inspect rank afterwards, but no oracle field is consumed in this function.
    """
    visible = visible_inputs(ep)
    q = visible["query_address"]
    if metric not in ("cosine", "euclidean"):
        raise ValueError("unknown baseline metric")
    scores = []
    for row in visible["history"] + visible["writes"]:
        x = row["address"]
        if metric == "euclidean":
            score = -sum((a-b)**2 for a,b in zip(q,x))
        else:
            dot = sum(a*b for a,b in zip(q,x))
            norm = math.sqrt(sum(a*a for a in q)*sum(b*b for b in x))
            score = dot / max(norm,1e-12)
        scores.append(score)
    selected = max(range(len(scores)), key=lambda i:scores[i])
    values = visible["history"] + visible["writes"]
    return values[selected]["value"] ^ visible["current_bit"], selected


def diagnostic_rows(*, seeds: Sequence[int] = (17, 23, 31),
                    histories: Sequence[int] = (32,128,512),
                    delays: Sequence[int] = (1,4,16,32),
                    fractions: Sequence[float] = HARD_FRACTIONS,
                    noises: Sequence[float] = NOISE_LEVELS,
                    pairs_per_cell: int = 8) -> list[dict]:
    if pairs_per_cell < 1:
        raise ValueError("invalid count")
    out = []
    for seed in seeds:
        for h in histories:
            for d in delays:
                for fraction in fractions:
                    for noise in noises:
                        total = hit_cos = hit_l2 = rank_cos = rank_l2 = 0
                        for pair_index in range(pairs_per_cell):
                            # Development-exclusive namespace, never the
                            # confirmatory namespace of INTEGRATION-001.
                            dev_seed = (83_000_000 + seed*10_000_019
                                        + h*1_000_003 + d*10_007
                                        + round(fraction*100)*103
                                        + round(noise*100)*503 + pair_index)
                            p = make_pair(dev_seed,h,d,fraction,noise)
                            for ep in (p.left,p.right):
                                c, j_c = similarity_prediction(ep,metric="cosine")
                                l2, j_l = similarity_prediction(ep,metric="euclidean")
                                hit_cos += c == ep.target
                                hit_l2 += l2 == ep.target
                                rank_cos += j_c == ep.target_slot
                                rank_l2 += j_l == ep.target_slot
                                total += 1
                        out.append({
                            "seed":seed,"H":h,"D":d,
                            "hard_fraction":fraction,"noise":noise,
                            "pairs":pairs_per_cell,
                            "cosine_accuracy":hit_cos/total,
                            "euclidean_accuracy":hit_l2/total,
                            "cosine_recall_at_1":rank_cos/total,
                            "euclidean_recall_at_1":rank_l2/total,
                            "oracle_accuracy":1.0,
                            "exact_model_visible_identity_reuse":False,
                            "candidate_scores_computed_per_query":h+d,
                        })
    return out
