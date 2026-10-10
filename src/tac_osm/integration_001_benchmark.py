"""INTEGRATION-001 synthetic benchmark instrument: paired causal histories.

No model or experiment is run by this module. It generates immutable examples
for a preregistered integration test with uniform target slots, opaque addresses,
and exact paired-history contrasts. Features are deliberately generic; downstream
experiments must still prove learnability and prevent model-visible leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random

HISTORY_SIZES = (32, 128, 512)
DELAYS = (1, 4, 16, 32)
K_VALUES = (4, 8, 16)
KEY_BYTES = 16


@dataclass(frozen=True)
class Observation:
    key: str
    value: int


@dataclass(frozen=True)
class Episode:
    history: tuple[Observation, ...]
    writes: tuple[Observation, ...]
    query_key: str
    current_bit: int
    target_slot: int
    target: int


@dataclass(frozen=True)
class PairedEpisode:
    left: Episode
    right: Episode


def _new_keys(rng: random.Random, n: int) -> tuple[str, ...]:
    keys: set[str] = set()
    while len(keys) < n:
        keys.add(rng.randbytes(KEY_BYTES).hex())
    # set iteration order is not stable across hash seeds; always sort.
    return tuple(sorted(keys))


def make_pair(seed: int, history_size: int, delay: int) -> PairedEpisode:
    if history_size not in HISTORY_SIZES or delay not in DELAYS:
        raise ValueError("unregistered history size or delay")
    rng = random.Random(seed)
    keys = _new_keys(rng, history_size + delay)
    keys = tuple(rng.sample(keys, len(keys)))
    target_slot = rng.randrange(history_size)
    current_bit = rng.randrange(2)
    values = [rng.randrange(2) for _ in range(history_size)]
    # Randomize whether the first half of the pair holds 0 or 1.
    values[target_slot] = rng.randrange(2)
    base = tuple(Observation(k, v) for k, v in zip(keys[:history_size], values))
    opposite = tuple(
        Observation(item.key, 1 - item.value if i == target_slot else item.value)
        for i, item in enumerate(base)
    )
    writes = tuple(
        Observation(k, rng.randrange(2)) for k in keys[history_size:]
    )
    # Intervening writes deliberately share the same representation/format.
    # The target key is requested at read time, never its index.
    q = base[target_slot].key
    left = Episode(base, writes, q, current_bit, target_slot,
                   base[target_slot].value ^ current_bit)
    right = Episode(opposite, writes, q, current_bit, target_slot,
                    opposite[target_slot].value ^ current_bit)
    pair = PairedEpisode(left, right)
    validate_pair(pair, history_size, delay)
    return pair


def model_inputs(episode: Episode) -> dict:
    """Exactly the features available to a learned system; omit target_slot/label."""
    return {
        "history": [{"key": o.key, "value": o.value} for o in episode.history],
        "writes": [{"key": o.key, "value": o.value} for o in episode.writes],
        "query_key": episode.query_key,
        "current_bit": episode.current_bit,
    }


def validate_pair(pair: PairedEpisode, history_size: int, delay: int) -> None:
    a, b = pair.left, pair.right
    assert len(a.history) == len(b.history) == history_size
    assert len(a.writes) == len(b.writes) == delay
    assert a.writes == b.writes
    assert a.query_key == b.query_key
    assert a.current_bit == b.current_bit
    assert a.target_slot == b.target_slot
    assert a.target != b.target
    assert sum(x != y for x, y in zip(a.history, b.history)) == 1
    assert a.history[a.target_slot].key == a.query_key
    assert b.history[b.target_slot].key == b.query_key
    assert len({o.key for o in a.history + a.writes}) == history_size + delay
    assert a.target == a.history[a.target_slot].value ^ a.current_bit
    assert b.target == b.history[b.target_slot].value ^ b.current_bit
    assert "target" not in model_inputs(a)
    assert "target_slot" not in model_inputs(a)
    assert model_inputs(a)["writes"] == model_inputs(b)["writes"]


def fingerprint(pair: PairedEpisode) -> str:
    record = {"left": model_inputs(pair.left), "right": model_inputs(pair.right),
              "targets": [pair.left.target, pair.right.target]}
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_pool(seed_namespace: int, seeds: tuple[int, ...],
              history_size: int, delay: int, pairs_per_seed: int) -> list[PairedEpisode]:
    if pairs_per_seed < 1:
        raise ValueError("nonpositive pair count")
    pool: list[PairedEpisode] = []
    fingerprints: set[str] = set()
    for seed in seeds:
        for idx in range(pairs_per_seed):
            pair = make_pair(seed_namespace + seed * 1000003 + idx,
                             history_size, delay)
            fp = fingerprint(pair)
            if fp in fingerprints:
                raise AssertionError("duplicate evaluation episode")
            fingerprints.add(fp)
            pool.append(pair)
    return pool
