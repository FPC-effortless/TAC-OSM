"""INTEGRATION-001 development-only falsification controls.

These controls are **not** the confirmatory experiment, contain no learned
model, and cannot satisfy the frozen 001 contract. Their purpose is to detect
trivial exact-key shortcuts and whether real, same-format distractor writes
actually make the target harder to retrieve.

The main preregistered grid and thresholds remain untouched. Never promote
an accuracy from this file to an INTEGRATION-001 PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random
from typing import Sequence

from tac_osm.integration_001_benchmark import (
    Episode, Observation, PairedEpisode, make_pair, validate_pair, model_inputs,
)

INTERFERENCE_STRENGTHS = (0.0, 0.25, 0.5, 1.0)
KEY_BITS = 128


def bit_distance(a: str, b: str) -> int:
    """Hamming distance between equal-size opaque hexadecimal keys."""
    if len(a) != 32 or len(b) != 32:
        raise ValueError("address keys must be 16 bytes")
    return (int(a, 16) ^ int(b, 16)).bit_count()


def near_key(key: str, bit: int) -> str:
    if not 0 <= bit < KEY_BITS:
        raise ValueError("bit outside key")
    return format(int(key, 16) ^ (1 << bit), "032x")


def make_interference_pair(seed: int, history_size: int,
                           delay: int, strength: float) -> PairedEpisode:
    """Inject distinct same-format 1-bit neighbor distractors into an exact pair.

    No target identity is duplicated and the paired present stays identical.
    This strengthens distractor similarity but **does not** make retrieval
    genuinely ambiguous, because the exact query key is still exposed.
    """
    if strength not in INTERFERENCE_STRENGTHS:
        raise ValueError("unregistered interference strength")
    pair = make_pair(seed, history_size, delay)
    neighbor_count = round(delay * strength)
    target_key = pair.left.query_key
    used = {obs.key for obs in pair.left.history}
    updated = list(pair.left.writes)
    rng = random.Random(seed + 71_027_119)
    positions = rng.sample(range(KEY_BITS), KEY_BITS)
    for i in range(neighbor_count):
        # Always distinct identities; this is a *confusability* diagnostic,
        # not a same-key overwrite experiment.
        proposed = near_key(target_key, positions[i])
        if proposed in used:
            raise AssertionError("unexpected 128-bit neighbor collision")
        used.add(proposed)
        updated[i] = Observation(proposed, updated[i].value)
    # Remaining random keys are overwhelmingly unlikely to collide; enforce.
    keys = [x.key for x in pair.left.history] + [x.key for x in updated]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate identity after confusable-write injection")
    writes = tuple(updated)
    result = PairedEpisode(
        replace(pair.left, writes=writes), replace(pair.right, writes=writes)
    )
    validate_pair(result, history_size, delay)
    assert sum(bit_distance(x.key, target_key) == 1 for x in writes) == neighbor_count
    return result


@dataclass(frozen=True)
class ExactControlTrace:
    target_found: bool
    output: int
    index_build_writes: int
    intervening_write_count: int
    index_refresh_writes: int
    query_index_lookups: int
    executed_values: int
    total_primitive_actions: int


def exact_lookup_control(episode: Episode, *,
                         include_intervening: bool = True) -> ExactControlTrace:
    """Non-learned dictionary solution with counted index construction.

    Each history observation and intervening write updates an index. The
    target's exact identity is supplied in the query; target_slot is NEVER
    read. Work counts are explanatory primitive actions, not GPU timings,
    and the initial build must not be omitted from an end-to-end comparison.
    """
    visible = model_inputs(episode)
    index: dict[str, int] = {}
    writes = visible["writes"] if include_intervening else []
    for row in visible["history"]:
        index[row["key"]] = row["value"]
    for row in writes:
        index[row["key"]] = row["value"]
    matched = visible["query_key"] in index
    if not matched:
        raise AssertionError("target not realizable from visible observation")
    payload = index[visible["query_key"]]
    pred = payload ^ visible["current_bit"]
    initial, refresh, lookup, execute = len(visible["history"]), len(writes), 1, 1
    return ExactControlTrace(
        target_found=True, output=pred, index_build_writes=initial,
        intervening_write_count=len(writes),
        index_refresh_writes=refresh, query_index_lookups=lookup,
        executed_values=execute,
        total_primitive_actions=initial + refresh + lookup + execute,
    )


def reset_control(episode: Episode) -> int:
    """Deterministic history-independent classifier; 50% on opposite pairs."""
    present = model_inputs(episode)
    return present["current_bit"]


def dense_query_cost(episode: Episode) -> int:
    """Instrument one full candidate scan; this is O(H+D), not selective."""
    visible = model_inputs(episode)
    comparisons = 0
    for row in (*visible["history"], *visible["writes"]):
        # Count all candidates actually inspected, not merely top-K outputs.
        _ = row["key"] == visible["query_key"]
        comparisons += 1
    return comparisons


def diagnostic_rows(*, seeds: Sequence[int] = (11, 19, 23),
                    history_sizes: Sequence[int] = (32, 128, 512),
                    delays: Sequence[int] = (1, 4, 16, 32),
                    strengths: Sequence[float] = INTERFERENCE_STRENGTHS,
                    pairs_per_cell: int = 5) -> list[dict]:
    """Independent **development** diagnostic namespace, not confirmatory."""
    if pairs_per_cell < 1:
        raise ValueError("invalid pair count")
    out = []
    for seed in seeds:
        for h in history_sizes:
            for delay in delays:
                for strength in strengths:
                    exact_correct = reset_correct = nowrite_correct = 0
                    scan_count = index_cost = 0
                    neighbor_count = 0
                    for pair_number in range(pairs_per_cell):
                        # Development-exclusive RNG namespace, intentionally not
                        # the contract's training or evaluation namespaces.
                        dev_seed = 97_000_000 + 10_000_019 * seed + (
                            1_000_003 * h + 10_007 * delay +
                            101 * int(strength * 100) + pair_number
                        )
                        pair = make_interference_pair(dev_seed, h, delay, strength)
                        neighbor_count += sum(
                            bit_distance(w.key, pair.left.query_key) == 1
                            for w in pair.left.writes
                        )
                        for ep in (pair.left, pair.right):
                            t = exact_lookup_control(ep)
                            nw = exact_lookup_control(ep, include_intervening=False)
                            exact_correct += t.output == ep.target
                            nowrite_correct += nw.output == ep.target
                            reset_correct += reset_control(ep) == ep.target
                            scan_count += dense_query_cost(ep)
                            index_cost += t.total_primitive_actions
                    n = 2 * pairs_per_cell
                    out.append({
                        "seed": seed, "H": h, "D": delay,
                        "interference_strength": strength,
                        "pairs": pairs_per_cell,
                        "exact_key_lookup_accuracy": exact_correct / n,
                        "no_intervening_write_accuracy": nowrite_correct / n,
                        "reset_accuracy": reset_correct / n,
                        "one_bit_neighbor_writes_per_pair":
                            neighbor_count / pairs_per_cell,
                        "dense_candidate_checks_per_query": scan_count / n,
                        "exact_index_total_actions_per_query": index_cost / n,
                    })
    return out
