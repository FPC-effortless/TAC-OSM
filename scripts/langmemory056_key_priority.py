#!/usr/bin/env python3
"""LANGMEMORY056: predeclared fixed-byte key-priority eviction falsifier.

Original CPython publisher FIT source ONLY; no neural fit/checkpoint/DEV/TEST.
Contrast equal 1344/4096-byte raw history, the old two-way K1 source key
table, frequency-priority and frequency-with-recency-decay key eviction.
All candidate predictions precede reading x[t]; nothing learns from labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory047_bounded import RawRing, FixedKeyTable, fnv

BUDGETS = (1344, 4096)
RECENT = 64
WIDTH = 8
KEYLEN = 3
HORIZON = 8192
START = 128
STRIDE = 8
POLICIES = ("lfu", "decayed")
# Preregistered before observing results, no fitted constants.
AGE_QUANTUM = 128


class PriorityKeyTable:
    """Exact 3-byte keys, one majority candidate, frequency priority, clock.

    8 bytes/slot: 3 key | 1 vote candidate | 1 signed vote magnitude
    | 1 saturated key occurrence count (zero means empty) | 2 last seen.
    64 bytes of each total budget are reserved for the same real recent ring
    as the prior K1 baseline. No Python dictionary or unbounded source cache.
    """

    def __init__(self, budget: int, policy: str):
        if budget not in BUDGETS or policy not in POLICIES:
            raise ValueError("unregistered budget or policy")
        self.budget = budget
        self.policy = policy
        self.payload = bytearray(budget - RECENT)
        self.slots = len(self.payload) // WIDTH
        if self.slots % 2 or self.slots * WIDTH != len(self.payload):
            raise ValueError("record geometry differs from K1 reference")
        self.sets = self.slots // 2
        self.evictions = 0
        self.key_hits = 0
        self.key_misses = 0
        self.probes = 0
        self.updates = 0

    def offsets(self, key: bytes) -> tuple[int, int]:
        if type(key) is not bytes or len(key) != KEYLEN:
            raise ValueError("requires exact 3-byte causal key")
        first = (fnv(key) % self.sets) * 2 * WIDTH
        return first, first + WIDTH

    def _get(self, offset: int, key: bytes) -> bool:
        return self.payload[offset + 5] != 0 and self.payload[offset:offset + 3] == key

    def _find(self, key: bytes) -> int | None:
        candidates = self.offsets(key)
        for offset in candidates:
            self.probes += 1
            if self._get(offset, key):
                return offset
        return None

    def read(self, key: bytes) -> int | None:
        offset = self._find(key)
        if offset is None:
            self.key_misses += 1
            return None
        self.key_hits += 1
        return self.payload[offset + 3] if self.payload[offset + 4] else None

    def _last(self, offset: int) -> int:
        return int.from_bytes(self.payload[offset + 6:offset + 8], "little")

    def _priority(self, offset: int, clock: int) -> int:
        freq = self.payload[offset + 5]
        if self.policy == "lfu":
            return freq
        age = max(0, clock - self._last(offset))
        # Monotone in observed frequency, decays with age. Integer only.
        return (freq * AGE_QUANTUM) // (AGE_QUANTUM + age)

    def update(self, key: bytes, value: int, clock: int) -> None:
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError("non-byte observation")
        if type(clock) is not int or not 0 <= clock < 65536:
            raise ValueError("unrepresentable source clock")
        offset = self._find(key)
        if offset is None:
            choices = self.offsets(key)
            offset = next((i for i in choices if self.payload[i + 5] == 0), None)
            if offset is None:
                # Tie-break old keys first, then slot address. No label used.
                offset = min(choices, key=lambda i:
                             (self._priority(i, clock), self._last(i), i))
                self.evictions += 1
            self.payload[offset:offset + WIDTH] = key + bytes((value, 1, 1)) + clock.to_bytes(2, "little")
        else:
            self.payload[offset + 5] = min(255, self.payload[offset + 5] + 1)
            vote = self.payload[offset + 4]
            if not vote:
                self.payload[offset + 3] = value
                self.payload[offset + 4] = 1
            elif self.payload[offset + 3] == value:
                self.payload[offset + 4] = min(255, vote + 1)
            else:
                self.payload[offset + 4] = vote - 1
            self.payload[offset + 6:offset + 8] = clock.to_bytes(2, "little")
        self.updates += 1

    def actual_payload_bytes(self) -> int:
        return len(self.payload) + RECENT


def metrics() -> dict:
    return {"queries": 0, "covered": 0, "correct": 0, "lookup_work": 0,
            "lookup_ns": 0, "update_ns": 0}


def score(stat: dict, answer: int | None, label: int, work: int, ns: int) -> None:
    stat["queries"] += 1
    stat["lookup_work"] += work
    stat["lookup_ns"] += ns
    if answer is not None:
        stat["covered"] += 1
        stat["correct"] += int(answer == label)


def owner_probe(raw: bytes, owner: str, horizon: int = HORIZON) -> dict:
    if type(raw) is not bytes or not raw or type(owner) is not str or not owner:
        raise ValueError("genuine raw original-source bytes and owner required")
    if horizon not in (1536, 4096, HORIZON):
        raise ValueError("frozen horizon only")
    stride = 4 if horizon < HORIZON else STRIDE
    limit = min(horizon, len(raw))
    recent = RawRing(RECENT)
    rings = {b: RawRing(b) for b in BUDGETS}
    legacy = {b: FixedKeyTable(b, 2) for b in BUDGETS}
    proposed = {(b, p): PriorityKeyTable(b, p) for b in BUDGETS for p in POLICIES}
    names = ([f"raw_{b}" for b in BUDGETS]
             + [f"k1_lru_{b}" for b in BUDGETS]
             + [f"k1_{p}_{b}" for b in BUDGETS for p in POLICIES])
    stats = {name: metrics() for name in names}
    cases = 0
    identities = hashlib.sha256()
    for t in range(limit):
        if t >= KEYLEN:
            key = raw[t - KEYLEN:t]
            if t >= START and t % stride == 0:
                label = raw[t]  # EVALUATOR ONLY, not passed to predictors before score.
                cases += 1
                identities.update(owner.encode() + b":" + str(t).encode() + b":" + key)
                tick = time.perf_counter_ns()
                recent_answer, recent_work = recent.lookup(key)
                recent_ns = time.perf_counter_ns() - tick
                for budget, ring in rings.items():
                    tick = time.perf_counter_ns()
                    answer, work = ring.lookup(key)
                    score(stats[f"raw_{budget}"], answer, label, work,
                          time.perf_counter_ns() - tick)
                for budget, tab in legacy.items():
                    tick = time.perf_counter_ns()
                    answer = tab.read(key)
                    score(stats[f"k1_lru_{budget}"],
                          recent_answer if recent_answer is not None else answer,
                          label, recent_work + 2, time.perf_counter_ns() - tick + recent_ns)
                for (budget, policy), tab in proposed.items():
                    before = tab.probes
                    tick = time.perf_counter_ns()
                    answer = tab.read(key)
                    score(stats[f"k1_{policy}_{budget}"],
                          recent_answer if recent_answer is not None else answer,
                          label, recent_work + tab.probes - before,
                          time.perf_counter_ns() - tick + recent_ns)
            # Insert only after all arms have predicted the actual x[t].
            for budget, tab in legacy.items():
                tick = time.perf_counter_ns()
                tab.update(key, raw[t], t)
                stats[f"k1_lru_{budget}"]["update_ns"] += time.perf_counter_ns() - tick
            for (budget, policy), tab in proposed.items():
                tick = time.perf_counter_ns()
                tab.update(key, raw[t], t)
                stats[f"k1_{policy}_{budget}"]["update_ns"] += time.perf_counter_ns() - tick
        tick = time.perf_counter_ns()
        recent.observe(raw[t])
        ns = time.perf_counter_ns() - tick
        for b in BUDGETS:
            stats[f"k1_lru_{b}"]["update_ns"] += ns
            for p in POLICIES:
                stats[f"k1_{p}_{b}"]["update_ns"] += ns
        for b, ring in rings.items():
            tick = time.perf_counter_ns()
            ring.observe(raw[t])
            stats[f"raw_{b}"]["update_ns"] += time.perf_counter_ns() - tick
    if not cases or any(v["queries"] != cases for v in stats.values()):
        raise ValueError("not all arms received identical genuine target positions")
    if any(m.actual_payload_bytes() != b for (b, p), m in proposed.items()):
        raise ValueError("memory payload mismatch")
    return {"owner": owner, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "cases": cases, "case_sha256": identities.hexdigest(),
            "arms": stats, "diagnostics": {
                f"k1_{p}_{b}": {"payload_bytes": m.actual_payload_bytes(),
                                "slots": m.slots, "evictions": m.evictions,
                                "key_hits": m.key_hits, "key_misses": m.key_misses,
                                "probes": m.probes, "updates": m.updates}
                for (b, p), m in proposed.items()}}


def run() -> dict:
    original = fit_sources()
    if len(original) != 14:
        raise ValueError("original licensed FIT owner cohort changed")
    results = {}
    for horizon in (1536, 4096, HORIZON):
        rows = [owner_probe(raw, owner, horizon)
                for owner, raw in sorted(original.items())]
        total = sum(r["cases"] for r in rows)
        arms = {name: {k: sum(row["arms"][name][k] for row in rows)
                       for k in metrics()} for name in rows[0]["arms"]}
        for stat in arms.values():
            stat["accuracy_all_positions"] = stat["correct"] / total
            stat["coverage"] = stat["covered"] / total
        if horizon == HORIZON and (total != 11784
                                  or arms["raw_1344"]["correct"] != 4152
                                  or arms["k1_lru_1344"]["correct"] != 2148):
            raise ValueError("original publisher long-horizon references drifted")
        results[str(horizon)] = {
            "original_positions": total, "arms": arms,
            "per_owner": rows}
    return {
        "study": "LANGMEMORY056",
        "status": "SOURCE_ONLY_NO_FIT_KEY_PRIORITY_INTERNAL_DIAGNOSTIC",
        "source": "genuine CPython 3.13.0 publisher FIT owners, previously exposed",
        "cohorts": results,
        "memory": "8 byte fixed exact-key record + 64 byte recent raw ring",
        "fixed_byte_budgets": list(BUDGETS),
        "priority": {"lfu": "saturated key frequency",
                     "decayed": "floor(frequency*128/(128+source_clock_age))",
                     "replacement": "lowest priority; tie oldest; 2-way sets"},
        "training_updates": 0, "gpu_minutes": 0, "protected_TEST": "NOT_ACCESSED",
        "checkpoint_replay": "NOT_APPLICABLE_NO_FIT",
        "source_only": True, "independent_publisher_holdout": False,
        "claims": "NO_NEURAL_GAIN_OR_ARCHITECTURE_SUPERIORITY",
        "limitations": [
            "same 14 authentic original FIT owners previously exposed",
            "frequency saturation/aging policy predetermined, not a trained gate",
            "real bytearray payload excludes Python allocator and wrapper bookkeeping",
            "CPU wall times measure Python implementations, not mobile/energy",
            "no original neural checkpoint loaded; no DEV/TEST accessed",
            "raw ring has different query work; equal bytes not equal latency",
        ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to replace prior empirical source receipt")
    value = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"study": value["study"],
                      "source_only": True, "gpu_minutes": 0,
                      "summaries": {
                          k: {a: {"correct": v["correct"], "covered": v["covered"]}
                              for a, v in row["arms"].items()}
                          for k, row in value["cohorts"].items()}}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
