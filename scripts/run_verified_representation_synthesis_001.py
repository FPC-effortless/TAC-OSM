#!/usr/bin/env python3
"""Confirmatory runner for TACOSM-VRS-DYNAMIC-REPRESENTATION-001.

The proposer is external and frozen. This runner only validates the frozen
artifact and evaluates fixed representations on a deterministic synthetic
domain. The checked-in smoke fixture is never admissible as confirmatory
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "TACOSM-VRS-DYNAMIC-REPRESENTATION-001"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (64, 128, 256, 512)
K_LEVELS = (4, 8, 16, 32)
TRIALS = 64
HORIZONS = (1, 4, 16, 32)
NEAR = 0.35
CAP_FLOOR = 0.80
SUCCESS_TOL = 0.02
WORK_PER_CANDIDATE = 12
VAR_FLOOR = 1e-6
CORR_LIMIT = 0.98
DOMAIN_VERSION = "rover-dynamics-v1"
ACTIONS = ("drive", "climb", "cool", "repair")
DELTA = {
    "drive": (0.08, 0.00, 0.05, 0.03),
    "climb": (0.06, 0.08, 0.08, 0.04),
    "cool": (0.01, 0.00, -0.15, 0.00),
    "repair": (0.04, 0.00, -0.02, -0.12),
}
FEATURE_KEYS = {"name", "description", "anchors", "unit_interval", "source"}
ANCHOR_LEVELS = {"low": 0.0, "mid": 0.5, "high": 1.0}


@dataclass(frozen=True)
class State:
    id: str
    power: float
    grip: float
    heat: float
    wear: float
    affinity: float

    def raw(self):
        return (self.power, self.grip, self.heat, self.wear, self.affinity)


@dataclass(frozen=True)
class Task:
    id: str
    required_power: float
    roughness: float
    volatility: float
    terrain: float

    def raw(self):
        # The query coordinate is deliberately a public task description, not
        # an outcome-derived statistic.
        return (
            self.required_power,
            self.roughness,
            self.volatility,
            self.terrain,
            0.5,
        )


def clamp(x):
    return max(0.0, min(1.0, x))


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


CALIBRATION_STATES = tuple(
    State(f"cal:{i}", *vals)
    for i, vals in enumerate(
        (
            (0.00, 0.25, 0.25, 0.25, 0.25),
            (0.50, 0.25, 0.25, 0.25, 0.25),
            (1.00, 0.25, 0.25, 0.25, 0.25),
            (0.25, 0.00, 0.25, 0.25, 0.25),
            (0.25, 0.50, 0.25, 0.25, 0.25),
            (0.25, 1.00, 0.25, 0.25, 0.25),
            (0.25, 0.25, 1.00, 0.25, 0.25),
            (0.25, 0.25, 0.50, 0.25, 0.25),
            (0.25, 0.25, 0.00, 0.25, 0.25),
            (0.25, 0.25, 0.25, 1.00, 0.25),
            (0.25, 0.25, 0.25, 0.50, 0.25),
            (0.25, 0.25, 0.25, 0.00, 0.25),
            (0.25, 0.25, 0.25, 0.25, 0.00),
            (0.25, 0.25, 0.25, 0.25, 0.50),
            (0.25, 0.25, 0.25, 0.25, 1.00),
        )
    )
)

SOURCES = {
    "power": lambda s: s.power,
    "grip": lambda s: s.grip,
    "thermal_stability": lambda s: 1 - s.heat,
    "mechanical_integrity": lambda s: 1 - s.wear,
    "terrain_affinity": lambda s: s.affinity,
}


def states(seed, m):
    r = random.Random(seed * 1000003 + m * 7919 + 101)
    return tuple(
        State(
            f"s:{seed}:{m}:{i}",
            r.random(),
            r.random(),
            r.random() * 0.8,
            r.random() * 0.8,
            r.random(),
        )
        for i in range(m)
    )


def tasks(seed, m):
    r = random.Random(seed * 2000003 + m * 6151 + 211)
    return tuple(
        Task(
            f"q:{seed}:{m}:{i}",
            0.45 + 0.45 * r.random(),
            r.random(),
            r.random(),
            r.random(),
        )
        for i in range(TRIALS)
    )


def score(s, t):
    w = (0.28, 0.24, 0.18, 0.10, 0.20)
    return 1.0 - sum(a * abs(x - y) for a, x, y in zip(w, s.raw(), t.raw()))


def step(s, action):
    dp, dg, dh, dw = DELTA[action]
    return State(
        s.id + "->" + action,
        clamp(s.power - dp),
        clamp(s.grip - dg),
        clamp(s.heat + dh),
        clamp(s.wear + dw),
        s.affinity,
    )


def safe(s, action):
    z = step(s, action)
    return int(
        z.power >= 0.15
        and z.grip >= 0.15
        and z.heat <= 0.92
        and z.wear <= 0.92
    )


def fixed_cases(seed, pool):
    # Pair identities are fixed without representation coordinates.  Every
    # sampled base pair gets all four one-step actions; longer horizons use a
    # seeded action schedule.  Thus the tractable one-step action space is
    # exhaustive and the multi-step schedule remains representation-blind.
    r = random.Random(seed * 3000017 + 991)
    ids = [s.id for s in pool]
    base_pairs = []
    for i in range(min(64, max(32, len(ids) // 2))):
        a, b = r.sample(ids, 2)
        base_pairs.append((a, b))
    out = []
    serial = 0
    for a, b in base_pairs:
        for act in ACTIONS:
            out.append(
                {
                    "id": f"p:{seed}:{serial}",
                    "a": a,
                    "b": b,
                    "actions": [act],
                    "horizon": 1,
                }
            )
            serial += 1
        for h in HORIZONS[1:]:
            out.append(
                {
                    "id": f"p:{seed}:{serial}",
                    "a": a,
                    "b": b,
                    "actions": [r.choice(ACTIONS) for _ in range(h)],
                    "horizon": h,
                }
            )
            serial += 1
    return tuple(out)


class ProposalError(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    raw: dict

    @classmethod
    def load(cls, path):
        p = cls(json.loads(Path(path).read_text(encoding="utf-8")))
        p.audit()
        return p

    @property
    def mode(self):
        return self.raw["mode"]

    @property
    def dim(self):
        return len(self.raw["features"])

    def _validate_vector(self, value, label):
        if not isinstance(value, (list, tuple)) or len(value) != self.dim:
            raise ProposalError(f"{label} must have exactly {self.dim} dimensions")
        out = tuple(float(x) for x in value)
        if not all(math.isfinite(x) for x in out):
            raise ProposalError(f"{label} contains non-finite values")
        return out

    def audit(self):
        required = {
            "proposal_id",
            "proposer",
            "prompt_hash",
            "domain_hash",
            "state_universe_hash",
            "task_universe_hash",
            "features",
            "mode",
            "build_cost_units",
            "build_cost_unit",
            "provenance",
        }
        miss = required - set(self.raw)
        if miss:
            raise ProposalError(f"proposal missing fields: {sorted(miss)}")
        if self.mode not in {"scored", "expression"}:
            raise ProposalError("invalid proposal mode")
        if not self.raw["features"]:
            raise ProposalError("proposal has no features")
        prov = self.raw["provenance"]
        for key in (
            "proposer_model_identifier",
            "prompt_hash",
            "domain_description_hash",
            "generation_timestamp",
            "decoding_settings",
            "outcome_visibility",
        ):
            if key not in prov:
                raise ProposalError(f"provenance missing {key}")
        if prov["outcome_visibility"] != "none":
            raise ProposalError("outcome_visibility must be 'none'")
        if prov["prompt_hash"] != self.raw["prompt_hash"]:
            raise ProposalError("proposal/provenance prompt hash mismatch")
        if prov["domain_description_hash"] != self.raw["domain_hash"]:
            raise ProposalError("proposal/provenance domain hash mismatch")
        for f in self.raw["features"]:
            unknown = set(f) - FEATURE_KEYS
            if unknown:
                raise ProposalError(
                    f"unregistered feature fields: {sorted(unknown)}"
                )
            anchors = f.get("anchors", [])
            if len(anchors) != 3:
                raise ProposalError("each feature must have exactly three anchors")
            names = [a.get("name") for a in anchors]
            if len(set(names)) != 3:
                raise ProposalError("anchor names must be unique")
            for a in anchors:
                if set(a) - {"name", "state_id", "value"}:
                    raise ProposalError("anchor contains unregistered fields")
        if self.mode == "scored":
            if not self.raw.get("scored_states"):
                raise ProposalError("scored proposal has no scored_states")
            if not self.raw.get("scored_queries"):
                raise ProposalError("scored proposal has no scored_queries")
            if set(self.raw.get("scored_calibration_states", {})) != {
                s.id for s in CALIBRATION_STATES
            }:
                raise ProposalError(
                    "scored proposal must cover exactly the frozen calibration states"
                )
            for sid, value in self.raw["scored_calibration_states"].items():
                self._validate_vector(value, f"calibration state {sid}")
            for sid, value in self.raw["scored_states"].items():
                self._validate_vector(value, f"scored state {sid}")
            for qid, value in self.raw["scored_queries"].items():
                self._validate_vector(value, f"scored query {qid}")
        else:
            for idx, f in enumerate(self.raw["features"]):
                if f.get("source") not in SOURCES:
                    raise ProposalError(
                        f"expression fixture feature {idx} has unsupported source"
                    )

    def require_confirmatory(self):
        if (
            self.mode != "scored"
            or self.raw["proposer"].lower() == "deterministic-fixture"
        ):
            raise ProposalError("only an external frozen scored proposal is confirmatory")

    def vec(self, s):
        if self.mode == "scored":
            table = (
                self.raw["scored_calibration_states"]
                if s.id.startswith("cal:")
                else self.raw["scored_states"]
            )
            try:
                return tuple(float(x) for x in table[s.id])
            except KeyError as e:
                raise ProposalError(f"missing scored state {s.id}") from e
        vals = [
            SOURCES[f.get("source")](s)
            for f in self.raw["features"]
            if f.get("source") in SOURCES
        ]
        if len(vals) != self.dim:
            raise ProposalError("unsupported smoke feature source or dimension")
        return tuple(vals)

    def qvec(self, t):
        if self.mode == "scored":
            try:
                return tuple(float(x) for x in self.raw["scored_queries"][t.id])
            except KeyError as e:
                raise ProposalError(f"missing scored task {t.id}") from e
        if self.dim != 5:
            raise ProposalError("expression fixture must use five canonical features")
        return t.raw()


def corr(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    den = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(da, db)) / den


def mask_for(rows):
    d = len(rows[0])
    cols = [[r[j] for r in rows] for j in range(d)]
    keep = []
    for j, c in enumerate(cols):
        if statistics.pvariance(c) < VAR_FLOOR:
            continue
        if any(abs(corr(c, cols[i])) >= CORR_LIMIT for i in keep):
            continue
        keep.append(j)
    if not keep:
        raise ProposalError("feature filtering removed every representation dimension")
    return tuple(keep)


def effective_rank(rows, tol=1e-8):
    if not rows:
        return 0
    cols = [[float(r[j]) for r in rows] for j in range(len(rows[0]))]
    basis = []
    for col in cols:
        v = list(col)
        for b in basis:
            den = sum(x * x for x in b)
            if den:
                a = sum(x * y for x, y in zip(v, b)) / den
                v = [x - a * y for x, y in zip(v, b)]
        if math.sqrt(sum(x * x for x in v)) > tol:
            basis.append(v)
    return len(basis)


def masked(v, keep):
    return tuple(v[i] for i in keep)


def representation_distance(a, b):
    if len(a) != len(b):
        raise ProposalError("representation vectors must have equal dimension")
    if not a:
        raise ProposalError("representation dimension must be positive")
    return math.dist(a, b) / math.sqrt(len(a))


def route(rows, q, k):
    order = sorted(
        range(len(rows)),
        key=lambda i: (math.dist(rows[i], q), i),
    )
    return order[: min(k, len(order))]


def evaluate(pool, ts, reps, qs, k):
    succ = []
    raw = []
    for t in ts:
        optimum = max(score(s, t) for s in pool)
        raw_i = max(range(len(pool)), key=lambda i: (score(pool[i], t), -i))
        chosen = route(reps, qs[t.id], k)
        win = max(chosen, key=lambda i: (score(pool[i], t), -i))
        succ.append(int(score(pool[win], t) >= optimum - SUCCESS_TOL))
        raw.append(int(score(pool[raw_i], t) >= optimum - SUCCESS_TOL))
    return {
        "n": len(ts),
        "success": statistics.fmean(succ),
        "raw_success": statistics.fmean(raw),
        "work_units": k * WORK_PER_CANDIDATE,
        "routing_units": len(pool) * (len(reps[0]) if reps else 0),
    }


def dynamic(proposal, pool, cases, keep):
    byid = {s.id: s for s in pool}
    near = checked = viol = 0
    for c in cases:
        a, b = byid[c["a"]], byid[c["b"]]
        if representation_distance(masked(proposal.vec(a), keep), masked(proposal.vec(b), keep)) > NEAR:
            continue
        near += 1
        if len(c["actions"]) != c["horizon"]:
            raise ProposalError(f"horizon mismatch {c['id']}")
        ca, cb = a, b
        for act in c["actions"]:
            checked += 1
            if safe(ca, act) != safe(cb, act):
                viol += 1
            ca, cb = step(ca, act), step(cb, act)
    if near == 0 or checked == 0:
        return {
            "applicable": False,
            "near_pairs": near,
            "checked_transitions": checked,
        }
    return {
        "applicable": True,
        "near_pairs": near,
        "checked_transitions": checked,
        "violation_count": viol,
        "violation_rate": viol / checked,
        "passed": viol == 0,
    }


def run(proposal_path, smoke=False):
    # Generate the fixed universes and pair schedules before loading or scoring
    # any proposal. This is the anti-selection-bias boundary.
    selected_seeds = SEEDS[:1] if smoke else SEEDS
    selected_m = M_LEVELS[:1] if smoke else M_LEVELS
    universe_states = []
    universe_tasks = []
    cases = {}
    for seed in selected_seeds:
        for m in selected_m:
            ps = states(seed, m)
            ts = tasks(seed, m)
            universe_states.extend(ps)
            universe_tasks.extend(ts)
            cases[(seed, m)] = fixed_cases(seed, ps)

    p = Proposal.load(proposal_path)
    expected_domain_hash = digest(
        {
            "domain_version": DOMAIN_VERSION,
            "state_fields": ["power", "grip", "heat", "wear", "affinity"],
            "actions": DELTA,
            "outcome": "post-action safety over power/grip/heat/wear",
        }
    )
    if p.raw["domain_hash"] != expected_domain_hash:
        raise ProposalError("domain hash mismatch")

    if p.mode == "scored":
        state_hash = digest([asdict(s) for s in universe_states])
        task_hash = digest([asdict(t) for t in universe_tasks])
        if p.raw["state_universe_hash"] != state_hash:
            raise ProposalError("state universe hash mismatch")
        if p.raw["task_universe_hash"] != task_hash:
            raise ProposalError("task universe hash mismatch")
        expected_states = {s.id for s in universe_states}
        expected_tasks = {t.id for t in universe_tasks}
        if set(p.raw["scored_states"]) != expected_states:
            raise ProposalError(
                "scored state coverage is not exactly the frozen evaluation universe"
            )
        if set(p.raw["scored_queries"]) != expected_tasks:
            raise ProposalError(
                "scored query coverage is not exactly the frozen evaluation universe"
            )

    if not smoke:
        p.require_confirmatory()

    anchor_checks = 0
    calibration_by_id = {s.id: s for s in CALIBRATION_STATES}
    for idx, f in enumerate(p.raw["features"]):
        levels = {a.get("name"): float(a.get("value")) for a in f.get("anchors", [])}
        if set(levels) != set(ANCHOR_LEVELS):
            raise ProposalError("each feature must provide low/mid/high calibration anchors")
        for name, expected in ANCHOR_LEVELS.items():
            if abs(levels[name] - expected) > 1e-6:
                raise ProposalError(f"anchor {name} must have value {expected}")
        seen_states = set()
        for a in f.get("anchors", []):
            if a["state_id"] in seen_states:
                raise ProposalError("a feature reuses an anchor state")
            seen_states.add(a["state_id"])
            s = calibration_by_id.get(str(a["state_id"]))
            if s is None:
                raise ProposalError(
                    f"anchor uses non-calibration state: {a['state_id']}"
                )
            if abs(p.vec(s)[idx] - float(a["value"])) > 1e-6:
                raise ProposalError(f"anchor mismatch: {a['state_id']}")
            anchor_checks += 1
    if not anchor_checks:
        raise ProposalError("no calibration anchors")

    out = {
        "experiment_id": CONTRACT,
        "status": "smoke" if smoke else "measured",
        "proposal_digest": digest(p.raw),
        "proposal_id": p.raw["proposal_id"],
        "protocol": {
            "seeds": list(selected_seeds),
            "M": list(selected_m),
            "K": list(K_LEVELS if not smoke else (4, 8)),
            "near_threshold": NEAR,
            "success_tolerance": SUCCESS_TOL,
            "work_per_candidate": WORK_PER_CANDIDATE,
        },
        "anchor_checks": anchor_checks,
        "by_M": {},
    }

    for m in selected_m:
        rows = []
        dynamic_rows = {}
        by_k = {k: [] for k in (K_LEVELS if not smoke else (4, 8))}
        for seed in selected_seeds:
            pool = states(seed, m)
            ts = tasks(seed, m)
            raw_reps = [s.raw() for s in pool]
            scalar = [[statistics.fmean(s.raw())] for s in pool]
            semantic = [p.vec(s) for s in pool]
            keep = mask_for([p.vec(s) for s in CALIBRATION_STATES])
            sem = [masked(v, keep) for v in semantic]
            raw_q = {t.id: t.raw() for t in ts}
            scalar_q = {t.id: (statistics.fmean(t.raw()),) for t in ts}
            semantic_q = {t.id: masked(p.qvec(t), keep) for t in ts}
            ks = K_LEVELS if not smoke else (4, 8)
            for k in ks:
                rr = evaluate(pool, ts, raw_reps, raw_q, k)
                sr = evaluate(pool, ts, scalar, scalar_q, k)
                mr = evaluate(pool, ts, sem, semantic_q, k)
                rows.append(
                    {
                        "seed": seed,
                        "K": k,
                        "raw": rr,
                        "scalar": sr,
                        "semantic": mr,
                    }
                )
                by_k[k].append(mr)
            dynamic_rows[str(seed)] = dynamic(
                p, pool, cases[(seed, m)], keep
            )

        pooled = {}
        for k, cells in by_k.items():
            sem = sum(c["success"] for c in cells) / len(cells)
            raw = sum(c["raw_success"] for c in cells) / len(cells)
            scalar = sum(
                r["scalar"]["success"]
                for r in rows
                if r["K"] == k
            ) / len(cells)
            semantic_exec_fraction = k / m
            pooled[str(k)] = {
                "semantic_success": sem,
                "raw_reference_success": raw,
                "scalar_success": scalar,
                "capability_retention": sem / max(raw, 1e-12),
                "execution_work_fraction": semantic_exec_fraction,
                "semantic_minus_scalar_success": sem - scalar,
            }

        eligible = [
            (int(k), v)
            for k, v in pooled.items()
            if v["semantic_success"] >= CAP_FLOOR * v["raw_reference_success"]
        ]
        selected = (
            min(eligible, key=lambda kv: kv[1]["execution_work_fraction"])
            if eligible
            else None
        )

        seed_blocks = {seed: {} for seed in selected_seeds}
        for r in rows:
            seed_blocks[r["seed"]][r["K"]] = {
                "success": r["semantic"]["success"],
                "raw_success": r["raw"]["raw_success"],
            }

        boot_vals = []
        rng = random.Random(7919)
        for _ in range(4000):
            sampled = [
                selected_seeds[rng.randrange(len(selected_seeds))]
                for _ in selected_seeds
            ]
            choices = []
            for k in by_k:
                cells = [seed_blocks[seed][k] for seed in sampled]
                sem = sum(c["success"] for c in cells) / len(cells)
                raw = sum(c["raw_success"] for c in cells) / len(cells)
                if sem >= CAP_FLOOR * raw:
                    choices.append(k / m)
            if choices:
                boot_vals.append(min(choices))
        boot_vals.sort()
        boot = (
            None
            if not boot_vals
            else {
                "mean": statistics.fmean(boot_vals),
                "lo": boot_vals[min(100, len(boot_vals) - 1)],
                "hi": boot_vals[max(0, len(boot_vals) - 101)],
            }
        )

        primary = None
        if selected is not None:
            sk, sv = selected
            raw_route = m * 5
            sem_route = m * len(keep)
            raw_steady = raw_route + sk * WORK_PER_CANDIDATE
            sem_steady = sem_route + sk * WORK_PER_CANDIDATE
            build_cost = p.raw["build_cost_units"]
            denom = max(raw_steady - sem_steady, 0)
            break_even = None
            if p.raw["build_cost_unit"] == "execution_work_units" and denom > 0:
                break_even = build_cost / denom
            primary = {
                "K": sk,
                "execution_work_fraction": sv["execution_work_fraction"],
                "capability_retention": sv["capability_retention"],
                "representation_dimension": len(keep),
                "steady_state_raw_cost_units": raw_steady,
                "steady_state_semantic_cost_units": sem_steady,
                "representation_build_cost_units": build_cost,
                "break_even_queries": break_even,
            }

        geometry_rows = [p.vec(s) for s in states(0, m)]
        geometry_sem = [masked(v, keep) for v in geometry_rows]
        out["by_M"][str(m)] = {
            "rows": rows,
            "pooled_by_K": pooled,
            "dynamic": dynamic_rows,
            "geometry": {
                "dimension_before_filter": p.dim,
                "dimension_after_filter": len(keep),
                "effective_rank": effective_rank(geometry_sem),
            },
            "primary": primary,
            "seed_primary_bootstrap": boot,
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = run(args.proposal, args.smoke)
    path = ROOT / "artifacts" / f"{CONTRACT}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
