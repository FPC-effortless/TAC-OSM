"""Target-side metrics for genuine-source byte and authored ARC evaluations.

No model, optimizer, data download or gold-derived output constraints.
First-byte likelihood, full likelihood and free-running exactness are distinct.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np


def nextbyte_metrics(logits, targets, valid=None):
    """Aggregate genuine next-byte CE/accuracy with explicit target denominator."""
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if logits.shape[:-1] != targets.shape or logits.shape[-1] != 256 or not np.isfinite(logits).all():
        raise ValueError("finite aligned 256-byte logits required")
    if targets.dtype.kind not in "iu" or np.any(targets < 0) or np.any(targets > 255):
        raise ValueError("integer byte targets required")
    valid = np.ones(targets.shape, bool) if valid is None else np.asarray(valid)
    if valid.shape != targets.shape or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("nonempty explicit valid-target mask required")
    shifted = logits - logits.max(-1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    nll = -np.take_along_axis(lp, targets[..., None], -1).squeeze(-1)
    correct = logits.argmax(-1) == targets
    return {"target_bytes": int(valid.sum()), "NLL_nats_per_byte": float(nll[valid].mean()),
            "bits_per_byte": float(nll[valid].mean()) / math.log(2),
            "nextbyte_accuracy": float(correct[valid].mean()), "correct_bytes": int(correct[valid].sum()),
            "scope": "teacher-forced one-step byte predictions; not exact free-running task success"}


def decode_arc_output(raw):
    """Independent exact implementation of original compact byte-grid grammar."""
    raw = bytes(raw)
    if b";" not in raw:
        raise ValueError("missing target-independent output delimiter")
    grid = raw.split(b";", 1)[0]
    if len(grid) < 3 or grid[0] not in b"123456" or grid[1] not in b"123456":
        raise ValueError("invalid grid dimensions")
    rows, cols = grid[0] - 48, grid[1] - 48
    if len(grid[2:]) != rows * cols or any(v not in b"0123456789" for v in grid[2:]):
        raise ValueError("invalid grid cell payload")
    return [[v - 48 for v in grid[2 + i * cols:2 + (i + 1) * cols]] for i in range(rows)]


def score_episodes(log_probs, generated, actors, judge, available=None):
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.shape != (len(actors), 256) or len(generated) != len(actors):
        raise ValueError("complete actor-aligned prediction cohort required")
    available = np.ones(len(actors), bool) if available is None else np.asarray(available)
    if available.dtype != np.bool_ or available.shape != (len(actors),):
        raise ValueError("explicit availability for all originally selected cases required")
    rows = []
    for i, actor in enumerate(actors):
        gold = judge[actor["id"]]
        row = {"id": actor["id"], "domain": actor["domain"], "group": gold["group"], "cluster": gold["cluster_id"],
               "status": "MEASURED" if available[i] else "INCONCLUSIVE_UNAVAILABLE"}
        if not available[i]:
            rows.append(row); continue
        if not np.isfinite(lp[i]).all() or abs(np.logaddexp.reduce(lp[i])) > 1e-4:
            raise ValueError("finite normalized actor probabilities required")
        raw = bytes(generated[i]); limit = actor["generation_limit"]
        if len(raw) != limit or limit != {"lexical": 4, "arc": 40}[actor["domain"]]:
            raise ValueError("generation limit must be fixed independently of gold length")
        target = bytes(gold["target_bytes"])
        if int(lp[i].argmax()) != raw[0]:
            raise ValueError("recorded greedy first byte disagrees with actor distribution")
        if actor["domain"] == "lexical":
            valid, exact = True, raw == target
        else:
            try:
                pred = decode_arc_output(raw); valid = True
            except ValueError:
                pred, valid = None, False
            exact = valid and pred == decode_arc_output(target)
        probabilities = np.exp(lp[i])
        row.update(first_byte_NLL=-float(lp[i, target[0]]), first_byte_accuracy=int(raw[0] == target[0]),
                   exact_success=int(exact), format_valid=int(valid),
                   confidence=float(probabilities.max()),
                   brier=float(np.square(probabilities).sum() - 2 * probabilities[target[0]] + 1))
        rows.append(row)
    return rows


def cohort_summary(rows):
    groups = defaultdict(list)
    for r in rows: groups[(r["domain"], r["group"])].append(r)
    result = []
    for (domain, group), rr in sorted(groups.items()):
        measured = [r for r in rr if r["status"] == "MEASURED"]
        result.append({"domain": domain, "group": group, "original_cases": len(rr), "measured_cases": len(measured),
                       "unavailable_cases": len(rr) - len(measured),
                       "first_byte_NLL": float(np.mean([r["first_byte_NLL"] for r in measured])) if measured else None,
                       "first_byte_accuracy": float(np.mean([r["first_byte_accuracy"] for r in measured])) if measured else None,
                       "exact_successes": sum(r["exact_success"] for r in measured),
                       "exact_rate": float(np.mean([r["exact_success"] for r in measured])) if measured else None,
                       "format_valid": sum(r["format_valid"] for r in measured),
                       "complete_cohort_status": "MEASURED" if len(measured) == len(rr) else "INCONCLUSIVE_FULL_COHORT"})
    return result


def paired_cluster_effect(values, clusters, *, seed=101001, draws=5000):
    """Case effects should first average paired model seeds within each case.

    This interval is exploratory, especially with four lexical sources. It is
    not a multiple-comparison-adjusted discovery or independent qualification.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) != len(clusters) or len(v) == 0 or not np.isfinite(v).all():
        raise ValueError("finite aligned nonempty paired case effects required")
    names = sorted(set(clusters))
    means = np.array([v[np.array([c == n for c in clusters])].mean() for n in names])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(names), (draws, len(names)))].mean(1)
    return {"case_mean": float(v.mean()), "equal_cluster_mean": float(means.mean()), "clusters": len(names),
            "CI95_exploratory": np.quantile(samples, [.025, .975]).tolist(), "draws": draws,
            "not_independent_or_multiplicity_adjusted": True}
