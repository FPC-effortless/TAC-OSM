"""Fail-closed anti-cheat and failure-localization tools for TAC-OSM research.

Engineering diagnostics only: passing these probes does NOT establish genuine
learning, absence of every covert channel, causal identification, or capability.
Keep source-owned data and sealed holdouts outside the diagnostic report.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import numbers
import re
from typing import Any, Callable, Iterator

# Explicit allowed inputs are preferable to a growing blacklist. The blacklist
# also scans nested dictionaries/dataclasses that an allowlisted input contains.
DENIED_NAMES = frozenset({
    "answer", "answers", "label", "labels", "target", "targets",
    "reward", "rewards", "outcome", "outcomes", "oracle", "gold",
    "gold_index", "correct_action", "expected_action", "solution",
    "next_measured_state", "new_intervention_onset",
    "next_intervention_onset_when_policy_controlled", "future_actions",
    "future_state", "test_label", "evaluation_label",
})
DENIED_FRAGMENTS = ("oracle", "gold", "ground_truth", "true_edge",
                    "correct_action", "hindsight", "post_action",
                    "future_", "next_", "label", "target", "reward")
STATUSES = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})


def _name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _denied(name: str) -> bool:
    n = _name(name)
    return n in DENIED_NAMES or any(piece in n for piece in DENIED_FRAGMENTS)


@dataclass(frozen=True)
class Finding:
    check: str
    status: str
    reason: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unknown anticheat status")

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "status": self.status,
                "reason": self.reason, "evidence": self.evidence}


class AuditGateError(RuntimeError):
    """Research run is not eligible to claim integrity."""


class AuditLedger:
    """Auditable, machine-readable gate; missing checks never pass implicitly.

    Run IDs/digests and check names are retained. Raw source records, secret
    features, answer values and model weights are intentionally NOT logged.
    """

    def __init__(self, run_id: str, required: Sequence[str]):
        if not run_id or not required or len(set(required)) != len(required):
            raise ValueError("run_id and unique required check names required")
        self.run_id = run_id
        self.required = tuple(required)
        self.findings: dict[str, Finding] = {}

    def add(self, finding: Finding) -> None:
        if finding.check in self.findings:
            raise AuditGateError("duplicate check: " + finding.check)
        self.findings[finding.check] = finding

    def report(self) -> dict[str, Any]:
        missing = sorted(set(self.required) - self.findings.keys())
        items = [self.findings[k].as_dict() for k in sorted(self.findings)]
        status = ("FAIL" if any(x["status"] == "FAIL" for x in items)
                  else "INCONCLUSIVE" if missing or any(
                      x["status"] == "INCONCLUSIVE" for x in items)
                  else "PASS")
        return {"schema": "tac-osm-anticheat/v1", "run_id": self.run_id,
                "status": status, "required": list(self.required),
                "missing": missing, "findings": items,
                "qualification": "engineering_only_not_capability_evidence"}

    def assert_clean(self) -> None:
        report = self.report()
        if report["status"] != "PASS":
            raise AuditGateError(
                "anti-cheat gate not PASS: " + report["status"] +
                " missing=" + repr(report["missing"]) +
                " findings=" + repr([(k, v.status, v.reason)
                                     for k, v in sorted(self.findings.items())
                                     if v.status != "PASS"]))


def _walk(value: Any, path: str, visited: set[int],
          problems: list[str], *, depth: int = 0) -> None:
    """Inspect reachable containers, never invoke arbitrary object descriptors."""
    if depth > 32:
        problems.append(path + ": nesting limit exceeded")
        return
    if value is None or isinstance(value, (str, bytes, numbers.Number)):
        return
    # Tensor/array payloads are opaque: their semantic provenance MUST be
    # certified by their containing named feature's producer and taint probe.
    if type(value).__module__.split(".")[0] in ("torch", "numpy") and (
            hasattr(value, "shape") and hasattr(value, "dtype")):
        return
    if id(value) in visited:
        return
    visited.add(id(value))
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(path + ": non-string field name")
                continue
            child = path + "." + key
            if _denied(key):
                problems.append(child + ": forbidden field")
            _walk(item, child, visited, problems, depth=depth + 1)
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            child = path + "." + f.name
            if _denied(f.name):
                problems.append(child + ": forbidden field")
            _walk(getattr(value, f.name), child, visited, problems, depth=depth + 1)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _walk(item, f"{path}[{i}]", visited, problems, depth=depth + 1)
    else:
        problems.append(path + ": uninspectable object " + type(value).__name__)


def audit_input_boundary(inputs: Mapping[str, Any],
                         allowed_roots: Sequence[str],
                         *, check: str = "input_boundary") -> Finding:
    """Allowlist root features and deny outcome/oracle fields at ANY depth."""
    if not isinstance(inputs, Mapping) or not allowed_roots:
        return Finding(check, "FAIL", "invalid boundary contract", {})
    allowed = set(allowed_roots)
    violations: list[str] = []
    for key, val in inputs.items():
        if not isinstance(key, str):
            violations.append("root: non-string field")
            continue
        if key not in allowed:
            violations.append("root." + key + ": not in declared allowlist")
        if _denied(key):
            violations.append("root." + key + ": forbidden field")
        _walk(val, "root." + key, set(), violations)
    if len(set(allowed_roots)) != len(allowed_roots):
        violations.append("duplicate allowlist entry")
    return Finding(check, "FAIL" if violations else "PASS",
                   "boundary violations" if violations else "allowlist and nested scan clean",
                   {"paths": sorted(set(violations)),
                    "allowed_roots": sorted(allowed),
                    "observed_roots": sorted(str(k) for k in inputs)})


def _plain(value: Any) -> Any:
    """Canonicalize probes without ever using repr(object) or object identity."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("only string-keyed dictionaries are auditable")
        return {k: _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError("nonfinite value in audit probe")
        return f
    if type(value).__module__.split(".")[0] in ("torch", "numpy") and (
            hasattr(value, "shape") and hasattr(value, "tolist")):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "values": _plain(value.detach().cpu().tolist()
                                 if hasattr(value, "detach") else value.tolist())}
    raise TypeError("unsupported audit value: " + type(value).__name__)


def fingerprint(value: Any) -> str:
    """Diagnostic-only content digest; NOT proof of authentic data provenance."""
    return hashlib.sha256(json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def probe_future_invariance(
        before: Any, poisoned: Any,
        prepare: Callable[[Any], Any], targets: Callable[[Any], Any],
        *, check: str = "future_invariance") -> Finding:
    """Counterfactual source poisoning: change only AFTER-cutoff outcomes.

    Both sides must be independently built from unmodified original records
    by the caller. If the poison did not change targets, the probe is INVALID.
    """
    try:
        a, b = prepare(copy.deepcopy(before)), prepare(copy.deepcopy(poisoned))
        ta, tb = targets(copy.deepcopy(before)), targets(copy.deepcopy(poisoned))
        ah, bh, th, uh = map(fingerprint, (a, b, ta, tb))
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        return Finding(check, "INCONCLUSIVE", "invalid preparation/probe: " +
                       type(exc).__name__, {})
    if th == uh:
        return Finding(check, "INCONCLUSIVE", "poison did not alter factual target", {})
    if ah != bh:
        return Finding(check, "FAIL", "future-only mutation reached decision inputs",
                       {"before_inputs_sha256": ah, "after_inputs_sha256": bh})
    return Finding(check, "PASS", "future target changed but inputs stayed identical",
                   {"inputs_sha256": ah, "targets_changed": True})


def probe_prediction_invariance(predict: Callable[[Any], Any],
                                clean_input: Any, poisoned_input: Any,
                                *, check: str = "prediction_invariance") -> Finding:
    """Run fixed-weight/seed prediction on paired permitted/poisoned payloads.

    A caller MUST reset model state and RNG inside predict for every call;
    nondeterminism is diagnosed rather than interpreted as a hidden pathway.
    """
    try:
        x = fingerprint(predict(copy.deepcopy(clean_input)))
        repeat = fingerprint(predict(copy.deepcopy(clean_input)))
        y = fingerprint(predict(copy.deepcopy(poisoned_input)))
    except (ValueError, TypeError, AssertionError, RuntimeError) as exc:
        return Finding(check, "INCONCLUSIVE", "prediction probe exception: " +
                       type(exc).__name__, {})
    if x != repeat:
        return Finding(check, "INCONCLUSIVE", "predictor is nondeterministic", {})
    return Finding(check, "PASS" if x == y else "FAIL",
                   "no poisoned-data influence" if x == y
                   else "poisoned information changed prediction",
                   {"clean_prediction_sha256": x, "poison_prediction_sha256": y})


def audit_partitions(splits: Mapping[str, Sequence[Any]],
                     *, check: str = "split_isolation") -> Finding:
    """Exact-ID duplication across FIT/DEV/TEST is prohibited.

    Does not prove no near-duplicates, identity alias, temporal overlap, or
    dataset provenance. Those require separate source-specific tests.
    """
    required = ("FIT", "DEV", "TEST")
    if set(splits) != set(required):
        return Finding(check, "FAIL", "required FIT/DEV/TEST partitions missing", {})
    try:
        ids = {s: [fingerprint(v) for v in splits[s]] for s in required}
    except (ValueError, TypeError) as exc:
        return Finding(check, "FAIL", "invalid partition id: " + str(exc), {})
    if any(not ids[s] for s in required):
        return Finding(check, "FAIL", "empty partition", {})
    within = [s for s in required if len(ids[s]) != len(set(ids[s]))]
    overlaps = [a + "/" + b for i, a in enumerate(required)
                for b in required[i + 1:] if set(ids[a]) & set(ids[b])]
    return Finding(check, "FAIL" if within or overlaps else "PASS",
                   "duplicate or overlapping source IDs" if within or overlaps
                   else "exact source IDs distinct",
                   {"counts": {s: len(ids[s]) for s in required},
                    "duplicate_within": within, "overlap_pairs": overlaps})


def audit_train_only_fit(fit_ids: Sequence[Any], train_ids: Sequence[Any],
                         *, check: str = "preprocessing_fit_scope") -> Finding:
    """A fitted encoder/normalizer must see ONLY registered training rows."""
    try:
        registered = set(fingerprint(i) for i in train_ids)
        actual = set(fingerprint(i) for i in fit_ids)
    except (ValueError, TypeError) as exc:
        return Finding(check, "FAIL", "invalid fit receipt: " + str(exc), {})
    leaked = actual - registered
    return Finding(check, "PASS" if actual and not leaked else "FAIL",
                   "fit scope limited to training" if actual and not leaked
                   else "unknown/nontraining fit IDs or empty fit",
                   {"fit_count": len(actual), "unexpected_count": len(leaked)})


def audit_forward_time(splits: Mapping[str, Sequence[tuple[Any, float]]],
                       *, embargo: float = 0.0,
                       check: str = "forward_time") -> Finding:
    """No backward time travel across splits for an entity with an embargo.

    Rows are (entity_id, observed_timestamp), sourced independently of labels.
    Allows same entity in chronological split; cross-entity generalization is
    a different, separately preregistered question.
    """
    if set(splits) != {"FIT", "DEV", "TEST"} or not math.isfinite(embargo) or embargo < 0:
        return Finding(check, "FAIL", "invalid time protocol", {})
    seen: dict[str, dict[str, list[float]]] = {}
    try:
        for split, rows in splits.items():
            if not rows:
                raise ValueError("empty " + split)
            for entity, timestamp in rows:
                t = float(timestamp)
                if not math.isfinite(t):
                    raise ValueError("nonfinite time")
                eid = fingerprint(entity)
                seen.setdefault(eid, {}).setdefault(split, []).append(t)
        bad = 0
        for by_split in seen.values():
            for left, right in (("FIT", "DEV"), ("DEV", "TEST"), ("FIT", "TEST")):
                if left in by_split and right in by_split and (
                        max(by_split[left]) + embargo >= min(by_split[right])):
                    bad += 1
    except (ValueError, TypeError) as exc:
        return Finding(check, "FAIL", "invalid time row: " + str(exc), {})
    return Finding(check, "FAIL" if bad else "PASS",
                   "time/embargo violations" if bad else "chronological boundaries preserved",
                   {"violating_entity_boundaries": bad, "embargo": embargo,
                    "entities": len(seen)})


def torch_gradient_diagnostics(model: Any, *,
                               expected_prefixes: Sequence[str] = (),
                               check: str = "gradient_health") -> Finding:
    """Run AFTER backward(), BEFORE optimizer.zero_grad(); no optimizer steps.

    Identifies disconnected/zero-gradient named modules and nonfinite gradient
    blocks. Zero gradients on unused heads are descriptive, not automatically
    proof of training failure.
    """
    blocks: dict[str, dict[str, Any]] = {}
    try:
        params = list(model.named_parameters())
        if not params:
            raise ValueError("model has no parameters")
        for name, param in params:
            prefix = name.rsplit(".", 1)[0] if "." in name else name
            row = blocks.setdefault(prefix, {"parameter_count": 0, "disconnected": [],
                                               "zero": [], "nonfinite": [],
                                               "gradient_l2_sq": 0.0})
            row["parameter_count"] += param.numel()
            if not param.requires_grad:
                continue
            if param.grad is None:
                row["disconnected"].append(name)
            elif not bool(param.grad.isfinite().all()):
                row["nonfinite"].append(name)
            else:
                l2 = float(param.grad.detach().float().square().sum().item())
                row["gradient_l2_sq"] += l2
                if l2 == 0.0:
                    row["zero"].append(name)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return Finding(check, "INCONCLUSIVE", "gradient inspection unavailable: " +
                       type(exc).__name__, {})
    absent = [p for p in expected_prefixes if not any(
        n == p or n.startswith(p + ".") for n in blocks)]
    inactive = [p for p in expected_prefixes if not any(
        (n == p or n.startswith(p + ".")) and
        blocks[n]["gradient_l2_sq"] > 0 and not blocks[n]["nonfinite"]
        for n in blocks)]
    bad = any(v["nonfinite"] for v in blocks.values()) or bool(absent or inactive)
    for row in blocks.values():
        row["gradient_l2"] = math.sqrt(row.pop("gradient_l2_sq"))
    return Finding(check, "FAIL" if bad else "PASS",
                   "nonfinite or expected gradient pathway inactive" if bad
                   else "inspected gradient blocks; unrequired zero gradients descriptive",
                   {"blocks": blocks, "missing_expected": absent,
                    "inactive_expected": inactive})


@contextmanager
def trace_torch_modules(model: Any) -> Iterator[list[dict[str, Any]]]:
    """Forward hooks localize exploding/NaN activations without copying data.

    Observe single forward evaluation, not a scientific qualification. Must be
    enclosed in a context manager to remove hooks even after exceptions.
    """
    observations: list[dict[str, Any]] = []
    handles = []

    def inspect(value: Any, seen: list[dict[str, Any]]) -> None:
        if hasattr(value, "detach") and hasattr(value, "isfinite"):
            t = value.detach()
            seen.append({"shape": list(t.shape), "finite": bool(t.isfinite().all()),
                         "max_abs": float(t.abs().max()) if t.numel() else 0.0})
        elif isinstance(value, (list, tuple)):
            for v in value:
                inspect(v, seen)
        elif isinstance(value, Mapping):
            for v in value.values():
                inspect(v, seen)

    def hook(name: str) -> Callable[..., None]:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            seen: list[dict[str, Any]] = []
            inspect(output, seen)
            observations.append({"module": name, "calls": 1,
                                 "outputs": seen,
                                 "finite": bool(seen) and all(v["finite"] for v in seen)})
        return record

    try:
        for name, module in model.named_modules():
            if name and not any(module.children()):  # leaf modules only
                handles.append(module.register_forward_hook(hook(name)))
        yield observations
    finally:
        for handle in handles:
            handle.remove()


def trace_torch_noninterference(model: Any,
                                forward: Callable[[Any, Any], Any],
                                legal_input: Any, denied_only_changed: Any,
                                *, check: str = "graph_noninterference") -> Finding:
    """Locate FIRST leaf module whose output depends on forbidden information.

    Operates on fresh model copies at fixed checkpoint under eval/no_grad.
    The caller must poison *only* illegal metadata, use a deterministic
    forward, and avoid external uncaptured state/global side effects.
    Functional operations outside a module are reported as untraced paths.
    This diagnostic copies tensor values to hash them: do not run it in
    production inference or over private raw logs without authorization.
    """
    try:
        import torch
    except ImportError:
        return Finding(check, "INCONCLUSIVE", "torch unavailable", {})

    def execute(sample: Any) -> tuple[str, list[tuple[str, str]]]:
        replica = copy.deepcopy(model)
        replica.eval()
        observations: list[tuple[str, str]] = []
        handles = []
        for name, module in replica.named_modules():
            if not name or any(module.children()):
                continue

            def hook(_module: Any, _inputs: Any, output: Any,
                     label: str = name) -> None:
                observations.append((label, fingerprint(output)))

            handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                result = forward(replica, copy.deepcopy(sample))
            return fingerprint(result), observations
        finally:
            for handle in handles:
                handle.remove()

    try:
        a = execute(legal_input)
        repeat = execute(legal_input)
        b = execute(denied_only_changed)
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        return Finding(check, "INCONCLUSIVE", "graph probe unavailable: " +
                       type(exc).__name__, {})
    if a != repeat:
        return Finding(check, "INCONCLUSIVE", "repeated legal forward differs; RNG/cache not sealed", {})
    first = None
    for index in range(max(len(a[1]), len(b[1]))):
        if index >= len(a[1]) or index >= len(b[1]):
            first = "<changed_call_schedule>"
            break
        if a[1][index] != b[1][index]:
            first = b[1][index][0]
            break
    if first is None and a[0] != b[0]:
        first = "<untraced_functional_or_external_path>"
    clean = first is None
    return Finding(check, "PASS" if clean else "FAIL",
                   "no traced forbidden-data dependency" if clean
                   else "forbidden-data dependence first observed at " + first,
                   {"first_changed_module": first,
                    "legal_output_sha256": a[0],
                    "poison_output_sha256": b[0],
                    "leaf_calls": len(a[1]),
                    "poison_leaf_calls": len(b[1])})
