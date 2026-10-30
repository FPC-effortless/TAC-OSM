"""No-fit persistence engineering probes; no intelligence or utility certificate.

Callbacks must be source reviewed. Acquisition is replayed once and prediction
receives only a separately supplied query. The probe cannot sandbox Python
closures, authenticate outcomes, or show that a structure was learned. Adapter
state views must expose actual NumPy arrays, including every retained cache.
Non-array state leaves use explicit dotted/indexed metadata paths. Unsupported
state objects and omitted storage leave accounting INCONCLUSIVE.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
from itertools import islice
import json
import math
import time
from typing import Any, Callable, Mapping

import numpy as np


FORBIDDEN_QUERY_FIELDS = frozenset({
    "history", "source", "raw_source", "source_history", "observations",
    "acquisition", "acquisition_events", "labels", "truth", "outcome",
})
REQUIRED_COSTS = (
    "upstream_acquisition_seconds", "upstream_retrieval_seconds",
    "upstream_verification_seconds", "learning_seconds", "model_bytes",
    "peak_working_bytes", "state_container_overhead_bytes",
)


@dataclass(frozen=True)
class PersistenceAdapter:
    """Small callback bridge; no dependency on a particular neural framework.

    A ResearchProbeAdapter can be bridged with start/advance/forecast callbacks,
    provided its forecast accepts the query-only object and all persisted state
    is exposed. Do not wrap a reset-window model as a streaming model.
    metadata_fields declares exact non-array leaf paths, e.g. ``clock`` or
    ``owners[0]``; an empty tuple explicitly declares no metadata. Cache names
    are state-view names, not additional unmetered stores.
    """
    name: str
    initial: Callable[[], Any]
    observe: Callable[[Any, Any], Any]
    predict: Callable[[Any, Mapping[str, Any]], Mapping[str, np.ndarray]]
    state_views: Callable[[Any], Mapping[str, np.ndarray]]
    patch_state: Callable[[Any, str, np.ndarray], Any]
    query_fields: tuple[str, ...]
    metadata_fields: tuple[str, ...] | None = None
    cache_components: tuple[str, ...] | None = None
    verify: Callable[[Any, Any], Any] | None = None
    metadata_limit_bytes: int = 4096


def _check(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def _array(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "bifu":
        raise ValueError("numeric NumPy arrays required")
    if value.size > 1_000_000 or not np.isfinite(value).all():
        raise ValueError("oversize or nonfinite array")
    return value


def _views(adapter: PersistenceAdapter, state: Any) -> dict[str, np.ndarray]:
    result = adapter.state_views(state)
    if not isinstance(result, Mapping) or any(not isinstance(k, str) or not k for k in result):
        raise ValueError("named state mapping required")
    return {k: _array(v) for k, v in result.items()}


def _inventory(state: Any) -> dict[str, Any]:
    arrays: dict[int, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    unsupported: list[str] = []
    active: set[int] = set()

    def walk(value: Any, path: str) -> None:
        if isinstance(value, np.ndarray):
            arrays[id(value)] = _array(value)
            return
        if value is None or isinstance(value, (str, bool, int, float, np.generic)):
            leaf = value.item() if isinstance(value, np.generic) else value
            if isinstance(leaf, float) and not math.isfinite(leaf):
                raise ValueError("nonfinite state metadata")
            metadata[path or "<root>"] = leaf
            return
        identity = id(value)
        if identity in active:
            unsupported.append(path + ":cycle")
            return
        active.add(identity)
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    unsupported.append(path + ":nonstring_key")
                    continue
                walk(item, f"{path}.{key}" if path else key)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}" if path else field.name)
        elif isinstance(value, tuple) and hasattr(value, "_fields"):
            for name, item in zip(value._fields, value):
                walk(item, f"{path}.{name}" if path else name)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        else:
            unsupported.append(path + ":" + type(value).__name__)
        active.remove(identity)

    walk(state, "")
    allocations: dict[int, np.ndarray] = {}
    opaque_backings: list[str] = []
    for array in arrays.values():
        backing = array
        seen: set[int] = set()
        while isinstance(backing.base, np.ndarray) and id(backing) not in seen:
            seen.add(id(backing))
            backing = backing.base
        if backing.base is None and backing.flags.owndata:
            allocations[id(backing)] = backing
        else:
            opaque_backings.append(type(backing.base).__name__)
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded)
    # Identity is used only for coverage; the digest remains stable after cloning.
    for value in arrays.values():
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    # A thin view can retain and access a much larger owning allocation. Its
    # hidden backing bytes must affect purity checks as well as resource meters.
    for backing in allocations.values():
        digest.update(str((backing.shape, backing.dtype.str)).encode())
        digest.update(backing.tobytes())
    known_resident_bytes = sum(v.nbytes for v in allocations.values())
    return {"arrays": arrays, "metadata": metadata, "metadata_bytes": len(encoded),
            "tensor_bytes": sum(v.nbytes for v in arrays.values()),
            "resident_array_bytes": None if opaque_backings else known_resident_bytes,
            "known_resident_array_bytes": known_resident_bytes,
            "opaque_backings": opaque_backings,
            "unsupported": unsupported, "digest": digest.hexdigest()}


def _query(adapter: PersistenceAdapter, query: Any) -> dict[str, Any]:
    if not isinstance(query, Mapping) or not adapter.query_fields:
        raise ValueError("explicit nonempty query-only field declaration required")
    if len(set(adapter.query_fields)) != len(adapter.query_fields) or set(query) != set(adapter.query_fields):
        raise ValueError("query differs from declared query-only fields")

    def validate(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or key.lower() in FORBIDDEN_QUERY_FIELDS:
                    raise ValueError("raw acquisition/source or truth field in query")
                validate(item)
        elif isinstance(value, np.ndarray):
            if _array(value).size > 4096:
                raise ValueError("query exceeds bounded query-only payload")
        elif isinstance(value, (list, tuple)):
            if len(value) > 4096:
                raise ValueError("query sequence too large")
            for item in value:
                validate(item)
        elif isinstance(value, (str, bool, int, float)) or value is None:
            if isinstance(value, str) and len(value.encode()) > 2048:
                raise ValueError("query string too large")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite query")
        else:
            raise ValueError("unsupported query payload")

    validate(query)
    return copy.deepcopy(dict(query))


def _outputs(adapter: PersistenceAdapter, state: Any, query: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out = adapter.predict(state, copy.deepcopy(query))
    if not isinstance(out, Mapping) or not out or any(not isinstance(k, str) or not k for k in out):
        raise ValueError("prediction requires named finite outputs")
    return {k: _array(v).copy() for k, v in out.items()}


def _difference(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> float:
    if left.keys() != right.keys() or any(left[k].shape != right[k].shape for k in left):
        raise ValueError("prediction names/shapes changed")
    return max((float(np.max(np.abs(left[k].astype(float) - right[k].astype(float))))
                if left[k].size else 0.) for k in left)


def _poison(value: Any) -> tuple[Any, dict[str, int]]:
    """Mutate retained mutable references, without claiming immutable coverage.

    Replacing a scalar slot changes its containing list/dict, but cannot change
    an immutable scalar independently retained by a callback. Such leaves keep
    coverage partial even when their mutable container was successfully changed.
    """
    coverage = {"mutated_input_leaves": 0, "unpoisoned_input_leaves": 0}
    seen: set[int] = set()

    def walk(item: Any, replace_slot: bool = False) -> Any:
        if isinstance(item, (np.ndarray, dict, list, tuple)):
            if id(item) in seen:
                return item
            seen.add(id(item))
        if isinstance(item, np.ndarray):
            _array(item)
            if not item.size or not item.flags.writeable:
                coverage["unpoisoned_input_leaves"] += 1
                return item
            before = item.copy()
            if item.dtype.kind == "b":
                item[...] = ~item
            elif item.dtype.kind in "iu":
                item[...] = np.bitwise_xor(item, 1)
            else:
                item[...] = np.where(item == 0, .125, -item)
            coverage["mutated_input_leaves"] += int(not np.array_equal(before, item))
            return item
        if isinstance(item, (dict, list)):
            if not item:
                coverage["unpoisoned_input_leaves"] += 1
            keys = list(item) if isinstance(item, dict) else range(len(item))
            for key in keys:
                original = item[key]
                changed = walk(original, replace_slot=True)
                if changed is not original:
                    item[key] = changed
                    coverage["mutated_input_leaves"] += 1
            return item
        if isinstance(item, tuple):
            if not item:
                coverage["unpoisoned_input_leaves"] += 1
            for child in item:
                walk(child)
            return item
        coverage["unpoisoned_input_leaves"] += 1
        if not replace_slot:
            return item
        if isinstance(item, bool):
            return not item
        if isinstance(item, (int, float, np.number)):
            return -item if item != 0 else 1
        if isinstance(item, str):
            return item + "#poison"
        return item

    return walk(value), coverage


def probe_persistence(
    adapter: PersistenceAdapter, acquisition_events: Any, query_only: Mapping[str, Any], *,
    donor_events: Any = None, feedback: Any = None, unavailable_feedback: Any = None,
    resource_costs: Mapping[str, float] | None = None, truth: Any = None,
    evaluate: Callable[[Mapping[str, np.ndarray], Any], Mapping[str, float]] | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Intervene on retained state, never infer usefulness from changed outputs.

    Caller-owned inputs are cloned. A missing cost is unknown, never zero.
    External truth reaches only evaluate(), after all predictions are committed.
    Timings are local engineering timings, not a matched performance benchmark.
    """
    started = time.perf_counter()
    report: dict[str, Any] = {"schema": "tac-osm.langmath095.persistence.v1",
        "adapter": getattr(adapter, "name", None), "checks": {}, "interventions": {},
        "qualification": "ENGINEERING_ONLY_NOT_CAPABILITY_OR_THEORY_EVIDENCE",
        "training_steps": 0, "protected_TEST": "NOT_ACCESSED",
        "usefulness": _check("INCONCLUSIVE", reason="output dependence does not establish future utility"),
        "learned_structure": _check("INCONCLUSIVE", reason="requires authenticated trained source and transfer study")}
    checks = report["checks"]
    try:
        if not isinstance(adapter, PersistenceAdapter) or not adapter.name:
            raise ValueError("explicit source-reviewed PersistenceAdapter required")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("finite nonnegative tolerance required")
        if type(adapter.metadata_limit_bytes) is not int or adapter.metadata_limit_bytes < 0:
            raise ValueError("nonnegative metadata byte limit required")
        query = _query(adapter, query_only)
        events = copy.deepcopy(list(islice(iter(acquisition_events), 10001)))
        if not events or len(events) > 10000:
            raise ValueError("bounded nonempty acquisition episode required")
        source_inventory = _inventory(events)
        checks["acquisition_payload"] = _check(
            "INCONCLUSIVE" if source_inventory["unsupported"] else "PASS",
            unsupported=source_inventory["unsupported"])
        checks["query_information_boundary"] = _check("PASS", coverage="declared_fields_only_source_review_still_required")
        tic = time.perf_counter()
        state = adapter.initial()
        for event in events:
            state = adapter.observe(state, event)
        acquisition_seconds = time.perf_counter() - tic
        views = _views(adapter, state)
        inventory = _inventory(state)
        if inventory["metadata_bytes"] > adapter.metadata_limit_bytes:
            raise ValueError("state metadata exceeds declared byte limit")
        declared_ids = {id(value) for value in views.values()}
        missing_arrays = len(set(inventory["arrays"]) - declared_ids)
        external_views = len(declared_ids - set(inventory["arrays"]))
        metadata_complete = (adapter.metadata_fields is not None and
                             set(adapter.metadata_fields) == set(inventory["metadata"]))
        cache_complete = (adapter.cache_components is not None and
                          set(adapter.cache_components).issubset(views))
        state_complete = not (missing_arrays or external_views or inventory["unsupported"] or
                              inventory["opaque_backings"]) and metadata_complete and cache_complete
        metered_inventory = inventory
        cache_payload_bytes = sum(views[k].nbytes for k in (adapter.cache_components or ()) if k in views)
        checks["state_inventory"] = _check("PASS" if state_complete else "INCONCLUSIVE",
            unexposed_arrays=missing_arrays, external_state_views=external_views,
            unsupported=inventory["unsupported"], opaque_array_backings=inventory["opaque_backings"],
            metadata_paths=sorted(inventory["metadata"]), metadata_declared=metadata_complete,
            cache_declared=cache_complete, coverage="reachable_declared_state_and_numpy_backings_only")
        tic = time.perf_counter()
        baseline = _outputs(adapter, state, query)
        retrieval_seconds = time.perf_counter() - tic
        checks["query_state_purity"] = _check("PASS" if inventory["digest"] == _inventory(state)["digest"] else "FAIL")
        repeated = _outputs(adapter, state, query)
        if inventory["digest"] != _inventory(state)["digest"]:
            checks["query_state_purity"] = _check("FAIL")
        checks["deterministic_query"] = _check("PASS" if _difference(baseline, repeated) <= tolerance else "FAIL")

        def intervene(name: str, replacements: Mapping[str, np.ndarray]) -> None:
            before = _inventory(state)["digest"]
            patched = copy.deepcopy(state)
            for component, value in replacements.items():
                if value.shape != views[component].shape:
                    raise ValueError("donor state shape mismatch")
                patched = adapter.patch_state(patched, component, value.copy())
            patched_before = _inventory(patched)["digest"]
            difference = _difference(baseline, _outputs(adapter, patched, query))
            if _inventory(patched)["digest"] != patched_before:
                checks["query_state_purity"] = _check("FAIL")
            if _inventory(state)["digest"] != before:
                raise ValueError("intervention mutated caller-owned retained state")
            report["interventions"][name] = {"max_abs_output_difference": difference,
                "status": "PASS" if difference > tolerance else "INCONCLUSIVE",
                "meaning": "state_sensitivity_only_not_usefulness",
                "out_of_distribution_possible": True}

        intervene("reset_all_components", {k: np.zeros_like(v) for k, v in views.items()})
        difference = report["interventions"]["reset_all_components"]["max_abs_output_difference"]
        checks["retained_state_dependence"] = _check("PASS" if difference > tolerance else "FAIL",
            max_abs_output_difference=difference, meaning="current_query_output_dependence_only")
        for name, value in views.items():
            intervene("zero:" + name, {name: np.zeros_like(value)})
        if donor_events is not None:
            donor = adapter.initial()
            donor_source = copy.deepcopy(list(islice(iter(donor_events), 10001)))
            if not donor_source or len(donor_source) > 10000:
                raise ValueError("bounded nonempty donor episode required")
            _inventory(donor_source)
            for event in donor_source:
                donor = adapter.observe(donor, event)
            donor_views = _views(adapter, donor)
            if donor_views.keys() != views.keys():
                raise ValueError("donor state component names differ")
            intervene("donor_all_components", donor_views)
            for name in views:
                intervene("donor:" + name, {name: donor_views[name]})

        events, poison_coverage = _poison(events)
        mutations = poison_coverage["mutated_input_leaves"]
        before_poison_query = _inventory(state)["digest"]
        after_poison = _outputs(adapter, state, query)
        if before_poison_query != _inventory(state)["digest"]:
            checks["query_state_purity"] = _check("FAIL")
        source_difference = _difference(baseline, after_poison)
        checks["no_source_replay_after_acquisition"] = _check(
            "FAIL" if source_difference > tolerance else "INCONCLUSIVE"
            if not mutations or poison_coverage["unpoisoned_input_leaves"] else "PASS",
            **poison_coverage, max_abs_output_difference=source_difference,
            limitation="trusted callback closures and external stores require separate source review")

        verification_seconds = 0.
        if feedback is not None or unavailable_feedback is not None:
            if adapter.verify is None:
                checks["verification"] = _check("INCONCLUSIVE", reason="verification callback unavailable")
            else:
                if unavailable_feedback is not None:
                    prior = copy.deepcopy(state)
                    corrected = adapter.verify(prior, copy.deepcopy(unavailable_feedback))
                    checks["unavailable_feedback_does_not_write"] = _check(
                        "PASS" if _inventory(corrected)["digest"] == _inventory(state)["digest"] else "FAIL")
                if feedback is not None:
                    tic = time.perf_counter()
                    corrected = adapter.verify(copy.deepcopy(state), copy.deepcopy(feedback))
                    verification_seconds = time.perf_counter() - tic
                    corrected_inventory = _inventory(corrected)
                    corrected_views = _views(adapter, corrected)
                    corrected_complete = (
                        not corrected_inventory["unsupported"] and not corrected_inventory["opaque_backings"] and
                        set(corrected_inventory["arrays"]) == {id(v) for v in corrected_views.values()} and
                        adapter.metadata_fields is not None and
                        set(adapter.metadata_fields) == set(corrected_inventory["metadata"]) and
                        adapter.cache_components is not None and
                        set(adapter.cache_components).issubset(corrected_views))
                    if corrected_inventory["metadata_bytes"] > adapter.metadata_limit_bytes:
                        raise ValueError("verified state metadata exceeds declared byte limit")
                    state_complete = state_complete and corrected_complete
                    checks["verified_state_inventory"] = _check("PASS" if corrected_complete else "INCONCLUSIVE")
                    if (corrected_inventory["known_resident_array_bytes"] + corrected_inventory["metadata_bytes"] >
                            metered_inventory["known_resident_array_bytes"] + metered_inventory["metadata_bytes"]):
                        metered_inventory = corrected_inventory
                    cache_payload_bytes = max(cache_payload_bytes, sum(
                        corrected_views[k].nbytes for k in (adapter.cache_components or ()) if k in corrected_views))
                    correction_outputs = _outputs(adapter, corrected, query)
                    if corrected_inventory["digest"] != _inventory(corrected)["digest"]:
                        checks["query_state_purity"] = _check("FAIL")
                    checks["verification"] = _check("PASS", max_abs_output_difference=_difference(baseline, correction_outputs),
                        meaning="finite_postfeedback_dependency_not_verified_future_utility")
                checks["feedback_preserves_committed_prediction"] = _check(
                    "PASS" if _inventory(state)["digest"] == inventory["digest"] else "FAIL")

        costs = dict(resource_costs or {})
        missing = [name for name in REQUIRED_COSTS if name not in costs or costs[name] is None]
        for name, value in costs.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("resource costs must be finite nonnegative numbers")
        checks["complete_resource_accounting"] = _check("PASS" if not missing and state_complete else "INCONCLUSIVE",
            missing_fields=missing, state_coverage_complete=state_complete,
            meaning="declared_fields_complete_not_independent_cost_or_global_storage_verification")
        report["resources"] = {"local_acquisition_seconds": acquisition_seconds,
            "local_retrieval_seconds": retrieval_seconds, "local_verification_seconds": verification_seconds,
            "state_array_payload_bytes": metered_inventory["tensor_bytes"], "state_metadata_payload_bytes": metered_inventory["metadata_bytes"],
            "resident_array_bytes": metered_inventory["resident_array_bytes"],
            "declared_cache_payload_bytes": cache_payload_bytes,
            "state_accounting_basis": "largest_known_numpy_backing_allocation_plus_metadata_and_declared_container_overhead",
            "accounting_scope": "reachable_declared_state_only_callback_globals_model_and_external_costs_are_unverified_declarations",
            "declared_upstream": costs,
            "total_episode_seconds": None if missing else acquisition_seconds + retrieval_seconds + verification_seconds +
                sum(costs[name] for name in REQUIRED_COSTS[:4]),
            "total_state_bytes": None if missing or not state_complete else metered_inventory["resident_array_bytes"] + metered_inventory["metadata_bytes"] + costs["state_container_overhead_bytes"],
            "local_diagnostic_total_seconds": time.perf_counter() - started}
        if truth is not None or evaluate is not None:
            if truth is None or evaluate is None:
                checks["external_outcome_evaluation"] = _check("INCONCLUSIVE", reason="both truth and separate evaluator required")
            else:
                metrics = dict(evaluate(copy.deepcopy(baseline), copy.deepcopy(truth)))
                if not metrics or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
                    raise ValueError("external evaluator must return finite named scalar metrics")
                report["external_metrics"] = metrics
                checks["external_outcome_evaluation"] = _check("PASS", meaning="evaluator_executed_not_outcome_authentication")
    except Exception as exc:
        checks["contract_execution"] = _check("FAIL", reason=str(exc))
    statuses = [item["status"] for item in checks.values()]
    report["status"] = "FAIL" if "FAIL" in statuses else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS"
    return report
