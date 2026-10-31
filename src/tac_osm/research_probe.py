"""Architecture- and version-agnostic PyTorch research diagnostic protocol.

An adapter describes ACTUAL model IO/state semantics. No hardcoded memory
names, shapes, model family, action axes, or number of prediction heads.
Unsupported diagnostics stay INCONCLUSIVE. Engineering/no-fit probes only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import math
import statistics
import time
from typing import Any

import torch
from torch import nn

from .research_torch_basics import architecture_inventory, gradient_conflict_atlas
from .research_interpretation import interpret_snapshot


Named = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ResearchProbeAdapter:
    """Explicit, trusted mapping from one source model contract to the probes.

    - sample is a source-authorized, decision-time object, never a label oracle.
    - state can be any Python value; state_views exposes its named tensors.
      Floating, integer and boolean state tensors are supported; discrete
      identities are compared exactly and are not treated as differentiable.
      State tensor lengths may grow (for example an observed raw prefix).
    - forecast must not update model buffers/parameters or caller-owned state.
    - patch_state must replace the named state component WITHOUT touching the
      caller-owned original state. This may form an out-of-distribution state.
    - batch_item is OPTIONAL and makes entity-isolation testable.
    - gradient_graph returns ({output_name: tensor}, {input_name: leaf tensor})
      with genuine differentiable outputs on a local replayed state.
    - permute_candidates returns (changed_sample, restore_outputs_fn);
      changed_sample must differ ONLY by candidate order.
    Callbacks run ONLY inside an authorized research Python process. Do not
    dynamically import arbitrary adapters from untrusted JSON or checkpoints.
    """
    name: str
    validate: Callable[[nn.Module, Any], None]
    steps: Callable[[Any], int]
    start: Callable[[nn.Module, Any], Any]
    advance: Callable[[nn.Module, Any, Any, int], Any]
    forecast: Callable[[nn.Module, Any, Any], Named]
    state_views: Callable[[Any], Named]
    patch_state: Callable[[Any, str, torch.Tensor], Any]
    batch_size: Callable[[Any], int]
    batch_item: Callable[[Any, int], Any] | None = None
    gradient_graph: Callable[[nn.Module, Any, Any], tuple[Named, Named]] | None = None
    permute_candidates: Callable[
        [Any], tuple[Any, Callable[[Named], Named]]
    ] | None = None


def _tensor_map(items: Named, *, allow_empty: bool = False,
                allow_discrete: bool = False) -> dict[str, torch.Tensor]:
    if not isinstance(items, Mapping) or (not items and not allow_empty):
        raise ValueError("adapter must expose a named tensor mapping")
    out = {}
    for name, tensor in items.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("adapter mapping has invalid named tensor")
        discrete = tensor.dtype in (torch.bool, torch.uint8, torch.int8,
                                    torch.int16, torch.int32, torch.int64)
        if (not (torch.is_floating_point(tensor) or (allow_discrete and discrete))
                or not torch.isfinite(tensor).all()):
            raise ValueError("nonfloating or nonfinite tensor at: " + name)
        out[name] = tensor
    return out


def _state(adapter: ResearchProbeAdapter, value: Any) -> dict[str, torch.Tensor]:
    return _tensor_map(adapter.state_views(value), allow_empty=True, allow_discrete=True)


def _outputs(adapter: ResearchProbeAdapter, model: nn.Module,
             state: Any, sample: Any) -> dict[str, torch.Tensor]:
    return _tensor_map(adapter.forecast(model, state, sample))


def _snapshot(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in value.items()}


def _module_snap(model: nn.Module) -> dict[str, torch.Tensor]:
    return _snapshot({
        **dict(model.named_parameters()),
        **{"buffer:" + k: v for k, v in model.named_buffers()},
    })


def _unchanged(before: Mapping[str, torch.Tensor],
               after: Mapping[str, torch.Tensor]) -> bool:
    return before.keys() == after.keys() and all(
        torch.equal(before[k], after[k]) for k in before)


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()).item())


def _stats(value: torch.Tensor) -> dict[str, Any]:
    if not torch.is_floating_point(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "storage_bytes": value.numel() * value.element_size(),
                "kind": "discrete_state_not_continuous_latent"}
    x = value.detach().float()
    return {"shape": list(x.shape), "l2": _norm(x),
            "mean_abs": float(x.abs().mean()) if x.numel() else 0.,
            "max_abs": float(x.abs().max()) if x.numel() else 0.}


def _changes(left: Named, right: Named) -> dict[str, float]:
    if left.keys() != right.keys():
        raise ValueError("forecast head names changed across diagnostic intervention")
    out = {}
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError("forecast shape changed across diagnostic intervention")
        out[name] = float((left[name] - right[name]).detach().float().abs().mean())
    return out


def _state_change(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    if before.shape != after.shape:
        return {"change_l2": None, "previous_shape": list(before.shape),
                "shape_changed": True,
                "storage_growth_bytes": after.numel()*after.element_size() -
                                        before.numel()*before.element_size()}
    if not torch.is_floating_point(after):
        return {"change_l2": None, "shape_changed": False,
                "changed_elements": int(torch.ne(before, after).sum())}
    return {"change_l2": _norm(after-before), "shape_changed": False}


def _check(adapter: ResearchProbeAdapter, model: nn.Module, sample: Any) -> tuple[int, int]:
    if not isinstance(adapter, ResearchProbeAdapter) or not adapter.name:
        raise ValueError("explicit named architecture adapter required")
    adapter.validate(model, sample)
    steps = adapter.steps(sample)
    batch = adapter.batch_size(sample)
    if (type(steps) is not int or not 0 <= steps <= 10000
            or type(batch) is not int or batch <= 0):
        raise ValueError("adapter steps and batch must be finite nonnegative/positive integers")
    return steps, batch


def _replay(adapter: ResearchProbeAdapter, model: nn.Module,
            sample: Any, *, trace: bool = False) -> tuple[Any, list[dict[str, Any]]]:
    state = adapter.start(model, sample)
    timeline = []
    prev = _state(adapter, state)
    for t in range(adapter.steps(sample)):
        before = _snapshot(prev)
        state = adapter.advance(model, state, sample, t)
        now = _state(adapter, state)
        if before.keys() != now.keys():
            raise ValueError("state component names changed during replay")
        if trace:
            timeline.append({
                "event_index": t,
                "components": {
                    k: {**_stats(now[k]), **_state_change(before[k], now[k])}
                    for k in now},
            })
        prev = now
    return state, timeline


def _forecast_pure(adapter: ResearchProbeAdapter, model: nn.Module,
                   state: Any, sample: Any) -> tuple[Named, bool]:
    prior_state = _snapshot(_state(adapter, state))
    prior_module = _module_snap(model)
    result = _outputs(adapter, model, state, sample)
    pure = (_unchanged(prior_state, _state(adapter, state))
            and _unchanged(prior_module, _module_snap(model)))
    return result, pure


def probe_memory(model: nn.Module, sample: Any,
                 adapter: ResearchProbeAdapter) -> dict[str, Any]:
    """State trajectory and per-head zero-patch diagnostics, any state topology."""
    replica = copy.deepcopy(model).eval()
    steps, batch = _check(adapter, replica, sample)
    with torch.no_grad():
        state, timeline = _replay(adapter, replica, sample, trace=True)
        output, pure = _forecast_pure(adapter, replica, state, sample)
        names = _state(adapter, state)
        deltas: dict[str, Any] = {}
        for name, tensor in names.items():
            # Snapshot BEFORE calling adapter patch, to detect in-place mutation.
            original = _snapshot(_state(adapter, state))
            patch = adapter.patch_state(state, name, torch.zeros_like(tensor))
            # Protect original against a bug in patch_state even if forecast
            # on the patched state completes successfully.
            altered, _ = _forecast_pure(adapter, replica, patch, sample)
            if not _unchanged(original, _state(adapter, state)):
                raise ValueError("adapter patch_state mutated original caller-owned state")
            deltas[name] = {"head_mean_absolute_change": _changes(output, altered),
                            "intervention": "zero_patch_out_of_distribution_possible"}
        zero = adapter.start(replica, sample)
        zero_out, _ = _forecast_pure(adapter, replica, zero, sample)
        deltas["<reset_all>"] = {
            "head_mean_absolute_change": _changes(output, zero_out),
            "intervention": "state_reset_no_history",
        }
        if adapter.permute_candidates is None:
            permutation = {"status": "INCONCLUSIVE_NOT_DECLARED"}
        else:
            permuted_sample, restore = adapter.permute_candidates(sample)
            adapter.validate(replica, permuted_sample)
            rearranged, _ = _forecast_pure(adapter, replica, state, permuted_sample)
            compared = _tensor_map(restore(rearranged))
            errors = _changes(output, compared)
            permutation = {"status": "PASS" if max(errors.values()) <= 1e-5 else "FAIL",
                           "max_mean_absolute_error": max(errors.values()),
                           "head_mean_absolute_error": errors}
    return {
        "schema": "tac-osm-observatory/model-agnostic-memory-v1",
        "adapter": adapter.name, "batch": batch, "events": steps,
        "state_components": sorted(names), "timeline": timeline,
        "interventions": deltas,
        "query_does_not_write_state": pure,
        "candidate_permutation": permutation,
        "coverage": "stateless_model" if not names else "named_adapter_state",
        "state_dtypes": {name: str(t.dtype) for name, t in names.items()},
        "state_storage_bytes": sum(t.numel()*t.element_size() for t in names.values()),
        "qualification": "local_no_fit_diagnostics_not_capability_evidence",
    }


def probe_dependencies(model: nn.Module, sample: Any,
                       adapter: ResearchProbeAdapter) -> dict[str, Any]:
    """No model-specific inputs, output heads, state shape or gradient scalar."""
    if adapter.gradient_graph is None:
        return {"status": "INCONCLUSIVE_UNINSTRUMENTED",
                "reason": "adapter does not define a differentiable input/output graph"}
    replica = copy.deepcopy(model).eval()
    _check(adapter, replica, sample)
    with torch.no_grad():
        state, _ = _replay(adapter, replica, sample)
    outputs, inputs = adapter.gradient_graph(replica, state, sample)
    heads = _tensor_map(outputs)
    sources = _tensor_map(inputs)
    if not all(t.requires_grad for t in sources.values()):
        raise ValueError("dependency source tensors must require gradients")
    report = {}
    for name, head in heads.items():
        if not head.requires_grad:
            report[name] = {
                src: {"connected": False, "finite": None, "gradient_l2": None}
                for src in sources}
            continue
        gradients = torch.autograd.grad(head.sum(), tuple(sources.values()),
                                        retain_graph=True, allow_unused=True)
        report[name] = {
            key: {"connected": grad is not None,
                  "finite": bool(torch.isfinite(grad).all()) if grad is not None else None,
                  "gradient_l2": _norm(grad) if grad is not None else None}
            for key, grad in zip(sources, gradients)
        }
    return {
        "schema": "tac-osm-observatory/model-agnostic-dependency-v1",
        "status": "MEASURED", "adapter": adapter.name,
        "heads": report,
        "qualification": "local_gradient_not_global_data_flow_proof",
    }


def probe_entity_isolation(model: nn.Module, sample: Any,
                           adapter: ResearchProbeAdapter, *,
                           tolerance: float = 1e-5) -> dict[str, Any]:
    """Requires an adapter-declared legal entity slice, otherwise INCONCLUSIVE."""
    if adapter.batch_item is None:
        return {"status": "INCONCLUSIVE_UNINSTRUMENTED",
                "reason": "batch_item contract not declared"}
    replica = copy.deepcopy(model).eval()
    _, batch = _check(adapter, replica, sample)
    if batch < 2:
        return {"status": "INCONCLUSIVE_ONE_ENTITY"}
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("isolation tolerance must be nonnegative and finite")
    with torch.no_grad():
        state, _ = _replay(adapter, replica, sample)
        output, _ = _forecast_pure(adapter, replica, state, sample)
        whole_state = _state(adapter, state)
        maximum = 0.
        discrete_mismatch = False
        for i in range(batch):
            one_sample = adapter.batch_item(sample, i)
            one = copy.deepcopy(model).eval()
            _, one_batch = _check(adapter, one, one_sample)
            if one_batch != 1:
                raise ValueError("batch_item did not isolate exactly one entity")
            own_state, _ = _replay(adapter, one, one_sample)
            own_output, _ = _forecast_pure(adapter, one, own_state, one_sample)
            for a, b in ((whole_state, _state(adapter, own_state)),
                         (output, own_output)):
                if a.keys() != b.keys():
                    raise ValueError("entity output/state names changed when sliced")
                for k in a:
                    if a[k].ndim < 1 or a[k].shape[0] != batch or b[k].shape != a[k][i:i + 1].shape:
                        raise ValueError("entity-isolation tensors must have declared batch-leading axis")
                    if not torch.is_floating_point(a[k]):
                        # Integer IDs near 2**63 and boolean masks must never
                        # be rounded to float or subtracted with overflow.
                        discrete_mismatch |= not torch.equal(a[k][i:i + 1], b[k])
                    elif a[k].numel():
                        maximum = max(maximum, float((a[k][i:i + 1] - b[k]).abs().max()))
    return {"schema": "tac-osm-observatory/model-agnostic-isolation-v1",
            "adapter": adapter.name, "status": "PASS" if maximum <= tolerance and not discrete_mismatch else "FAIL",
            "entities": batch, "max_absolute_difference": maximum,
            "tolerance": tolerance, "within_tolerance": maximum <= tolerance and not discrete_mismatch,
            "discrete_state_exactly_equal": not discrete_mismatch,
            "qualification": "model_batch_isolation_not_source_identity_authentication"}


def probe_runtime(model: nn.Module, sample: Any,
                  adapter: ResearchProbeAdapter, *,
                  warmups: int = 2, repeats: int = 8) -> dict[str, Any]:
    """Measure local replay+inference and state bytes, independent of model form."""
    replica = copy.deepcopy(model).eval()
    steps, batch = _check(adapter, replica, sample)
    if type(warmups) is not int or type(repeats) is not int or warmups < 0 or not 2 <= repeats <= 1000:
        raise ValueError("warmups>=0 and 2<=repeats<=1000 required")
    first = next(replica.parameters(), None)
    device = first.device if first is not None else torch.device("cpu")
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def timed():
        sync()
        tic = time.perf_counter()
        state, _ = _replay(adapter, replica, sample)
        _outputs(adapter, replica, state, sample)
        sync()
        duration = (time.perf_counter() - tic) * 1000
        memory_bytes = sum(t.numel() * t.element_size()
                           for t in _state(adapter, state).values())
        return duration, memory_bytes
    with torch.no_grad():
        for _ in range(warmups):
            timed()
        trials = [timed() for _ in range(repeats)]
    durations = sorted(p[0] for p in trials)
    return {"schema": "tac-osm-observatory/model-agnostic-runtime-v1",
            "adapter": adapter.name, "batch": batch, "events": steps,
            "device": str(device), "threads": torch.get_num_threads(),
            "state_bytes_batch": trials[-1][1],
            "end_to_end_ms_median": statistics.median(durations),
            "end_to_end_ms_p95": durations[math.ceil(.95 * repeats) - 1],
            "end_to_end_ms_min": durations[0],
            "end_to_end_ms_max": durations[-1],
            "qualification": "local_timing_not_matched_performance_or_capability"}


def inspect_variant(model: nn.Module, sample: Any,
                    adapter: ResearchProbeAdapter, *,
                    isolation: bool = True, profile: bool = False) -> dict[str, Any]:
    """One uniform output schema for any adapter-backed PyTorch architecture."""
    _check(adapter, model, sample)
    def invoked(net: nn.Module):
        state, _ = _replay(adapter, net, sample)
        return _outputs(adapter, net, state, sample)
    report = {
        "schema": "tac-osm-observatory/model-agnostic-report-v1",
        "adapter": adapter.name,
        "architecture": architecture_inventory(model, invocations={"replay_and_forecast": invoked}),
        "memory": probe_memory(model, sample, adapter),
        "dependencies": probe_dependencies(model, sample, adapter),
        "entity_isolation": probe_entity_isolation(model, sample, adapter)
                            if isolation else {"status": "INCONCLUSIVE_NOT_REQUESTED"},
        "runtime": probe_runtime(model, sample, adapter) if profile else
                   {"status": "INCONCLUSIVE_NOT_REQUESTED"},
        "qualification": "no_fit_model_local_engineering_only",
    }
    coverage = {
        "memory_probe": "MEASURED",
        "dependency_probe": report["dependencies"].get("status", "UNKNOWN"),
        "entity_isolation": report["entity_isolation"].get("status", "UNKNOWN"),
        "candidate_permutation": report["memory"]["candidate_permutation"]["status"],
        "prediction_state_purity": ("PASS" if report["memory"]["query_does_not_write_state"]
                                    else "FAIL"),
        "runtime": report["runtime"].get("status", "MEASURED"),
    }
    report["coverage"] = coverage
    report["interpretation"] = interpret_snapshot(report)
    report["integrity_status"] = (
        "FAIL" if "FAIL" in coverage.values() else
        "INCONCLUSIVE" if any(k.startswith(("INCONCLUSIVE", "UNKNOWN"))
                            for k in coverage.values()) else "PASS")
    return report
