"""TAC-OSM Research Observatory: inspect actual PyTorch model mechanisms.

Non-invasive, NO-FIT engineering diagnostics. All numeric effects are
checkpoint-local; untrained fixture traces have ZERO scientific capability
weight. This module neither authenticates real data nor certifies causality.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
import copy
import math
from typing import Any

import torch
from torch import nn

from .predictive_state_agent import Belief


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.detach().float()).item())


def _delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().mean().item())


def _summary(t: torch.Tensor) -> dict[str, Any]:
    if not _finite(t):
        raise ValueError("nonfinite state or activation; inspect numerical stability")
    return {"shape": list(t.shape), "l2": _norm(t),
            "max_abs": float(t.detach().abs().max().item()) if t.numel() else 0.,
            "mean_abs": float(t.detach().abs().mean().item()) if t.numel() else 0.}


def architecture_inventory(
    model: nn.Module,
    *,
    invocations: Mapping[str, Callable[[nn.Module], Any]] | None = None,
) -> dict[str, Any]:
    """Actual instantiated module/parameter inventory and per-phase call trace.

    A list of sequential calls is NOT a dataflow DAG. No guessed edges or
    source-equivalent mathematical proof are emitted. Each invocation operates
    on an independent cloned, eval-mode model with no optimizer or feedback.
    """
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trainable": 0, "deployed_learned": 0, "tensors": 0})
    names = set()
    for name, p in model.named_parameters(remove_duplicate=True):
        group = name.split(".")[0]
        rows[group]["trainable"] += p.numel() if p.requires_grad else 0
        rows[group]["deployed_learned"] += p.numel()
        rows[group]["tensors"] += 1
        names.add(name)
    declared = {name: type(module).__name__ for name, module in model.named_modules()}
    phases = {}
    for phase, invocation in (invocations or {}).items():
        replica = copy.deepcopy(model).eval()
        seen: list[dict[str, Any]] = []
        handles = []

        def register(path: str, layer: nn.Module) -> None:
            def hook(_module: nn.Module, _args: Any, output: Any) -> None:
                tensors = []

                def visit(x: Any) -> None:
                    if isinstance(x, torch.Tensor):
                        tensors.append({"shape": list(x.shape), "finite": _finite(x)})
                    elif isinstance(x, (tuple, list)):
                        for v in x:
                            visit(v)
                    elif isinstance(x, Mapping):
                        for v in x.values():
                            visit(v)
                visit(output)
                seen.append({"module": path, "output_tensors": tensors})
            handles.append(layer.register_forward_hook(hook))
        for path, layer in replica.named_modules():
            if path and not list(layer.children()):
                register(path, layer)
        try:
            with torch.no_grad():
                invocation(replica)
        finally:
            for h in handles:
                h.remove()
        phases[phase] = {
            "ordered_leaf_calls": seen,
            "meaning": "runtime call order only; not a computational DAG",
            "unobserved_modules": sorted(set(declared) - {""} -
                                         {x["module"] for x in seen}),
        }
    return {
        "schema": "tac-osm-observatory/architecture-v1",
        "module_type": type(model).__name__,
        "total_trainable": sum(x["trainable"] for x in rows.values()),
        "total_deployed_learned": sum(x["deployed_learned"] for x in rows.values()),
        "component_parameters": dict(sorted(rows.items())),
        "module_types": declared, "phases": phases,
        "qualification": "inspection_only_not_trained_or_proven",
    }


def _native_inputs(model: nn.Module, history: torch.Tensor,
                   candidates: torch.Tensor) -> None:
    if history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("history BxTxC and candidates BxAxD required")
    b, t, c = history.shape
    if b < 1 or t < 1 or candidates.shape[0] != b or candidates.shape[1] < 1:
        raise ValueError("nonempty batch, time, candidates required")
    if c != model.context_size or candidates.shape[-1] != model.action_size:
        raise ValueError("actual model feature dimensions differ")
    if not _finite(history) or not _finite(candidates):
        raise ValueError("nonfinite source inputs")
    if t > 10000:
        raise ValueError("observatory capped at 10000 temporal events per invocation")


def _observe(model: nn.Module, history: torch.Tensor) -> Belief:
    state = model.initial(history.shape[0])
    for t in range(history.shape[1]):
        state = model.observe(state, history[:, t])
    return state


def memory_tomography(model: nn.Module, history: torch.Tensor,
                      candidates: torch.Tensor) -> dict[str, Any]:
    """Quantify recurrent state motion and prediction sensitivity to state edits.

    Lesion/zero-patch interventions leave the real history and candidates
    unchanged. These states may be out-of-distribution: deltas are local
    dependence diagnostics, NOT feature importance or causal world dynamics.
    """
    replica = copy.deepcopy(model).eval()
    _native_inputs(replica, history, candidates)
    timeline = []
    with torch.no_grad():
        state = replica.initial(history.shape[0])
        for t in range(history.shape[1]):
            old = state
            state = replica.observe(state, history[:, t])
            entry: dict[str, Any] = {"event_index": t}
            for name in ("fast", "slow", "memory"):
                cur, prev = getattr(state, name), getattr(old, name)
                entry[name] = {**_summary(cur), "change_l2": _norm(cur - prev)}
            timeline.append(entry)
        context = history[:, -1]
        base = replica.predict(state, context, candidates)
        if not all(_finite(p) for p in base):
            raise ValueError("nonfinite original prediction")
        changes = {}
        for part in ("fast", "slow", "memory"):
            patch = Belief(**{
                k: torch.zeros_like(getattr(state, k)) if k == part
                else getattr(state, k)
                for k in ("fast", "slow", "memory")})
            adjusted = replica.predict(patch, context, candidates)
            changes[part] = {
                "outcome_mean_absolute_change": _delta(base[0], adjusted[0]),
                "future_mean_absolute_change": _delta(base[1], adjusted[1]),
                "intervention": "zero_patch_out_of_distribution_possible",
            }
        no_history = replica.predict(replica.initial(history.shape[0]),
                                     context, candidates)
        changes["all_state"] = {
            "outcome_mean_absolute_change": _delta(base[0], no_history[0]),
            "future_mean_absolute_change": _delta(base[1], no_history[1]),
            "intervention": "state_reset_no_history",
        }
        before = [p.detach().clone() for p in state]
        _ = replica.predict(state, context, candidates)
        pure_query = all(torch.equal(p, q) for p, q in zip(before, state))
        order = list(reversed(range(candidates.shape[1])))
        reordered = replica.predict(state, context, candidates[:, order])
        permutation_max_error = max(
            float((x[:, order] - y).abs().max().item())
            for x, y in zip(base, reordered))
    return {
        "schema": "tac-osm-observatory/memory-v1",
        "events": len(timeline), "batch": history.shape[0],
        "timeline": timeline, "interventions": changes,
        "query_does_not_write_state": pure_query,
        "candidate_permutation_max_abs_error": permutation_max_error,
        "qualification": "no_fit_local_mechanism_probe_not_retention_capability",
    }


def input_dependency_map(model: nn.Module, history: torch.Tensor,
                         candidates: torch.Tensor) -> dict[str, Any]:
    """Local autograd sensitivity, separately for outcome and future heads.

    An unconnected input produces 'connected': False. Zero gradient when
    connected means only this input/head/checkpoint/point had zero derivative;
    it is never a proof that the pathway does not exist globally.
    """
    replica = copy.deepcopy(model).eval()
    _native_inputs(replica, history, candidates)
    with torch.no_grad():
        original = _observe(replica, history)
    state = Belief(*(p.detach().clone().requires_grad_(True) for p in original))
    context = history[:, -1].detach().clone().requires_grad_(True)
    actions = candidates.detach().clone().requires_grad_(True)
    outputs = replica.predict(state, context, actions)
    inputs = {"current_context": context, "candidate_action": actions,
              "fast_state": state.fast, "slow_state": state.slow,
              "associative_memory": state.memory}
    result = {}
    for title, head in zip(("outcome", "next_observation"), outputs):
        scalar = head.sum()
        gradients = torch.autograd.grad(scalar, tuple(inputs.values()),
                                        retain_graph=True, allow_unused=True)
        result[title] = {
            name: {
                "connected": grad is not None,
                "finite": _finite(grad) if grad is not None else None,
                "gradient_l2": _norm(grad) if grad is not None else None,
            } for name, grad in zip(inputs, gradients)
        }
    return {"schema": "tac-osm-observatory/dependency-v1",
            "heads": result, "method": "local_autograd_sum_of_outputs",
            "qualification": "local_gradient_not_global_structural_noninterference"}


def gradient_conflict_atlas(model: nn.Module,
                            losses: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compare *actual provided* differentiable loss gradients without stepping.

    Caller supplies the genuine loss terms and targets for an authorized
    FIT/DEV batch. No loss targets are made up here. Cosines are calculated
    only across parameters reached by BOTH losses in each named component;
    no overlapping live gradients => cosine=null (INCONCLUSIVE).
    """
    if len(losses) < 2:
        raise ValueError("two or more genuine differentiable losses required")
    named = [(name, p) for name, p in model.named_parameters()
             if p.requires_grad]
    if not named:
        raise ValueError("no trainable model parameters")
    values = {}
    for label, loss in losses.items():
        if loss.numel() != 1 or not bool(loss.requires_grad) or not _finite(loss):
            raise ValueError(f"loss {label} must be finite differentiable scalar")
        gradients = torch.autograd.grad(
            loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        values[label] = {name: g.detach().float() if g is not None else None
                         for (name, _), g in zip(named, gradients)}
    labels = list(values)
    pairs = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            group_rows = {}
            groups = sorted({name.split(".")[0] for name, _ in named})
            for group in [*groups, "<all_shared>"]:
                names = [name for name, _ in named if group == "<all_shared>"
                         or name.split(".")[0] == group]
                common = [(values[left][n], values[right][n]) for n in names
                          if values[left][n] is not None and
                          values[right][n] is not None]
                if any(not _finite(x) or not _finite(y) for x, y in common):
                    group_rows[group] = {"cosine": None, "status": "NONFINITE_GRADIENT"}
                    continue
                if not common:
                    group_rows[group] = {"cosine": None, "status": "NO_SHARED_GRADIENT"}
                    continue
                dot = sum(float((a * b).sum().item()) for a, b in common)
                na = math.sqrt(sum(float((a * a).sum().item()) for a, _ in common))
                nb = math.sqrt(sum(float((b * b).sum().item()) for _, b in common))
                cosine = max(-1., min(1., dot / (na * nb))) if na and nb else None
                group_rows[group] = {
                    "cosine": cosine,
                    "status": ("CONFLICT_AT_THIS_POINT" if cosine is not None and cosine < 0
                               else "NO_LOCAL_CONFLICT" if cosine is not None
                               else "ZERO_GRADIENT"),
                    "shared_parameter_tensors": len(common),
                    "left_l2": na, "right_l2": nb,
                }
            pairs.append({"left": left, "right": right, "components": group_rows})
    return {"schema": "tac-osm-observatory/gradient-conflicts-v1",
            "pairs": pairs, "qualification": "local_gradients_not_training_outcome"}


def interpret_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic investigation suggestions; no diagnosis asserted as fact."""
    warnings = []
    memory = report.get("memory")
    if isinstance(memory, Mapping):
        if not memory.get("query_does_not_write_state", True):
            warnings.append({"area": "prediction", "priority": "high",
                             "hypothesis": "candidate query unexpectedly writes state"})
        if memory.get("candidate_permutation_max_abs_error", 0) > 1e-5:
            warnings.append({"area": "action_order", "priority": "high",
                             "hypothesis": "candidate order changes corresponding output"})
        edits = memory.get("interventions", {})
        if (all(k in edits for k in ("fast", "slow", "memory", "all_state"))
                and all(edits[k]["outcome_mean_absolute_change"] == 0 and
                        edits[k]["future_mean_absolute_change"] == 0
                        for k in ("fast", "slow", "memory", "all_state"))):
            warnings.append({"area": "memory", "priority": "review",
                             "hypothesis": "none of the state lesions affected this particular prediction; check task relevance, gradients and trained checkpoints"})
    grads = report.get("gradients")
    if isinstance(grads, Mapping):
        for pair in grads.get("pairs", []):
            for name, status in pair["components"].items():
                if status["status"] == "CONFLICT_AT_THIS_POINT":
                    warnings.append({"area": "losses", "priority": "review",
                                     "hypothesis": f"local competing gradients for {pair['left']}/{pair['right']} in {name}"})
    return {"schema": "tac-osm-observatory/interpretation-v1",
            "alerts": warnings,
            "rule": "hypotheses only; corroborate on authentic FIT/DEV trajectories"}



def entity_isolation_probe(model: nn.Module, history: torch.Tensor,
                           candidates: torch.Tensor, *,
                           tolerance: float = 1e-5) -> dict[str, Any]:
    """Detect cross-entity influence from batching identical causal histories.

    Compare per-entity state and outputs when processed together versus alone.
    Does not test caller identity routing, source joins or global caches outside
    the model clone. Numerical kernel differences can require calibrated tol.
    """
    replica = copy.deepcopy(model).eval()
    _native_inputs(replica, history, candidates)
    if history.shape[0] < 2:
        raise ValueError("entity isolation requires >=2 independent trajectories")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("nonnegative numerical tolerance required")
    with torch.no_grad():
        batch_state = _observe(replica, history)
        batched = replica.predict(batch_state, history[:, -1], candidates)
        max_diff = 0.
        for i in range(history.shape[0]):
            only_history = history[i:i + 1]
            only_candidates = candidates[i:i + 1]
            one_state = _observe(replica, only_history)
            one_out = replica.predict(one_state, only_history[:, -1], only_candidates)
            for left, right in zip(batch_state, one_state):
                max_diff = max(max_diff, float((left[i:i + 1] - right).abs().max().item()))
            for left, right in zip(batched, one_out):
                max_diff = max(max_diff, float((left[i:i + 1] - right).abs().max().item()))
    return {"schema": "tac-osm-observatory/entity-isolation-v1",
            "entities": history.shape[0], "max_absolute_difference": max_diff,
            "tolerance": tolerance, "within_tolerance": max_diff <= tolerance,
            "qualification": "batching_invariance_not_source_identity_verification"}


def profile_native_runtime(model: nn.Module, history: torch.Tensor,
                           candidates: torch.Tensor, *,
                           warmups: int = 2, repeats: int = 8) -> dict[str, Any]:
    """Replay exactly the same source input and candidate query, no optimizer.

    Measures end-to-end local inference through ALL history updates followed
    by one action query. It is NOT a controlled architecture speed comparison.
    CPU/GPU clocks, thermal state, thread count and I/O can alter results.
    """
    import statistics
    import time

    replica = copy.deepcopy(model).eval()
    _native_inputs(replica, history, candidates)
    if not isinstance(warmups, int) or not isinstance(repeats, int) or (
            isinstance(warmups, bool) or isinstance(repeats, bool) or
            warmups < 0 or repeats < 2 or repeats > 1000):
        raise ValueError("warmups>=0 and 2<=repeats<=1000 required")
    device = history.device
    if next(replica.parameters()).device != device or candidates.device != device:
        raise ValueError("model and source inputs must share one device")
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def timed_once() -> tuple[float, int]:
        synchronize()
        start = time.perf_counter()
        state = _observe(replica, history)
        replica.predict(state, history[:, -1], candidates)
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        state_bytes = sum(x.numel() * x.element_size() for x in state)
        return elapsed_ms, state_bytes
    with torch.no_grad():
        for _ in range(warmups):
            timed_once()
        samples = []
        state_bytes = 0
        for _ in range(repeats):
            ms, state_bytes = timed_once()
            samples.append(ms)
    samples.sort()
    ix95 = math.ceil(.95 * len(samples)) - 1
    return {
        "schema": "tac-osm-observatory/runtime-v1",
        "device": str(device), "torch_version": str(torch.__version__),
        "threads": torch.get_num_threads(), "warmups": warmups,
        "repeats": repeats, "history_events": history.shape[1],
        "batch": history.shape[0], "candidate_count": candidates.shape[1],
        "end_to_end_ms_median": statistics.median(samples),
        "end_to_end_ms_p95": samples[ix95],
        "end_to_end_ms_min": samples[0], "end_to_end_ms_max": samples[-1],
        "state_bytes_batch": state_bytes,
        "interpretation": "local replay+query timing; not matched benchmark or real-time guarantee",
    }
