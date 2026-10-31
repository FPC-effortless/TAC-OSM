"""Evaluator-only SENSE-TOK-006 interventions and teacher-forced route regret.

No training, optimizer, data download, TEST access or capability qualification.
Callers must supply authorized FIT/DEV inputs. Targets are used only after
model prediction. Latent codes are lossy; payload-ledger reconstruction is
not code-only reconstruction or compression.
"""
from __future__ import annotations

import copy
from math import log
from types import MethodType

import torch
from torch import Tensor
from torch.nn import functional as F

from .adaptive_codebook_token import AdaptiveCodebookByteTokenGRU
from .adaptive_token_controls import validate_previous_ids


def _validate_targets(ids: Tensor, targets: Tensor) -> None:
    validate_previous_ids(ids)
    if (not isinstance(targets, Tensor) or targets.dtype != torch.long
            or targets.shape != ids.shape or targets.device != ids.device
            or not bool(((targets >= 0) & (targets < 256)).all())):
        raise ValueError("byte targets must match input shape/device and be in 0..255")
    if not torch.equal(ids[:, 1:], targets[:, :-1]):
        raise ValueError("input must be BOS followed by exactly the preceding targets")


def bits_per_byte(logits: Tensor, targets: Tensor) -> float:
    if (logits.shape != (*targets.shape, 256) or not logits.numel()
            or targets.dtype != torch.long or logits.device != targets.device
            or not bool(torch.isfinite(logits).all())
            or not bool(((targets >= 0) & (targets < 256)).all())):
        raise ValueError("finite byte logits and matching byte targets required")
    return float(F.cross_entropy(logits.reshape(-1, 256).double(),
                                 targets.reshape(-1), reduction="mean") / log(2))


def codebook_interventions(model: AdaptiveCodebookByteTokenGRU, ids: Tensor,
                           targets: Tensor | None = None) -> dict:
    """Freeze observed baseline routing, then intervene on code VALUES only.

    A joint key/value row permutation is a relabelling control (up to argmax
    ties/numerics). Wrong-value and zero-value arms retain the original keys
    and hard assignment, and replay identical gate outputs. All work is on
    evaluation copies; original model parameters, mode and gradients survive.
    """
    if not isinstance(model, AdaptiveCodebookByteTokenGRU):
        raise ValueError("S005 codebook graph required")
    validate_previous_ids(ids)
    if targets is not None:
        _validate_targets(ids, targets)
    original = copy.deepcopy(model).eval()
    gate_outputs = []
    handle = original.mode_gate.register_forward_hook(
        lambda _m, _x, out: gate_outputs.append(out.detach().clone()))
    try:
        with torch.no_grad():
            baseline, base_audits, _ = original(ids, audit=True)
    finally:
        handle.remove()
    rows = {}
    for kind in ("joint_row_permutation", "wrong_values_fixed_keys", "zero_values_fixed_keys"):
        candidate = copy.deepcopy(model).eval()
        if kind == "joint_row_permutation":
            with torch.no_grad():
                candidate.token_codes.weight.copy_(candidate.token_codes.weight.flip(0).clone())
        else:
            selector = candidate._assignment

            def changed_values(self, items, selector=selector, kind=kind):
                _, index = selector(items)  # original keys and original ID
                value = self.token_codes.weight[(index + 1) % self.code_count].unsqueeze(0)
                if kind == "zero_values_fixed_keys":
                    value = torch.zeros_like(value)
                return value, index

            candidate._assignment = MethodType(changed_values, candidate)
        outputs = iter(gate_outputs)
        calls = []

        def frozen_gate(_module, _inputs, _output):
            calls.append(1)
            return next(outputs).clone()

        handle = candidate.mode_gate.register_forward_hook(frozen_gate)
        try:
            with torch.no_grad():
                changed, audits, _ = candidate(ids, audit=True)
        finally:
            handle.remove()
        if len(calls) != len(gate_outputs) or any(
            a.base.decisions != b.base.decisions for a, b in zip(base_audits, audits)
        ):
            raise ValueError("code-value intervention changed the frozen route schedule")
        ids_preserved = all(
            [c.code_id for c in a.codes] == [c.code_id for c in b.codes]
            for a, b in zip(base_audits, audits))
        if kind != "joint_row_permutation" and not ids_preserved:
            raise ValueError("value intervention changed the original selector IDs")
        rows[kind] = {
            "max_logit_change": float((baseline - changed).abs().max()),
            "route_schedule_preserved": True,
            "selector_ids_preserved": ids_preserved,
            "bits_per_byte": bits_per_byte(changed, targets) if targets is not None else None,
        }
    return {
        "evidence_class": "EVALUATOR_ONLY_INTERVENTION_NOT_CAPABILITY_EVIDENCE",
        "token_events": sum(len(a.codes) for a in base_audits),
        "baseline_bits_per_byte": bits_per_byte(baseline, targets) if targets is not None else None,
        "interventions": rows,
        "value_knockout_may_be_out_of_distribution": True,
        "joint_permutation_invariance_requires_no_relevant_argmax_tie": True,
    }


def decision_regret(model: AdaptiveCodebookByteTokenGRU, ids: Tensor, targets: Tensor,
                    *, decision_at: int, horizon: int, tie_tolerance: float = 1e-12) -> dict:
    """Realized hindsight regret for ONE internal routing intervention.

    Teacher-forced future observations are revealed in their original order.
    The action at decision_at changes; subsequent policy choices may respond.
    This is not a rollout, an unbiased policy gradient, or a real-world causal
    action effect. Aggregate only on registered DEV decisions/horizons.
    """
    if not isinstance(model, AdaptiveCodebookByteTokenGRU):
        raise ValueError("S005 codebook graph required")
    _validate_targets(ids, targets)
    if (ids.shape[0] != 1 or type(decision_at) is not int
            or not 1 <= decision_at < ids.shape[1]
            or type(horizon) is not int or horizon < 1
            or type(tie_tolerance) not in (int, float)
            or not 0 <= tie_tolerance < float("inf")):
        raise ValueError("one example, legal observed decision and positive horizon required")
    # Never silently shorten an endpoint near the end of a source segment.
    end = decision_at + horizon
    if end > ids.shape[1]:
        raise ValueError("complete registered horizon unavailable")
    replica = copy.deepcopy(model).eval()
    losses, updates = {}, {}
    with torch.no_grad():
        actual, (receipt,), _ = replica(ids, audit=True)
        selected = receipt.base.decisions[decision_at - 1].lower()
        for action in ("byte", "token"):
            output, (audit,), _ = replica(
                ids, audit=True, intervention_at=decision_at, intervention_mode=action)
            if not torch.equal(actual[:, :decision_at], output[:, :decision_at]):
                raise ValueError("decision intervention changed prior predictions")
            losses[action] = bits_per_byte(output[:, decision_at:end],
                                           targets[:, decision_at:end])
            updates[action] = sum(decision_at <= e.available_after < end
                                  for e in audit.base.events)
            if action == selected and not torch.equal(actual, output):
                raise ValueError("selected-action fork failed exact baseline replay")
    gap = losses["byte"] - losses["token"]
    oracle = None if abs(gap) <= tie_tolerance else ("byte" if gap < 0 else "token")
    return {"decision_at": decision_at, "horizon": horizon,
            "selected_action": selected, "loss_bits_per_byte": losses,
            "realized_regret_bits_per_byte": losses[selected] - min(losses.values()),
            "oracle_action": oracle,
            "oracle_agreement": None if oracle is None else selected == oracle,
            "committed_updates_within_horizon": updates,
            "compute_units": "event_counts_not_FLOPs_or_latency",
            "estimand": "teacher_forced_internal_action_with_policy_continuation",
            "qualification": "realized_hindsight_diagnostic_not_expected_optimality"}
