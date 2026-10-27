"""GP-UNC-004: truth-isolated streaming forecast/outcome event contract.

Constructed-only source-boundary diagnostic. Issue payloads NEVER carry truth;
outcomes enter through separate events at their claimed availability time.
The caller still must independently authenticate the original data, clock,
trained checkpoint and source/evaluator permissions before ANY real fit/test.
No real-source execution or independent capability claim is authorized here.
"""
from __future__ import annotations

import heapq
import math
import re
from collections.abc import Iterable, Mapping

from .gp_prequential_nofit import ReplayForecast
from .gp_rankone_nofit import RankOneProjectedResidualGP
from .gp_uncertainty_nofit import _vector

_ISSUE = frozenset(("kind", "row_id", "entity_id", "features",
                    "base_prediction", "feature_at", "issued_at",
                    "checkpoint_sha256"))
_LABEL = frozenset(("kind", "row_id", "available_at", "truth"))
_HEX = re.compile(r"[0-9a-f]{64}")


def _finite(value, name):
    if (type(value) not in (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def replay_truth_isolated_constructed(
    events: Iterable[Mapping], inducing_by_entity: Mapping,
    *, checkpoint_sha256: str, adaptive_feedback: bool = False,
    max_pending: int = 256, max_records: int = 10_000,
    max_entities: int = 32, noise_variance: float = 0.1
) -> tuple[ReplayForecast, ...]:
    """Process one event at a time, never pre-reading withheld future truths.

    Adaptive feedback must be separately authorized and preregistered for
    genuine original-source experiments. Here all tests are constructed.
    Both known label arrivals and missing delayed labels consume queue capacity;
    per-run row IDs consume O(max_records) audit memory beyond fixed GP state.
    """
    if not isinstance(checkpoint_sha256, str) or _HEX.fullmatch(checkpoint_sha256) is None:
        raise ValueError("frozen reference checkpoint sha256 required")
    if type(adaptive_feedback) is not bool:
        raise ValueError("adaptive_feedback must be explicit boolean")
    for label, value, upper in (("max_pending", max_pending, 10_000),
                                ("max_records", max_records, 100_000),
                                ("max_entities", max_entities, 128)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{label} must be bounded positive integer")
    if (not isinstance(inducing_by_entity, Mapping) or not inducing_by_entity or
            len(inducing_by_entity) > max_entities or
            any(not isinstance(k, str) or not k for k in inducing_by_entity)):
        raise ValueError("inducing dictionary must be nonempty with bounded entity IDs")

    models = {entity: RankOneProjectedResidualGP(basis, noise_variance=noise_variance)
              for entity, basis in inducing_by_entity.items()}
    seen = set()
    issued = {}  # row_id -> (entity, features, original base, feature time, issued time)
    released = []  # only outcomes ALREADY arrived, but not yet eligible by strict cutoff
    output = []
    last_event_at = -math.inf
    ordinal = 0

    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("each event must be an exact-key mapping")
        kind = event.get("kind")
        if kind == "issue":
            if set(event) != _ISSUE:
                raise ValueError("issue event must exclude truth and future outcome fields")
            now = _finite(event["issued_at"], "issued_at")
        elif kind == "label_arrived":
            if set(event) != _LABEL:
                raise ValueError("label event requires exact delayed-outcome contract")
            now = _finite(event["available_at"], "available_at")
        else:
            raise ValueError("unknown original-source event kind")
        if now < last_event_at:
            raise ValueError("event stream reversed source-time ordering")
        last_event_at = now

        if kind == "label_arrived":
            row_id = event["row_id"]
            if not isinstance(row_id, str) or row_id not in issued:
                raise ValueError("unknown, duplicate or out-of-order original outcome")
            truth = _finite(event["truth"], "truth")
            entity, x, base, feature_at, original_issue_at = issued[row_id]
            if now <= original_issue_at:
                raise ValueError("no feedback at or before forecast issuance")
            # The label has physically arrived before it is retained anywhere.
            del issued[row_id]
            heapq.heappush(released, (now, ordinal, entity, x, truth-base,
                                      feature_at, original_issue_at))
            ordinal += 1
        else:
            row_id, entity = event["row_id"], event["entity_id"]
            if (not isinstance(row_id, str) or not row_id or
                    not isinstance(entity, str) or entity not in models):
                raise ValueError("unknown entity or invalid row ID")
            if row_id in seen or len(seen) >= max_records:
                raise ValueError("duplicate issued forecast or max_records exhausted")
            if event["checkpoint_sha256"] != checkpoint_sha256:
                raise ValueError("checkpoint identity differs from frozen registration")
            x = _vector(event["features"])
            base = _finite(event["base_prediction"], "base_prediction")
            feature_at = _finite(event["feature_at"], "feature_at")
            if feature_at > now:
                raise ValueError("future feature at decision time")
            # Strict temporal fence: equality does NOT admit an outcome
            # whose arrival order is ambiguous at the same timestamp.
            while released and released[0][0] < now:
                avail, _, old_entity, old_x, residual, old_feature_at, old_issue = heapq.heappop(released)
                if adaptive_feedback:
                    models[old_entity].observe(
                        old_x, residual, feature_at=old_feature_at,
                        issued_at=old_issue, label_at=avail, cutoff=now)
            gp = models[entity]
            p = gp.predict(x)
            output.append(ReplayForecast(
                row_id, entity, now, base, base+p.mean, p.latent_variance,
                p.observed_variance, gp.observations))
            issued[row_id] = (entity, x, base, feature_at, now)
            seen.add(row_id)

        if len(issued) + len(released) > max_pending:
            raise ValueError("source-delayed observation queues exceed bounded audit budget")
    return tuple(output)
