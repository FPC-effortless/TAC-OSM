"""Identity-only controls for offline frozen-memory research.

These are privileged diagnostic addresses, never legal native model inputs.
The caller must supply actual writer identities from an audited source trace.
No source value, label, pair position or model score participates in selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True, order=True)
class WriteIdentity:
    owner: str
    hour: int

    def __post_init__(self):
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("nonempty original owner required")
        if type(self.hour) is not int:
            raise ValueError("original integer source hour required")


def select_intervening_writes(
    writes: list[WriteIdentity], *, target: WriteIdentity,
    query_hour: int, count: int, seed: int,
) -> dict:
    """Select without replacement using a frozen identity-only hash ordering.

    Strictly intervening, same-owner writes only. Missing candidates block;
    no opposite-class pair substitution, duplicated key or random fallback.
    The complete eligible pool and selected identities are retained for replay.
    """
    if type(query_hour) is not int or query_hour <= target.hour:
        raise ValueError("query must follow target")
    if type(count) is not int or count < 1:
        raise ValueError("positive control count required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative frozen seed required")
    if any(not isinstance(w, WriteIdentity) for w in writes):
        raise ValueError("only audited writer identities accepted")
    if len(set(writes)) != len(writes):
        raise ValueError("duplicate source writer identities")
    if target not in writes:
        raise ValueError("target missing from original writer trace")
    pool = sorted(w for w in writes if w.owner == target.owner
                  and target.hour < w.hour < query_hour)
    if len(pool) < count:
        raise ValueError("insufficient same-owner intervening writes; BLOCKED")
    context = ["tac-osm.history-controls.v1", seed, target.owner,
               target.hour, query_hour]

    def rank(w):
        payload = json.dumps(context + [w.hour], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest(), w.hour

    selected = sorted(pool, key=rank)[:count]
    receipt = {
        "schema": context[0], "seed": seed,
        "owner": target.owner, "target_hour": target.hour,
        "query_hour": query_hour, "eligible_hours": [w.hour for w in pool],
        "selected_hours": [w.hour for w in selected],
        "privileged_diagnostic_only": True,
        "source_trace_authenticity": "CALLER_MUST_VERIFY",
        "scientific_qualification": False,
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return {**receipt, "selection_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
