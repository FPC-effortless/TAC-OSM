"""Authored ARC control metadata and invertible surface relabeling.

This is a DERIVED SECONDARY EVALUATION DIAGNOSTIC, never primary operational
data or training augmentation. It is not a new held-out ARC task/family.
"""
from __future__ import annotations
import hashlib
import json


COLOR_ROTATION = (0, 2, 3, 4, 5, 6, 7, 8, 9, 1)


def _validate_permutation(permutation):
    if tuple(sorted(permutation)) != tuple(range(10)):
        raise ValueError("bijection of the ten color IDs required")


def relabel_grid(raw, permutation=COLOR_ROTATION):
    """Change cell color IDs only; retain the exact two dimension bytes."""
    _validate_permutation(permutation); raw = bytes(raw)
    if len(raw) < 3 or raw[0] not in b"123456" or raw[1] not in b"123456":
        raise ValueError("compact grid dimensions required")
    if len(raw[2:]) != (raw[0] - 48) * (raw[1] - 48) or any(v not in b"0123456789" for v in raw[2:]):
        raise ValueError("compact grid cell payload required")
    return raw[:2] + bytes(48 + permutation[v - 48] for v in raw[2:])


def relabel_observations(acquisition, query, permutation=COLOR_ROTATION):
    """Only original past demonstrations and current query enter this path."""
    acquisition, query = bytes(acquisition), bytes(query)
    if not acquisition.startswith(b"R;") or not acquisition.endswith(b";") or not query.endswith(b">"):
        raise ValueError("original ARC acquisition/query envelopes required")
    pairs = acquisition[2:-1].split(b";")
    changed = []
    for pair in pairs:
        if pair.count(b">") != 1: raise ValueError("complete observed demonstration pair required")
        before, after = pair.split(b">")
        changed.append(relabel_grid(before, permutation) + b">" + relabel_grid(after, permutation))
    return b"R;" + b";".join(changed) + b";", relabel_grid(query[:-1], permutation) + b">"


def relabel_target_for_scoring(target, permutation=COLOR_ROTATION):
    """Evaluator-only target transform; never call while constructing actors."""
    target = bytes(target)
    if not target.endswith(b";") or target.count(b";") != 1:
        raise ValueError("original compact target plus fixed delimiter required")
    return relabel_grid(target[:-1], permutation) + b";"


def derived_actor(actor, permutation=COLOR_ROTATION):
    if set(actor) != {"id", "domain", "acquisition_bytes", "query_bytes", "generation_limit"} or actor["domain"] != "arc":
        raise ValueError("observed-only original ARC actor required")
    a, q = relabel_observations(actor["acquisition_bytes"], actor["query_bytes"], permutation)
    assert len(a) == len(actor["acquisition_bytes"]) and len(q) == len(actor["query_bytes"])
    identity = hashlib.sha256(json.dumps([actor["id"], list(permutation)], separators=(",", ":")).encode()).hexdigest()
    return {**actor, "id": "derived-color-" + identity, "acquisition_bytes": list(a), "query_bytes": list(q)}


def control_metadata(bundle):
    originals = {a["id"]: a for a in bundle.dev_actors}
    arc = [a for a in bundle.dev_actors if a["domain"] == "arc"]
    donors = {r["id"]: r for r in bundle.dev_donors}
    binding = {r["id"]: r for r in bundle.dev_binding_controls}
    return {"cohort": "eight original authored ARC DEV tasks", "original_task_count": len(arc),
            "legal_donor_available": sum(bool(donors[a["id"]]["available"]) for a in arc),
            "binding_shuffle_available": sum(bool(binding[a["id"]]["available"]) for a in arc),
            "derived_surface_control": {"mapping": list(COLOR_ROTATION), "bijection": True, "background_zero_preserved": True,
                "actor_inputs": "past demonstration and current query bytes only", "gold_mapping": "scorer only after prediction sealing",
                "classification": "derived secondary authored-benchmark evaluation, not independently sourced primary observations",
                "new_rule_family": False, "training_augmentation": False, "invariance_expected": "Tests robustness to a fixed invertible color-ID codebook change; task identities and rule evidence are otherwise the same."},
            "unavailable": {"known_rule_family_labels": True, "new_executable_code_transformations": True,
                            "independently_heldout_rule_compositions": True, "latent_structure_truth_labels": True},
            "claim_limit": "Task-disjoint ARC transfer and color-ID robustness are not proof of hidden executable structure recovery or a theory of intelligence."}
