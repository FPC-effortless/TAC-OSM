"""Architecture-neutral diagnostic interpretation: hypotheses, never proof.

Works with native compatibility and arbitrary ResearchProbeAdapter schemas.
Does not assert a research model is learning, causal, or qualified.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def interpret_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    alerts = []

    def warn(area: str, priority: str, hypothesis: str) -> None:
        alerts.append({"area": area, "priority": priority, "hypothesis": hypothesis})

    memory = report.get("memory")
    if isinstance(memory, Mapping):
        if memory.get("query_does_not_write_state") is False:
            warn("prediction", "high", "prediction changed model buffers or caller-owned state")
        permutation = memory.get("candidate_permutation")
        if (isinstance(permutation, Mapping) and permutation.get("status") == "FAIL"
                or memory.get("candidate_permutation_max_abs_error", 0) > 1e-5):
            warn("action_order", "high",
                 "declared candidate permutation alters restored prediction outputs")
        interventions = memory.get("interventions")
        if isinstance(interventions, Mapping):
            # Core: named arbitrary state components, named arbitrary heads.
            entries = [(k, v) for k, v in interventions.items()
                       if k not in ("<reset_all>", "all_state")]
            core_complete = entries and all(
                isinstance(v, Mapping) and
                isinstance(v.get("head_mean_absolute_change"), Mapping) and
                bool(v["head_mean_absolute_change"])
                for _, v in entries)
            if core_complete and all(
                all(isinstance(d, (int, float)) and d == 0
                    for d in item["head_mean_absolute_change"].values())
                for _, item in entries):
                warn("memory", "review",
                     "all measured named-state lesions had zero local output effect; inspect task dependence, source history and gradients")
            # Historical native helper: only if the complete native readout
            # lesion set exists; do not warn on missing/untraced entries.
            elif all(k in interventions for k in ("fast", "slow", "memory", "all_state")) and all(
                    "outcome_mean_absolute_change" in interventions[k] and
                    "future_mean_absolute_change" in interventions[k] and
                    interventions[k]["outcome_mean_absolute_change"] == 0 and
                    interventions[k]["future_mean_absolute_change"] == 0
                    for k in ("fast", "slow", "memory", "all_state")):
                warn("memory", "review",
                     "all native memory lesions had zero local output effect; inspect task dependence and gradients")

    isolation = report.get("entity_isolation")
    if isinstance(isolation, Mapping) and isolation.get("status") == "FAIL":
        warn("entity_isolation", "high",
             "per-entity predictions differ when equivalent inputs are batched")

    sources = report.get("dependencies")
    if isinstance(sources, Mapping) and sources.get("status", "").startswith("INCONCLUSIVE"):
        warn("coverage", "review",
             "local input-to-output gradient dependencies have not been instrumented")

    gradients = report.get("gradients")
    if not isinstance(gradients, Mapping):
        gradients = report.get("gradient_conflicts")
    if isinstance(gradients, Mapping):
        for pair in gradients.get("pairs", ()):
            if not isinstance(pair, Mapping):
                continue
            for name, observation in pair.get("components", {}).items():
                if (isinstance(observation, Mapping)
                        and observation.get("status") == "CONFLICT_AT_THIS_POINT"):
                    warn("losses", "review",
                         f"local competing gradients for {pair['left']}/{pair['right']} in {name}")

    return {"schema": "tac-osm-observatory/interpretation-v1",
            "alerts": alerts,
            "rule": "hypotheses only; corroborate on authentic FIT/DEV trajectories"}
