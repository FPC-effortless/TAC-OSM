"""Pre-registered ablation catalog for the fused TAC-OSM research chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FusionAblationConfig:
    name: str
    seed: int = 0

    fact_enabled: bool = True
    fact_write: bool = True
    fact_intervention: str = "persistent"

    memory_enabled: bool = True
    memory_timescales: int = 4
    local_memory: bool = True
    global_memory: bool = True
    priming: bool = True
    memory_intervention: str = "persistent"

    coordinator_mode: str = "sparse"
    top_k_modules: int = 2
    disabled_modules: tuple[int, ...] = ()

    shared_operators: bool = False
    adaptive_halting: bool = True

    verifier_mode: str = "path"
    verifier_commit_success_only: bool = True
    repair: bool = True

    provenance: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "provenance"}
        data["provenance"] = self.provenance
        return data


def full_config(seed: int = 0) -> FusionAblationConfig:
    return FusionAblationConfig(name="full_fused", seed=seed)


def all_fusion_matrices(seed: int = 0) -> dict[str, list[FusionAblationConfig]]:
    return {
        "component": component_matrix(seed),
        "memory": memory_matrix(seed),
        "coordination": coordination_matrix(seed),
        "priming": priming_matrix(seed),
        "specialization": specialization_matrix(seed),
        "halting": halting_matrix(seed),
        "verification": verification_matrix(seed),
        "fact_state": fact_state_matrix(seed),
        "integrity": integrity_matrix(seed),
        "lesion": lesion_matrix(seed),
    }


def component_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="no_fact_state", fact_enabled=False, fact_write=False),
        replace(f, name="no_experience_memory", memory_enabled=False),
        replace(f, name="no_local_memory", local_memory=False),
        replace(f, name="no_global_memory", global_memory=False),
        replace(f, name="no_priming", priming=False),
        replace(f, name="dense_coordination", coordinator_mode="dense"),
        replace(f, name="single_module", coordinator_mode="single", top_k_modules=1),
        replace(f, name="all_modules", coordinator_mode="all", top_k_modules=8),
        replace(f, name="shared_operators", shared_operators=True),
        replace(f, name="fixed_halting", adaptive_halting=False),
        replace(f, name="no_verifier", verifier_mode="none", repair=False),
        replace(f, name="final_verifier", verifier_mode="final", repair=False),
        replace(f, name="no_repair", repair=False),
    ]


def memory_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="memory_fast_only", memory_timescales=1),
        replace(f, name="memory_two_timescales", memory_timescales=2),
        replace(f, name="memory_no_local", local_memory=False),
        replace(f, name="memory_no_global", global_memory=False),
        replace(f, name="memory_reset", memory_intervention="reset"),
        replace(f, name="memory_shuffle", memory_intervention="shuffled"),
        replace(f, name="memory_random", memory_intervention="random"),
        replace(f, name="memory_corrupt_50", memory_intervention="corrupted", memory_timescales=4),
        replace(f, name="memory_wrong_module", memory_intervention="wrong_key"),
    ]


def coordination_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="coord_sparse_k1", top_k_modules=1),
        replace(f, name="coord_sparse_k4", top_k_modules=4),
        replace(f, name="coord_dense", coordinator_mode="dense"),
        replace(f, name="coord_single", coordinator_mode="single", top_k_modules=1),
        replace(f, name="coord_all", coordinator_mode="all", top_k_modules=8),
        replace(f, name="coord_disabled", coordinator_mode="disabled"),
    ]


def priming_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="priming_off", priming=False),
        replace(f, name="priming_off_no_memory", priming=False, memory_enabled=False),
    ]


def specialization_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        replace(f, name="specialists_independent", shared_operators=False),
        replace(f, name="specialists_shared", shared_operators=True),
    ]


def halting_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        replace(f, name="halting_adaptive", adaptive_halting=True),
        replace(f, name="halting_fixed", adaptive_halting=False),
    ]


def verification_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        replace(f, name="verifier_none", verifier_mode="none", repair=False),
        replace(f, name="verifier_final", verifier_mode="final", repair=False),
        replace(f, name="verifier_path", verifier_mode="path", repair=False),
        f,
        replace(f, name="verifier_path_no_commit_gate", verifier_commit_success_only=False),
        replace(f, name="verifier_path_no_repair", repair=False),
    ]


def fact_state_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="facts_reset", fact_intervention="reset"),
        replace(f, name="facts_shuffled", fact_intervention="shuffled"),
        replace(f, name="facts_random", fact_intervention="random"),
        replace(f, name="facts_corrupt_50", fact_intervention="corrupted"),
        replace(f, name="facts_wrong_key", fact_intervention="wrong_key"),
        replace(f, name="facts_no_write", fact_write=False),
        replace(f, name="facts_disabled", fact_enabled=False, fact_write=False),
    ]


def integrity_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="integrity_dynamic_reset", memory_intervention="reset"),
        replace(f, name="integrity_dynamic_shuffle", memory_intervention="shuffled"),
        replace(f, name="integrity_no_future_outcome_channel"),
    ]


def lesion_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    return [
        replace(full_config(seed), name=f"lesion_module_{m}", disabled_modules=(m,))
        for m in range(8)
    ]


def replace(
    config: FusionAblationConfig,
    *,
    name: str,
    **changes: Any,
) -> FusionAblationConfig:
    values = {**config.__dict__, **changes, "name": name}
    return FusionAblationConfig(**values)
