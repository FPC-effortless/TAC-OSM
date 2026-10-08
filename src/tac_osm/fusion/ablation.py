"""Pre-registered ablation catalog for the fused TAC-OSM research chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FusionAblationConfig:
    name: str
    seed: int = 0
    memory_enabled: bool = True
    memory_timescales: int = 4
    local_memory: bool = True
    global_memory: bool = True
    priming: bool = True
    memory_intervention: str = "persistent"
    coordinator_mode: str = "sparse"
    top_k_modules: int = 2
    shared_operators: bool = False
    adaptive_halting: bool = True
    verifier_mode: str = "path"
    verifier_commit_success_only: bool = True
    repair: bool = True

    provenance: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return self.name

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "memory_enabled": self.memory_enabled,
            "memory_timescales": self.memory_timescales,
            "local_memory": self.local_memory,
            "global_memory": self.global_memory,
            "priming": self.priming,
            "memory_intervention": self.memory_intervention,
            "coordinator_mode": self.coordinator_mode,
            "top_k_modules": self.top_k_modules,
            "shared_operators": self.shared_operators,
            "adaptive_halting": self.adaptive_halting,
            "verifier_mode": self.verifier_mode,
            "verifier_commit_success_only": self.verifier_commit_success_only,
            "repair": self.repair,
            "provenance": self.provenance,
        }


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
        "integrity": integrity_matrix(seed),
        "lesion": lesion_matrix(seed),
    }


def component_matrix(seed: int) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        f,
        replace(f, name="no_memory", memory_enabled=False),
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
        replace(f, name="memory_fast_slow", memory_timescales=2),
        replace(f, name="memory_slow_only", memory_timescales=4, local_memory=False),
        replace(f, name="memory_reset", memory_intervention="reset"),
        replace(f, name="memory_shuffle", memory_intervention="shuffled"),
        replace(f, name="memory_random", memory_intervention="random"),
        replace(
            f, name="memory_corrupt_50", memory_intervention="corrupted"
        ),
        replace(f, name="memory_wrong_key", memory_intervention="wrong_key"),
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
        replace(f, name="coord_disabled", coordinator_mode="disabled", top_k_modules=2),
    ]


def priming_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [f, replace(f, name="priming_off", priming=False)]


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
    ]


def integrity_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    f = full_config(seed)
    return [
        replace(f, name="persistent_control"),
        replace(f, name="state_reset", memory_intervention="reset"),
        replace(f, name="same_present_different_past", memory_intervention="persistent"),
    ]


def lesion_matrix(seed: int = 0) -> list[FusionAblationConfig]:
    # Lesions are applied by the runner after construction; the config name is
    # sufficient to identify the requested lesion without adding another model
    # branch. Module IDs 0..7 are all first-class lesion targets.
    return [
        replace(full_config(seed), name=f"lesion_module_{m}")
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
