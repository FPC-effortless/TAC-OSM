"""Research runner and measurement functions for fused TAC-OSM.

The benchmark reports capability and mechanism metrics separately. In
particular, a smaller active set is never treated as a success unless accuracy
and task validity remain intact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from .ablation import FusionAblationConfig, all_fusion_matrices, full_config
from .builder import build_fused_model


@dataclass(frozen=True)
class FusionMetrics:
    name: str
    seed: int
    steps: int
    accuracy: float
    successes: int
    verified_steps: int
    verified_commits: int
    verification_failures: int
    writes: int
    repair_signals: int
    mean_modules_selected: float
    mean_module_candidates: float
    mean_routing_work: float
    mean_halting_steps: float
    module_selection_counts: dict[str, int]
    accuracy_by_family: dict[str, float]
    steps_by_family: dict[str, int]
    false_commit_count: int
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_arm(
    config: FusionAblationConfig,
    *,
    n_steps: int = 24,
) -> FusionMetrics:
    built = build_fused_model(config, n_steps=n_steps)
    steps = built.model.run()

    family_successes: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    verified = 0
    commits = 0
    repairs = 0
    false_commits = 0
    total_selected = 0
    total_module_candidates = 0
    total_routing_work = 0
    total_halting = 0
    selection_counts: dict[str, int] = {}

    for step in steps:
        family_counts[step.family] = family_counts.get(step.family, 0) + 1
        if step.outcome.success:
            family_successes[step.family] = family_successes.get(step.family, 0) + 1
        verified += int(step.verification_passed)
        commits += int(step.wrote_experience)
        repairs += int(step.repair is not None)
        false_commits += int(step.wrote_experience and not step.outcome.success)
        total_selected += len(step.coordination.selected_modules)
        total_module_candidates += len(step.coordination.candidate_module_ids)
        total_routing_work += step.coordination.routing_work
        total_halting += sum(
            execution.halting_steps for execution in step.module_executions
        )
        for module_id in step.coordination.selected_modules:
            key = str(module_id)
            selection_counts[key] = selection_counts.get(key, 0) + 1

    accuracy_by_family = {
        family: family_successes.get(family, 0) / count
        for family, count in family_counts.items()
    }
    execution_count = sum(len(step.module_executions) for step in steps)

    return FusionMetrics(
        name=config.name,
        seed=config.seed,
        steps=len(steps),
        accuracy=sum(int(s.outcome.success) for s in steps) / max(1, len(steps)),
        successes=sum(int(s.outcome.success) for s in steps),
        verified_steps=verified,
        verified_commits=commits,
        verification_failures=len(steps) - verified,
        writes=commits,
        repair_signals=repairs,
        mean_modules_selected=total_selected / max(1, len(steps)),
        mean_module_candidates=total_module_candidates / max(1, len(steps)),
        mean_routing_work=total_routing_work / max(1, len(steps)),
        mean_halting_steps=total_halting / max(1, execution_count),
        module_selection_counts=selection_counts,
        accuracy_by_family=accuracy_by_family,
        steps_by_family=family_counts,
        false_commit_count=false_commits,
        manifest=built.manifest,
    )


def iter_matrix(
    matrix: Iterable[FusionAblationConfig],
    *,
    seed: int,
) -> Iterable[FusionAblationConfig]:
    """Yield the matrix with the requested seed substituted deterministically."""
    for config in matrix:
        yield FusionAblationConfig(**{
            **config.__dict__,
            "seed": seed,
        })


def run_suite(
    suite: str = "component",
    *,
    seed: int = 0,
    n_steps: int = 24,
) -> list[FusionMetrics]:
    matrices = all_fusion_matrices(seed)
    if suite == "smoke":
        configs = [full_config(seed)]
    elif suite == "all":
        configs = [
            config
            for matrix in matrices.values()
            for config in matrix
        ]
    elif suite in matrices:
        configs = matrices[suite]
    else:
        raise ValueError(f"unknown suite {suite!r}; expected smoke, all, or {sorted(matrices)}")

    # Duplicate full-control cells are deliberately common across matrices.
    # Deduplicate by the complete configuration, not by the label.
    unique: list[FusionAblationConfig] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps(config.describe(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(config)

    return [run_arm(config, n_steps=n_steps) for config in unique]


def run_multi_seed_suite(
    suite: str = "component",
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_steps: int = 24,
) -> list[FusionMetrics]:
    """Run the exact same suite for independently seeded matched arms."""
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    results: list[FusionMetrics] = []
    for seed in seeds:
        results.extend(run_suite(suite, seed=seed, n_steps=n_steps))
    return results


def summarize_metrics(metrics: Sequence[FusionMetrics]) -> list[dict[str, Any]]:
    """Aggregate per-arm accuracy and mechanism metrics across seeds."""
    grouped: dict[str, list[FusionMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(metric.name, []).append(metric)
    summary: list[dict[str, Any]] = []
    for name, group in sorted(grouped.items()):
        accuracies = [m.accuracy for m in group]
        mean_accuracy = mean(accuracies)
        sd = stdev(accuracies) if len(accuracies) > 1 else 0.0
        ci95 = 1.96 * sd / (len(accuracies) ** 0.5) if len(accuracies) > 1 else 0.0
        summary.append({
            "name": name,
            "n_seeds": len(group),
            "accuracy_mean": mean_accuracy,
            "accuracy_sd": sd,
            "accuracy_ci95_halfwidth": ci95,
            "mean_modules_selected": mean(m.mean_modules_selected for m in group),
            "mean_module_candidates": mean(m.mean_module_candidates for m in group),
            "mean_routing_work": mean(m.mean_routing_work for m in group),
            "mean_halting_steps": mean(m.mean_halting_steps for m in group),
            "false_commit_count_total": sum(m.false_commit_count for m in group),
        })
    return summary


def write_results(metrics: Sequence[FusionMetrics], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([m.to_dict() for m in metrics], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
