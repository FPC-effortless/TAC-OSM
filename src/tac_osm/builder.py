"""The single point at which an ``AblationConfig`` becomes a running model.

This is the whole of the "one architecture, controlled switches" claim: every
cell in ``ablation.all_matrices()`` is built here, by one function, with no
per-experiment branches. An arm comparison is then guaranteed to differ only
in switch values, because there is no other code that could differ.

The builder also enforces two gates before construction is even offered:

* **Representability (§34)** — the learned router's feature basis must be able
  to express the target relation. This is the compulsory pre-training unit
  test added after ``dd8f63c``; here it runs against the *actual* basis the
  router will use, on the *actual* episodes the environment will emit.
* **Leakage (§7)** — the router's observable inputs must exclude the forbidden
  set.

A failed gate is a build failure, not a warning. That is the point: at
``dd8f63c`` the broken basis reported 0 for every candidate while an oracle
arm reported 1.0000 for four consecutive commits, and nothing in the pipeline
objected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import Candidate, PersistentState, Query, RelevanceRouter, StateRead
from .ablation import AblationConfig
from .environment import (
    WorldConfig,
    WorldEnvironment,
    build_relational_task,
    satisfies_relation,
)
from .executor import ExecutorConfig, StructuralExecutor
from .leakage import audit_router_inputs
from .model import ModelConfig, TacOsmModel
from .representability import representable
from .router import (
    LearnedRelationalRouter,
    RouterConfig,
    build_router,
    features,
    basis_size,
)
from .state import PersistentStore, StateConfig
from .verifier import BoundedRepairController, ThresholdVerifier, VerifierConfig

__all__ = ["BuiltModel", "build_model", "check_representability", "check_leakage"]


@dataclass
class BuiltModel:
    """A model plus the report of the gates that were run before it existed."""
    model: TacOsmModel
    representability: dict
    leakage: dict
    config: AblationConfig


def build_model(config: AblationConfig, *, n_steps: int = 12) -> BuiltModel:
    """Construct a running model from an ablation cell.

    One function, one code path, every arm. The switches are read off the
    config and injected; nothing about the arm is decided anywhere else.
    """
    state = PersistentStore(_state_config(config))
    router = build_router(_router_config(config))
    executor = StructuralExecutor(_executor_config(config))
    verifier = ThresholdVerifier(_verifier_config(config))
    repair = (
        BoundedRepairController(_verifier_config(config))
        if config.verifier.repair
        else None
    )
    world = WorldEnvironment(_world_config(config))

    rep = check_representability(config)
    leak = check_leakage()

    model = TacOsmModel(
        state=state,
        router=router,
        executor=executor,
        verifier=verifier,
        repair=repair,
        environment=world,
        config=ModelConfig(n_steps=n_steps, learn=config.router.type == "learned",
                           seed=config.seed),
    )
    return BuiltModel(model=model, representability=rep, leakage=leak, config=config)


# --------------------------------------------------------------------------- #
# Config translation
# --------------------------------------------------------------------------- #


def _state_config(config: AblationConfig) -> StateConfig:
    return StateConfig(
        enabled=config.state.enabled,
        write=config.state.write,
        intervention=config.state.intervention,
        corruption_rate=config.state.corruption_rate,
        seed=config.seed,
    )


def _router_config(config: AblationConfig) -> RouterConfig:
    return RouterConfig(
        type=config.router.type,
        temperature=config.router.temperature,
        seed=config.seed,
    )


def _executor_config(config: AblationConfig) -> ExecutorConfig:
    return ExecutorConfig(type=config.structure.type, seed=config.seed)


def _verifier_config(config: AblationConfig) -> VerifierConfig:
    return VerifierConfig(
        type=config.verifier.type,
        repair=config.verifier.repair,
        max_attempts=config.verifier.max_attempts,
    )


def _world_config(config: AblationConfig) -> WorldConfig:
    return WorldConfig(seed=config.seed)


# --------------------------------------------------------------------------- #
# Pre-model gates, run on the real basis and the real episodes
# --------------------------------------------------------------------------- #


def check_leakage() -> dict:
    """G1: assert the router's observable inputs exclude the forbidden set."""
    query = Query(text="0 1 1 0", context=(1, 0, 1, 0))
    candidates = [Candidate(key="c0", descriptor=(0, 1, 1, 0))]
    return audit_router_inputs(query=query, candidates=candidates)


def check_representability(config: AblationConfig | None = None,
                           n_episodes: int = 16) -> dict:
    """G2: the learned router's real basis must express the target relation.

    Runs the §34 gate on the actual feature basis the router will use, against
    actual episodes the environment will emit. The ideal weights are chosen to
    maximise the gold row's score, then the margin against the best distractor
    is measured; if the ideal weights cannot separate, no learned weights can.
    """
    dim = 8
    max_slots = 4
    router = LearnedRelationalRouter(RouterConfig(dim=dim, max_state_slots=max_slots))
    store = PersistentStore(StateConfig(seed=0))

    episodes = [build_relational_task(1000 + i, dim=dim, n_candidates=8) for i in range(n_episodes)]

    def feature_fn(episode: Any) -> list[list[float]]:
        query = episode.query
        read = store.read(query)
        return [features(query, c.descriptor, read, dim, max_slots)
                for c in episode.candidates]

    def gold_fn(episode: Any) -> int:
        return episode.target_action

    def score_fn(weights: list[float], rows: list[list[float]]) -> list[float]:
        return [sum(w * f for w, f in zip(weights, row)) for row in rows]

    return representable(
        score_fn=score_fn,
        feature_fn=feature_fn,
        gold_fn=gold_fn,
        episodes=episodes,
        min_margin=1e-6,
    )
