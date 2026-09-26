"""``O_t`` — the environment that closes the loop.

This module is new to TAC-OSM. None of the three source laboratories supplies
an environment: TAC-transformer is a language model, CASM is a substrate with
a *task* generator but no persistent world, and the Stage-2c harness is a
one-shot bandit with no temporal extension at all. So the environment is the
one component that must be written here rather than adapted.

## The design problem

The environment has to do four things at once, and they pull in opposite
directions:

1. **It must be learnable by a tiny linear router**, so that the loop's first
   running version demonstrates the loop rather than the difficulty of
   optimisation. The strongest routing evidence in the portfolio is Stage 2c
   — a REINFORCE-trained linear scorer over a hand-designed basis, no torch.
   So the environment must present that interface: a bit query, a bit context
   marking positions, and bit-vector candidate descriptors.
2. **It must require persistence.** The loop's central claim is that
   information written at step ``t`` can be *used* at step ``t + k``, so the
   environment must contain tasks that cannot be solved from the current
   observation alone.
3. **It must not leak.** The anti-leakage boundary forbids the router from
   seeing outcomes, gold indices, or correct actions. Everything the router
   can observe about a task is in the ``Query`` — public bits, the mark
   vector, and the *address* of any state entry involved, but never the
   answer.
4. **It must have nontrivial candidate sets.** A router over one candidate
   cannot route, so candidates must be numerous enough for static and random
   arms to separate (Stage 2c: 8 candidates, chance 0.125).

The tension is resolved by keeping the task itself extremely simple — one
relation, one target bit vector — while making *where the target comes from*
depend on the family.

## One relation, three sources of the target

Relevance is the Stage-2c relation throughout, adapted verbatim from
``stage2c_relational.py::satisfies_relation`` (``42f9814``: relevance is
``agreement(q, c) restricted to the positions x marks``. The gold candidate
matches a **target** bit vector on every marked position and anti-matches the
query on the unmarked ones, so total query agreement is actively
uninformative. The families differ only in what the target *is*:

* ``relational`` — the target is the public query bits. Solvable by routing
  alone; needs no state. This is the discriminating family, because a static
  total-agreement scorer scores ~0 here.
* ``state_lookup`` — the target is a bit vector the environment wrote to
  persistent state earlier under a named key. Solvable only by reading state.
* ``replay`` — the target is a written vector on the marked positions while
  the anti-match is taken against the *public* query bits. Neither routing
  alone nor lookup alone solves it. This is the family the integration claim
  rests on.

## Query encoding

``Query`` is frozen in the interface, so the two things the router needs —
the public bits and the state address — are carried in the one free field:

    Query.text == f"{space-joined bits}\t{state address}"

The address is empty for ``relational`` tasks and the bits are empty for
``state_lookup`` tasks. Both are legitimate router inputs: the bits define the
relation and the address says where to look. Neither is the answer.

## A note on ``Candidate.action``

In the Stage-2c source the gold candidate carried ``action == 0`` as a
construction tag. That is a leak in disguise: ``action`` uniquely identified
gold, so a router could learn "pick action 0" and score 1.0 without ever
reading the relation. Here ``action`` is assigned *after* shuffling and equals
the candidate's own position, so it carries no information beyond the order
the candidates happen to be presented in.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Sequence

from . import Candidate, Outcome, Query, StateUpdate

__all__ = [
    "Task",
    "RelationalTaskSpec",
    "StateLookupTaskSpec",
    "ReplayTaskSpec",
    "build_relational_task",
    "build_lookup_task",
    "build_replay_task",
    "satisfies_relation",
    "parse_query",
    "WorldConfig",
    "WorldEnvironment",
]


# --------------------------------------------------------------------------- #
# Task specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    """A task the environment will present.

    ``target_action`` is **never** placed on the ``Query``. The query carries
    public bits, the mark vector, and a state address; the leakage auditor
    inspects router inputs, so no gold information crosses into routing.
    ``target_action`` lives here, environment-side, because the environment
    needs the answer to score the outcome — and nowhere else.
    """

    query: Query
    target_action: int
    candidates: tuple[Candidate, ...]
    family: str = "relational"
    detail: Any = None

    def public(self) -> Query:
        """The router-visible view. Identical to ``query``; explicit for safety."""
        return Query(
            text=self.query.text,
            context=self.query.context,
            step=self.query.step,
            provenance=self.query.provenance,
        )


@dataclass(frozen=True)
class RelationalTaskSpec:
    """Carried in ``Task.detail`` for the relational family.

    ``gold_index`` is recorded for diagnostics and oracle arms only; the
    router never sees this object.
    """

    dim: int
    n_marked: int
    noise: float
    gold_descriptor: tuple[int, ...]
    query_bits: tuple[int, ...]
    gold_index: int


@dataclass(frozen=True)
class StateLookupTaskSpec:
    """The persistence-requiring family.

    ``key`` is the persistent-state address the task is about. It reaches the
    router through ``Query.text`` — legitimate addressing information.
    ``written_bits`` is the answer and is hidden.
    """

    key: str
    written_bits: tuple[int, ...]
    n_marked: int


@dataclass(frozen=True)
class ReplayTaskSpec:
    """Hybrid: a written vector combined with the current public query.

    The correct action matches the written vector on the marked positions,
    so the router must locate the key in state *and* apply the relation.
    """

    key: str
    written_bits: tuple[int, ...]
    n_marked: int


# --------------------------------------------------------------------------- #
# The Stage-2c relevance relation, adapted verbatim
# --------------------------------------------------------------------------- #


def satisfies_relation(
    query: tuple[int, ...], context: tuple[int, ...], descriptor: tuple[int, ...]
) -> bool:
    """Relevance: agreement with the query on the positions the context marks.

    Adapted from ``stage2c_relational.py::satisfies_relation`` (``42f9814``).
    Public and observable, so tests can verify the gold candidate is the
    unique satisfier without consulting ``gold_index`` to *construct*
    anything.
    """
    return all(
        descriptor[j] == query[j]
        for j in range(len(query))
        if j < len(context) and context[j] == 1
    )


def n_marked_for(dim: int) -> int:
    """How many positions the context marks. Scales with dim, fixed across tasks."""
    return max(2, dim // 4)


def parse_query(query: Query) -> tuple[tuple[int, ...], str]:
    """Split a query into (public bits, state address).

    Inverse of the encoding documented at the top of this module. The address
    is empty when the task needs no state; the bits are empty when the task
    is a pure lookup. Used by the router and by tests.
    """
    bits_part, _, address = query.text.partition("\t")
    bits = tuple(int(b) for b in bits_part.split()) if bits_part.strip() else ()
    return bits, address


def _rand_bits(rng: random.Random, dim: int) -> list[int]:
    return [rng.randrange(2) for _ in range(dim)]


def _agree_with(
    rng: random.Random, target: Sequence[int], strength: float, dim: int
) -> list[int]:
    d = _rand_bits(rng, dim)
    for j in range(dim):
        if rng.random() < strength:
            d[j] = target[j]
    return d


def _force_violation(
    rng: random.Random, desc: list[int], marks: Sequence[int], target: Sequence[int]
) -> list[int]:
    """Guarantee ``desc`` fails the relation on one marked position.

    Without this the random distractor modes can accidentally satisfy the
    relation, making gold non-unique and capping the oracle arm below 1.0.
    The environment must be unambiguous for the control arms to be
    interpretable. Adapted from ``stage2c_relational.py``.
    """
    j = rng.choice(list(marks))
    desc[j] = 1 - target[j]
    return desc


def _build_candidates(
    rng: random.Random,
    *,
    seed: int,
    target: Sequence[int],
    query_bits: Sequence[int],
    marks: Sequence[int],
    dim: int,
    n_candidates: int,
    noise: float,
    family: str,
) -> tuple[list[Candidate], tuple[int, ...]]:
    """Build a candidate set whose unique relation-satisfier is the target.

    Gold matches ``target`` on every marked position and anti-matches
    ``query_bits`` elsewhere with probability ``1 - noise``. Distractors are
    each forced to violate the relation on at least one marked position:

    - query-deceptive: agrees strongly with the query overall, violates every
      marked position. Beats a static total-agreement scorer, fails the
      relation.
    - near-miss: right on all but one marked position.
    - random: random bits, then forced to violate one marked position.

    Returns the candidate list and the gold descriptor.
    """
    gold_desc: list[int] = []
    for j in range(dim):
        if j in marks:
            gold_desc.append(int(target[j]))
        else:
            gold_desc.append(
                1 - int(query_bits[j]) if rng.random() < 1.0 - noise else rng.randrange(2)
            )
    gold_desc_t = tuple(gold_desc)

    candidates: list[Candidate] = [
        Candidate(
            key=f"k{seed:07d}g0",
            descriptor=gold_desc_t,
            action=0,
            provenance=family,
        )
    ]

    made = 1
    guard = 0
    while made < n_candidates and guard < 100 * n_candidates:
        guard += 1
        mode = rng.randrange(3)
        if mode == 0:
            desc = _agree_with(rng, query_bits, 0.9, dim)
            for j in marks:
                desc[j] = 1 - int(target[j])
        elif mode == 1:
            desc = _rand_bits(rng, dim)
            miss = rng.choice(list(marks))
            for j in marks:
                desc[j] = 1 - int(target[j]) if j == miss else int(target[j])
        else:
            desc = _force_violation(rng, _rand_bits(rng, dim), marks, target)
        desc_t = tuple(desc)
        if desc_t == gold_desc_t:
            continue
        candidates.append(
            Candidate(
                key=f"k{seed:07d}d{made}", descriptor=desc_t, action=made, provenance=family
            )
        )
        made += 1

    while made < n_candidates:
        candidates.append(
            Candidate(
                key=f"k{seed:07d}d{made}",
                descriptor=tuple(_force_violation(rng, _rand_bits(rng, dim), marks, target)),
                action=made,
                provenance=family,
            )
        )
        made += 1

    return candidates, gold_desc_t


def _with_gold_descriptor(detail: Any, gold_descriptor: tuple[int, ...],
                          gold_index: int) -> Any:
    """Rewrite the pre-shuffle gold fields with the post-shuffle truth.

    ``_build_candidates`` returns the descriptor it built for gold and the
    caller may compute a gold index, but ``_finalise`` shuffles the list and
    *rebuilds* every ``Candidate`` in the process. Anything derived before
    that point therefore describes a task that does not exist: the gold index
    points at the pre-shuffle position while the true satisfier sits elsewhere,
    and an oracle arm built on the stale index scores ~0 without the failure
    being attributable to the mechanism under test.
    """
    if detail is None:
        return detail
    if isinstance(detail, RelationalTaskSpec):
        return dataclasses.replace(
            detail, gold_descriptor=gold_descriptor, gold_index=gold_index
        )
    return detail


def _finalise(
    rng: random.Random,
    candidates: list[Candidate],
    *,
    query_bits: Sequence[int],
    context: tuple[int, ...],
    address: str,
    step: int,
    family: str,
    detail: Any,
) -> Task:
    """Shuffle, assign positional actions, and wrap in a ``Task``.

    Actions are assigned *after* the shuffle so that no field on a candidate
    identifies gold. ``target_action`` is the shuffled position of the
    candidate whose key marks it as gold (suffix ``g0``). The spec's
    ``gold_descriptor`` is likewise re-derived from the *placed* candidate,
    because the shuffle and the rebuild happen after the spec was built.
    """
    rng.shuffle(candidates)
    gold_index = next(i for i, c in enumerate(candidates) if c.key.endswith("g0"))
    placed = [
        Candidate(key=c.key, descriptor=c.descriptor, action=i, provenance=c.provenance)
        for i, c in enumerate(candidates)
    ]
    gold_descriptor = tuple(placed[gold_index].descriptor)
    text = " ".join(str(b) for b in query_bits) + "\t" + address
    return Task(
        query=Query(text=text, context=context, step=step, provenance=family),
        target_action=gold_index,
        candidates=tuple(placed),
        family=family,
        detail=_with_gold_descriptor(detail, gold_descriptor, gold_index),
    )


# --------------------------------------------------------------------------- #
# Task builders
# --------------------------------------------------------------------------- #


def build_relational_task(
    seed: int, *, dim: int = 8, n_candidates: int = 8, noise: float = 0.10, step: int = 0
) -> Task:
    """Context-gated relational relevance, adapted from Stage 2c.

    The target is the public query, so this family needs no state. It is the
    discriminating family: the static total-agreement scorer scores ~0 here
    while the learned Stage-2c router scores ~0.89 at 8 candidates.
    """
    _validate_shape(dim, n_candidates)
    rng = random.Random(seed)
    query_bits = tuple(_rand_bits(rng, dim))

    marks = rng.sample(range(dim), n_marked_for(dim))
    context = tuple(1 if j in marks else 0 for j in range(dim))

    candidates, gold_desc = _build_candidates(
        rng,
        seed=seed,
        target=query_bits,
        query_bits=query_bits,
        marks=marks,
        dim=dim,
        n_candidates=n_candidates,
        noise=noise,
        family="relational",
    )
    return _finalise(
        rng,
        candidates,
        query_bits=query_bits,
        context=context,
        address="",
        step=step,
        family="relational",
        detail=RelationalTaskSpec(
            dim=dim,
            n_marked=len(marks),
            noise=noise,
            gold_descriptor=gold_desc,
            query_bits=query_bits,
            gold_index=0,
        ),
    )


def build_lookup_task(
    seed: int,
    state: "StateWriter",
    *,
    dim: int = 8,
    n_candidates: int = 8,
    noise: float = 0.10,
    horizon: int = 0,
    step: int = 0,
) -> Task:
    """A task whose target was written to persistent state earlier.

    ``horizon`` is the number of intervening steps between the write and this
    query. The environment does not enforce it, but it is recorded so a
    persistence measurement can vary it deliberately.

    The write happens here, at task construction, because the environment is
    the component that owns hidden world state. It is a *raw* write,
    pre-verification, which is deliberate: the verifier governs the loop's
    *learning* writes, not the world's own state. That distinction is what
    keeps ``write`` from becoming a reward channel.
    """
    _validate_shape(dim, n_candidates)
    rng = random.Random(seed)
    address = f"world_{seed % 1000:03d}"
    written = tuple(_rand_bits(rng, dim))

    state.write(
        StateUpdate(
            key=address, value=written, task_key=address, success_score=1.0, step=step
        )
    )

    # No public bits for a pure lookup: the target lives entirely in state.
    query_bits: tuple[int, ...] = ()
    marks = rng.sample(range(dim), n_marked_for(dim))
    context = tuple(1 if j in marks else 0 for j in range(dim))

    candidates, _ = _build_candidates(
        rng,
        seed=seed,
        target=written,
        query_bits=written,
        marks=marks,
        dim=dim,
        n_candidates=n_candidates,
        noise=noise,
        family="state_lookup",
    )
    return _finalise(
        rng,
        candidates,
        query_bits=query_bits,
        context=context,
        address=address,
        step=step,
        family="state_lookup",
        detail=StateLookupTaskSpec(
            key=address, written_bits=written, n_marked=len(marks)
        ),
    )


def build_replay_task(
    seed: int,
    state: "StateWriter",
    *,
    dim: int = 8,
    n_candidates: int = 8,
    noise: float = 0.10,
    horizon: int = 0,
    step: int = 0,
) -> Task:
    """Hybrid: a written target combined with the public query.

    The gold candidate matches the *written* vector on the marked positions
    and anti-matches the *public query* bits elsewhere. Neither routing alone
    nor lookup alone solves it: the router must locate ``address`` in state,
    read it, and then apply the context-gated agreement rule.
    """
    _validate_shape(dim, n_candidates)
    rng = random.Random(seed)
    address = f"replay_{seed % 1000:03d}"
    written = tuple(_rand_bits(rng, dim))

    state.write(
        StateUpdate(
            key=address, value=written, task_key=address, success_score=1.0, step=step
        )
    )

    query_bits = tuple(_rand_bits(rng, dim))
    marks = rng.sample(range(dim), n_marked_for(dim))
    context = tuple(1 if j in marks else 0 for j in range(dim))

    candidates, _ = _build_candidates(
        rng,
        seed=seed,
        target=written,
        query_bits=query_bits,
        marks=marks,
        dim=dim,
        n_candidates=n_candidates,
        noise=noise,
        family="replay",
    )
    return _finalise(
        rng,
        candidates,
        query_bits=query_bits,
        context=context,
        address=address,
        step=step,
        family="replay",
        detail=ReplayTaskSpec(key=address, written_bits=written, n_marked=len(marks)),
    )


def _validate_shape(dim: int, n_candidates: int) -> None:
    if dim < 4:
        raise ValueError(f"dim must be >= 4 to mark >=2 positions, got {dim}")
    if n_candidates < 2:
        raise ValueError(f"need at least 2 candidates, got {n_candidates}")


# --------------------------------------------------------------------------- #
# The environment
# --------------------------------------------------------------------------- #


class StateWriter:
    """Structural supertype for the ``state`` argument of task builders.

    Exists only so the annotations above resolve without a circular import.
    Real implementations live in ``state.py`` and satisfy ``PersistentState``.
    """


@dataclass
class WorldConfig:
    """What the world contains. Validated at construction, then frozen by use."""

    dim: int = 8
    n_candidates: int = 8
    noise: float = 0.10
    families: tuple[str, ...] = ("relational", "state_lookup", "replay")
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dim < 4:
            raise ValueError(f"dim must be >= 4, got {self.dim}")
        if self.n_candidates < 2:
            raise ValueError(f"n_candidates must be >= 2, got {self.n_candidates}")
        unknown = set(self.families) - {"relational", "state_lookup", "replay"}
        if unknown:
            raise ValueError(f"unknown task families: {sorted(unknown)}")


class WorldEnvironment:
    """``O_t`` — owns hidden state, observations, transitions, and outcomes.

    Implements the ``Environment`` protocol from ``__init__.py``. The temporal
    contract is load-bearing: ``transition`` is the only place ``Outcome`` is
    produced, and the model calls it *after* routing, so no router can observe
    the outcome it is selecting for.

    The environment holds no persistent state of its own across calls — that
    is ``PersistentState``'s role, and one of the ablations replaces the state
    implementation without touching the environment. The environment *writes*
    to it during task construction, which is how world facts enter the store
    the loop later reads.
    """

    def __init__(self, config: WorldConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config if config is not None else WorldConfig()
        if seed is not None:
            self.config.seed = seed
        self._task_index = 0
        self._step = 0
        self._current: Task | None = None

    # -- task supply ------------------------------------------------------- #

    def next_task(self, state: StateWriter) -> Task:
        """Present one task, cycling the configured families in order.

        The task stream is deterministic given the seed: the same seed and the
        same family order produce the same tasks, so two ablation arms see an
        identical episode stream. That determinism is what makes an arm
        comparison a comparison of the model rather than of the world.
        """
        families = self.config.families
        if not families:
            raise ValueError("WorldConfig.families is empty")
        family = families[self._task_index % len(families)]
        self._task_index += 1

        seed = self.config.seed * 100_003 + self._task_index * 7919
        if family == "relational":
            task = build_relational_task(
                seed,
                dim=self.config.dim,
                n_candidates=self.config.n_candidates,
                noise=self.config.noise,
                step=self._step,
            )
        elif family == "state_lookup":
            task = build_lookup_task(
                seed,
                state,
                dim=self.config.dim,
                n_candidates=self.config.n_candidates,
                noise=self.config.noise,
                step=self._step,
            )
        elif family == "replay":
            task = build_replay_task(
                seed,
                state,
                dim=self.config.dim,
                n_candidates=self.config.n_candidates,
                noise=self.config.noise,
                step=self._step,
            )
        else:
            raise ValueError(f"unreachable family: {family}")

        self._current = task
        return task

    def issue(self, task: Task) -> Query:
        """Bind ``task`` as the live task and return its router-visible query.

        ``next_task`` already binds the task it returns; this exists so tests
        can pin a specific task and inspect the loop's response to it.
        """
        self._current = task
        return task.public()

    # -- the protocol ------------------------------------------------------ #

    def transition(self, state: StateWriter, action: int, query: Query) -> Outcome:
        """Apply ``action`` and reveal the outcome.

        ``action`` is the index of the candidate the model selected. The
        environment scores it against the task it issued for this step, so the
        outcome is defined by the world, never by the model.
        """
        if self._current is None:
            raise RuntimeError("no task has been issued for this step")
        task = self._current
        success = action == task.target_action
        outcome = Outcome(
            success=success,
            value=float(success),
            feedback="ok" if success else "wrong_candidate",
            detail=task.detail,
        )
        self._step += 1
        self._current = None
        return outcome

    def success(self, query: Query, outcome: Outcome) -> bool:
        """Score the outcome. The router never sees this."""
        return bool(outcome.success)
