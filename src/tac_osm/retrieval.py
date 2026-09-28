"""The retrieval boundary — a cheap index that narrows ``H`` to ``K`` before scoring.

## What this module is

``H → cheap index → K candidates → existing scorer → R``

M2.1 (historically Stage F1, pre-registered in ``docs/TACOSM-RETRIEVAL-001.md``
and ``docs/ROADMAP.md``). The index is the *only new mechanism*: the scorer is
the router's own :meth:`~tac_osm.router.LearnedRelationalRouter.score`,
unchanged and reused as-is, because the claim under test is a cost-quality
trade and not a scorer-quality claim (see the "what this experiment is not"
section of the pre-registration).

The reason the module exists is C5: until a retrieval boundary sits between
routing and execution, ``|R|`` is not a defined quantity and the claim is
unstatable rather than merely untested (``docs/CLAIMS.md``). This module is
what makes ``|R|`` measurable.

## Why the index is a *filter* and not a second scorer

The index must be **cheap** — that is the whole point of the trade — so it
cannot itself be a learned scorer over the router's full basis, or it would
cost the same as the exhaustive arm and the experiment would have no cost
axis to measure. It is a *hard filter*: a vector-space pruning rule that
decides, from information the router already legitimately sees, which
candidates are worth scoring, and it never consults ``target_action``,
``outcome``, ``gold_index`` or any forbidden field (see
:mod:`tac_osm.leakage`).

That has a consequence the design commits to rather than hides: **an index
built from cheap features can be a better *filter* than the learned scorer
when the scorer is badly trained, and a worse one when the scorer is good.**
The measurement therefore reports both directions and does not assume which
one holds — the C-vs-B comparison is defined so that either outcome is a
readable result, exactly as the pre-registration requires.

## The retrieval budget

``K`` is the *budget*: the number of candidates the scorer sees after
filtering. ``K < H`` is the only regime where the index is doing anything at
all, so ``K >= H`` is not a budget — it recalls everything by construction.
:func:`budgets` drops those levels rather than clamping, because a clamped
column would silently report a different ``K`` from the one a reader asked
for. The same rule F0 and MATCHED-001 use, in one place now rather than three.

## Layer 0 — infrastructure, not evidence

This module establishes no scientific claim. It provides the mechanism a
pre-registered experiment measures; the result belongs to
``TACOSM-RETRIEVAL-001``'s F1 run, recorded in ``results/retrieval_001.json``,
and to no other document. A negative F1 result — the index costs more quality
than it saves computation — withdraws nothing from Layer 2 and is reported as
the negative result it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from . import Candidate, Query, StateRead
from .environment import parse_query
from .router import _bits

__all__ = [
    "RetrievalIndex",
    "retrieve",
    "budgets",
    "top_k",
    "gold_rank",
    "retrieval_miss",
    "ranking_miss",
    "FAILURE_CATEGORIES",
]


# --------------------------------------------------------------------------- #
# The retrieval budget, defined once
# --------------------------------------------------------------------------- #


def budgets(h: int, k_levels: Sequence[int]) -> tuple[int, ...]:
    """The ``K`` levels that are well-defined at this candidate count.

    A retrieval budget at or above the candidate population is not a budget —
    it recalls everything by construction. The interesting part of the curve
    is always ``K < H``, so levels at or above ``H`` are **dropped rather
    than clamped**: a clamped column would silently report a different ``K``
    from the one a reader asked for, and a reader comparing an indexed arm at
    ``K=16`` against an exhaustive arm at ``H=8`` would be comparing two
    different experiments.

    This function is the one definition; F0 and MATCHED-001 each carried a
    private copy, and three copies of a rule is three chances for the rule to
    drift.
    """
    return tuple(int(k) for k in k_levels if int(k) < int(h))


def top_k(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Indices of the ``k`` highest scores, ties broken by position.

    The selection rule F0 used, so the recall columns are directly comparable
    across experiments. Ties are broken *against* gold throughout the
    retrieval statistics: a tie for first is reported as rank 2, because the
    router samples among the tied set and cannot be said to rank gold first.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return tuple(order[: max(0, min(int(k), len(scores)))])


def gold_rank(scores: Sequence[float], gold: int) -> int:
    """1-indexed rank of gold (1 == best), ties counting against gold.

    A candidate *strictly above* gold pushes gold's rank down by one; a
    candidate *tied with* gold does not, so gold tied with one other
    candidate is rank 1 and gold below two others is rank 3. The convention
    is stated here because it is not the same as :func:`top_k`'s —
    ``top_k`` breaks ties by position, so a tie for the last slot is resolved
    one way by it and reported another way by this function. Both are the
    definitions the experiments since HS-001 used; the difference is that
    they answer different questions, one about membership in a set and one
    about position in an ordering.
    """
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #


#: The failure taxonomy, as a closed set. Every failed step is attributed to
#: exactly one category, so a drop in ``exact_success`` can be read as a
#: *kind* of failure rather than a quantity. The taxonomy is what makes the
#: C-vs-B comparison interpretable: without it a quality loss in C could be
#: an index that discards gold (``retrieval_miss``) or a scorer that mis-ranks
#: within the retrieved set (``ranking_miss``) — different failures with the
#: same headline number, and only the first is an argument against the index.
FAILURE_CATEGORIES = (
    "retrieval_miss",
    "ranking_miss",
    "execution_failure",
    "verification_failure",
)


@dataclass(frozen=True)
class RetrievalIndex:
    """A cheap pre-filter that narrows ``H`` candidates to ``K`` before scoring.

    ``H → [index] → K candidates → [scorer] → R``. The index scores every
    candidate with a fixed, unlearned agreement rule and retains the top ``K``
    by *index* score. It is deliberately not the learned scorer: the whole
    point of the trade is that this stage costs far less than the learned
    one, so its rule must be cheap by construction.

    **What the rule is, and why it is what it is.** The signal the index uses
    is *agreement with the written state vector on the positions the context
    marks*, plus public-bits agreement on the marked positions:

        score(c) = w_state  · |{j ∈ marks : written[j] == c[j]}|
                 + w_query  · |{j ∈ marks : query[j]  == c[j]}|
                 + w_cheap  · |{j ∈ marks : c[j] == 1}|           (cheap tie-break)

    Both terms are present because **the term that separates gold depends on
    the family**, and the environment's three families partition cleanly:

    * ``relational`` publishes its target *as* the public query bits, so the
      query term is the separator and the state term is inert — the family
      carries no address, so the written row is empty.
    * ``state_lookup`` and ``replay`` hide the target in the written state
      vector, so the state term is the separator. Here the query term is not
      merely inert but actively misleading, in the way worth recording because
      it is the mistake a reader would expect: the gold candidate's
      public-bits agreement is *below* the best distractor's, because the
      environment's query-deceptive distractors are constructed to agree with
      the query on every unmarked position while violating the relation on
      the marked ones. A public-bits index is a *better* filter for the
      distractors than for gold on these two families, and at large H it
      loses gold entirely.

    The sum of the two terms — the registered default — separates gold on all
    three families, which is what makes one unlearned rule serve the whole
    task stream. The state term is called primary because it is the one that
    carries the persistence families, not because the query term is absent:
    the same signal the analytic vector's ``slot_gated`` block computes (see
    :func:`tac_osm.router.analytic_weights`).

    **The sum is a trade, not a conjunction — and it cancels on ``replay``.**
    ``replay`` publishes query bits *and* an address, so both terms are
    active, and the query term is anti-correlated with gold there. The two
    therefore partially cancel: at the registered default weights the index
    makes gold the unique argmax on ``replay`` only about a third of the
    time, and its retention rate at large H is below the state-only index's.
    That is not a defect in the rule and it is not a tuning question — it is
    the reason the module reports the C-vs-B comparison in both directions,
    and the reason the arm-D ceiling is *measured* rather than assumed. The
    separation is exact when the two terms are not both active: query-only on
    ``relational``, state-only on the persistence families.

    **Why this is a filter and not a second scorer.** It is state-sparse and
    parameter-free: no learned weights, no per-slot expansion, one pass over
    the marked positions per candidate. The learned scorer's basis is
    ``1 + 4·dim + 2·dim·max_slots`` features per candidate; the index's is
    ``n_marked``. That cost ratio is what ``index_operations`` measures, and
    it is why an index that were merely another scorer would have no cost
    axis to report. The pre-registration forbids the alternative reading: a
    better scorer is an M1 result measured under M1's protocol, and would
    change the baseline this experiment is defined against.

    **Leakage.** The index sees only the ``Query``, the candidates' public
    ``descriptor`` and the state read — the same three inputs the router has,
    and nothing else. It never sees ``target_action``, ``outcome`` or
    ``gold_index``; :mod:`tac_osm.leakage` is applied to the index's inputs
    in the test suite exactly as it is applied to the router's. A filter that
    needed the answer would not be a filter, it would be an oracle, and arm D
    exists to measure that ceiling honestly under its own name.
    """

    #: How many candidates the scorer is allowed to see. The retrieval
    #: boundary: ``|R| == K`` once the index is in the loop, which is what
    #: makes C5 statable.
    k: int
    #: Weight on agreement with the written state vector, on marked
    #: positions. The term that separates gold on ``state_lookup`` and
    #: ``replay``; inert on ``relational``, which carries no state address.
    #: See the class docstring for the family-by-family partition.
    weight_state: float = 1.0
    #: Weight on agreement with the public query bits, on marked positions.
    #: The term that separates gold on ``relational``, where the public bits
    #: *are* the target; anti-correlated with gold on the two persistence
    #: families, whose query-deceptive distractors are built to agree with
    #: the query on the unmarked positions.
    weight_query: float = 1.0
    #: Weight on a cheap mark-position indicator, used as a deterministic
    #: tie-break only. It carries no information about which candidate is
    #: relevant, and it exists so a tie among ``n_marked``-many candidates
    #: does not silently resolve by list position — which would favour the
    #: pre-shuffle gold-at-0 layout.
    weight_cheap: float = 0.0
    #: Seed for the tie-break RNG, so a run is reproducible.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    # -- the cheap pass --------------------------------------------------- #

    def score(self, query: Query, candidates: Sequence[Candidate],
              read: StateRead) -> list[float]:
        """Cheap index scores, one per candidate.

        ``O(n_marked)`` per candidate, against the learned scorer's
        ``1 + 4·dim + 2·dim·max_slots`` features per candidate. The ratio is
        what the experiment's ``index_operations`` column measures.
        """
        q_bits = _bits(query.text)
        context = tuple(query.context)
        _, address = parse_query(query)
        #: The addressed slot only. Unaddressed slots have no relation to
        #: this task but their agreement features are still signed, so
        #: including them injects noise that grows with history — the same
        #: reasoning as the learned basis's ``_addressed`` gate.
        written: tuple[int, ...] = ()
        if read is not None and address:
            for key, row in zip(read.keys, read.values):
                if key == address and row:
                    written = tuple(row)
                    break

        out: list[float] = []
        for c in candidates:
            desc = tuple(c.descriptor)
            s = 0.0
            for j in range(min(len(context), len(desc))):
                if not context[j]:
                    continue
                if written and j < len(written) and written[j] == desc[j]:
                    s += self.weight_state
                if j < len(q_bits) and q_bits[j] == desc[j]:
                    s += self.weight_query
                if self.weight_cheap:
                    s += self.weight_cheap if desc[j] == 1 else 0.0
            out.append(s)
        return out

    def retrieve(self, query: Query, candidates: Sequence[Candidate],
                 read: StateRead) -> tuple[tuple[int, ...], list[float]]:
        """Narrow ``candidates`` to the top-``K`` by index score.

        Returns the **indices** of the retained candidates (into the input
        sequence) and the index scores themselves, so a measurement reports
        both what was kept and why.
        """
        idx = self.score(query, candidates, read)
        if self.k >= len(candidates):
            return tuple(range(len(candidates))), idx
        # Highest index score first; ties are broken by a seeded RNG so the
        # retention order is reproducible and does not silently prefer early
        # positions (which would favour the pre-shuffle gold-at-0 layout).
        import random

        rng = random.Random(self.seed)
        order = sorted(range(len(idx)), key=lambda i: (-idx[i], rng.random()))
        return tuple(sorted(order[: self.k])), idx


def retrieve(index: RetrievalIndex | None, query: Query,
             candidates: Sequence[Candidate], read: StateRead,
             ) -> tuple[tuple[int, ...], list[float]]:
    """Apply the index, or pass every candidate through when there is none.

    The pass-through is arm A — no index in the loop, so the retained set is
    the whole population and ``|R| == H``. Arm B is *not* a pass-through:
    after amendment A1 it keeps the scorer's own top-K, so its retained set is
    smaller than the population even with no index applied. Having one entry
    point for both the indexed and the unindexed case is what keeps them on
    one code path — they differ in whether a filter ran, not in how the
    scorer is called.
    """
    if index is None:
        return tuple(range(len(candidates))), [0.0] * len(candidates)
    return index.retrieve(query, candidates, read)


# --------------------------------------------------------------------------- #
# The failure taxonomy
# --------------------------------------------------------------------------- #


def retrieval_miss(gold_in_retrieved: bool) -> str | None:
    """``retrieval_miss`` when gold was not in the set the index retained.

    The index (or the budget) lost the gold candidate. This is the *only*
    category that is an argument against the index, which is why the taxonomy
    exists rather than a single failure count.
    """
    return "retrieval_miss" if not gold_in_retrieved else None


def ranking_miss(gold_in_retrieved: bool, gold_is_selected: bool) -> str | None:
    """``ranking_miss`` when gold was retrievable but the scorer did not pick it.

    Gold survived the index and the scorer still ranked it below a
    distractor — a scorer failure inside the retrieved set, and *not* an
    argument against the index. Reporting the two as one number is what would
    make a bad scorer look like a bad index.
    """
    if gold_in_retrieved and not gold_is_selected:
        return "ranking_miss"
    return None


def classify_step(*, success: bool, gold_in_retrieved: bool,
                  gold_is_selected: bool, verification_passed: bool) -> str | None:
    """Attribute one failed step to exactly one category, or ``None`` on success.

    The taxonomy is *closed*: a failure that fits none of the four categories
    is a bug in the measurement rather than a new kind of failure, and the
    caller raises rather than inventing a fifth bucket. That closure is what
    makes the C-vs-B table interpretable, because a reader can account for
    every lost step exactly once.

    Order matters and is the precedence the pre-registration implies: a
    retrieval miss is blamed on the index before anything downstream is
    considered, because downstream components cannot recover what the index
    discarded.
    """
    if success:
        return None
    if not gold_in_retrieved:
        return "retrieval_miss"
    if not gold_is_selected:
        return "ranking_miss"
    if not verification_passed:
        return "verification_failure"
    return "execution_failure"


# --------------------------------------------------------------------------- #
# The cost terms
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostTerms:
    """The six cost terms of the pre-registered cost model.

    All six are reported per ``(arm, H, K)``, along the same axis, so a
    capability-versus-computation curve is a plot of one table rather than a
    narrative. Two of them are architectural constants in v0.1 rather than
    measurements, and saying so in the field is what keeps the column from
    being read as a scaling result.
    """

    candidates_inspected: int
    router_operations: int
    index_operations: int
    #: ``executor_nodes`` is reported because C5 is untested and the column
    #: must not be read as a scaling result: in v0.1 it is fixed at
    #: ``max_nodes = 10`` by the executor, so it is an *architectural
    #: constant*, not a measurement.
    executor_nodes: int
    verifier_operations: int
    #: Measured, not derived. The only term that depends on all the others at
    #: once, which is why it is reported rather than computed from them.
    #:
    #: Note the asymmetry with :attr:`router_operations`: that term is a
    #: *modelled* count of the scoring work each arm's architecture would
    #: perform, whereas this one is a wall-clock observation. Comparing the
    #: two is comparing a design claim against a runtime measurement, and
    #: the F1 script's reporting states the distinction rather than letting
    #: the two columns be read as one.
    wall_clock: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_inspected": self.candidates_inspected,
            "router_operations": self.router_operations,
            "index_operations": self.index_operations,
            "executor_nodes": self.executor_nodes,
            "verifier_operations": self.verifier_operations,
            "wall_clock": self.wall_clock,
        }


@dataclass(frozen=True)
class ArmResult:
    """One ``(arm, H, K)`` cell of the F1 measurement.

    The capability metrics and the cost terms for one arm, together with the
    failure taxonomy counts that make a capability drop attributable. Frozen,
    because a measurement cell must not be editable after it is reported —
    the same reasoning as the contract and the checkpoint.
    """

    arm: str
    h: int
    k: int
    #: ``P(gold ∈ the arm's retained set AND is the argmax over that set)``.
    #: The capability half of the cost-quality trade. Defined after amendment
    #: A1: arm B's retained set is its own top-K, so the primary is a single
    #: measure over each arm's *own* retained set rather than a top-K over a
    #: full-population score — which is what makes the arms comparable at all.
    recall_at_k: float
    #: ``P(gold is the argmax of the scores)``. Reported alongside recall so
    #: the two are not confused — the error that motivated the C4 gate was
    #: reading one as the other.
    routing_at_1: float
    #: Mean rank of gold under the score, ties against gold.
    gold_rank_mean: float
    #: The loop's own task success, the thing being improved. Differs from
    #: ``routing_at_1`` because the learned router samples rather than taking
    #: the argmax, and the gap is the cost of sampling, not an execution
    #: failure.
    exact_success: float
    #: ``P(task success | gold ∈ top-K)``. A low value means the loss is not
    #: retrieval at all, so the top-K curve is measuring something other than
    #: task-relevant relevance — the control F0 introduced and F1 keeps.
    acc_given_retrieved: float
    costs: CostTerms
    #: Per-category failure counts, keyed by :data:`FAILURE_CATEGORIES`.
    failures: dict[str, int] = field(default_factory=dict)
    n_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "h": self.h,
            "k": self.k,
            "recall@K": self.recall_at_k,
            "routing@1": self.routing_at_1,
            "gold_rank": self.gold_rank_mean,
            "exact_success": self.exact_success,
            "acc|retrieved": self.acc_given_retrieved,
            "costs": self.costs.to_dict(),
            "failures": dict(self.failures),
            "n_steps": self.n_steps,
        }
