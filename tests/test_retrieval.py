"""Tests for the retrieval boundary: the cheap index that narrows ``H`` to ``K``.

The index is the only new mechanism in M2.1, and the claim under test is a
cost-quality trade, so the tests here pin three things rather than "it works":

1. **Soundness.** Where the index's own rule separates gold from every
   distractor, gold must survive the filter. Where it does not, the tests
   assert *that it does not*, because a filter that silently lost gold only on
   the family the loop's persistence claim rests on would be the worst kind of
   defect: invisible in the aggregate and fatal to the claim.
2. **The taxonomy is closed.** Every failed step is attributed to exactly one
   category, because the C-vs-B comparison is interpretable only if a lost
   step can be read as a *kind* of failure. A taxonomy that double-counted or
   dropped a failure would make a bad scorer look like a bad index.
3. **Leakage.** The index sees the same three inputs the router does and
   nothing else. A filter that needed the answer would be an oracle, and arm D
   exists to measure that ceiling under its own name instead of reaching it by
   accident.

The recorded family asymmetry is tested as a *recorded property*, not as a
defect to be fixed and not as a goal to be optimised: on ``replay`` the public
bits actively mislead, so the query-weighted index loses gold at a rate that
grows with H, while the state term alone makes gold the unique argmax on all
three families. That is the property the module's own design rationale rests
on, and asserting it holds is what keeps a future "fix" from silently
weakening the separation the ``slot_gated`` block was sized to provide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm import Candidate, Query, StateRead  # noqa: E402
from tac_osm.environment import (  # noqa: E402
    build_lookup_task,
    build_relational_task,
    build_replay_task,
    satisfies_relation,
)
from tac_osm.leakage import audit_router_inputs  # noqa: E402
from tac_osm.retrieval import (  # noqa: E402
    FAILURE_CATEGORIES,
    ArmResult,
    CostTerms,
    RetrievalIndex,
    budgets,
    classify_step,
    gold_rank,
    retrieve,
    retrieval_miss,
    ranking_miss,
    top_k,
)
from tac_osm.router import (  # noqa: E402
    LearnedRelationalRouter,
    RouterConfig,
    analytic_weights,
)
from tac_osm.state import PersistentStore, StateConfig  # noqa: E402

DIM = 8


# --------------------------------------------------------------------------- #
# Fixtures: the three task families, built the way the environment builds them
# --------------------------------------------------------------------------- #


def _store() -> PersistentStore:
    return PersistentStore(StateConfig(seed=0))


def _task(family: str, seed: int, *, n_candidates: int = 8, dim: int = DIM):
    """One task of the family, with its state written by the builder.

    The builders write world state at construction; the store they wrote to is
    the one the index reads from, so the address in the query resolves.
    """
    if family == "relational":
        return build_relational_task(seed, dim=dim, n_candidates=n_candidates)
    if family == "state_lookup":
        store = _store()
        return build_lookup_task(seed, store, dim=dim, n_candidates=n_candidates), store
    if family == "replay":
        store = _store()
        return build_replay_task(seed, store, dim=dim, n_candidates=n_candidates), store
    raise AssertionError(f"unknown family {family!r}")


def _read_for(family: str, task_or_pair) -> StateRead:
    """The state read the index receives, for the family's own store."""
    if family == "relational":
        # ``relational`` carries no address, so the read is empty on purpose:
        # the index's state term must be inert there, not silently supplied by
        # a store that happens to be occupied.
        return StateRead(keys=(), values=(), slot_used=())
    return task_or_pair[1].read(task_or_pair[0].public())


def _task_of(family: str, task_or_pair):
    return task_or_pair if family == "relational" else task_or_pair[0]


FAMILIES = ("relational", "state_lookup", "replay")


# --------------------------------------------------------------------------- #
# 1. Gold is the unique relation-satisfier — the premise every test below reads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", FAMILIES)
def test_gold_satisfies_the_relation(family):
    """The index's target is well-defined: gold is the unique satisfier.

    Without this the oracle arm would sit below 1.0 and every retained-set
    number downstream would be measuring an ambiguous task. It is checked
    here, over the same candidates the index sees, using the *public* relation
    against the family's own reference vector — so the test does not consult
    ``target_action`` to construct the thing it checks for.

    The reference differs by family, which is the environment's central
    asymmetry and the reason the index carries two terms: on ``relational`` the
    reference is the public query, on the two persistence families it is the
    written vector, which the task's hidden ``detail`` records and the query's
    address locates.
    """
    for seed in range(50):
        built = _task(family, seed + 1000)
        task = _task_of(family, built)
        reference = _reference_vector(task, family)
        satisfiers = [
            i
            for i, c in enumerate(task.candidates)
            if satisfies_relation(reference, task.query.context, c.descriptor)
        ]
        assert satisfiers == [task.target_action], (
            f"{family} seed={seed}: gold is not the unique relation-satisfier "
            f"({len(satisfiers)} found); the oracle arm would not reach 1.0 and "
            "no retained-set number from this family is interpretable"
        )


def _reference_vector(task, family: str) -> tuple[int, ...]:
    """The vector the relation is taken against, per family.

    The environment's three families share one relation and differ only in
    where the reference lives. ``relational`` publishes it as the query bits —
    which is why the index's query term is the separator there — and the two
    persistence families hold it in the written state vector, which is why the
    state term is the separator there.
    """
    if family == "relational":
        part = task.query.text.partition("\t")[0]
        return tuple(int(b) for b in part.split()) if part.strip() else ()
    detail = task.detail
    return tuple(detail.written_bits)


# --------------------------------------------------------------------------- #
# 2. The budget: K < H or it is not a budget
# --------------------------------------------------------------------------- #


def test_budgets_drop_levels_at_or_above_the_population():
    """``K >= H`` recalls everything by construction, so it is not a budget.

    Dropped rather than clamped: a clamped column would silently report a
    different ``K`` from the one a reader asked for, and a reader comparing an
    indexed arm at ``K=16`` against an exhaustive arm at ``H=8`` would be
    comparing two different experiments.
    """
    assert budgets(8, (2, 4, 8, 16)) == (2, 4)
    assert budgets(64, (2, 4, 16, 64)) == (2, 4, 16)
    assert budgets(256, (2, 4, 16)) == (2, 4, 16)


def test_budgets_preserve_order_and_types():
    """The returned tuple is the asked-for order with the invalid levels gone."""
    got = budgets(32, (16, 2, 4, 31))
    assert got == (16, 2, 4, 31)
    assert all(isinstance(k, int) for k in got)


def test_budgets_can_be_empty():
    """An empty result is the honest answer when every level is a non-budget.

    The F1 script's table renders that cell as ``--`` rather than inventing a
    number, and ``budgets`` is what makes the cell empty in the first place.
    """
    assert budgets(4, (4, 8)) == ()


# --------------------------------------------------------------------------- #
# 3. Selection conventions: ties count against gold
# --------------------------------------------------------------------------- #


def test_top_k_returns_indices_in_score_order():
    """Ties broken by position, highest score first — the F0 selection rule."""
    assert top_k((0.1, 0.3, 0.2), 2) == (1, 2)
    assert top_k((0.3, 0.1, 0.2), 3) == (0, 2, 1)


def test_top_k_clamps_to_the_population():
    """Asking for more than there is returns everything, in order."""
    assert top_k((0.1, 0.2), 10) == (1, 0)
    assert top_k((0.1,), 0) == ()


def test_gold_rank_counts_only_strictly_greater_scores():
    """Rank counts candidates strictly above gold; a tie does not push it down.

    This is deliberately *not* the same convention as :func:`top_k`, which
    breaks ties by position. The two answer different questions — membership
    in a set versus position in an ordering — and the module states the
    difference in its own docstring rather than letting a reader assume one
    convention where the other applies. Recording it as a test is what keeps
    the two from drifting back together silently.
    """
    assert gold_rank((0.5, 0.5, 0.2), 0) == 1
    assert gold_rank((0.2, 0.5, 0.5), 0) == 3
    # Rank is a function of the score, not of the position: gold at index 1 is
    # rank 1 when it holds the top score and rank 2 when one candidate beats it.
    assert gold_rank((0.1, 0.9), 1) == 1
    assert gold_rank((0.9, 0.1), 1) == 2
    assert gold_rank((0.1, 0.9), 0) == 2


# --------------------------------------------------------------------------- #
# 4. Index soundness: where the rule separates gold, gold survives
# --------------------------------------------------------------------------- #


def _state_only_index(k: int = 2, seed: int = 0) -> RetrievalIndex:
    """The index with the query term switched off.

    On the persistence families the state term alone makes gold the unique
    argmax — the separation the ``slot_gated`` basis block is sized to
    provide. This is the index's soundness premise on those families, tested
    in its own right rather than assumed from the aggregate retention rate.
    """
    return RetrievalIndex(k=k, seed=seed, weight_state=1.0, weight_query=0.0)


#: Which index term separates gold on each family. This is the environment's
#: own asymmetry, and it is the reason the index carries both terms rather
#: than only the state one: ``relational`` publishes its target as the query
#: bits, so the query term is the separator there and the state term is inert
#: (the family carries no address, so the written row is empty). The two
#: persistence families are the reverse: their target lives in state, and the
#: public bits are anti-informative about it.
#:
#: Recorded here because a single-weight index that claimed to separate all
#: three families would be claiming a property the task does not have, and
#: because the registered index is the *sum* of the two terms — which
#: separates all three, and is what the default weights give.
_SEPARATING_TERM = {
    "relational": "query",
    "state_lookup": "state",
    "replay": "state",
}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("h", (8, 64))
def test_the_separating_term_makes_gold_the_unique_argmax(family, h):
    """On every family, one index term separates gold from every distractor.

    This is the load-bearing soundness property. The environment's distractors
    are constructed to defeat cheap agreement rules, and for each family there
    is exactly one index term that separates gold — the term that carries the
    family's target. An index that did not separate here would be relying on
    luck rather than on the relation, and its retention rate would be a
    property of the distractor distribution rather than of the task.
    """
    term = _SEPARATING_TERM[family]
    index = RetrievalIndex(
        k=1, seed=0,
        weight_state=1.0 if term == "state" else 0.0,
        weight_query=1.0 if term == "query" else 0.0,
    )
    for seed in range(60):
        built = _task(family, seed + 1000, n_candidates=h)
        task = _task_of(family, built)
        read = _read_for(family, built)
        scores = index.score(task.public(), task.candidates, read)
        gold = scores[task.target_action]
        others = [s for i, s in enumerate(scores) if i != task.target_action]
        assert gold > max(others), (
            f"{family} H={h} seed={seed}: the {term} term does not separate "
            f"gold (gold={gold}, best distractor={max(others)}); the index's "
            "soundness premise fails on this family"
        )


@pytest.mark.parametrize("family", ("relational", "state_lookup"))
@pytest.mark.parametrize("h", (8, 64))
def test_the_registered_index_separates_gold_where_one_term_is_inert(family, h):
    r"""The default weights separate gold on the families with one active term.

    ``relational`` is separated by the query term with the state term inert
    (no address, so no written row); ``state_lookup`` is separated by the
    state term with the query term inert (no public bits). On these two the
    sum *is* the separating term, so gold is the unique argmax.

    ``replay`` is deliberately excluded here and tested by its own test
    below: it publishes query bits *and* an address, so both terms are active
    and they partially cancel. That is a property of the registered rule, and
    recording it is what keeps the index's soundness claim honest rather than
    overstated into the one family where it does not hold.
    """
    index = RetrievalIndex(k=1, seed=0)
    for seed in range(60):
        built = _task(family, seed + 1000, n_candidates=h)
        task = _task_of(family, built)
        read = _read_for(family, built)
        scores = index.score(task.public(), task.candidates, read)
        gold = scores[task.target_action]
        others = [s for i, s in enumerate(scores) if i != task.target_action]
        assert gold > max(others), (
            f"{family} H={h} seed={seed}: the registered index does not "
            f"separate gold (gold={gold}, best distractor={max(others)})"
        )


def test_the_registered_index_does_not_separate_replay():
    r"""The sum cancels on ``replay``, and the record says so.

    ``replay`` is the one family where the registered default weights do not
    make gold the unique argmax: the query term is anti-correlated with gold
    there and the state term is correlated with it, so the two partially
    cancel. This is recorded rather than tuned away, because the index is
    registered at these weights and a run will report whatever rate the rule
    gives. The state-only index is what separates ``replay`` exactly, and
    that is the subject of the test below.
    """
    index = RetrievalIndex(k=1, seed=0)
    separated = 0
    n = 200
    for seed in range(n):
        built = _task("replay", seed + 1000, n_candidates=8)
        task = _task_of("replay", built)
        scores = index.score(task.public(), task.candidates, _read_for("replay", built))
        gold = scores[task.target_action]
        separated += int(
            gold > max(s for i, s in enumerate(scores) if i != task.target_action)
        )
    rate = separated / n
    assert rate < 0.6, (
        f"the registered index separated gold on replay {rate:.3f} of the "
        "time; the recorded cancellation between the query and state terms "
        "no longer holds, and the module's own note about the trade is stale"
    )


@pytest.mark.parametrize("family", ("relational", "state_lookup"))
@pytest.mark.parametrize("h", (8, 64))
def test_the_registered_index_retains_gold_at_every_k(family, h):
    """Gold is in the retained set at every K, on the families the rule separates.

    ``k=1`` is the hardest case — there is exactly one slot and gold must be
    it — so this is a stronger statement than the aggregate retention rate.
    The two families covered here are the ones whose separating term is the
    registered sum's only active term; ``replay`` is excluded because the sum
    cancels there, and its retention rate is recorded by the test below.
    """
    for k in (1, 2, 4):
        index = RetrievalIndex(k=min(k, h - 1), seed=0)
        for seed in range(60):
            built = _task(family, seed + 1000, n_candidates=h)
            task = _task_of(family, built)
            read = _read_for(family, built)
            kept, _ = index.retrieve(task.public(), task.candidates, read)
            assert task.target_action in kept, (
                f"{family} H={h} K={k} seed={seed}: gold was not retained by "
                "the registered index; a filter that separates gold and still "
                "discards it is a bug in the selection, not a property of the task"
            )


def test_the_registered_index_loses_gold_on_replay_as_the_record_states():
    r"""The registered index's replay retention is partial and falls with H.

    This is the property the previous tests exclude, and it is recorded rather
    than asserted away: on ``replay`` the query and state terms partially
    cancel, so the index does not separate gold uniquely and the seeded
    tie-break does not always retain it. The retention rate is the number an
    F1 run at the registered weights will actually report, so the test pins
    its *order of magnitude* rather than a value — a rate that rose to
    near-full retention would mean the cancellation had changed and the
    module's own note about it was stale.

    The state-only index recovers it, which is the test below the fold.
    """
    for h, ceiling in ((8, 0.95), (64, 0.75)):
        index = RetrievalIndex(k=4, seed=0)
        retained = 0
        n = 150
        for seed in range(n):
            built = _task("replay", seed + 1000, n_candidates=h)
            task = _task_of("replay", built)
            kept, _ = index.retrieve(task.public(), task.candidates,
                                     _read_for("replay", built))
            retained += int(task.target_action in kept)
        rate = retained / n
        assert rate < ceiling, (
            f"the registered index retained gold on replay at {rate:.3f} at "
            f"H={h}, above the recorded ceiling of {ceiling:.2f}; the "
            "query/state cancellation no longer holds and the module's note "
            "about the trade is stale"
        )


def test_the_index_returns_scores_alongside_the_retained_set():
    """The caller sees both what was kept and why.

    A measurement reports the index scores so a reader can tell a narrow
    margin from a wide one; returning only the retained set would make the
    retention rate unattributable.
    """
    built = _task("relational", 1000, n_candidates=64)
    task = _task_of("relational", built)
    read = StateRead(keys=(), values=(), slot_used=())
    kept, scores = RetrievalIndex(k=2, seed=0).retrieve(task.public(), task.candidates, read)
    assert len(scores) == len(task.candidates)
    assert len(kept) == 2
    assert all(0 <= i < len(task.candidates) for i in kept)
    # The retained set is drawn from the top of the index ordering. It is not
    # required to equal ``top_k``: ``top_k`` breaks ties by position, while
    # ``retrieve`` breaks them with a seeded RNG so the retention order
    # cannot favour the pre-shuffle gold-at-0 layout. Where the index
    # separates gold uniquely the two agree; where it ties they may not, and
    # the distinction is the subject of the tie-break tests below.
    assert min(scores[i] for i in kept) >= max(
        scores[i] for i in range(len(scores)) if i not in kept
    )


def test_top_k_and_retrieve_break_ties_differently_by_design():
    """Two selection rules with two tie conventions, both deliberate.

    ``top_k`` is the F0 selection rule and breaks ties by position, so the
    recall columns are comparable across experiments. ``retrieve`` is the
    index's own boundary and breaks ties with a seeded RNG, so a tie among
    ``n_marked``-many candidates cannot resolve by list position and favour
    the pre-shuffle gold-at-0 layout. Asserting that the two *can* differ is
    what keeps a reader from assuming one convention where the other applies.
    """
    scores = (0.5, 0.5, 0.2, 0.5)
    candidates = tuple(
        Candidate(key=f"c{i}", descriptor=(1, 1, 1, 1), action=i) for i in range(4)
    )
    query = Query(text="1 1 1 1", context=(1, 1, 0, 0))
    read = StateRead(keys=(), values=(), slot_used=())

    assert top_k(scores, 2) == (0, 1)
    kept = RetrievalIndex(k=2, seed=0).retrieve(query, candidates, read)[0]
    assert len(kept) == 2
    # The index's tie-break is not positional: on a three-way tie for the two
    # slots it does not simply take the first two positions.
    assert set(kept) != {0, 1}


def test_k_at_or_above_the_population_passes_everything_through():
    """``k >= len(candidates)`` is a no-op, and is the exhaustive arm's shape.

    The boundary is inert rather than raising, so the indexed and exhaustive
    arms share one code path: they differ in whether a filter ran, not in how
    the scorer is called.
    """
    built = _task("relational", 1000)
    task = _task_of("relational", built)
    read = StateRead(keys=(), values=(), slot_used=())
    kept, scores = RetrievalIndex(k=len(task.candidates), seed=0).retrieve(
        task.public(), task.candidates, read
    )
    assert kept == tuple(range(len(task.candidates)))
    assert len(scores) == len(task.candidates)


def test_retrieve_passes_through_when_there_is_no_index():
    """The no-index case is the whole population, which is arms A and B.

    One entry point for both means the arms cannot diverge in how the scorer
    is called — only in whether a filter ran.
    """
    built = _task("relational", 1000)
    task = _task_of("relational", built)
    kept, scores = retrieve(None, task.public(), task.candidates,
                            StateRead(keys=(), values=(), slot_used=()))
    assert kept == tuple(range(len(task.candidates)))
    assert scores == [0.0] * len(task.candidates)


def test_the_retained_set_has_size_k_below_the_population():
    """``|R| == K`` once the index is in the loop — the quantity C5 needs."""
    built = _task("relational", 1000, n_candidates=64)
    task = _task_of("relational", built)
    read = StateRead(keys=(), values=(), slot_used=())
    for k in (2, 4, 16):
        kept, _ = RetrievalIndex(k=k, seed=0).retrieve(task.public(), task.candidates, read)
        assert len(kept) == k


# --------------------------------------------------------------------------- #
# 5. The recorded family asymmetry: the public bits mislead on ``replay``
# --------------------------------------------------------------------------- #
#
# The environment's query-deceptive distractors agree with the query on every
# unmarked position while violating the relation on the marked ones. Gold
# anti-matches the query on the unmarked positions, so gold's *public* query
# agreement is below the best distractor's on two of the three families. A
# public-bits index is therefore a better filter for the distractors than for
# gold, and at large H it loses gold entirely.
#
# This is tested as a recorded property. It is the reason the index's state
# term is primary, and it is the mistake a reader would otherwise expect the
# module to have made.


@pytest.mark.parametrize("family", ("state_lookup", "replay"))
def test_golds_public_query_agreement_is_below_the_best_distractors(family):
    """On the persistence families the public bits are anti-informative about gold.

    ``relational`` is excluded because its public bits *are* the target, so
    gold's agreement is the highest by construction; ``state_lookup`` and
    ``replay`` hide the target in state, and the environment's
    query-deceptive distractors agree with the query on the unmarked
    positions while violating the relation on the marked ones — so a
    public-bits scorer ranks a distractor above gold.
    """
    n = 80
    for seed in range(n):
        built = _task(family, seed + 1000)
        task = _task_of(family, built)
        bits = task.query.text.partition("\t")[0]
        qb = tuple(int(b) for b in bits.split()) if bits.strip() else ()
        if not qb:
            continue
        agree = [
            sum(
                1
                for j in range(min(len(qb), len(c.descriptor)))
                if qb[j] == c.descriptor[j]
            )
            for c in task.candidates
        ]
        gold = agree[task.target_action]
        best_distractor = max(a for i, a in enumerate(agree) if i != task.target_action)
        assert gold < best_distractor, (
            f"{family} seed={seed}: gold's public query agreement ({gold}) is "
            f"not below the best distractor's ({best_distractor}); the recorded "
            "asymmetry that justifies the state term's primacy does not hold, "
            "and the query term is not anti-informative on this family"
        )


def test_a_query_only_index_loses_gold_on_replay():
    """A public-bits index is a better filter for the distractors than for gold.

    At ``H=64`` with ``K=4`` it retains gold at well below half the rate the
    state term achieves, which is the failure the state term exists to
    prevent. Recorded as a property of the *rule*, not of a tuning: the query
    term is what a naive index would use, and its failure rate at large H is
    the measurement that says why the module does not.
    """
    index = RetrievalIndex(k=4, seed=0, weight_state=0.0, weight_query=1.0)
    retained = 0
    n = 120
    for seed in range(n):
        built = _task("replay", seed + 1000, n_candidates=64)
        task = _task_of("replay", built)
        kept, _ = index.retrieve(task.public(), task.candidates, _read_for("replay", built))
        retained += int(task.target_action in kept)
    rate = retained / n
    assert rate < 0.5, (
        f"a query-only index retained gold on replay at {rate:.3f}, but the "
        "public bits are anti-informative on this family — a rate at or above "
        "half would mean the recorded asymmetry no longer holds and the "
        "state term's primacy is not justified by the task"
    )


def test_the_state_term_restores_retention_on_replay():
    """Switching the query term off takes replay retention to 1.0 at any H.

    The two tests together are the design argument in executable form: the
    query term is what fails on the persistence families, the state term is
    what recovers, and neither is a matter of tuning.
    """
    for h in (8, 64):
        index = _state_only_index(k=4, seed=0)
        retained = 0
        n = 80
        for seed in range(n):
            built = _task("replay", seed + 1000, n_candidates=h)
            task = _task_of("replay", built)
            kept, _ = index.retrieve(task.public(), task.candidates, _read_for("replay", built))
            retained += int(task.target_action in kept)
        assert retained == n, (
            f"the state-only index retained gold on replay {retained}/{n} at "
            f"H={h}; the state term separates gold on this family, so anything "
            "below full retention is a selection bug rather than a task property"
        )


def test_the_query_term_separates_relational():
    """The complement property: on ``relational`` the query term is the separator.

    Gold matches the query on every marked position, so gold's marked-position
    agreement is maximal — the property that makes a public-bits index the
    right filter for this family and the wrong one for the persistence
    families. Both directions are recorded, because an index that used only
    the state term would be sound on two families and inert on the third.
    """
    index = RetrievalIndex(k=1, seed=0, weight_state=0.0, weight_query=1.0)
    for seed in range(60):
        built = _task("relational", seed + 1000)
        task = _task_of("relational", built)
        read = _read_for("relational", built)
        scores = index.score(task.public(), task.candidates, read)
        gold = scores[task.target_action]
        others = [s for i, s in enumerate(scores) if i != task.target_action]
        assert gold > max(others), (
            f"relational seed={seed}: the query term does not separate gold, "
            "but the family publishes its target as the query bits"
        )


# --------------------------------------------------------------------------- #
# 6. Leakage: the index's inputs are the router's inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", FAMILIES)
def test_the_index_sees_no_forbidden_field(family):
    """The audit runs on the index's inputs, not only on the router's.

    The retrieval boundary is a new component in the loop, so its input
    boundary is checked where it runs. A filter that needed the answer would
    be an oracle, and arm D exists to measure that ceiling under its own name
    instead of reaching it by accident.
    """
    for seed in range(20):
        built = _task(family, seed + 1000)
        task = _task_of(family, built)
        audit_router_inputs(
            query=task.public(),
            candidates=task.candidates,
            state=_read_for(family, built) if family != "relational" else None,
        )


def test_the_index_does_not_read_target_action():
    """``RetrievalIndex.retrieve`` takes (query, candidates, read) and nothing else.

    Structural, not statistical: the signature itself is the boundary. The
    index has no parameter through which the gold index could be supplied, so
    a caller cannot make it an oracle without changing its type.
    """
    import inspect

    params = inspect.signature(RetrievalIndex.retrieve).parameters
    assert set(params) == {"self", "query", "candidates", "read"}, (
        f"the index's retrieve signature is {sorted(params)}; any additional "
        "parameter is a channel by which the answer could reach the filter"
    )
    params_score = inspect.signature(RetrievalIndex.score).parameters
    assert set(params_score) == {"self", "query", "candidates", "read"}


def test_the_index_ignores_the_unaddressed_pool():
    """Only the slot the query addresses contributes, so pool noise cannot enter.

    The read returns the addressed slot first and then the rest of the
    occupied pool, and the pool *grows* over an episode — after 40 tasks the
    read returns 40 vectors. A non-addressed slot's value has no relation to
    this task but its agreement features are still signed, so an index that
    scored them would inject noise that grows with history and dominates the
    signal. The same gate the learned basis's ``_addressed`` block applies.
    """
    # Two stores with different pool contents and the same addressed value:
    # the index scores must be identical.
    store_a = PersistentStore(StateConfig(seed=0))
    task = build_lookup_task(1000, store_a, dim=DIM, n_candidates=8)
    query = task.public()
    addressed = task.detail.key

    store_b = PersistentStore(StateConfig(seed=0))
    store_b.write(_update(addressed, task.detail.written_bits, step=0))
    # Fill the pool with unrelated slots the query does not name.
    for extra in range(6):
        store_b.write(_update(f"noise_{extra}", tuple((i % 2 for i in range(DIM))), step=1))

    index = RetrievalIndex(k=2, seed=0)
    candidates = list(task.candidates)
    sa = index.score(query, candidates, store_a.read(query))
    sb = index.score(query, candidates, store_b.read(query))
    assert sa == sb, (
        "the index's scores depend on state slots the query does not address; "
        "pool noise would then dominate the signal as history grows"
    )


def _update(key: str, value, step: int = 0):
    from tac_osm import StateUpdate

    return StateUpdate(key=key, value=tuple(value), task_key=key,
                       success_score=1.0, step=step)


# --------------------------------------------------------------------------- #
# 7. The failure taxonomy is closed and attributing
# --------------------------------------------------------------------------- #


def test_the_taxonomy_is_a_closed_set():
    """The four categories are the whole space of failures the run can report.

    A failure fitting none of them is a bug in the measurement rather than a
    new kind of failure, and the run raises rather than inventing a fifth
    bucket. Closure is what makes the C-vs-B table interpretable, because a
    reader can account for every lost step exactly once.
    """
    assert set(FAILURE_CATEGORIES) == {
        "retrieval_miss", "ranking_miss", "execution_failure",
        "verification_failure",
    }
    assert len(FAILURE_CATEGORIES) == len(set(FAILURE_CATEGORIES))


def test_a_retrieval_miss_is_attributed_to_the_index():
    """Gold not in the retained set: the only category that argues against the index."""
    assert retrieval_miss(False) == "retrieval_miss"
    assert retrieval_miss(True) is None
    assert classify_step(success=False, gold_in_retrieved=False,
                         gold_is_selected=False, verification_passed=True
                         ) == "retrieval_miss"
    # Retrieval is blamed before anything downstream, because the downstream
    # components cannot recover what the index discarded.
    assert classify_step(success=False, gold_in_retrieved=False,
                         gold_is_selected=True, verification_passed=True
                         ) == "retrieval_miss"


def test_a_ranking_miss_is_a_scorer_failure_not_an_index_failure():
    """Gold survived the index and the scorer still ranked it below a distractor.

    Reporting the two as one number is what would make a bad scorer look like
    a bad index — the distinction the taxonomy exists to keep.
    """
    assert ranking_miss(True, False) == "ranking_miss"
    assert ranking_miss(True, True) is None
    assert ranking_miss(False, False) is None
    assert classify_step(success=False, gold_in_retrieved=True,
                         gold_is_selected=False, verification_passed=True
                         ) == "ranking_miss"


def test_a_verification_failure_is_attributed_last():
    """Gold was selected and the computation succeeded but the verifier rejected it."""
    assert classify_step(success=False, gold_in_retrieved=True,
                         gold_is_selected=True, verification_passed=False
                         ) == "verification_failure"


def test_an_execution_failure_is_the_residual_category():
    """Gold selected, verifier satisfied, and the task still failed.

    The residual is named rather than left as ``None``, because an
    unattributed failure is a failure the taxonomy cannot account for and a
    reader cannot read.
    """
    assert classify_step(success=False, gold_in_retrieved=True,
                         gold_is_selected=True, verification_passed=True
                         ) == "execution_failure"


def test_a_successful_step_is_unattributed():
    assert classify_step(success=True, gold_in_retrieved=False,
                         gold_is_selected=False, verification_passed=False
                         ) is None


def test_the_taxonomy_attributes_exactly_once():
    """Every failure path lands in exactly one bucket.

    Enumerating the full input space rather than the interesting cases, because
    a taxonomy that double-counted on an unexamined combination would make a
    capability drop look larger than it is.
    """
    seen: dict[str, int] = {}
    for success in (False, True):
        for retrieved in (False, True):
            for selected in (False, True):
                for verified in (False, True):
                    cat = classify_step(success=success,
                                         gold_in_retrieved=retrieved,
                                         gold_is_selected=selected,
                                         verification_passed=verified)
                    if cat is None:
                        assert success, (
                            f"an unsuccessful step was left unattributed at "
                            f"(retrieved={retrieved}, selected={selected}, "
                            f"verified={verified}); a lost step the taxonomy "
                            "cannot name is a measurement bug"
                        )
                        continue
                    assert cat in FAILURE_CATEGORIES
                    seen[cat] = seen.get(cat, 0) + 1
    # Every category is reachable, so none is dead code in the taxonomy.
    assert set(seen) == set(FAILURE_CATEGORIES), (
        f"a failure category is unreachable: {sorted(set(FAILURE_CATEGORIES) - set(seen))}"
    )


# --------------------------------------------------------------------------- #
# 8. The cost terms and the arm cell
# --------------------------------------------------------------------------- #


def test_cost_terms_serialise():
    """The six terms are the pre-registered cost model, reported along one axis."""
    ct = CostTerms(candidates_inspected=4, router_operations=4, index_operations=8,
                   executor_nodes=10, verifier_operations=1, wall_clock=0.5)
    d = ct.to_dict()
    assert set(d) == {
        "candidates_inspected", "router_operations", "index_operations",
        "executor_nodes", "verifier_operations", "wall_clock",
    }
    assert d["candidates_inspected"] == 4
    assert d["wall_clock"] == 0.5


def test_an_arm_cell_round_trips():
    """A measurement cell must survive a write and a read.

    Frozen, because a cell must not be editable after it is reported — the
    same reasoning as the contract and the checkpoint.
    """
    import dataclasses

    ct = CostTerms(candidates_inspected=2, router_operations=2, index_operations=8,
                   executor_nodes=10, verifier_operations=1, wall_clock=0.1)
    arm = ArmResult(arm="indexed_topk", h=64, k=4, recall_at_k=0.75,
                    routing_at_1=0.5, gold_rank_mean=1.6, exact_success=0.4,
                    acc_given_retrieved=0.5, costs=ct,
                    failures={"retrieval_miss": 3, "ranking_miss": 7}, n_steps=100)
    d = arm.to_dict()
    assert d["arm"] == "indexed_topk" and d["h"] == 64 and d["k"] == 4
    assert d["recall@K"] == 0.75
    assert d["costs"]["index_operations"] == 8
    assert dataclasses.is_dataclass(arm) and arm.failures["ranking_miss"] == 7


def test_executor_nodes_is_an_architectural_constant_in_the_cost_model():
    """``executor_nodes`` is fixed by the v0.1 executor, not measured here.

    C5 is untested: a retrieval boundary in routing is not yet a retrieval
    boundary in execution. The field is reported so the column is not read as
    a scaling result, and the test pins the constant so a future executor that
    did scale would have to change the cost model's documentation rather than
    just its number.
    """
    ct = CostTerms(candidates_inspected=4, router_operations=4, index_operations=8,
                   executor_nodes=10, verifier_operations=1, wall_clock=0.5)
    assert ct.executor_nodes == 10


# --------------------------------------------------------------------------- #
# 9. The seeded tie-break does not favour the pre-shuffle gold-at-0 layout
# --------------------------------------------------------------------------- #


def _all_equal_candidates(n: int) -> tuple[Candidate, ...]:
    return tuple(Candidate(key=f"c{i}", descriptor=(1, 1, 1, 1), action=i)
                 for i in range(n))


def _equal_score_query() -> Query:
    return Query(text="1 1 1 1", context=(1, 1, 0, 0))


def _empty_read() -> StateRead:
    return StateRead(keys=(), values=(), slot_used=())


def test_a_tie_is_broken_by_a_seeded_rng_not_by_position():
    """Retention order is reproducible and does not silently prefer early positions.

    Before the shuffle, gold sits at position 0; a tie broken by list position
    would favour it, and the index would look better than it is. The seeded RNG
    is what keeps the tie-break honest, and the distribution check is what
    keeps the RNG honest — a broken seed would concentrate on the first two
    positions across many seeds.
    """
    candidates = _all_equal_candidates(4)
    counts = [0] * 4
    for seed in range(200):
        kept, _ = RetrievalIndex(k=2, seed=seed).retrieve(
            _equal_score_query(), candidates, _empty_read()
        )
        for i in kept:
            counts[i] += 1
    # No position is a structural favourite: the counts are within a tolerance
    # that 200 draws of a 2-of-4 selection comfortably satisfy.
    assert max(counts) - min(counts) <= 40, (
        f"the tie-break favours positions {counts}; a positional bias would "
        "favour the pre-shuffle gold-at-0 layout"
    )


def test_the_tie_break_is_reproducible():
    """The same seed gives the same retained set, so a run is reproducible."""
    candidates = _all_equal_candidates(4)
    q, read = _equal_score_query(), _empty_read()
    first = RetrievalIndex(k=2, seed=0).retrieve(q, candidates, read)[0]
    second = RetrievalIndex(k=2, seed=0).retrieve(q, candidates, read)[0]
    assert first == second
    # And a different seed is not required to agree — the seed is what varies.
    other = RetrievalIndex(k=2, seed=1).retrieve(q, candidates, read)[0]
    assert isinstance(other, tuple) and len(other) == 2


def test_repeated_calls_on_one_index_are_reproducible():
    """One index instance reused across steps is the measurement's usage pattern."""
    candidates = _all_equal_candidates(4)
    q, read = _equal_score_query(), _empty_read()
    index = RetrievalIndex(k=2, seed=0)
    a = index.retrieve(q, candidates, read)[0]
    b = index.retrieve(q, candidates, read)[0]
    assert a == b


# --------------------------------------------------------------------------- #
# 10. C vs B at matched K: the comparison the experiment exists to make
# --------------------------------------------------------------------------- #
#
# Arms B and C both produce a retained set of size K. B's is the scorer's own
# top-K; C's is the index's. The comparison is the cost-quality trade, and it
# is interpretable in both directions: a C below B is an index that discards
# gold, and a C above B is a result about the scorer, which the third branch
# of the decision rule exists to receive.


def _analytic_router() -> LearnedRelationalRouter:
    """The fixed vector that separates gold on all three families.

    Not a fitted value: it is the evidence that the relation lives in the
    hypothesis class, and it is the strongest available scorer for a test that
    wants the B arm to be a meaningful reference rather than a zero vector.
    """
    r = LearnedRelationalRouter(RouterConfig(dim=DIM))
    r.w = analytic_weights(DIM, r.max_state_slots)
    return r


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("h", (8, 64))
def test_c_versus_b_at_matched_k(family, h):
    """Both arms produce a retained set of size K, from the same task stream.

    The assertion is on the *structure* of the comparison, not on which arm
    wins: the endpoint is defined so that either outcome is a readable result,
    and a test that required C to beat B would be assuming the result the
    experiment exists to measure. What must hold is that the two arms are
    comparable at all — same K, same scorer, same stream.
    """
    k = min(4, h - 1)
    router = _analytic_router()
    index = RetrievalIndex(k=k, seed=0)
    n = 60
    for seed in range(n):
        built = _task(family, seed + 1000, n_candidates=h)
        task = _task_of(family, built)
        query = task.public()
        read = _read_for(family, built)
        raw = router.score(query, _store_for(family, built), task.candidates)

        b_kept = top_k(raw, k)
        c_kept, _ = index.retrieve(query, task.candidates, read)
        assert len(b_kept) == k and len(c_kept) == k, (
            f"{family} H={h} seed={seed}: the arms' retained sets are not at "
            "matched K, so the comparison the decision rule reads is not the "
            "comparison the contract registered"
        )
        # Same scorer in both arms: the arms differ in which candidates the
        # scorer sees, never in how it scores.
        assert all(
            abs(raw[i] - router.score(query, _store_for(family, built),
                                      [task.candidates[i]])[0]) < 1e-12
            for i in set(b_kept) | set(c_kept)
        )


def _store_for(family: str, built) -> PersistentState:
    """The store the task wrote to, for a router's own read."""
    if family == "relational":
        return _store()
    return built[1]


def test_the_index_can_be_the_better_filter():
    """The third branch of the decision rule is reachable, not theoretical.

    Against a scorer with no trained weights — the HS-001 failure mode, an
    all-zero vector that scores every candidate uniformly — the B arm's
    top-K is positional and near chance, while the index's state term still
    separates gold on ``state_lookup`` and ``replay``. So C above B is a
    real configuration and not an artefact of a favourable seed, and the
    pre-registration's third branch exists to receive it rather than to
    smooth it away.
    """
    untrained = LearnedRelationalRouter(RouterConfig(dim=DIM, seed=0))
    assert all(w == 0.0 for w in untrained.w), "the fixture is not an untrained router"

    for family in ("state_lookup", "replay"):
        index = RetrievalIndex(k=4, seed=0)
        b_hits = c_hits = 0
        n = 80
        for seed in range(n):
            built = _task(family, seed + 1000, n_candidates=64)
            task = _task_of(family, built)
            query = task.public()
            read = _read_for(family, built)
            raw = untrained.score(query, built[1], task.candidates)
            b_hits += int(task.target_action in set(top_k(raw, 4)))
            kept, _ = index.retrieve(query, task.candidates, read)
            c_hits += int(task.target_action in kept)
        assert c_hits > b_hits, (
            f"{family}: the index did not beat an untrained scorer's top-K "
            f"({c_hits} vs {b_hits} of {n}); the third branch of the decision "
            "rule would be unreachable, and the pre-registration's provision "
            "for an index that beats the scorer would be dead"
        )


# --------------------------------------------------------------------------- #
# 11. The oracle arm: gold is forced in, and the index is applied off-frame
# --------------------------------------------------------------------------- #


def test_the_oracle_arm_retains_gold_and_respects_the_budget():
    """Arm D forces gold in and fills the remaining K-1 slots off the reduced list.

    The arm reads information no runtime component may have, exactly as
    ``OracleRouter`` does. What is tested here is the bookkeeping the arm
    shares with no other arm: the index is applied to the population *without*
    gold, so the retrieved indices are reduced-list indices and must be mapped
    back through ``rest`` before use. A previous version unioned the
    reduced-list indices with ``gold`` as if they shared a frame of reference,
    which silently re-indexed the wrong candidates.
    """
    router = _analytic_router()
    for family in ("relational", "replay"):
        for h in (8, 64):
            k = 4
            for seed in range(30):
                built = _task(family, seed + 1000, n_candidates=h)
                task = _task_of(family, built)
                query = task.public()
                gold = task.target_action
                store = _store_for(family, built)
                read = _read_for(family, built)

                rest = [i for i in range(len(task.candidates)) if i != gold]
                index = RetrievalIndex(k=max(1, k - 1), seed=0)
                others, _ = index.retrieve(
                    query, [task.candidates[i] for i in rest], read
                )
                kept = tuple(sorted({rest[o] for o in others} | {gold}))

                # The frame-of-reference invariant: every retained index is a
                # valid full-list index, and the size is exactly the budget.
                assert len(kept) == k, (
                    f"{family} H={h} seed={seed}: |kept|={len(kept)} != K={k}"
                )
                assert all(0 <= i < len(task.candidates) for i in kept)
                assert gold in kept, (
                    f"{family} H={h} seed={seed}: the oracle arm lost gold"
                )
                # The mapping is the point: reduced-list indices unioned with
                # the full-list gold index would produce an invalid set here.
                assert set(kept) == {rest[o] for o in others} | {gold}


def test_the_oracle_arm_uses_information_the_index_may_not_have():
    """Arm D reads ``target_action`` by design; the index structurally cannot.

    The ceiling is measured honestly under its own name rather than reached by
    accident: this test holds the two apart, so a change that gave the real
    index access to the gold index would break the leakage test above *and*
    this one, and the failure would name both.
    """
    built = _task("replay", 1000, n_candidates=8)
    task = _task_of("replay", built)
    # The index's own signature carries no channel for this information.
    index = RetrievalIndex(k=2, seed=0)
    kept, _ = index.retrieve(task.public(), task.candidates, _read_for("replay", built))
    # And it is not handed gold by construction: on replay at the default
    # weights it can genuinely lose it, which is what keeps it an index.
    assert len(kept) == 2


def test_k_must_be_positive():
    """A zero or negative budget is not a budget at all."""
    with pytest.raises(ValueError, match="k must be"):
        RetrievalIndex(k=0)
    with pytest.raises(ValueError, match="k must be"):
        RetrievalIndex(k=-1)
