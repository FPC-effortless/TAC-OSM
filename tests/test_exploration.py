"""Tests for the F2 exploration surface: ``ExplorationSchedule``, ``evaluate``,
and ``arm_exploration``.

TACOSM-LEARN-001 pre-registers two named training-time interventions and then
fixes the endpoint at ``epsilon = 0``, ``temperature = 0.5``. The scientific
claim of the experiment is that the interventions change *which parameters are
learned* and that the endpoint measures those parameters rather than the
exploratory policy. That claim rests on three properties this file pins:

1. The registered values are the only reachable ones. ``arm_exploration`` is a
   lookup by name, so there is no code path to a schedule with caller-chosen
   constants — an "exploration" result from a tuned knob would not be a result
   about the registered intervention.
2. The schedules anneal as the pre-registration describes, and stop at the
   endpoints it names: epsilon to 0, tau to the evaluation temperature.
3. ``evaluate`` never consults the exploration schedule. This is the load-
   bearing one: it is how ``epsilon = 0`` at the endpoint is a structural
   property rather than a convention a caller could forget.

A measurement that breaks any of these fails here rather than in a results
table, where it would read as a finding about routing.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, PersistentState, Query  # noqa: E402
from tac_osm.router import (  # noqa: E402
    EPS_0,
    SCHEDULE_LENGTH,
    TAU_0,
    TAU_END,
    ExplorationSchedule,
    LearnedRelationalRouter,
    arm_exploration,
)

# The three registered arms. ``baseline`` is None rather than a schedule with
# both flags off, because the baseline is MATCHED-001's protocol verbatim and
# ``route`` distinguishes the two paths by ``exploration is None``.
BASELINE = "baseline"
EPSILON_GREEDY = "epsilon_greedy"
TEMPERATURE = "temperature"


# -- the registered values --------------------------------------------- #

def test_registered_constants_are_the_pre_registered_numbers():
    """The constants are the ones named in docs/TACOSM-LEARN-001.md."""
    assert EPS_0 == 0.30
    assert TAU_0 == 2.0
    assert TAU_END == 0.5
    assert SCHEDULE_LENGTH == 500


def test_arm_exploration_returns_the_three_registered_arms():
    assert arm_exploration(BASELINE) is None
    assert arm_exploration(EPSILON_GREEDY) is not None
    assert arm_exploration(TEMPERATURE) is not None


def test_arm_exploration_baseline_is_none_not_a_dead_schedule():
    """``baseline`` is None so ``route`` takes its pre-F2 path verbatim.

    A schedule with both flags off would produce the same behaviour, but it
    would make the baseline a *thing the F2 code constructed* rather than the
    MATCHED-001 protocol. The reproduction gate depends on the distinction.
    """
    assert arm_exploration("baseline") is None


def test_arm_exploration_rejects_unknown_names():
    """A typo must raise, not fall back to the baseline.

    Silently training the baseline and reporting it as an exploration arm is
    the warm-start confusion the pre-registration's Layer 1 section exists to
    prevent: the column would look like an intervention that did nothing.
    """
    with pytest.raises(KeyError):
        arm_exploration("epsilon-greedy")  # hyphen, not underscore
    with pytest.raises(KeyError):
        arm_exploration("EPSILON_GREEDY")
    with pytest.raises(KeyError):
        arm_exploration("")


def test_arm_exploration_does_not_expose_values_to_a_caller():
    """The returned object is one of the three registered schedules.

    Constructing a schedule is possible, but the measurement path goes through
    this lookup, which takes a name only. That is what makes the registered
    values the reachable ones.
    """
    eps = arm_exploration(EPSILON_GREEDY)
    assert eps is not None and eps.epsilon_greedy and not eps.temperature
    assert eps.eps_0 == EPS_0
    assert eps.n_steps == SCHEDULE_LENGTH

    tau = arm_exploration(TEMPERATURE)
    assert tau is not None and tau.temperature and not tau.epsilon_greedy
    assert tau.tau_0 == TAU_0
    assert tau.tau_end == TAU_END


def test_exploration_schedule_defaults_are_the_registered_values():
    """The dataclass defaults *are* the registered values, not a restatement."""
    eps = ExplorationSchedule(epsilon_greedy=True)
    assert eps.eps_0 == EPS_0
    assert eps.tau_0 == TAU_0
    assert eps.tau_end == TAU_END
    assert eps.n_steps == SCHEDULE_LENGTH


# -- the anneals -------------------------------------------------------- #

def test_epsilon_anneals_linearly_to_zero():
    eps = ExplorationSchedule(epsilon_greedy=True)
    assert eps.epsilon(0) == pytest.approx(EPS_0)
    assert eps.epsilon(SCHEDULE_LENGTH // 2) == pytest.approx(EPS_0 / 2)
    assert eps.epsilon(SCHEDULE_LENGTH) == pytest.approx(0.0)


def test_epsilon_stays_at_zero_past_the_schedule_end():
    """A run longer than the registered length must not keep exploring."""
    eps = ExplorationSchedule(epsilon_greedy=True)
    assert eps.epsilon(SCHEDULE_LENGTH * 4) == 0.0


def test_epsilon_is_zero_when_the_flag_is_off():
    assert ExplorationSchedule(epsilon_greedy=False).epsilon(0) == 0.0


def test_tau_anneals_linearly_to_the_evaluation_temperature():
    tau = ExplorationSchedule(temperature=True)
    assert tau.tau(0) == pytest.approx(TAU_0)
    mid = SCHEDULE_LENGTH // 2
    assert tau.tau(mid) == pytest.approx((TAU_0 + TAU_END) / 2)
    assert tau.tau(SCHEDULE_LENGTH) == pytest.approx(TAU_END)


def test_tau_clamps_at_tau_end_past_the_schedule_end():
    tau = ExplorationSchedule(temperature=True)
    assert tau.tau(SCHEDULE_LENGTH * 4) == TAU_END


def test_tau_is_the_evaluation_temperature_when_the_flag_is_off():
    """A schedule that only sets epsilon must not change the temperature."""
    assert ExplorationSchedule(epsilon_greedy=True).tau(0) == TAU_END


def test_schedule_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        ExplorationSchedule(eps_0=-0.1)
    with pytest.raises(ValueError):
        ExplorationSchedule(eps_0=1.1)
    with pytest.raises(ValueError):
        ExplorationSchedule(tau_end=0.0)
    # tau_0 below tau_end would anneal *up*, the opposite of "cool".
    with pytest.raises(ValueError):
        ExplorationSchedule(tau_0=0.2, tau_end=0.5)
    with pytest.raises(ValueError):
        ExplorationSchedule(n_steps=0)


# -- the endpoint is structurally exploration-free ---------------------- #

def test_fresh_router_has_no_exploration_schedule():
    """A router straight from the constructor has ``exploration`` unset, which
    is how an evaluation cell reaches ``epsilon = 0`` without being told to:
    the arm's schedule lived on the training-time router, not this one."""
    r = LearnedRelationalRouter()
    assert r.exploration is None


def _discriminating_index(store, candidates):
    """A feature index on which the candidates do not all score identically.

    The basis is large (``1 + 4*dim + 2*dim*max_slots`` = 97 by default) and
    its layout is an implementation detail, so a test cannot name an index or
    which candidate it favours. This scans for one that merely separates the
    field, which is what makes the ``evaluate`` assertions about the argmax
    rather than about a particular winner.
    """
    probe = LearnedRelationalRouter()
    for i in range(probe.n):
        probe.w = [0.0] * probe.n
        probe.w[i] = 1.0
        s = probe.score(_query(), store, candidates)
        if len(set(s)) > 1:
            return i
    raise AssertionError(  # pragma: no cover
        "no feature separates the candidates; the basis has 4*dim agreement "
        "blocks, so one should")


def _query():
    # ``_bits`` parses the text as space-separated integers, and ``context`` is
    # a tuple of ints read positionally: the basis is over query/descriptor
    # agreement, so both have to be bits.
    return Query(text="1 0 1 0 0 0 0 0", context=(1, 0, 0, 1, 0, 0, 0, 0),
                 step=0, provenance="t")


def test_evaluate_ignores_an_attached_exploration_schedule():
    """The load-bearing property of the experiment.

    ``route`` on the scheduled router would explore — epsilon is 0.30 at step
    0, so a call to ``route`` could return any of the four candidates.
    ``evaluate`` must return the same answer on both routers, because the
    endpoint is a property of the weights, not of the policy that sampled
    during training.
    """
    store = _fresh_state()
    candidates = [
        Candidate(key=f"c{i}", descriptor=[float(i), 1.0], action=i,
                  provenance="t")
        for i in range(4)
    ]
    idx = _discriminating_index(store, candidates)

    plain = LearnedRelationalRouter()
    plain.w = [0.0] * plain.n
    plain.w[idx] = 1.0
    scores = plain.score(_query(), store, candidates)
    assert len(set(scores)) > 1, "the feature must separate the candidates"

    exploring = LearnedRelationalRouter()
    exploring.w = list(plain.w)
    exploring.exploration = arm_exploration(EPSILON_GREEDY)
    a = plain.evaluate(_query(), store, candidates)
    b = exploring.evaluate(_query(), store, candidates)
    assert a == b
    # And it is the argmax the weights imply, not a draw from the policy.
    assert a[0] == scores.index(max(scores))


def test_evaluate_returns_the_raw_dot_products_not_the_softmax():
    """The endpoint is in score units, so margins are comparable across H.

    Softmax probabilities are normalised over the candidate count, so a margin
    in probability units shrinks with H even for a perfect scorer; the raw
    dot products do not, which is why the endpoint reads them.
    """
    r = LearnedRelationalRouter()
    store = _fresh_state()
    candidates = [
        Candidate(key=f"c{i}", descriptor=[float(i), 1.0], action=i,
                  provenance="t")
        for i in range(3)
    ]
    idx = _discriminating_index(store, candidates)
    r.w = [0.0] * r.n
    r.w[idx] = 1.0
    scores = r.score(_query(), store, candidates)
    selected, raw = r.evaluate(_query(), store, candidates)
    assert selected == scores.index(max(scores))
    # Raw scores, not probabilities: they do not sum to 1 and are not divided
    # by the candidate count.
    assert raw == pytest.approx(scores)
    assert sum(raw) != pytest.approx(1.0)


def test_evaluate_picks_the_argmax_of_the_raw_scores():
    r = LearnedRelationalRouter()
    store = _fresh_state()
    candidates = [
        Candidate(key=f"c{i}", descriptor=[float(i), 1.0], action=i,
                  provenance="t")
        for i in range(4)
    ]
    idx = _discriminating_index(store, candidates)
    r.w = [0.0] * r.n
    r.w[idx] = 1.0
    scores = r.score(_query(), store, candidates)
    selected, raw = r.evaluate(_query(), store, candidates)
    assert selected == scores.index(max(scores))
    assert max(raw) == max(scores)
    assert len(raw) == len(candidates)


def _fresh_state() -> PersistentState:
    from tac_osm.state import PersistentStore, StateConfig
    return PersistentStore(StateConfig(seed=0, n_slots=64))
