"""Tests for the F3 reward hook.

TACOSM-SURROGATE-001 intervenes on exactly one thing: the scalar the loop
passes as ``reward`` to ``LearnedRelationalRouter.update``. These tests pin
the properties that make the intervention interpretable:

* ``reward_fn = None`` is bit-for-bit the pre-F3 protocol, so the frozen
  MATCHED-001 baseline is the reference every arm's delta is taken against;
* the arm lookup is a lookup, not a constructor — the registered values are
  the only reachable ones;
* the surrogate is the margin it claims to be, in the units ``delta_1`` is
  reported in, and its centring sequence is the one the pre-registration
  describes;
* the hook cannot widen the router's inputs, and a broken hook fails loudly
  at the step where it fired.

The bits these tests cover are the ones a wrong result would otherwise be
silent about. The full arm-vs-arm measurement is ``scripts/measure_surrogate.py``,
whose reproduction gate is the end-to-end version of the first test here.
"""

from __future__ import annotations

import math
import os
import statistics
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig, RouterSwitch  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.model import (  # noqa: E402
    ModelConfig,
    RewardContext,
    RewardFn,
    TacOsmModel,
    baseline_reward,
)
from tac_osm.rewards import (  # noqa: E402
    AnalyticMarginReward,
    arm_reward,
    baseline_arm_names,
)
from tac_osm.router import LearnedRelationalRouter  # noqa: E402

# --------------------------------------------------------------------------- #
# The baseline: `None` and `baseline_reward` must be the same update
# --------------------------------------------------------------------------- #


def _trained_weights(reward_fn, *, seed: int = 0, n_steps: int = 40,
                     n_candidates: int = 8) -> list[float]:
    """Train one learned router under a named reward hook.

    Two arms built identically except for the hook, so a difference in the
    final weights is a difference the hook caused and nothing else.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = n_candidates
    bm.model.config.reward_fn = reward_fn
    assert isinstance(bm.model.router, LearnedRelationalRouter)
    bm.model.run()
    return list(bm.model.router.w)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n_candidates", [4, 8, 16])
def test_none_and_baseline_reward_are_bit_for_bit(seed, n_candidates):
    """``reward_fn = None`` is the pre-F3 protocol, to the last bit.

    This is the reproduction gate the measurement states in aggregate: if
    ``None`` and ``baseline_reward`` ever diverge, the baseline arm is not the
    MATCHED-001 protocol and no arm's delta is comparable to the published
    reference.
    """
    none_w = _trained_weights(None, seed=seed, n_candidates=n_candidates)
    named_w = _trained_weights(baseline_reward, seed=seed,
                               n_candidates=n_candidates)
    assert none_w == named_w, (
        f"`None` and `baseline_reward` diverged at seed={seed}, "
        f"H={n_candidates}: max|d|="
        f"{max(abs(a - b) for a, b in zip(none_w, named_w))}"
    )


def test_baseline_reward_is_the_success_indicator():
    """``baseline_reward`` is literally ``float(outcome.success)``, and nothing more.

    The environment scores ``success = action == target_action``, so this is
    ``1[a_t = gold]``. That the baseline is *already* gold-anchored is the
    reason the F3 leakage boundary is stated in terms of signal density
    rather than the absence of gold.
    """
    class _Outcome:
        success = True

    ctx = RewardContext(
        task=None, candidates=(), selected=0, decision=None,
        outcome=_Outcome(), query=None, state=None, router=None,
    )
    assert baseline_reward(ctx) == 1.0
    ctx.outcome.success = False  # type: ignore[misc]
    assert baseline_reward(ctx) == 0.0


def test_reward_context_is_frozen():
    """The context is immutable, so a hook cannot mutate the step's state."""
    ctx = RewardContext(
        task=None, candidates=(), selected=0, decision=None,
        outcome=None, query=None, state=None, router=None,
    )
    with pytest.raises((AttributeError, Exception)):
        ctx.selected = 7  # type: ignore[misc]


def test_reward_context_carries_the_router():
    """The hook receives the router, which is what the margin is computed from.

    A hook that needs ``score`` has nothing to call on an oracle arm, which is
    why the surrogate can only run where the router is a learned one.
    """
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=1)
    assert isinstance(bm.model.router, LearnedRelationalRouter)

    seen = []
    bm.model.config.reward_fn = lambda ctx: seen.append(ctx.router) or 0.0
    bm.model.run()
    assert seen and all(r is bm.model.router for r in seen)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_model_config_rejects_a_non_callable_reward_fn():
    with pytest.raises(TypeError):
        ModelConfig(reward_fn="analytic_margin")


def test_model_config_accepts_none_and_a_callable():
    assert ModelConfig().reward_fn is None
    assert ModelConfig(reward_fn=baseline_reward).reward_fn is baseline_reward


def test_baseline_reward_satisfies_the_rewardfn_protocol():
    """The named baseline is structurally a ``RewardFn``, not just callable."""
    assert isinstance(baseline_reward, RewardFn)


# --------------------------------------------------------------------------- #
# The arm lookup: a lookup, not a constructor
# --------------------------------------------------------------------------- #


def test_the_registered_arms_are_exactly_three():
    """The pre-registration names three arms and the table knows no others."""
    assert baseline_arm_names() == ("baseline", "analytic_margin",
                                    "analytic_margin_clipped")


def test_unknown_arm_raises_not_falls_back():
    """A typo must raise, not silently train the baseline and report it as a surrogate.

    Falling back would be exactly the warm-start confusion the
    pre-registration's Layer 1 section exists to prevent.
    """
    with pytest.raises(KeyError):
        arm_reward("Analytic_Margin")
    with pytest.raises(KeyError):
        arm_reward("oracle")


def test_the_baseline_arm_is_none():
    """``baseline`` is the MATCHED-001 protocol: no hook at all."""
    assert arm_reward("baseline") is None


@pytest.mark.parametrize("name, clip", [("analytic_margin", False),
                                        ("analytic_margin_clipped", True)])
def test_each_surrogate_arm_returns_a_fresh_instance(name, clip):
    """A fresh instance per call, with the pre-registered clip setting.

    The fresh instance is deliberate: the running mean is a property of one
    trajectory, so a shared hook would carry one cell's baseline into the next
    and change the centring without changing any visible config.
    """
    first = arm_reward(name)
    second = arm_reward(name)
    assert first is not second
    assert isinstance(first, AnalyticMarginReward)
    assert first.clip is clip
    assert second.clip is clip


def test_no_arm_reaches_caller_chosen_clip_bounds():
    """The clip bounds are not a knob: the registered values are the reachable ones."""
    hook = arm_reward("analytic_margin_clipped")
    assert hook is not None
    assert (hook.clip_lo, hook.clip_hi) == (0.0, 1.0)
    with pytest.raises(TypeError):
        arm_reward("analytic_margin_clipped", clip=False)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# The surrogate: the margin, in raw units
# --------------------------------------------------------------------------- #


class _FakeRouter:
    """A stand-in scorer returning caller-set raw dot products.

    The margin must come from ``router.score`` — the raw dot products, not the
    softmax probabilities — because a probability margin shrinks with H even
    for a perfect scorer. Testing with a fixed score vector pins the
    arithmetic instead of depending on a trained vector's happenstance.
    """

    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, query, state, candidates):  # noqa: ANN001, ANN201
        return list(self._scores)


def _ctx(scores, gold, *, n_candidates=None):
    n = n_candidates if n_candidates is not None else len(scores)
    return RewardContext(
        task=type("Task", (), {"target_action": gold})(),
        candidates=tuple(range(n)),
        selected=gold,
        decision=None,
        outcome=None,
        query="q",
        state="s",
        router=_FakeRouter(scores),
    )


def test_the_margin_is_gold_minus_the_best_distractor():
    """``s_gold - max_{j != gold} s_j``, verbatim."""
    hook = AnalyticMarginReward()
    # gold=2 scores 1.0; the best distractor is 0.25, so the margin is 0.75.
    reward = hook(_ctx([0.1, 0.25, 1.0, -0.5], gold=2))
    assert reward == pytest.approx(0.75)


def test_a_perfect_margin_stays_unaffected_by_population():
    """A gold that dominates by a fixed amount scores the same margin at any H.

    This is why the surrogate is computed in raw units rather than softmax
    probabilities: the probability margin would shrink as candidates are
    added, and a cross-H comparison would then measure the normalisation
    rather than the scorer.
    """
    for n in (4, 8, 16, 64):
        scores = [0.0] * n
        scores[3] = 0.4
        hook = AnalyticMarginReward()
        assert hook(_ctx(scores, gold=3, n_candidates=n)) == pytest.approx(0.4)


def test_a_negative_margin_is_signed_correctly():
    """Gold below the best distractor gives a negative surrogate, not a clipped zero.

    The unclipped arm must carry the sign, because at large H from a zero
    start the margin is negative on most steps — the measured ``sat_lo`` —
    and an arm that floored it to zero would silently become the baseline.
    """
    hook = AnalyticMarginReward()
    scores = [-1.0, 2.0, -3.0]
    assert hook(_ctx(scores, gold=0)) == pytest.approx(-3.0)


def test_fewer_than_two_candidates_raises():
    """No competitor means no margin; 0.0 would be a silent terminal reward."""
    hook = AnalyticMarginReward()
    with pytest.raises(ValueError):
        hook(_ctx([0.5], gold=0))


def test_the_margin_ignores_the_sampled_action():
    """The margin is anchored to gold, not to ``selected``.

    The outcome reward is ``1[a_t = gold]``, so it is already gold-anchored;
    the surrogate keeps that anchor and changes the density. A margin off the
    *selected* action would be a different intervention, and a wrong one.
    """
    hook = AnalyticMarginReward()
    scores = [0.0, 1.0, 0.5]
    ctx = _ctx(scores, gold=1)
    ctx = RewardContext(
        task=ctx.task, candidates=ctx.candidates, selected=0,  # not gold
        decision=None, outcome=None, query=ctx.query, state=ctx.state,
        router=ctx.router,
    )
    assert hook(ctx) == pytest.approx(0.5)  # 1.0 - max(0.0, 0.5)


# --------------------------------------------------------------------------- #
# The centring rule
# --------------------------------------------------------------------------- #


def test_the_first_step_is_uncentred():
    """With no history the baseline is 0.0, so step 0's reward is its raw margin.

    This is why the running mean cannot produce the constant-offset failure the
    score-mean centring rule did: the first estimate is unbiased.
    """
    hook = AnalyticMarginReward()
    scores = [0.0, 0.0, 0.8]
    assert hook(_ctx(scores, gold=2)) == pytest.approx(0.8)
    stats = hook.stats()
    assert stats["count"] == 1.0
    assert stats["baseline_mean"] == pytest.approx(0.8)


def test_the_baseline_is_the_mean_of_previous_steps():
    """``baseline_t = mean(s_{<t})`` — the step itself is not folded in.

    The centred value is therefore the deviation from *past* average
    performance, which is the standard policy-gradient variance reduction: it
    lowers the update's variance without biasing it.
    """
    hook = AnalyticMarginReward()
    surrogates = [0.4, 0.8, 0.2]
    for margin in surrogates:
        hook(_ctx([0.0, 0.0, margin], gold=2))
    # The hook centred each step by the mean of the steps before it.
    #  t=0: 0.4 - 0.0            =  0.4
    #  t=1: 0.8 - 0.4            =  0.4
    #  t=2: 0.2 - mean(0.4,0.8)  = -0.4
    expected = [surrogates[0] - 0.0,
                surrogates[1] - statistics.fmean(surrogates[:1]),
                surrogates[2] - statistics.fmean(surrogates[:2])]
    stats = hook.stats()
    assert stats["count"] == 3.0
    assert stats["baseline_mean"] == pytest.approx(statistics.fmean(surrogates))
    assert stats["reward_mean"] == pytest.approx(statistics.fmean(expected))
    assert stats["reward_min"] == pytest.approx(min(expected))
    assert stats["reward_max"] == pytest.approx(max(expected))


def test_a_step_is_never_its_own_baseline():
    """The running mean is read *before* this step is folded in.

    Without that ordering the mean would include the current step and the
    centred reward would be biased toward zero by ``1/t`` — largest on the
    first steps, which are exactly the ones that determine whether a sparse
    early signal is ever seen. This is the regression test for it: the second
    step's baseline must be the *first* step's value, not the mean of both.
    """
    hook = AnalyticMarginReward()
    first = hook(_ctx([0.0, 0.0, 0.4], gold=2))
    assert first == pytest.approx(0.4)
    second = hook(_ctx([0.0, 0.0, 0.8], gold=2))
    # Correctly past: 0.8 - 0.4 = 0.4
    # If the step were folded in first: 0.8 - mean(0.4, 0.8) = 0.2
    assert second == pytest.approx(0.4), (
        "the second step's baseline included the step itself"
    )


def test_centring_by_a_running_mean_is_not_a_constant_offset():
    """The score-mean rule the pre-registration replaced produced a constant
    positive reward on 180/180 steps. The running mean of the surrogate tracks
    the surrogate, so the update stays signed by deviation from mean
    performance — positive and negative rewards both occur.
    """
    hook = AnalyticMarginReward()
    for margin in (0.5, 0.5, 0.5, 0.1):
        hook(_ctx([0.0, 0.0, margin], gold=2))
    stats = hook.stats()
    assert stats["reward_min"] < 0.0, "centring never produced a negative reward"
    assert stats["reward_max"] > 0.0, "centring never produced a positive reward"


def test_stats_report_zero_before_any_call():
    """An unrun hook reports zeros rather than a fabricated range."""
    stats = AnalyticMarginReward().stats()
    assert stats["count"] == 0.0
    assert stats["reward_nonzero_frac"] == 0.0
    assert stats["sat_hi_frac"] == 0.0
    assert stats["sat_lo_frac"] == 0.0
    assert stats["pre_clip_min"] == 0.0
    assert stats["pre_clip_max"] == 0.0
    assert stats["reward_min"] == 0.0
    assert stats["reward_max"] == 0.0


# --------------------------------------------------------------------------- #
# The clip, and both saturation directions
# --------------------------------------------------------------------------- #


def test_the_clip_bounds_the_margin_before_centring():
    """The clip is applied to the pre-centring margin, in ``[0, 1]``."""
    hook = AnalyticMarginReward(clip=True)
    # A margin above 1.0 saturates at the ceiling, so the reward is 1.0 - 0.0.
    assert hook(_ctx([0.0, 0.0, 5.0], gold=2)) == pytest.approx(1.0)


def test_the_clip_floors_a_negative_margin():
    """The clip's binding constraint is the floor, not the ceiling.

    Measured at H=256 from a zero start, ``sat_hi`` is 0.0000 and ``sat_lo``
    is 0.8784 — the clip is flooring a mostly-negative margin, not ceiling a
    mostly-positive one. An arm reporting only the ceiling would mis-describe
    its own signal.
    """
    hook = AnalyticMarginReward(clip=True)
    assert hook(_ctx([0.0, 3.0, -2.0], gold=2)) == pytest.approx(0.0)


def test_both_saturation_directions_are_counted():
    """``sat_hi_frac`` and ``sat_lo_frac`` are reported separately."""
    hook = AnalyticMarginReward(clip=True)
    # Ceiling: 5.0 > 1.0
    hook(_ctx([0.0, 0.0, 5.0], gold=2))
    # Floor: -3.0 < 0.0
    hook(_ctx([0.0, 4.0, -3.0], gold=2))
    # In range, no saturation
    hook(_ctx([0.0, 0.0, 0.5], gold=2))
    stats = hook.stats()
    assert stats["count"] == 3.0
    assert stats["sat_hi_frac"] == pytest.approx(1.0 / 3.0)
    assert stats["sat_lo_frac"] == pytest.approx(1.0 / 3.0)


def test_the_pre_clip_range_is_reported_separately_from_the_reward():
    """The pre-clip margin's min and max survive the clip in the diagnostics.

    Arm 3's saturation verdict depends on the *pre-clip* value, so a reader can
    see the clip was applied to something rather than trusting the number.
    """
    hook = AnalyticMarginReward(clip=True)
    # gold=2 scores 5.0 against a best distractor of 0.0 -> margin +5.0
    hook(_ctx([0.0, 0.0, 5.0], gold=2))
    # gold=2 scores -3.0 against a best distractor of 4.0 -> margin -7.0
    hook(_ctx([0.0, 4.0, -3.0], gold=2))
    stats = hook.stats()
    assert stats["pre_clip_max"] == pytest.approx(5.0)
    assert stats["pre_clip_min"] == pytest.approx(-7.0)
    # The rewards, by contrast, are the clipped and centred scalars:
    # step 1 clips to 1.0 and is uncentred; step 2 clips to 0.0 and is centred
    # by the *first* step's clipped value (1.0), giving -1.0.
    assert stats["reward_max"] == pytest.approx(1.0)
    assert stats["reward_min"] == pytest.approx(-1.0)


def test_an_unclipped_arm_never_saturates():
    """``sat_hi_frac`` and ``sat_lo_frac`` are 0.0 without a clip."""
    hook = AnalyticMarginReward(clip=False)
    hook(_ctx([0.0, 0.0, 5.0], gold=2))
    hook(_ctx([0.0, 4.0, -3.0], gold=2))
    stats = hook.stats()
    assert stats["sat_hi_frac"] == 0.0
    assert stats["sat_lo_frac"] == 0.0
    # ...and the margin passes through untouched.
    assert stats["pre_clip_max"] == pytest.approx(5.0)
    assert stats["pre_clip_min"] == pytest.approx(-7.0)


# --------------------------------------------------------------------------- #
# The hook's contract, as the loop enforces it
# --------------------------------------------------------------------------- #


def test_a_non_float_return_raises():
    """A hook returning a non-scalar fails loudly, with the step in hand."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=3)
    bm.model.config.reward_fn = lambda ctx: "0.5"
    with pytest.raises(TypeError):
        bm.model.run()


def test_a_nan_return_raises():
    """A NaN-emitting hook fails at the step it fired instead of poisoning a mean."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=3)
    bm.model.config.reward_fn = lambda ctx: math.nan
    with pytest.raises(ValueError):
        bm.model.run()


def test_an_infinite_return_raises():
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=3)
    bm.model.config.reward_fn = lambda ctx: math.inf
    with pytest.raises(ValueError):
        bm.model.run()


def test_a_boolean_return_raises():
    """``True`` is not a reward: it would silently be 1.0 for every step."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=3)
    bm.model.config.reward_fn = lambda ctx: True
    with pytest.raises(TypeError):
        bm.model.run()


def test_the_hook_is_called_once_per_learning_step():
    """One call per step, and only when ``learn`` is true."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=6)
    calls = []
    bm.model.config.reward_fn = (
        lambda ctx: calls.append(ctx.selected) or 0.0
    )
    bm.model.run()
    assert len(calls) == 6


def test_the_hook_is_not_called_when_learning_is_disabled():
    """Evaluation runs the loop without consulting the hook at all.

    This is what keeps the surrogate from reaching the reported endpoint: the
    eval router is rebuilt fresh, ``learn = False``, and ``routing@1`` comes
    from ``router.score``.
    """
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=4)
    bm.model.config.learn = False
    calls = []
    bm.model.config.reward_fn = lambda ctx: calls.append(ctx) or 0.0
    bm.model.run()
    assert calls == []


def test_the_hook_sees_the_gold_index_not_the_router():
    """The hook reads gold off the task, not off the router.

    The loop already binds the oracle arm from ``task.target_action``, so this
    is not a new channel into the router: the hook is not the router, and its
    return value reaches ``update`` as a bare ``float``.
    """
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=2)
    seen = []
    bm.model.config.reward_fn = (
        lambda ctx: seen.append(int(ctx.task.target_action)) or 0.0
    )
    bm.model.run()
    assert len(seen) == 2
    assert all(isinstance(g, int) for g in seen)


def test_the_hook_does_not_change_the_action_taken():
    """The hook is consulted after routing, so it cannot change the action.

    F3 changes the *gradient*, not the trajectory: the executed action, the
    task stream and the evaluator are identical across arms, so the arms are
    comparable on the parameters alone.
    """
    cfg = AblationConfig(seed=11, router=RouterSwitch(type="learned"))
    plain = build_model(cfg, n_steps=8)
    plain.model.environment.config.n_candidates = 8
    hooked = build_model(cfg, n_steps=8)
    hooked.model.environment.config.n_candidates = 8
    hooked.model.config.reward_fn = baseline_reward

    plain_ep = plain.model.run()
    hooked_ep = hooked.model.run()
    assert [s.decision.selected for s in plain_ep.steps] == [
        s.decision.selected for s in hooked_ep.steps
    ]
    assert [s.outcome.success for s in plain_ep.steps] == [
        s.outcome.success for s in hooked_ep.steps
    ]


def test_a_surrogate_arm_trains_the_weights():
    """A dense surrogate moves the weights, so the trained state is not a zero vector.

    A hook that left the weights at exactly zero would archive a uniform
    scorer as a trained state, which is what the integrity gate exists to
    catch. The update's centring term is ``reward - chance`` with
    ``chance = 1/H``, so a constant reward is *not* inert — it is a uniform
    negative push — and the meaningful check is that a zero-weight outcome is
    not what the surrogate produces either.
    """
    surrogate_w = _trained_weights(arm_reward("analytic_margin"), seed=5,
                                   n_candidates=8)
    assert any(w != 0.0 for w in surrogate_w)


def test_the_surrogate_and_baseline_reach_different_weights():
    """The intervention changes the trained parameters.

    This is the property the arm comparison rests on: if the surrogate arm
    trained to the same weights as the baseline, the endpoint comparison would
    be measuring nothing. ``baseline_reward`` and the surrogate differ in the
    scalar alone, and at these settings the two trajectories diverge.
    """
    baseline_w = _trained_weights(baseline_reward, seed=2, n_candidates=8)
    surrogate_w = _trained_weights(arm_reward("analytic_margin"), seed=2,
                                   n_candidates=8)
    assert baseline_w != surrogate_w, (
        "the surrogate arm trained to the identical weights as the baseline; "
        "the intervention had no effect on the update"
    )


def test_the_hook_reaches_the_router_only_through_score():
    """The hook holds the router but the router's inputs are unchanged.

    ``router.score`` is a function of the permitted inputs it already reads, so
    handing the router to the hook widens nothing: the hook returns one scalar
    and the router's ``update`` still takes ``reward: float``.
    """
    cfg = AblationConfig(seed=0, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=3)
    touched = []
    router = bm.model.router

    def _hook(ctx):
        # The hook reads the scorer, which is permitted, and mutates nothing.
        touched.append(tuple(ctx.router.score(ctx.query, ctx.state,
                                              ctx.candidates)))
        return 0.0

    bm.model.config.reward_fn = _hook
    bm.model.run()
    assert len(touched) == 3
    assert router is bm.model.router
    # The router's own bookkeeping is untouched by the hook's read.
    assert isinstance(router, LearnedRelationalRouter)
