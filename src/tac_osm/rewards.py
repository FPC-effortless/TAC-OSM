"""``R_t``'s supervision — the F3 reward arms.

TACOSM-SURROGATE-001 (F3) intervenes on one thing only: the scalar the loop
passes as ``reward`` to ``LearnedRelationalRouter.update``. This module is
the named implementation, and the lookup that maps an arm name to it.

## Why the arm lives on the loop and not the router

F2's exploration knob is consulted *by the router*, inside ``route``: the
schedule decides which action is sampled, so it belongs on the component that
samples. The reward is not like that. It is computed by the *loop* from the
task, the decision and the outcome — none of which the router is permitted to
hold — and is passed into ``update`` as a bare ``float``. So the hook is
attached to ``ModelConfig`` and called once per step from
``TacOsmModel.step``. The router's API gains nothing: ``update`` still takes
``reward: float`` and still has no gold parameter, so the interface boundary
``leakage.py`` enforces is unchanged in every arm.

## The two boundaries the pre-registration separates

1. **Interface** — does the router, or ``features()``, see the gold index?
   No, in every arm. The surrogate is computed here, outside the router, and
   reaches ``update`` through the one scalar argument it already accepted.
2. **Supervision** — is the gradient equivalent to information a deployed
   router could obtain? Also no, and *neither is the baseline*: the
   environment sets ``target_action = gold_index`` and scores ``success =
   action == target_action``, so ``float(outcome.success)`` is itself
   ``1[a_t = gold]``. F3 holds (1) fixed and loosens (2) by a measured
   amount, in a named direction: the *density* of a signal of the same
   logical kind, with the anchor held fixed.

## The centring rule, and why it is a running mean

The surrogate is centred by a running mean of *itself*, not by the per-step
mean of the candidate scores. The first draft of the pre-registration used
the score mean; measured over 180 episodes per H, ``mean(scores)`` is
*negative* under the analytic vector (−0.99 at H=8, −1.65 at H=256) because
distractors score strongly negative on the marked positions. Subtracting a
negative adds, so the centred reward was positive on 180/180 steps with a
minimum of +1.75 — precisely the "constant positive reward" the centring
term was written to prevent. A running mean of the surrogate tracks the
surrogate instead, so the update stays signed by deviation from mean
performance.

This is the standard policy-gradient baseline: a state-independent estimate
of the reward mean, which lowers the update's variance without biasing it.
"""

from __future__ import annotations

from typing import Sequence

from .model import RewardContext

__all__ = ["AnalyticMarginReward", "arm_reward", "baseline_arm_names"]


class AnalyticMarginReward:
    """The F3 surrogate: the analytic margin to the gold action.

    Computes, on every training step::

        surrogate_t = s_gold(w_t) - max_{j != gold} s_j(w_t)

    from ``router.score`` — the raw dot products, in the same
    non-normalised units ``delta_1`` is reported in, so the margin is
    comparable across H. The value is then optionally clipped to ``[0, 1]``
    and centred by the running mean of the surrogate itself.

    **State, and why a caller gets a fresh instance.** The running mean is
    the only mutable state this class holds, and it is a property of one
    trajectory, not of an arm: :func:`arm_reward` therefore builds a new hook
    on every call. Two training cells sharing a baseline would quietly carry
    one trajectory's mean into the next, which would change the centring
    without changing any visible config — the exact silent-protocol-drift the
    single-construction-path rule exists to prevent. The lookup pins the
    arm's *definition*; the fresh instance keeps its *state* honest.
    """

    def __init__(self, *, clip: bool = False, clip_lo: float = 0.0,
                 clip_hi: float = 1.0) -> None:
        #: Whether the surrogate is clipped to ``[clip_lo, clip_hi]`` before
        #: centring. Arm 3 clips; arm 2 does not.
        self.clip = clip
        #: The clip bounds. Fixed by the pre-registration and not exposed on
        #: any CLI — see the "Reproduction" section of
        #: ``docs/TACOSM-SURROGATE-001.md``.
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        #: The running-mean baseline, kept as a total and a count so the mean
        #: is exact rather than a lossy accumulator.
        self._total = 0.0
        self._count = 0
        #: Diagnostics the measurement reports per H. The pre-clip margin is
        #: kept separately from the returned reward because arm 3's
        #: saturation verdict depends on the *pre-clip* value — the fraction
        #: of steps on which it exceeded 1.0.
        self.pre_clip_min: float | None = None
        self.pre_clip_max: float | None = None
        #: Saturation counts for the clip arm. ``n_saturated_hi`` is the
        #: number of steps whose pre-clip margin exceeded ``clip_hi`` — the
        #: case the pre-registration's saturation caveat is about, where the
        #: clip has removed the gradient's magnitude and left a constant 1.0.
        #: ``n_saturated_lo`` is the floor, which a *negative* margin hits.
        #: The floor matters as much as the ceiling at large H: measured at
        #: H=256 from a zero start, the margin's own maximum is +0.088, so
        #: the clip is flooring far more than it is ceiling, and an arm that
        #: reports only the ceiling would mis-describe its own signal.
        self.n_saturated_hi = 0
        self.n_saturated_lo = 0
        #: Stats of the reward actually returned — the post-clip,
        #: post-centring scalar the router consumed. Kept separately from the
        #: pre-clip margin because the pre-registration's density audit
        #: compares arms on the *non-zero rate of the reward the update saw*,
        #: and centring can zero a non-zero margin. The baseline arm's
        #: equivalent is computed by the measurement from the episode's
        #: ``outcome.success`` values, which is the same quantity.
        self._reward_total = 0.0
        self._reward_nonzero = 0
        self.reward_min: float | None = None
        self.reward_max: float | None = None

    # -- the margin -------------------------------------------------------- #

    def margin(self, ctx: RewardContext) -> float:
        """``s_gold - max_{j != gold} s_j``, from the router's own scorer.

        The scores are recomputed here rather than read off
        ``ctx.decision.scores`` for two reasons:

        * ``decision.scores`` are the *softmax probabilities* the decision
          sampled from, and a probability margin shrinks with H even for a
          perfect scorer, because the probabilities are normalised over the
          candidate count. The raw dot products have no such normalisation,
          so margins in these units are comparable across H — the same
          reason ``delta_1`` is reported in raw units.
        * recomputing is what makes the hook a function of the weights as
          they stand at the moment of the update, which is the quantity the
          margin is about, rather than of an object whose contract is
          "probabilities at decision time".

        The query, state and candidates come from the context the loop
        already holds, so nothing new is read off the environment:
        ``router.score`` reads ``state.read(query)``, which the router is
        permitted to read, and the gold index comes from
        ``ctx.task.target_action``, which the loop already needed to bind the
        oracle arm.

        **The empty-candidates case.** With fewer than two candidates there
        is no competitor to take a maximum over, so there is no margin and
        the surrogate is undefined. Returning 0.0 would be a silent zero
        reward — the terminal reward in disguise — so the caller sees a
        raised error instead. A hook that silently emitted the baseline's
        reward would make an "analytic margin" arm indistinguishable from the
        baseline arm.
        """
        if len(ctx.candidates) < 2:
            raise ValueError(
                f"the analytic margin needs >= 2 candidates to define a "
                f"competitor, got {len(ctx.candidates)}"
            )
        raw = self._scores(ctx)
        gold = self._gold_index(ctx)
        competitors = [s for i, s in enumerate(raw) if i != gold]
        margin = raw[gold] - max(competitors)
        self.pre_clip_min = margin if self.pre_clip_min is None else min(
            self.pre_clip_min, margin)
        self.pre_clip_max = margin if self.pre_clip_max is None else max(
            self.pre_clip_max, margin)
        return margin

    # -- the hook ---------------------------------------------------------- #

    def __call__(self, ctx: RewardContext) -> float:
        """The reward: the (clipped) margin, centred by its strictly-past mean.

        The order is margin → clip → centre → record. Clipping before
        centring means the baseline tracks the signal the update actually
        consumed, and centring after clipping means a saturated step's
        contribution to the running mean is the saturated value.
        """
        surrogate = self.margin(ctx)
        if self.clip:
            if surrogate > self.clip_hi:
                self.n_saturated_hi += 1
            elif surrogate < self.clip_lo:
                self.n_saturated_lo += 1
            surrogate = min(self.clip_hi, max(self.clip_lo, surrogate))
        reward = surrogate - self._baseline(surrogate)
        self._reward_total += reward
        self._reward_nonzero += int(reward != 0.0)
        self.reward_min = reward if self.reward_min is None else min(
            self.reward_min, reward)
        self.reward_max = reward if self.reward_max is None else max(
            self.reward_max, reward)
        return reward

    # -- helpers ----------------------------------------------------------- #

    def _baseline(self, surrogate: float) -> float:
        """The mean of the (clipped) surrogate over the steps *before* this one.

        Implements, exactly::

            baseline_t = mean(s_0, ..., s_{t-1}),  s_i = the value step i
                                                   returned from the clip

        with ``baseline_0 = 0.0``. The argument is the *post-clip* surrogate,
        because :meth:`__call__` clips before it centres, so the accumulator
        and ``stats()["baseline_mean"]`` describe the same quantity the reward
        was computed from. The pre-clip margin is tracked separately in
        ``pre_clip_min`` / ``pre_clip_max``.

        The mean is read *before* this step is folded in, so a step is never
        its own baseline. Without that ordering the running mean would include
        the current step and the centred reward would be biased toward zero by
        ``1/t`` — small in a long run, but nonzero on every step and largest on
        the first ones, which are exactly the steps that determine whether a
        sparse early signal is ever seen.

        With no history the baseline is 0.0 and the first step's centred value
        is its raw surrogate — the unbiased first estimate, and the reason the
        running mean cannot produce the constant-offset failure the score-mean
        rule did.

        **Must match the router's own centring term.** ``update`` computes
        ``reward - 1/H`` and this computes ``s_t - mean(s_{<t})``, so the two
        centring terms are the same kind of object: a state-independent
        estimate of the reward mean, subtracted to lower the update's variance
        without biasing it. Keeping this strictly-past is what keeps the
        parallel exact.
        """
        mean = self._total / self._count if self._count else 0.0
        self._total += surrogate
        self._count += 1
        return mean

    def _gold_index(self, ctx: RewardContext) -> int:
        """The gold candidate's index, from the task the loop already holds.

        The loop reads ``task.target_action`` to bind the oracle arm, so this
        is not a new channel into the router: the hook is not the router.
        """
        return int(ctx.task.target_action)

    def _scores(self, ctx: RewardContext) -> list[float]:
        """The router's raw dot products at the moment of the update."""
        return list(ctx.router.score(ctx.query, ctx.state, ctx.candidates))

    # -- diagnostics ------------------------------------------------------- #

    def stats(self) -> dict[str, float]:
        """The running mean, step count, pre-clip range and saturation rates.

        The pre-registration requires the mean, min and max surrogate seen
        during training to be reported per H, plus the fraction of steps on
        which the clip arm saturated. Those are accumulated here from the
        pre-clip margin, so a reader can verify the reward behaved as claimed
        rather than trusting a number the script printed. ``count`` is
        exposed so the measurement can assert the hook was actually called on
        every learning step.

        Both saturation directions are reported, because the clip has a floor
        as well as a ceiling. The pre-registration's caveat is written about
        the ceiling — a margin above 1.0 saturating to a constant unit
        reward — but at large H, from a zero start, the margin is small and
        *negative* on most steps, so the floor is what binds. Reporting only
        the ceiling would let an arm appear untested-by-saturation while the
        clip was silently zeroing most of its signal.
        """
        out: dict[str, float] = {
            "baseline_mean": self._total / self._count if self._count else 0.0,
            "count": float(self._count),
            "pre_clip_min": (0.0 if self.pre_clip_min is None
                             else float(self.pre_clip_min)),
            "pre_clip_max": (0.0 if self.pre_clip_max is None
                             else float(self.pre_clip_max)),
            "sat_hi_frac": (self.n_saturated_hi / self._count
                            if self._count else 0.0),
            "sat_lo_frac": (self.n_saturated_lo / self._count
                            if self._count else 0.0),
            "reward_mean": (self._reward_total / self._count
                            if self._count else 0.0),
            "reward_nonzero_frac": (self._reward_nonzero / self._count
                                    if self._count else 0.0),
            "reward_min": (0.0 if self.reward_min is None
                           else float(self.reward_min)),
            "reward_max": (0.0 if self.reward_max is None
                           else float(self.reward_max)),
        }
        if self._count:
            out["pre_clip_mean"] = self._total / self._count
        return out


# --------------------------------------------------------------------------- #
# The registered arms, and the lookup
# --------------------------------------------------------------------------- #

#: The three arms ``TACOSM-SURROGATE-001`` pre-registers.
#:
#: ``baseline`` is ``None`` because that arm *is* the MATCHED-001 protocol:
#: no hook, ``reward = float(outcome.success)``, which is what
#: :func:`tac_osm.model.baseline_reward` computes and what the loop does when
#: ``ModelConfig.reward_fn is None``. The two interventions are named hooks
#: with the constants the pre-registration fixed — no arm carries a value
#: this table does not name, and no value here is a knob a caller may tune.
_ARM_REWARD: dict[str, dict[str, bool]] = {
    "baseline": {},
    "analytic_margin": {"clip": False},
    "analytic_margin_clipped": {"clip": True},
}


def baseline_arm_names() -> tuple[str, ...]:
    """The registered arm names, for validation and for the help text."""
    return tuple(_ARM_REWARD)


def arm_reward(arm: str) -> AnalyticMarginReward | None:
    """The registered reward hook for an F3 arm.

    Takes the arm *name*, not the parameters: this is a lookup, not a
    constructor, so the only way to run a surrogate arm is to name one of the
    three the pre-registration defines. A caller cannot construct a hook with
    different clip bounds and reach the training path through this function —
    the registered values are the reachable ones.

    Raises ``KeyError`` for an unknown arm rather than falling back to the
    baseline. A typo that silently trained the baseline and reported it as a
    surrogate arm would be exactly the warm-start confusion the
    pre-registration's Layer 1 section exists to prevent — the same contract
    as F2's ``arm_exploration``.

    Returns ``None`` for the baseline arm, and a **fresh instance** for a
    surrogate arm. The fresh instance is deliberate: the running mean is a
    property of one trajectory, so a shared hook would carry one cell's
    baseline into the next. The caller keeps the returned object if it wants
    the per-arm reward statistics after training.
    """
    if arm not in _ARM_REWARD:
        raise KeyError(
            f"unknown F3 arm {arm!r}; expected one of {sorted(_ARM_REWARD)} "
            "(values are fixed by the TACOSM-SURROGATE-001 pre-registration "
            "and cannot be supplied by the caller)"
        )
    kwargs = _ARM_REWARD[arm]
    if not kwargs:
        return None
    return AnalyticMarginReward(**kwargs)
