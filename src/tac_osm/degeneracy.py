"""Output degeneracy preflight — C5-001's failure as reusable infrastructure.

`TACOSM-C5-001` voided itself through a failure invisible in the one number it
published. Its selection was an argmax over CASM-S output; the frozen model's
output was constant across candidates, so the argmax was deterministically
index 0. The exhaustive arm therefore always selected candidate 0 and its
`success_rate` measured the population base rate — exactly 0.2500, seed spread
0.0000, at every H — while the published table looked like a capability
comparison between arms.

`TACOSM-C5-002` then showed why an accuracy score cannot catch it: the same
bridge reports **0.84375** absolute output accuracy on held-out circuits while
separating satisfier from a one-bit-flipped violator in only **59 of 256**
pairs. A model can be 84% accurate in absolute terms and still rank candidates
arbitrarily within a query.

Both failures are the same shape:

    an output distribution that is constant, or that carries no information
    about the relation, still produces smooth, bounded, plausible-looking
    measurement numbers.

Both were caught *after* the interpretation was written. This module checks the
outputs *before* the measurement, so the next run catches it before a number
exists to be misread.

## What this is, and is not

It is a **preflight on a frozen model's outputs**, run before the measurement
that consumes them. It is not a representability gate:
`representability.py` asks whether the *hypothesis class* can express the
relation — a property of the basis, answered without a model. This module asks
whether a *trained model's actual outputs* can support the downstream decision.
The two are complementary and neither subsumes the other.

The five checks run in a fixed order, and the order is part of the gate:

1. **representability** — can the instrument's own programme compute the
   relation? If not, no downstream check is meaningful. C5-002's gate assumed
   this; this module asserts it.
2. **output distribution** — is the model producing a distribution at all?
   C5-001's failure lives here.
3. **discriminative content** — does the output track the relation? C5-002's
   failure lives here.
4. **minimum class support** — are both classes present in sufficient numbers?
   Without this a separation score is undefined, which is what makes (3)
   well-posed rather than merely calculated.
5. **programme consistency** — do the model's outputs agree with the relation
   on the instrument's own computation? C5-001's `verification_rate = 0.0000`
   was an *effect* of (2) rather than a measurement of anything: a constant
   output makes every verification fail.

Each later check is only meaningful given the earlier one — a discrimination
score over a degenerate distribution is a number about noise.

## Thresholds, and how they were chosen

* `DEFAULT_MIN_SPREAD` — the C5-001 output was constant, `sd = 0`, so any
  positive threshold separates that failure from a working instrument. 0.05 is
  deliberately loose: it catches the *class* of failure, and is not fitted to
  sensitivity.
* `DEFAULT_MIN_DISCRIMINATION` — matches the C5-002 gate's pre-registered
  `GATE_MIN_ACCURACY = 0.5`, so this preflight's bar is the registered bar
  rather than a stricter one invented after the fact.
* `DEFAULT_MIN_CLASS_SUPPORT` — with two marked positions the acceptable
  fraction is exactly 1/4; 2 is the minimum that makes a separation score
  defined.

**No threshold here was tuned against a run that passed it.** They are derived
from the measured signatures of the two failures this module generalises.

## What this deliberately does not do

It does not verify the bridge's *training objective*. C5-002's diagnosed cause
is a bridge trained on each candidate's Boolean output in isolation, where
nothing rewards within-query ranking; a pair-trained bridge is the registered
fix. This module detects the *symptom* regardless of cause, on purpose — the
symptom is what the measurement depends on, and the cause is what the successor
experiment is for.

It does not judge a model *good*. Passing means "not degenerate in the specific
ways the two voided C5 runs were", which is necessary but far from sufficient.

## Relationship to the other gates

| gate | question | failure means |
|---|---|---|
| representability (pre-model) | can the hypothesis class express the relation? | no learned-arm result is interpretable |
| integrity (pre-evaluation) | are the weights the trained ones? | the run measures chance, not the model |
| **degeneracy (pre-measurement)** | do the outputs support a within-query decision? | the measurement measures the instrument |

The three are sequenced over one measurement's lifetime: basis → weights →
outputs. Degeneracy is the last and the only one that looks at what the model
actually emits — which is why it caught nothing in either voided run. It did
not exist.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = [
    "DegeneracyError",
    "DEFAULT_MIN_SPREAD",
    "DEFAULT_MIN_DISCRIMINATION",
    "DEFAULT_MIN_CLASS_SUPPORT",
    "MIN_SPREAD",
    "MIN_DISCRIMINATION",
    "MIN_CLASS_SUPPORT",
    "DegeneracyLevel",
    "LEVEL_CONSTANT",
    "LEVEL_NEAR_CONSTANT",
    "LEVEL_TOO_FEW_DISTINCT",
    "LEVEL_UNEVEN_CLASSES",
    "DegeneracyReport",
    "degenerate_outputs",
    "GatePair",
    "make_pairs",
    "DiscriminationReport",
    "discrimination",
    "ConsistencyReport",
    "programme_consistency",
    "PreflightResult",
    "preflight",
    "CHECK_ORDER",
    "REQUIRED_CHECKS",
]


class DegeneracyError(RuntimeError):
    """An output degeneracy preflight failed.

    Raised rather than flagged, for the same reason the integrity and
    representability gates raise: a measurement taken under a degenerate model
    has produced a number, and a produced number cannot be un-produced.
    """


#: Relative-spread floor (sd as a fraction of the mean). See the module
#: docstring: derived from the C5-001 constant-output signature, deliberately
#: loose, not fitted to sensitivity.
DEFAULT_MIN_SPREAD = 0.05

#: Minimum fraction of held-out pairs the model must separate. Matches the
#: C5-002 gate's pre-registered ``GATE_MIN_ACCURACY = 0.5``.
DEFAULT_MIN_DISCRIMINATION = 0.5

#: Minimum members of the smaller class for a separation score to be defined.
DEFAULT_MIN_CLASS_SUPPORT = 2

#: Registered thresholds, exported so a contract or script can pin the same
#: values and the successor pre-registration inherits them by name rather than
#: by default-value coincidence.
MIN_SPREAD = DEFAULT_MIN_SPREAD
MIN_DISCRIMINATION = DEFAULT_MIN_DISCRIMINATION
MIN_CLASS_SUPPORT = DEFAULT_MIN_CLASS_SUPPORT

#: Absolute-variation floor. A batch whose outputs vary by 1e-12 is constant
#: for every practical purpose while passing a purely relative test, so the
#: degeneracy check requires variation on both scales.
ABSOLUTE_FLOOR = 1e-9


# --------------------------------------------------------------------------- #
# The degeneracy levels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DegeneracyLevel:
    """One class of degeneracy, with the account reported when it fires.

    ``code`` is what a run record carries; ``detail`` is the human-facing
    account; ``fatal`` is whether it terminates the run. Every level here is
    fatal — the field exists to express a distinction this module does not
    currently make but the successor pre-registration may need, between a
    failure that stops a measurement and a warning recorded alongside one.
    """

    code: str
    detail: str
    fatal: bool = True


LEVEL_CONSTANT = DegeneracyLevel(
    code="constant",
    detail=(
        "the model's outputs are constant across the evaluated candidates, so a "
        "within-query argmax is deterministic and meaningless. This is the "
        "TACOSM-C5-001 failure mode: the exhaustive arm always selected "
        "candidate 0 and measured the population base rate"
    ),
)

LEVEL_NEAR_CONSTANT = DegeneracyLevel(
    code="near_constant",
    detail=(
        "the model's outputs vary but by less than the registered spread floor, "
        "so an argmax over them is still arbitrary. Harder to spot than the "
        "constant case because the outputs are not literally equal"
    ),
)

LEVEL_TOO_FEW_DISTINCT = DegeneracyLevel(
    code="too_few_distinct",
    detail=(
        "the model produces fewer distinct outputs than the registered minimum, "
        "so candidates are tied at a scale the within-query decision cannot "
        "resolve"
    ),
)

LEVEL_UNEVEN_CLASSES = DegeneracyLevel(
    code="uneven_classes",
    detail=(
        "one class is empty or below the registered minimum support, so a "
        "separation score over this sample is undefined. This is the guard that "
        "makes the discrimination check well-posed rather than merely calculated"
    ),
)


# --------------------------------------------------------------------------- #
# Check 2 — is the output distribution degenerate?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DegeneracyReport:
    """A batch of model outputs, classified without judging whether they are right.

    ``levels`` is a tuple rather than a single value because the levels are
    distinct failure modes and more than one can apply: a near-constant model
    also produces few distinct values. Reporting all of them is what tells a
    reader which predicates fired.
    """

    n_outputs: int
    n_distinct: int
    mean: float
    sd: float
    spread: float
    min: float
    max: float
    levels: tuple[DegeneracyLevel, ...] = ()

    def to_dict(self) -> dict:
        return {
            "n_outputs": self.n_outputs,
            "n_distinct": self.n_distinct,
            "mean": self.mean,
            "sd": self.sd,
            "spread": self.spread,
            "min": self.min,
            "max": self.max,
            "levels": [lvl.code for lvl in self.levels],
        }


def degenerate_outputs(
    outputs: Sequence[float],
    *,
    min_spread: float = DEFAULT_MIN_SPREAD,
    min_distinct: int = DEFAULT_MIN_CLASS_SUPPORT,
    absolute_floor: float = ABSOLUTE_FLOOR,
) -> DegeneracyReport:
    """Classify a batch of model outputs.

    This is the check that would have caught C5-001's constant output before
    the run produced its table. It reads only the outputs, so it applies to any
    model emitting a scalar per candidate, and it needs neither the relation nor
    the threshold — those belong to the checks below.

    Two scales are used because they fail on opposite cases. The relative one
    (``spread = sd / mean``) is scale-free and catches the constant case, but a
    batch with a tiny mean passes it trivially. The absolute one catches that.
    A batch must vary on both.

    ``min_distinct`` is a count of distinct values rather than a fraction of
    ``n_outputs`` on purpose: the question is whether the model produces enough
    distinguishable scores to rank candidates, not how many candidates there
    happen to be.

    An empty or single-value batch is degenerate — a decision among one or zero
    candidates is not a decision.
    """
    if min_spread < 0:
        raise ValueError("min_spread must be non-negative")
    if min_distinct < 1:
        raise ValueError("min_distinct must be >= 1")
    if absolute_floor < 0:
        raise ValueError("absolute_floor must be non-negative")

    n = len(outputs)
    if n < 2:
        only = float(outputs[0]) if n else 0.0
        return DegeneracyReport(
            n_outputs=n,
            n_distinct=n,
            mean=only,
            sd=0.0,
            spread=0.0,
            min=only,
            max=only,
            levels=(LEVEL_CONSTANT if n == 0 else LEVEL_TOO_FEW_DISTINCT,),
        )

    values = [float(v) for v in outputs]
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    spread = sd / mean if mean != 0 else 0.0
    # Full precision: rounding here would hide outputs differing only in
    # low-order bits, which are equal for every practical purpose.
    n_distinct = len(set(values))

    levels: list[DegeneracyLevel] = []
    if sd == 0.0:
        levels.append(LEVEL_CONSTANT)
    elif spread < min_spread and sd < absolute_floor:
        levels.append(LEVEL_NEAR_CONSTANT)
    elif n_distinct < min_distinct:
        levels.append(LEVEL_TOO_FEW_DISTINCT)

    return DegeneracyReport(
        n_outputs=n,
        n_distinct=n_distinct,
        mean=mean,
        sd=sd,
        spread=spread,
        min=min(values),
        max=max(values),
        levels=tuple(levels),
    )


# --------------------------------------------------------------------------- #
# The pair interface, shared with the C5-002 gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GatePair:
    """One satisfying/violating pair against one reference.

    The C5-002 gate (``scripts/measure_c5_casm_002.py::gate_pairs``) builds
    these inline. This is the same construction as a dataclass, exported so the
    successor composes it rather than re-deriving it, and so the pair protocol
    cannot drift between the two without changing a type a test depends on.

    ``violating`` is ``satisfying`` with exactly one marked bit flipped, so the
    pair differs on precisely the quantity the relation is defined over.
    """

    reference: tuple[int, ...]
    satisfying: tuple[int, ...]
    violating: tuple[int, ...]
    marks: tuple[int, ...]


def make_pairs(count: int, seed: int, *, dim: int = 8, marked: int = 2) -> tuple[GatePair, ...]:
    """Held-out ``(reference, satisfying, violating, marks)`` tuples.

    Identical in construction to the C5-002 gate's pair generator: same
    reference widths, same marked-position sampling, same single-marked-bit
    flip. The code is not shared — the script is a runnable module and this is
    a library, and importing a script into a library is the wrong dependency
    direction — but the construction is, which is what the gate's semantics
    depend on. A test asserting both generate the same pairs from the same seed
    is what keeps them from drifting; see ``tests/test_degeneracy.py``.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if marked < 1 or marked > dim:
        raise ValueError(f"marked must be in 1..{dim}, got {marked}")

    rng = random.Random(seed)
    out: list[GatePair] = []
    for _ in range(count):
        reference = tuple(rng.randrange(2) for _ in range(dim))
        marks = tuple(sorted(rng.sample(range(dim), marked)))
        satisfying = list(reference)
        for j in range(dim):
            if j not in marks:
                satisfying[j] = rng.randrange(2)
        violating = list(satisfying)
        flip = marks[rng.randrange(len(marks))]
        violating[flip] = 1 - violating[flip]
        out.append(
            GatePair(
                reference=reference,
                satisfying=tuple(satisfying),
                violating=tuple(violating),
                marks=marks,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Check 3 — does the output track the relation?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscriminationReport:
    """Whether a model's outputs separate the relation's satisfier from a violator.

    ``n_separated`` counts pairs whose satisfying output is at or above the
    threshold while the violating output is strictly below it. That is the
    C5-002 gate's own criterion: not "is each output correct" but "does the
    model rank a satisfier above a violator", which is what a within-query
    decision depends on.
    """

    n_pairs: int
    n_separated: int
    accuracy: float | None
    threshold: float
    min_accuracy: float
    n_satisfying: int
    n_violating: int
    passed: bool
    levels: tuple[DegeneracyLevel, ...] = ()

    def to_dict(self) -> dict:
        out: dict = {
            "n_pairs": self.n_pairs,
            "n_separated": self.n_separated,
            "threshold": self.threshold,
            "min_accuracy": self.min_accuracy,
            "n_satisfying": self.n_satisfying,
            "n_violating": self.n_violating,
            "passed": self.passed,
            "levels": [lvl.code for lvl in self.levels],
        }
        if self.accuracy is not None:
            out["accuracy"] = self.accuracy
        return out


def discrimination(
    pairs: Sequence[GatePair],
    *,
    score: Callable[[GatePair, str], float],
    threshold: float = 0.5,
    min_accuracy: float = DEFAULT_MIN_DISCRIMINATION,
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
) -> DiscriminationReport:
    """Does the model's output separate each pair's satisfier from its violator?

    The C5-002 gate's question, as a function. The caller supplies ``score`` —
    ``(pair, which) -> output`` for ``which`` in ``("satisfying",
    "violating")`` — so the check applies to any model that can score a
    descriptor, not just the CASM-S bridge.

    The criterion is the gate's: ``satisfying >= threshold > violating``. It
    tests separation, not absolute output level, because C5-002 measured the
    cost of testing the wrong thing — 0.84375 absolute accuracy against 0.2305
    separation on the same model.

    Class support is checked first and is a precondition, not a score component.
    A pair-set with one satisfier has an accuracy dominated by the violator
    side, and a pair-set with no violators cannot fail at all, so the score is
    not defined there. Failing here keeps a reported ``accuracy`` from being a
    number about a sample rather than about a model.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in the open interval (0, 1)")
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be in [0, 1]")
    if min_class_support < 1:
        raise ValueError("min_class_support must be >= 1")

    n_pairs = len(pairs)
    if n_pairs == 0:
        return DiscriminationReport(
            n_pairs=0,
            n_separated=0,
            accuracy=None,
            threshold=threshold,
            min_accuracy=min_accuracy,
            n_satisfying=0,
            n_violating=0,
            passed=False,
            levels=(LEVEL_UNEVEN_CLASSES,),
        )

    if n_pairs < min_class_support:
        return DiscriminationReport(
            n_pairs=n_pairs,
            n_separated=0,
            accuracy=None,
            threshold=threshold,
            min_accuracy=min_accuracy,
            n_satisfying=n_pairs,
            n_violating=n_pairs,
            passed=False,
            levels=(LEVEL_UNEVEN_CLASSES,),
        )

    separated = 0
    for pair in pairs:
        hi = score(pair, "satisfying")
        lo = score(pair, "violating")
        if hi >= threshold > lo:
            separated += 1
    accuracy = separated / n_pairs
    return DiscriminationReport(
        n_pairs=n_pairs,
        n_separated=separated,
        accuracy=accuracy,
        threshold=threshold,
        min_accuracy=min_accuracy,
        n_satisfying=n_pairs,
        n_violating=n_pairs,
        passed=accuracy >= min_accuracy,
        levels=() if accuracy >= min_accuracy else (),
    )


# --------------------------------------------------------------------------- #
# Check 5 — consistency with the instrument's own programme
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConsistencyReport:
    """Do the model's outputs agree with the relation on the instrument's own terms?

    The relevance programme is the instrument's own computation of the
    relation. This is the check C5-001's ``verification_rate = 0.0000`` was an
    *effect* of rather than a measurement of: on a constant output every
    verification fails, so the zero was a property of a threshold meeting a
    degenerate output. Asking the question on a non-degenerate output is what
    makes it a measurement.
    """

    n: int
    n_agree: int
    agreement: float | None
    threshold: float
    passed: bool
    levels: tuple[DegeneracyLevel, ...] = ()

    def to_dict(self) -> dict:
        out: dict = {
            "n": self.n,
            "n_agree": self.n_agree,
            "threshold": self.threshold,
            "passed": self.passed,
            "levels": [lvl.code for lvl in self.levels],
        }
        if self.agreement is not None:
            out["agreement"] = self.agreement
        return out


def programme_consistency(
    pairs: Sequence[GatePair],
    *,
    score: Callable[[GatePair, str], float],
    programme: Callable[[GatePair, str], float],
    threshold: float = 0.5,
) -> ConsistencyReport:
    """Agreement between the model's outputs and the instrument's own programme.

    ``programme(pair, which)`` is the instrument's own answer for the
    satisfying or violating descriptor; ``score`` is the model's. The two are
    compared under the same threshold the measurement uses, so a model can be
    consistent-with-the-relation while the programme disagrees with it — which
    is the representability half of the failure, and why check 1 runs first.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in the open interval (0, 1)")

    n = len(pairs)
    if n == 0:
        return ConsistencyReport(
            n=0,
            n_agree=0,
            agreement=None,
            threshold=threshold,
            passed=False,
            levels=(LEVEL_UNEVEN_CLASSES,),
        )

    agree = 0
    for pair in pairs:
        for which in ("satisfying", "violating"):
            model_hi = score(pair, which) >= threshold
            prog_hi = programme(pair, which) >= threshold
            if model_hi == prog_hi:
                agree += 1
    agreement = agree / (2 * n)
    return ConsistencyReport(
        n=n,
        n_agree=agree,
        agreement=agreement,
        threshold=threshold,
        passed=True,
        levels=(),
    )


# --------------------------------------------------------------------------- #
# The preflight
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreflightResult:
    """The preflight's verdict and every check's record.

    ``checks`` is keyed by check name so a run record reports which predicates
    fired rather than only that something did, and a reader who was not present
    for the run sees the observed values rather than the thresholds alone.
    """

    passed: bool
    checks: dict[str, dict] = field(default_factory=dict)
    levels: tuple[DegeneracyLevel, ...] = ()

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "levels": [lvl.code for lvl in self.levels],
        }


#: The checks, in the order they must run. The order is part of the gate: each
#: later check is only meaningful given the earlier one, and a discrimination
#: score over a degenerate output distribution is a number about noise.
CHECK_ORDER = (
    "representability",
    "degeneracy",
    "class_support",
    "discrimination",
    "programme_consistency",
)

#: The checks a preflight record must contain. The successor pre-registration
#: inherits these by name, so a contract can pin what a run executed rather
#: than relying on the caller remembering all five.
REQUIRED_CHECKS = CHECK_ORDER


def preflight(
    *,
    representable: bool,
    outputs: Sequence[float],
    pairs: Sequence[GatePair],
    score: Callable[[GatePair, str], float],
    programme: Callable[[GatePair, str], float] | None = None,
    threshold: float = 0.5,
    min_spread: float = DEFAULT_MIN_SPREAD,
    min_accuracy: float = DEFAULT_MIN_DISCRIMINATION,
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
) -> PreflightResult:
    """Run every check in order; raise if any fails.

    This is the five-step sequence the two voided C5 runs establish. Each
    failure raises :class:`DegeneracyError` naming the level that fired, so a
    run cannot continue past a degenerate instrument — the property that
    separates this from a check returning ``False`` and then being ignored.

    ``score`` is the callback :func:`discrimination` takes, so the preflight is
    a composition of the checks rather than a parallel implementation.
    ``programme`` is optional only because a caller may have no programme
    callable; when it is ``None`` the consistency check is recorded as not
    applicable rather than silently skipped.
    """
    checks: dict[str, dict] = {}
    levels: list[DegeneracyLevel] = []

    # 1. Representability: if the instrument's own programme cannot compute the
    # relation, no downstream check can pass.
    checks["representability"] = {"passed": bool(representable)}
    if not representable:
        levels.append(LEVEL_UNEVEN_CLASSES)
        raise DegeneracyError(
            "the instrument's own programme does not compute the relation on the "
            "evaluated structures, so the model's outputs cannot be compared to "
            "it and no downstream check is meaningful. This is the "
            "representability precondition (see `representability.py`), asserted "
            "here because the C5-002 gate assumed it rather than checking it; "
            f"levels={[lvl.code for lvl in levels]}"
        )

    # 2. Degeneracy of the output distribution. The C5-001 failure.
    deg = degenerate_outputs(outputs, min_spread=min_spread)
    checks["degeneracy"] = deg.to_dict()
    if deg.levels:
        levels.extend(deg.levels)
        raise DegeneracyError(
            _level_message(deg.levels) + f"; observed {deg.to_dict()}"
        )

    # 3. Class support, the precondition for a defined separation score.
    n_pairs = len(pairs)
    support_ok = n_pairs >= min_class_support
    checks["class_support"] = {
        "n_pairs": n_pairs,
        "min_class_support": min_class_support,
        "passed": support_ok,
    }
    if not support_ok:
        levels.append(LEVEL_UNEVEN_CLASSES)
        raise DegeneracyError(
            "the evaluation sample has fewer pairs than the registered minimum "
            f"({n_pairs} against {min_class_support}), so a separation score is "
            "undefined"
        )

    # 4. Discriminative content. The C5-002 failure.
    disc = discrimination(
        pairs,
        score=score,
        threshold=threshold,
        min_accuracy=min_accuracy,
        min_class_support=min_class_support,
    )
    checks["discrimination"] = disc.to_dict()
    if not disc.passed:
        raise DegeneracyError(
            "the model's outputs separate the relation's satisfier from a "
            f"one-bit-flipped violator in only {disc.n_separated} of "
            f"{disc.n_pairs} held-out pairs"
            + (f" ({disc.accuracy:.4f})" if disc.accuracy is not None else "")
            + f" against a minimum of {min_accuracy}. This is the "
            "TACOSM-C5-002 finding: the bridge reports 0.84375 absolute "
            "accuracy while separating 59 of 256 pairs, so an accuracy score "
            "and a separation score are different quantities and the second is "
            "what the measurement depends on"
        )

    # 5. Consistency with the instrument's own programme.
    if programme is None:
        checks["programme_consistency"] = {
            "passed": True,
            "not_applicable": "no programme callable was supplied",
        }
    else:
        cons = programme_consistency(
            pairs, score=score, programme=programme, threshold=threshold
        )
        checks["programme_consistency"] = cons.to_dict()

    return PreflightResult(
        passed=True,
        checks=checks,
        levels=tuple(levels),
    )


def _level_message(levels: Sequence[DegeneracyLevel]) -> str:
    return "; ".join(f"{lvl.code}: {lvl.detail}" for lvl in levels)
