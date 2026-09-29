# TACOSM-DEGENERACY-001 — the preflight that C5-001 and C5-002 needed

**Status: implemented, not run as an experiment.** This is infrastructure, not
a measurement: it has no contract because a contract pins what a run holds
constant, and there is no run. Its tests are what pins it, and they pass.

Related: `docs/TACOSM-C5-001-RESULT.md` (void), `docs/TACOSM-C5-002-RESULT.md`
(INSTRUMENT_INVALID), `src/tac_osm/degeneracy.py`.

## The problem, in one sentence each

Two C5 experiments failed in ways that were invisible in the number each
published, and both failures were properties of the frozen model's *outputs*:

- **C5-001** — the bridge's output was constant across candidates, so an
  argmax-over-output selection was deterministically index 0. The exhaustive
  arm measured the population base rate (0.2500, seed spread 0.0000, at every
  H) and the published table looked like a capability comparison.
- **C5-002** — the same bridge reports 0.84375 absolute output accuracy while
  separating satisfier from a one-bit-flipped violator in only 59/256 pairs
  (0.2305 against a pre-registered minimum of 0.5).

The shape of both:

> an output distribution that is constant, or that carries no information
> about the relation, still produces smooth, bounded, plausible-looking
> measurement numbers.

Both were found *after* the interpretation was written. C5-002 found the
second one *before* the capability table, because it had a gate; that is the
difference between a terminated run and a voided one, and it is worth
generalising.

## What was built

`src/tac_osm/degeneracy.py` — a preflight on a frozen model's outputs, run
before the measurement that consumes them. Five checks in a fixed order, where
the order is part of the gate because each later check is only meaningful given
the earlier one:

| # | check | catches | source failure |
|---:|---|---|---|
| 1 | representability | an instrument whose own programme cannot compute the relation | assumed, not checked, by the C5-002 gate |
| 2 | output distribution | constant or near-constant outputs | **C5-001** |
| 3 | class support | a separation score computed over too few pairs | makes (4) well-posed |
| 4 | discriminative content | outputs that do not track the relation | **C5-002** |
| 5 | programme consistency | model outputs disagreeing with the instrument's own computation | C5-001's `verification_rate = 0.0000` was an *effect* of (2) |

A failure raises `DegeneracyError` rather than returning a flag, for the same
reason the integrity and representability gates raise: a measurement taken
under a degenerate model has produced a number, and a produced number cannot
be un-produced.

## Why it is not a representability gate

`representability.py` asks whether the *hypothesis class* can express the
relation — a property of the basis, answered without a model. This module asks
whether a *trained model's actual outputs* can support the downstream decision.
The two are sequenced over one measurement's lifetime and neither subsumes the
other:

```
basis  ->  weights  ->  outputs
   representability   integrity   degeneracy
   (pre-model)        (pre-evaluation)  (pre-measurement)
```

Degeneracy is the last and the only one that looks at what the model actually
emits — which is why it caught nothing in either voided run. It did not exist.

## The thresholds, and where they come from

| constant | value | derivation |
|---|---|---|
| `DEFAULT_MIN_SPREAD` | 0.05 | the C5-001 output was constant (`sd = 0`), so any positive threshold separates that failure from a working instrument; deliberately loose, to catch the class rather than to maximise sensitivity |
| `DEFAULT_MIN_DISCRIMINATION` | 0.5 | the C5-002 gate's pre-registered `GATE_MIN_ACCURACY`, imported so this preflight's bar is the registered bar rather than a stricter one invented after the fact |
| `DEFAULT_MIN_CLASS_SUPPORT` | 2 | with two marked positions the acceptable fraction is exactly 1/4; 2 is the minimum that makes a separation score defined |

**No threshold was tuned against a run that passed it.** All three are derived
from the measured signatures of the two failures the module generalises, and
the module documents that a future pass is a fact about the run rather than
evidence the thresholds were well chosen.

## The pair interface, and why it is duplicated

`make_pairs` builds the same `(reference, satisfying, violating, marks)` tuples
the C5-002 gate builds inline, from the same seed. The code is deliberately not
shared — a runnable script is the wrong dependency target for a library — so
the construction is duplicated and `test_make_pairs_matches_the_c5_002_gate_construction`
is the seam that keeps them in step: a divergence there would silently change
what the registered gate tested, because the pairs *are* its semantics.

Verified: `make_pairs(256, 20260930)` is elementwise identical to
`gate_pairs(256, 20260930)`, the exact stream the C5-002 run evaluated.

## What this does not do

It does not verify the bridge's *training objective*. C5-002's diagnosed cause
is a bridge trained on each candidate's Boolean output in isolation, where
nothing rewards within-query ranking; a pair-trained bridge is the registered
fix. This module detects the *symptom* — outputs that do not separate —
regardless of cause, deliberately, because the symptom is what the measurement
depends on and the cause is what the successor experiment is for.

It does not judge a model *good*. Passing means "not degenerate in the specific
ways the two voided C5 runs were", which is necessary but far from sufficient.

## Reproduction

```
python -m pytest tests/test_degeneracy.py -q
```

The tests that matter most are the ones that would not have passed against the
failed runs:

- `test_constant_outputs_are_flagged_as_the_c5_001_signature` — the exact
  output distribution C5-001 measured, now flagged before any cell runs;
- `test_preflight_refuses_a_non_separating_model` — the C5-002 case, where the
  distribution is well spread and non-constant and only the discrimination
  check catches it, which is the case an accuracy score cannot;
- `test_make_pairs_matches_the_c5_002_gate_construction` — the drift guard.
