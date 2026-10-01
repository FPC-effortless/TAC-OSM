# TACOSM-STATE-REP-007 RESULT

## Run provenance

- workflow: `36665305689`
- code head: `a859a3e346ed8ff9392c96710ef6e1f2215d9fa6`
- full test gate: **847 passed**
- contract checks: **passed**
- artifact: `TACOSM-STATE-REP-007`
- artifact id: `11075258348`
- artifact zip SHA-256: `30533666f5f574a8d98178d0d7dc725a3e8ac76b75c1125d8537a37bcc343782`

## Population curve

Learned and no-learning state addressors used identical architecture and
initialization, with only parameter updates enabled in the learned arm.

| M | Learned Top-1 | No-learning Top-1 | Chance | Learned − control | Address MACs |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.9922 | 0.5273 | 0.5000 | +0.4648 | 136 |
| 4 | 0.9828 | 0.2734 | 0.2500 | +0.7094 | 232 |
| 8 | 0.7141 | 0.1977 | 0.1250 | +0.5164 | 424 |
| 16 | 0.4125 | 0.1813 | 0.0625 | +0.2313 | 808 |
| 32 | 0.3344 | 0.1563 | 0.0313 | +0.1781 | 1,576 |

The learned arm exceeds the no-learning control at every registered M.
The analytic witness is 1.0000 at every M and seed.

## Seed behavior

At M=32 the learned arm varies substantially by seed:

- seed 0: 0.0000;
- seed 1: 0.1523;
- seed 2: 0.0000;
- seed 3: 1.0000;
- seed 4: 0.5195.

This matters: the pooled result remains above the control, but the learner is
not uniformly reliable at the largest tested population.

At M=8, the result reproduces the REP-006 direction while differing slightly
in exact values because REP-007 fixes its own seed/task generation schedule.

## Cost result

The measured arithmetic follows:

`C_state = 40 + 48M` MACs.

| M | Measured MACs | Scaling |
|---:|---:|---|
| 2 | 136 | O(M) |
| 4 | 232 | O(M) |
| 8 | 424 | O(M) |
| 16 | 808 | O(M) |
| 32 | 1,576 | O(M) |

Every state item is inspected. This is a direct linear scan, not an indexed or
sublinear address mechanism.

## What the experiment establishes

Under the registered synthetic semantic task, a learned content matcher can
retain a substantial advantage over a frozen random addressor as the persistent
state population increases from 2 to 32 items.

The result is strongest as a mechanism statement:

`semantic query + stored semantic value -> useful state address selection`

The fact that the learned arm remains well above 1/M chance at M=32 is useful
evidence that the relation is not simply collapsing immediately under a larger
state pool.

## What it does not establish

The absolute learned recall falls from approximately 0.99 at M=2 to 0.33 at
M=32. Therefore the result is not scale-invariant.

It does not establish:

- sublinear addressing;
- learned indexing;
- arbitrary-address memory retrieval;
- long-horizon semantic memory;
- natural-language memory retrieval;
- selective execution;
- C5 capability/computation parity.

## Architectural consequence

The evidence ladder now supports three distinct facts:

1. persistent state can cross a causal boundary;
2. the relevant state item can be selected from semantic content behind an
   opaque address;
3. the current implementation pays O(M) to discover that item.

The remaining architectural task is therefore not to prove that semantic
addressing exists. It is to change the cost of discovering it without losing
the capability already demonstrated.

## Next experiment

Build a real state index with a mandatory shortlist K << M.

The experiment should compare:

`full scan: C_address(M)`

against

`indexed: C_index(M) + C_route(K)`.

The index must preserve a held-out exact-recall endpoint and must be evaluated
against the same opaque addresses and same semantic query. Only after the index
maintains capability should C5 move to a matched capability-vs-compute test.

## Frozen evidence

REP-006, REP-005, REP-004, REP-003, REP-002, REP-001, and prior C5 evidence
remain separate historical measurements.