# TACOSM-STATE-REP-006 RESULT

## Run provenance

- clean registered run: workflow `36664979180`, run #8;
- code head: `b548642c397b4c9289761e65f11bd4327ccec86b`;
- full test gate: **841 passed**;
- contract/precondition gate: **passed**;
- registered artifact: `TACOSM-STATE-REP-006`, artifact id `11075093261`;
- corrected instrumentation rerun: subsequently opened after a secondary-endpoint audit; it is not folded into this primary result until it completes.

## Primary endpoint

State pool size was fixed at 8, with opaque randomized addresses. The query
contained the five-bit semantic requirement but no target address.

| Seed | Learned B state Top-1 | No-learning C state Top-1 | B − C |
|---:|---:|---:|---:|
| 0 | 0.5117 | 0.0156 | +0.4961 |
| 1 | 0.7422 | 0.2188 | +0.5234 |
| 2 | 0.2695 | 0.0156 | +0.2539 |
| 3 | 1.0000 | 0.4883 | +0.5117 |
| 4 | 1.0000 | 0.1914 | +0.8086 |
| **Pooled** | **0.7047 (902/1280)** | **0.1859 (238/1280)** | **+0.5188** |

Chance with eight state items is 0.125.

The learned arm exceeded the no-learning control in **all five seeds**. The
pooled absolute improvement is **51.88 percentage points**.

## Representability and interface controls

The analytic identity witness achieved state Top-1 **1.000** on every seed
and end-to-end structural selection **1.000**.

The registered state read inspected exactly **8 items** in every decision.
The learned addressor's arithmetic cost was **424 MACs** per decision at
M=8.

The reset intervention failed closed on all five seeds. The value-shuffle
control preserved the opaque address set on all five seeds.

These controls establish that the primary effect is not caused by a known
address being supplied to the router or by an unavailable state item.

## Secondary endpoint audit

The first clean run reported `end_to_end_top1_recall = 0.0` even on seeds where
`state_top1_recall = 1.0` and reported impossible target-rank behavior. Code
audit found the cause: `semantic_match_rank()` returns the rank of the selected
argmax candidate, which is necessarily 1, rather than the rank of the declared
target candidate.

Therefore the first run's **secondary end-to-end rank/recall fields are
quarantined** and are not used for interpretation.

A separate exact `semantic_target_rank()` diagnostic and corrected runner have
been added without changing the registered task or primary endpoint. The
correction is instrumentation-only.

## Interpretation

REP-006 supports the bounded mechanism claim:

`opaque state address + semantic query -> learned state-item selection`

can work on the registered synthetic task, and the observed gain is not a
single-seed artifact.

The result is materially stronger than REP-005's known-address condition because
the router must now identify which persistent item contains the relevant
semantic state.

However, the experiment is still small:

- 8 state items;
- one-step temporal delay;
- five-bit synthetic semantics;
- exhaustive O(M) state scan;
- no natural-language memory;
- no long-horizon memory load.

It therefore does not establish general semantic memory or scalable addressing.

## Compute

State addressing cost at M=8:

`C_state_address = 8*5 + 8*(8*5 + 8) = 424 MACs`

before dictionary reads, Python overhead, and comparisons.

The implementation still scans all M state items, so the addressing stage is
**O(M)**. This is not a sublinear retrieval result.

## Research decision

The main architectural bottleneck has shifted again:

`candidate representation` -> solved for this controlled topology task;
`persistent transport` -> demonstrated;
`semantic state addressing` -> demonstrated at M=8;
`learned addressing quality at scale` -> open;
`sublinear address/retrieval` -> open;
`selective execution vs H` -> open.

The next experiment should vary **M** while holding task entropy and semantic
difficulty fixed, then introduce a genuinely selective index. The required
measurement is:

`C_total = C_address(M) + C_route(K) + C_execute(|R|) + C_verify`

versus a conventional `C_full(H)` baseline at matched capability.

C5 remains **UNTESTED**.

## Frozen evidence

REP-005, REP-004, REP-003, REP-002, REP-001, and prior C5 artifacts remain
separate historical measurements.