# TACOSM-SUCCESSOR-REP-001 — RESULT

**Status: FAILED — falsifier triggered. See Verdict.**

This is the successor diagnostic defined by
[REP-001](TACOSM-SUCCESSOR-REP-001.md). It does not modify, replace, or
reinterpret C5-001, C5-002, or C5-003. C5 evidence files are unchanged.

## Provenance

| Item | Value |
|---|---|
| Artifact | `artifacts/TACOSM-SUCCESSOR-REP-001.json` |
| Branch | `successor/arch-v1` |
| Commit | `6062ebd` |
| Run | `36651165861` (REP-001 successor diagnostic) |
| Seeds | 5 (0,1,2,3,4) |
| Train / held-out episodes per seed | 128 / 128 |
| Candidates (N) | 8 |
| Executor | exact |
| Top-K | 1 |
| Dimension / latent | 8 / 8 |
| Task families | relational, state_lookup, replay |

Chance for N=8 with Top-K=1 is **1/8 = 0.125**.

## Raw results

Mean over 5 seeds, held-out split. Success = `decision.selected == target`.

| Condition | relational | state_lookup | replay |
|---|---|---|---|
| A analytic (Top-1 recall) | **1.000** | **1.000** | **1.000** |
| B learned (Top-1 recall) | 0.177 | 0.200 | 0.200 |
| C no-learning (Top-1 recall) | 0.186 | 0.177 | 0.191 |
| D oracle (episode success) | 1.000 | 1.000 | 1.000 |

Exact execution success rate is **1.000 in every condition, family, phase,
and seed**. The executor is not a source of error in any of these runs.

Condition A additionally reports a mean hard-negative margin of approximately
2.0 (relational 2.025, state_lookup 2.025, replay 2.0), i.e. the analytic
witness separates the target from the in-pool hardest negative by a large
margin in every family.

## Verdict

**The diagnostic FAILS. Falsifier 2 is triggered:**

> learned routing does not exceed the frozen no-learning baseline on held-out
> Top-1 rank/recall

Learned routing (B) does not exceed the no-learning baseline (C) on held-out
Top-1 recall:

| Family | B | C | delta | Higher |
|---|---|---|---|---|
| relational | 0.177 | 0.186 | -0.009 | C |
| state_lookup | 0.200 | 0.177 | +0.023 | B |
| replay | 0.200 | 0.191 | +0.010 | B |

The sign of B - C is inconsistent across task families, and the magnitudes
(|delta| <= 0.023) are within seed noise. Averaged over all three families,
B = 0.192 and C = 0.185, a difference of +0.007 — far smaller than the
per-seed spread.

Per-seed held-out Top-1 recall for the relational family, condition B, is
`[0.186, 0.209, 0.209, 0.186, 0.093]`. This scatter straddles chance (0.125)
with no consistent direction, which is the signature of a router that is not
learning the relevance relation at all rather than one that is learning it
slowly.

The total router update count over 128 training episodes per seed is
`[17, 27, 21, 23, 21]` — i.e. the router performs on the order of 20
parameter updates across 128 episodes, and still fails to move held-out
performance above the untrained baseline.

No other falsifier is triggered by these data (see Precondition audit below).

## Interpretation

The three observations that matter:

1. **B ~= C.** Learning the router representation produces no measurable
   benefit over the untrained router. This is not a slow-convergence or
   hyperparameter effect: the update budget is small (about 20 updates per
   128 episodes, because the router only updates on outcome success), and the
   held-out rank distribution does not move in a consistent direction.

2. **A = 1.0 while B and C ~= 0.19.** The analytic witness solves routing
   exactly (Top-1 recall 1.000, hard-negative margin ~2.0) using the same
   energy router with an imposed analytic relation. The learned
   representation does not reach it. The relation is therefore representable
   in this architecture but is not recoverable from the observed signals.

3. **Exact execution = 1.000 everywhere.** The execution plane is solved by
   construction in this diagnostic, so the entire failure is located in the
   selection plane. Every error is a candidate-selection error, not an
   execution error.

Together these say the failure is **not** an optimization failure. The
target relation is representable (condition A) and the executor is exact
(1.000 everywhere), but the signals the router can observe are insufficient
to distinguish the correct candidate from the in-pool negatives. This is the
identifiability failure identified in the C5-003 audit: the routing input is
derived from observables that do not determine the true wiring, so episodes
with different ground truth can be indistinguishable to the router.

The analytic witness succeeds precisely because it is *not* restricted to
those observables — the relevance relation is imposed rather than inferred,
which supplies the information the learned path cannot recover.

## Precondition audit

Preconditions for a valid run, from [REP-001](TACOSM-SUCCESSOR-REP-001.md):

- **repository CI passes the pre-model gates** — PASS. The REP-001 run
  completed successfully and the paired CI run on the same commit also
  passed.
- **the constructive representability test passes** — PASS. Condition A
  achieves Top-1 recall 1.000 with hard-negative margin ~2.0 in all three
  task families.
- **exact executor tests pass** — PASS. Exact execution success rate is 1.000
  in every condition/family/phase/seed, and the paired CI run passed.
- **frozen legacy reward tests pass** — PASS. No C5 or legacy evidence file
  was modified on this branch.
- **baseline test-count provenance current** — PASS on the `successor/arch-v1`
  branch; the top commit on the branch keeps the frozen baseline test count
  unchanged.
- **C5 evidence files are unchanged** — PASS. No C5-001/C5-002/C5-003
  artifact or document was modified by this branch. C5 evidence remains
  frozen.

## Secondary metrics

Held-out mean target rank (lower is better; chance is 4.5 for N=8):

| Condition | relational | state_lookup | replay |
|---|---|---|---|
| B learned | 4.59 | 4.40 | 4.58 |
| C no-learning | 4.65 | 4.39 | 4.63 |
| D oracle (rank of target) | 4.47 | 4.59 | 4.48 |

Condition D reports episode success 1.000 because the oracle selects the
target by construction. Its target rank (~4.5) is the honest number: even
knowing the target, the router energy places the target at chance rank,
exactly as in conditions B and C. The oracle arm is specified as outside the
router information boundary precisely so this distinction is visible.

The exact-execution result and the condition-A margin together isolate the
failure to the selection plane: the capability exists in the architecture
(A), the executor honours it exactly (1.000), but the learned router cannot
select for it (B ~= C ~= chance).

## Non-claims

Consistent with the preregistration, this result does not establish or
refute:

- general intelligence;
- compositional generalization;
- sublinear retrieval;
- economic superiority over Transformers;
- learned sparse execution;
- long-horizon planning;
- PLM validity.

Nor does it establish that the successor architecture cannot work. It
establishes that the specific learned router in Successor Architecture v1
does not learn the relevance relation under this protocol, and that the
defect is in the information available to the router rather than in the
executor or the optimizer.

## Next diagnostic

The failure points at the router's observation set, not at training. The
next step is to make the target relation identifiable to the router rather
than to tune the optimizer: route over an explicit candidate edge set so
that episodes with different true wiring are distinguishable, then re-run
this diagnostic under the same seeds and protocol.

The scaling extension (N and state-slot population sweeps) remains deferred,
as the preregistration requires the fixed diagnostic to pass first.
