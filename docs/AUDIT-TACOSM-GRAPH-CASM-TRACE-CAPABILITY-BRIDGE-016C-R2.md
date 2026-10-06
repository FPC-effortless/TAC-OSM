# G-CASM-016C-R2 — Corrected evidence-type bridge

Status: MEASURED — confirmatory artifact accepted; see audit findings below.

R2 supersedes 016C-R1, which is **invalidated**. R1 is the second consecutive
failure of this bridge experiment, and both failures have the same root cause:
candidate evidence and target evidence are extracted by two separate code paths
that disagree about type.

## What was actually wrong

In `run_graph_casm_trace_capability_bridge_016c_r1.py`, for the scalar arm:

```python
evidence = tuple(cache.scalar[(idx, row)][0] for idx in indices)   # ints
target_evidence = (cache.scalar[(task.target_index, row)][0],)     # a 1-tuple
```

`cache.scalar[(i, row)]` is `tuple[int, int]`, so `[0]` is an `int`. The
candidate sequence is therefore a tuple of ints and the target is a 1-tuple
containing one int. `compatible_bucket` compares them with `==`, and
`int == (int,)` is always `False`.

Every scalar compatible bucket was empty by construction.

This is not a subtle statistical artefact. It is a deterministic property of
the code path, and it held for all 800 scalar trials in the R1 artifact.

## The R1 artifact confirms it

Run `37326642412`, artifact `TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1`:

| M | scalar `target_bucket_size` | scalar `verified_success` | trace `verified_success` |
|---|---|---|---|
| 32 | 0 | 0.0000 | 1.0000 |
| 64 | 0 | 0.0000 | 1.0000 |
| 128 | 0 | 0.0000 | 1.0000 |
| 256 | 0 | 0.0000 | 0.9938 |
| 512 | 0 | 0.0000 | 0.8313 |

The distinct values of `target_bucket_size` and `shortlist_size` over all
scalar trials are `{0}`.

R1's reported primary endpoint was `mean_paired_delta = 0.83125`, bootstrap CI
`[0.78125, 0.89375]`. **That number is a comparison against an arm that was
structurally absent**, not a measured channel difference. The CI's tightness is
irrelevant: one side of the difference is identically zero by construction.

## The correction was never made

R1 was registered as a "corrected repeat" of 016C, and its contract states the
required regression in full:

> candidate scalar signatures and realized scalar target signature must have
> identical tuple type and at least one compatible candidate.

The diff between the 016C and R1 runners is an identifier rename and a filename
change. Nothing else. The relevant lines are byte-identical, so R1 inherited
016C's defect rather than fixing it. The contract described a correction that
the code never contained.

## Why the regression test did not catch it

R1's `tests/test_trace_capability.py::test_scalar_target_evidence_signature_matches_candidate_signature`
asserted:

```python
candidate_evidence = ((0,), (1,), (0,))
target_evidence = (0,)
assert all(isinstance(x, tuple) for x in candidate_evidence)
assert isinstance(target_evidence, tuple)
assert sum(sig == target_evidence for sig in candidate_evidence) == 2
```

Every value is hand-written. The test asserts that a tuple of tuples compares
against a tuple, which is true on the literal and false in the runner. Type
identity was checked against constants, never against generated evidence.

R2 replaces this with a runner-level invariant enforced on every trial, plus
offline tests against the real generator.

## The scalar partition itself was correct

The failure was confined to target extraction. R1's scalar
`budget_capped_utility` matched the analytic partition ceiling exactly at every
M:

| M | measured | analytic `2*min(8, M/2)/M` |
|---|---|---|
| 32 | 0.50000 | 0.50000 |
| 64 | 0.25000 | 0.25000 |
| 128 | 0.12500 | 0.12500 |
| 256 | 0.06250 | 0.06250 |
| 512 | 0.03125 | 0.03125 |

`budget_capped_utility` is computed from the candidate evidence alone, which is
well typed, so the partition was right. Only the target lookup was wrong. That
is consistent with the observed `target_bucket_size = 0` and non-zero
`budget_capped_utility`: the bucket selection was broken, the underlying
partition was not.

## The trace arm was valid descriptive evidence

The trace channel did not have the defect — its target evidence is a bare tuple,
matching its candidates. On its own terms it measured something real:

- trace success 1.000 at M ∈ {32, 64, 128}, 0.9938 at 256, 0.8313 at 512;
- against its own preregistered `budget_capped_utility` ceiling of 0.84805 at
  M=512, a gap of 0.0168;
- bucket ≤ 8 → success 70/70 = 1.000; bucket > 8 → 63/90 = 0.700.

This supports "trace permits highly effective target partitioning, approaching
its information ceiling." It does **not** support "trace outperforms scalar",
because that comparison was never measured. R2 measures it.

## The second defect: accounting

Independent of the type mismatch, `accounted_total_work_units` did not measure
total work. It summed the terminal verifier work with `best.expected_cost`,
which `action_scores` computes as `statistics.fmean(...)` over candidates — a
*per-candidate* mean, O(1) in M.

Meanwhile `action_scores` actually iterates `for idx in indices` over all M
candidates for each probe row to build signatures. That real O(M) scan was
excluded from the accounting. The measured
`probe_environment_work_units` was therefore flat at ≈16.9 for scalar across all
M, while the true acquisition scan grew linearly.

The recorded `verified_execution_work_fraction` (0.032 → 0.0028 as M went 32 →
512) is legitimate evidence for **selective execution** scaling — the verifier
touches fewer candidates as the shortlist shrinks. It is not evidence for
**total computation** scaling, because the probe acquisition cost was omitted.

R2 records probe work as a total (`per_candidate_mean × M`) and adds explicit
per-trial operation counters: signature construction, candidate scan, probe
selection, execution, verification and their sum. The per-candidate mean is
retained as a separate endpoint for comparability with G-CASM-015.

This distinction matters for C5. The claim is that execution cost scales with
the relevant subset, `C_execution ≈ O(r)`, not with total history `O(H)`. That
is a statement about total computation, so an accounting that omits the
dominant O(M) term cannot support it.

## What R2 changes

1. One typed extraction path per channel. `candidate_evidence` and
   `target_evidence` are separate functions that both index the cache the same
   way, so the arms cannot diverge in shape.
2. A runner-level invariant on every generated trial:
   `type(e_i) == type(e_t)` **and** `|{i : e_i == e_t}| >= 1`. The second
   condition matters because uniform typing alone could still yield an empty
   compatible bucket. Preflight fails closed on either. The invariant lives in
   `tac_osm.trace_capability` so it is testable without torch.
3. Probe work as a total, plus per-trial operation counters.
4. The R1 literal-only regression test is replaced by tests against the real
   generator, and the R1 failure mode is pinned as an explicit negative test.

No protocol parameter changes: same seeds, M levels, B=8, channels, frozen
generator commit and primary endpoint.

## Smoke verification

The smoke/preflight job passed on run `37354913663` (3m21s, peak RSS 524.6 MB,
status `measured`, tacosm commit `78f93b7`). It is 4 trials per arm at
M ∈ {32, 128}, so it establishes that the instrument is fixed — it is **not**
the confirmatory measurement.

The type defect is gone. Distinct scalar `target_bucket_size` values are
`{16}` at M=32 and `{63, 65}` at M=128. R1 recorded `{0}` for all 800 trials;
R2 records no zero anywhere.

| M | scalar bucket | scalar `budget_capped_utility` | scalar success | trace bucket | trace success |
|---|---|---|---|---|---|
| 32 | 16.0 | 0.50000 | 0.2500 | 1.0 | 1.0000 |
| 128 | 63/65 | 0.12500 | 0.0000 | 1/3 | 1.0000 |

The scalar arm is now doing real work: success 0.25 at M=32 and 0.0 at M=128,
against an analytic partition ceiling of `2·min(8, M/2)/M` = 0.5 and 0.125
respectively. The `budget_capped_utility` endpoint still matches that ceiling
exactly, as it did in R1 — confirming again that the partition was never the
broken part.

The accounting fix is visible too. Scalar `probe_environment_work_units` now
scales with M (550.0 → 2153.0) instead of sitting flat at ≈16.9, while the
per-candidate mean is retained (17.19 → 16.82). Per-trial operation counters
at M=32 / M=128, both arms:

| counter | scalar 32 | scalar 128 | trace 32 | trace 128 |
|---|---|---|---|---|
| signature construction | 512 | 2048 | 512 | 2048 |
| candidate scan | 512 | 2048 | 512 | 2048 |
| probe selection | 1 | 1 | 1 | 1 |
| execution / verification | 481.75 | 340.0 | 276.0 | 305.75 |
| total | 1506.75 | 4437.0 | 1301.0 | 4402.75 |
| `verified_candidates` | 7.0 | 8.0 | 1.0 | 1.75 |

The acquisition scan (signature construction + candidate scan) is now the
dominant accounted term at 2·M per trial, exactly the O(M) cost R1 omitted.
Probe selection is O(1) — one decision per probe row, independent of M — which
is the C6/C8 structure. `verified_candidates` tracks the bucket: scalar reaches
the B=8 cap at M=128, trace stays near 1–2. That contrast is the intended
selective-execution signal.

`verified_execution_work_fraction` is 0.0547 → 0.0099 for scalar as M goes
32 → 128, consistent with the R1 direction (0.032 → 0.0028) but now backed by
a total-work denominator that includes the scan.

Two caveats that keep this from being a result. Four trials cannot resolve the
primary endpoint `verified_success_delta_m512_b8`, which needs M=512. And
scalar success 0.0 at M=128 is a floor on 4 trials, not an estimate. The
full confirmatory job (`[run-graph-casm-trace-capability-bridge-016C-R2-full]`)
runs the preregistered seed grid and is the only run that can move a claim.

## Confirmatory result

Run `37382071406`, artifact `TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2`
(tacosm commit `ccc32b9`, generator `c315544`, status `measured`). This is the
preregistered grid: 5 seeds × 5 M levels × 32 tasks per M × 2 arms = **800
trials**. The smoke artifact is a separate upload and is not pooled into any of
these statistics.

### Validity gates

All gates are fail-closed and were checked over every one of the 1600 channel
records.

| gate | result |
|---|---|
| G1 — `target_bucket_size ≥ 1` every trial/channel | **PASS**. No zero anywhere. Distinct scalar bucket values at M=512 are `{255, 256, 257}` |
| G2 — evidence-type invariant, asserted per trial | **PASS**. Runner asserts uniform typing and target presence; leakage flags clean across all 5 seeds |
| G3 — `shortlist_size == min(8, target_bucket_size)` | **PASS**, 1600/1600 |
| G4 — accounting identity | **PASS** under the runner's own convention, with a finding (below) |
| G5 — protocol fidelity | **PASS**. M, seeds, B, eval_steps, channels and generator identical to the contract and to parent 015 |
| G6 — acquisition included in the total | **PASS**. Acquisition is 72–98% of `total_operations` |

The R1 failure mode is definitively absent. R1 recorded `target_bucket_size`
of `{0}` across all 800 scalar trials; R2 records no zero in any of the 1600
records.

### Primary endpoint

`verified_success_delta_m512_b8` — seed-level paired delta, M=512, B=8:

| seed | scalar | trace | delta |
|---|---|---|---|
| 0 | 0.0000 | 0.7812 | +0.7812 |
| 1 | 0.0000 | 0.7812 | +0.7812 |
| 2 | 0.0000 | 0.8750 | +0.8750 |
| 3 | 0.0000 | 0.7812 | +0.7812 |
| 4 | 0.0000 | 0.9375 | +0.9375 |
| **mean** | **0.0000** | **0.8313** | **+0.8313** |

Every seed is positive. This is the number R1 reported (0.83125) but here, for
the first time, **both arms are real**.

Verified success by M:

| M | scalar | trace | delta |
|---|---|---|---|
| 32 | 0.4938 | 1.0000 | 0.5062 |
| 64 | 0.2500 | 1.0000 | 0.7500 |
| 128 | 0.1125 | 1.0000 | 0.8875 |
| 256 | 0.0875 | 0.9938 | 0.9062 |
| 512 | 0.0000 | 0.8313 | 0.8313 |

The scalar arm is no longer broken: it recovers the expected finite-information
behaviour, succeeding 49.4% at M=32 and decaying monotonically to 0 at M=512.
R1's 0.0 at every M was the defect; R2's 0.0 at M=512 is the real information
limit of a single binary observation.

### Computation decomposition

C_total = C_acquisition + C_probe_selection + C_execution + C_verification,
with each term's mean over 160 trials per cell:

| term | scalar 32→512 | slope | trace 32→512 | slope |
|---|---|---|---|---|
| acquisition (`signature_construction` + `candidate_scan`) | 1024 → 16384 | **1.000** | 1024 → 16384 | **1.000** |
| probe selection | 1 → 1 | 0.000 | 1 → 1 | 0.000 |
| execution | 397 → 313 | −0.073 | 279 → 385 | 0.114 |
| verification | 397 → 313 | −0.073 | 279 → 385 | 0.114 |
| **total** | 1422 → 16698 | **0.894** | 1304 → 16770 | **0.924** |
| `verified_candidates` | 6.34 → 8.00 | 0.078 | 1.17 → 4.95 | 0.510 |

`C_acquisition ~ O(M)` exactly, both arms. Selection is exactly `O(1)`.
Execution is flat or negative — genuinely selective, not scanning history.
But `C_total ~ O(M^0.89)` / `O(M^0.92)` because acquisition dominates.

**This is the decision-relevant contrast and it resolves C5.**

### What R2 establishes

The clean causal chain from G-CASM-015 is now complete on its first two links:

- 015: activation trace → more informative observation channel
- R2: more informative structured evidence → higher verified selective capability

Trace reaches 0.8313 verified success at M=512 where scalar reaches 0, at
approximately comparable **recorded** `total_operations` (16770 vs 16698).

**The work-normalized interpretation is provisional.** `probe_environment_work_units`
is recorded but is *outside* `total_operations`, and the two arms have identical
acquisition counts. The correct current wording is "approximately comparable
recorded total_operations", not "comparable total work", because environment-side
acquisition/probe work is not yet inside the accounting boundary. If that work
later differs between channels it could change the ratio. See the G4 finding
below.

### Audit finding: `U_B` is not a first-B success probability

The information-theoretic ceiling remains a valid upper bound, but `U_B` is not
an exact predictor of first-B verified-success probability because the execution
rule depends on ordered target rank within the compatible bucket. Observed
deviations occur in both scalar and trace arms, indicating generator/order
interaction rather than a scalar-specific implementation defect.

Concretely, `verified_success` requires the target to sit among the **first B**
entries of the bucket — `shortlist()` returns `bucket[:budget]` — whereas
`budget_capped_utility` is the uniform-prior utility
`sum_e min(B, |H_e|) / M`. The first is an order-dependent realization; the
second is a prior expectation. They should not be equated:

- **counting ceiling:** `P(success) ≤ min(1, qB / M)` where q is the number of
  distinguishable evidence classes;
- **uniform-prior utility:** `U_B = sum_e min(B, |H_e|) / M`.

Deviations of `verified_success` from `min(B, bucket)/bucket`, both arms:

| M | scalar bucket | `min(B,b)/b` | scalar success | deviation | trace deviation |
|---|---|---|---|---|---|
| 32 | 16.0 | 0.500 | 0.4938 | −0.006 | 0.000 |
| 64 | 32.1 | 0.250 | 0.2500 | 0.000 | 0.000 |
| 128 | 64.0 | 0.125 | 0.1125 | −0.013 | 0.000 |
| 256 | 128.0 | 0.0625 | 0.0875 | **+0.025** | −0.003 |
| 512 | 256.0 | 0.0312 | 0.0000 | **−0.031** | −0.017 |

The deviations **flip sign** and grow with bucket size. Under a uniform target
rank they would be ≈0, so the target's rank within the compatible bucket is not
uniform. At M=512 the scalar target ranks consistently late — 0 successes
against 5 expected (p ≈ 0.006); at M=256 it ranks consistently early.

The trace arm shows the same effect, so this is a property of the generator's
candidate ordering interacting with first-B truncation, not a scalar-arm defect.
It does not weaken the capability result — the delta is 0.8313 either way — but
the scalar control is at or below its information ceiling, not exactly at it.

### Audit finding: G4 accounting convention

The runner records `execution_operations` and `verification_operations` as the
**same** `verifier_work` value (lines 168–169), because `verify_shortlist`
performs both in a single pass and returns on the first exact match. The
recorded `total_operations` therefore counts that work once:

```
total = signature_construction + candidate_scan + probe_selection + verifier_work
```

This identity holds exactly on all 1600 records (G4 passes under this
convention). The two fields are not independently measured cost terms, and the
five-term sum is not what the artifact records.

The headline exponent is sensitive to this choice:

| channel | convention A (runner, 1× verifier) | convention B (five-term, 2×) |
|---|---|---|
| scalar_row | 0.894 | 0.814 |
| activation_trace | 0.924 | 0.863 |

The qualitative conclusion is invariant — both are sublinear in the total but
acquisition-dominated, and `C_acquisition ~ O(M)` exactly in both. The
**exponent is not final**. Any quantitative complexity statement must settle
the convention first, or report both side by side.

**Disposition: audit finding only.** The R2 artifact is left immutable and is
not amended retroactively. Accounting cleanup — genuinely distinct cost terms
or an explicit nesting definition, a corrected identity, and regression tests —
is a separate PR, and a fresh confirmatory artifact is required if corrected
accounting is later used as quantitative evidence. This finding does not
explain the scalar/trace capability difference and does not affect the primary
endpoint.

### Interpretation limits

Three limits bound what R2 may be read as establishing.

1. **Execution flatness is partly by construction.** `shortlist()` truncates to
   `bucket[:B]`, so execution and verification are capped at B=8 by G3 itself.
   The measured flat/negative execution slopes are therefore not an independent
   asymptotic scaling law. The finding is *capability under cap*: trace reaches
   0.8313 while execution+verification stays flat at ≈279–385 units.
2. **Acquisition is scan-based, not necessary.** Each candidate's 16-row truth
   table is recomputed on every trial and charged to the query; the runner
   defines `candidate_scan_operations = M · len(INPUT_ROWS)` with
   `signature_construction_operations` identical. Indexed acquisition is
   untested. "Acquisition scales with the full candidate set" is a statement
   about this implementation, not about all possible designs.
3. **A fixed 64-class trace channel does not scale as configured.** Its counting
   ceiling is `64·B/M`, i.e. 12.5% at M=4096. Holding B fixed likely requires
   roughly `log2(M/B)` bits of evidence. An equal-evidence-bits arm — scalar
   with 6 rows — is not in this artifact.

## Status of claims

**C5 — NOT SUPPORTED in its original end-to-end formulation.** The experiment
demonstrates a B-capped selective execution/verification regime, but scan-based
acquisition remains `O(M)` under the current accounting convention and dominates
total measured operations. Indexed or learned sublinear acquisition was not
tested. This is cleaner than "partially supported", because the original C5
hypothesis was explicitly about *total* computation. R2 provides a decomposition
showing *where* the hypothesis fails — execution is genuinely selective, the
total is not sublinear — rather than partial confirmation of the original claim.

The capability result is separate from C5 and is not affected by it: structured
activation-trace evidence produces substantially higher exact verified selective
success than the scalar channel under the same M=512/B=8 protocol.

## Provenance

| field | value |
|---|---|
| supersedes | TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1 |
| invalidated artifact | TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1 (run 37326642412) |
| invalidated artifact | TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C (never produced a valid artifact) |
| parent experiment | TACOSM-GRAPH-CASM-STRUCTURED-ACTION-PROBE-015 |
| generator commit | `c31554413301e3c9d3e6b3f8c8c6be572a74a748` |
| primary endpoint | `verified_success_delta_m512_b8` |
| contract | `contracts/TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2.json` |
| runner | `scripts/run_graph_casm_trace_capability_bridge_016c_r2.py` |
| invariant | `src/tac_osm/trace_capability.py::assert_evidence_invariant` |
| tests | `tests/test_trace_capability.py` |


<!-- Confirmatory rerun trigger: [run-graph-casm-trace-capability-bridge-016C-R2-full] -->


Confirmatory trigger: [run-graph-casm-trace-capability-bridge-016C-R2-full]
