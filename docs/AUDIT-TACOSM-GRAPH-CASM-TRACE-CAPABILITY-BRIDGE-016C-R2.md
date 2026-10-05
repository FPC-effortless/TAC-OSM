# G-CASM-016C-R2 — Corrected evidence-type bridge

Status: PREREGISTERED / CORRECTED REPEAT 2.

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

## Status of claims

C5 remains **UNTESTED**. R2 is a correction to the instrument, not a claim
result, and no claim moves until a corrected control actually runs and passes
its own invariant.

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
