# LANGDECODE057 — batch finite-state checks, identical TAC graph

Frozen before original trained-model measurement, 2026-10-29. Internally
measured engineering only, not independently qualified. Preserve 056 negative.

Hypothesis: replacing nine small finite-value reductions with one bounded
concatenation and reduction reduces original incremental CPU decode overhead.
Keep both input and output validation, model mutation/eval/hook checks, token
range, shape/dtype/device, clock, mode, owner identity and snapshot guards.
Invalid finite-state diagnostics still identify the original bad field.

Original reference: `IncrementalTAC043`, unchanged, inherited from TAC commit
`07617627906d51000b1d9a0932f6b598fcb81554`. The wrapper overrides **only**
`validate`; `_advance`, `step`, `prefill`, `save_state` and `load_state` are the
same Python methods. Same 976 coordinates / 3904 FP32 bytes per owner.
Temporary validation concatenation uses 3904 bytes per owner, is released each
call and is excluded from persistent-state accounting. No total RSS/VRAM claim.

Mathematical lineage: original LANGSTREAM043 PNDS PR66, commit
`cf14fea287c949aa5aea52f2123d65cce87e0aa4`, `PNDS/LangStream043.lean` and
`foundation/v1/langstream043.py`, pinned in the inherited PNDS_MATH_LINK.json.
The transition graph and source semantics are unchanged. For finite tensors
X1..X9 of the same dtype/device, all(isfinite(cat(flatten(Xi)))) equals the
conjunction of all(isfinite(Xi)). Constructed tests cover every nonfinite field.
This is an execution-wrapper check, no new variant, no claim of completed
Lean/source/IEEE/AD equivalence; inherited universal obligations remain OPEN.

Original checkpoint: LANGLAYER040 graph-free48 seed1701/320, SHA256
`0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`,
GitHub run40797105091 artifact LANGLAYER040-CPU-state48_no_graph-seed1701.
Strict source schedule/optimizer/RNG/checkpoint checks reuse 052 loader.
Historical 499623p model is replayed unchanged, never fitted or promoted.

Compute: bounded GitHub CPU, FP32 eager batch1, two torch threads, one interop,
no CUDA/GPU minutes. Run phone planner and no-fit scientific preflight first.
Same authentic already-exposed CPython3.13 FIT owner as 052 (first sorted owner
with >=544 bytes), prompt128/512; no new DEV/TEST queries, loss, rewards or fit.
Inherited source loader verifies all publisher files including already-exposed
DEV files for historical receipt consistency; only FIT bytes reach inference.

Prediction boundary: each arm sees the same observed prefix and original
incremental greedy continuation only. No target labels or future source suffix
reach decode. Explicit owner tuple is metadata, never model features. Check
each generated-byte logit and all nine persistent fields **bitwise** against
original incremental; check original full growing-prefix logits at every step
with frozen atol3e-5/rtol1e-4 and identical argmax. Save/load original-format
snapshots in both directions and compare the next transition bitwise.
Any parity, source, state-size, snapshot or checkpoint-mutation failure blocks
acceptance regardless of speed. Broader graph/AD/anti-cheat probes unrun remain
INCONCLUSIVE, not blanket PASS.

Timing: original and batched wrappers on the same model/host/source/16 tokens,
alternate order, two warmups and seven measured repeats. Record raw decode,
prefill, setup and validation-only timings separately; compare per-step median
and p95, and whole-repeat prefill+decode. Greedy choice and token construction
are identical and excluded from both decode timers. Validation guards remain
inside each timer. Run parity separately from timing. No benchmark retuning.

Accept the optimization only if every equivalence guard passes and median
per-byte latency improves >=5% at BOTH prompts, with p95 no more than10% worse.
Otherwise preserve as negative/inconclusive and do not merge/promote. This is
one host and one checkpoint, not mobile throughput or universal speed evidence.

Review before execution: original callers, snapshot/checkpoint identity and
all validation guards inspected; only finite reductions change. Independent
review and mathematical readiness are not implied. Draft PR remains unmerged.
