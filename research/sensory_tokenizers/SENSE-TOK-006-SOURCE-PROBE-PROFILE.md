# SENSE-TOK-006: original DEV source, native probe and partial resource evidence

Status: zero model fits and optimizer steps. No sealed TEST access or GPU usage.
This extends preparation with actual publisher-matched DEV measurement.

## Original source evidence

The CPython v3.13.5 annotated tag `4a6ce4b7a462d5da01b8b66f769224a4a0b1de17`
resolves to the existing pinned implementation commit
`6cb20a219a860eaf687b2d968b41c480c7461909`. All six original files match
previously pinned Git blobs: 321,075 bytes total. The source receipt records
whole-file SHA256s and the original LICENSE hash. These are exposed DEV files;
no bytes were reassigned to FIT or sealed TEST. Publisher content equality
alone is not signed source/goal/rights/ACL qualification or holdout custody.

## Actual native-state probe

`dense_byte_adapter()` registers the dense BYTE model's actual 118-wide GRU
state and 32-wide current-byte representation with ResearchProbeAdapter.
Every streamed forecast matches the actual dense source forward at each tested
position. State interventions, pure readout, actual readout-gradient dependency
and entity isolation pass the local fixtures. The adapter explicitly rejects
S004/S005 and label-bearing samples. Candidate permutation is unsupported and
reported INCONCLUSIVE. Full adaptive pending-token/codebook state adapters
remain OPEN; dense coverage must not be generalized to them.

## Actual partial profile

One untrained seed, four original exposed DEV prefixes, 256 bytes each, CPU
single-thread forward plus backward, zero optimizer construction. Random
control probability .5 is an unqualified diagnostic fixture. These are FLOPs
reported by supported profiler operators, not full training FLOPs.

| Candidate | Counted operator FLOPs |
| --- | ---: |
| Dense BYTE | 562,571,688 |
| Forced TOKEN | 490,477,804 |
| Adaptive continuous latent | 717,052,013 |
| Adaptive discrete codebook | 720,169,449 |
| Frozen gate weights | 719,812,273 |
| Random gate | 708,315,253 |

Counted six-arm relative spread: 31.8941%. Adaptive continuous/discrete spread:
0.4329%. Neither establishes full compute parity: 93–150 operator categories
per arm have no FLOP count. Those lists include non-arithmetic bookkeeping
as well as genuinely uncounted numerical operations. A zero profiler count
is not evidence that an operation is free. Actual backward behavior also
varies with learned routing and cannot be inferred from one initialization.

The partial records do not support a six-arm matched-compute claim. Next,
complete arithmetic coverage on source-authorized representative FIT batches
and across routing regimes; then determine whether the primary adaptive pair
can meet all strict resource requirements. Do not silently delete controls,
pad dummy work, alter sample exposure, or promote these partial counts into
approved budgets. Any narrowed study needs a separate reviewed registration.

Recorded wall time includes profiler instrumentation, excludes later key-average
analysis, and was not isolated from other local checks. It is not inference
latency, a speedup result, or a CPU-time fairness gate. The resource receipt
keeps full training compute eligibility INCONCLUSIVE. Checkpoint training,
full graph/precision/gradient proof, complete adaptive probe coverage, external
signatures, sealed holdouts and protected supervised execution remain OPEN.

## Reproduction and checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/profile_sense_tok_source.py --source-root /path/to/cpython-v3.13.5/Lib --output partial-profile.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

All six original whole-file blob checks run before profiling. Three new
no-fit source-adapter cases pass. Complete local suite: 1,311 passed.
The source and resource JSON receipts accompany this report; no source
plaintext or runtime profiler trace is redistributed. No capability score,
learned representation claim, or training authorization follows from it.
