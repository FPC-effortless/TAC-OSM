# ME026 — equivalent native forecasts execute faster

The preregistered engineering gate **passed** for every native checkpoint:
full 128-event, batch-128 forecast inference is **1.266–1.329× faster** than its
original eager implementation on this CPU. Geometric mean speedup is 1.288×.
All six checkpoints and all nine paired repeats are retained. No training,
architecture selection or new outcomes were used to achieve this result.

| Frozen native checkpoint | Eager ms | Compiled ms | Same-work speedup | State bytes/entity |
|---|---:|---:|---:|---:|
| tac_rms / 14001 | 14.844 | 11.535 | 1.287× | 256 |
| tac_kernel / 14001 | 15.206 | 11.444 | 1.329× | 296 |
| tac_rms / 14002 | 14.928 | 11.692 | 1.277× | 256 |
| tac_kernel / 14002 | 15.102 | 11.775 | 1.282× | 296 |
| tac_rms / 14003 | 14.845 | 11.723 | 1.266× | 256 |
| tac_kernel / 14003 | 15.203 | 11.802 | 1.288× | 296 |

Same machine/process, float32, one Torch thread; three warmups and nine repeats
per implementation at batches 1, 32 and 128. Hardware: AMD EPYC 9V74 80-Core Processor.
Timing source and gate were pushed before execution at
`ecd7d973c10fd8bf021ed7161efa16f1f4943351`. All six numerical gates passed before
timing began. Raw CPU/wall samples, compile times, buffers and state sizes are
in recorded-runtime.json. Shared process peak RSS is not isolated inference
memory. Compilation/installation and training are not included in the
steady-state speedup. Results need remeasurement on other hardware/runtimes.

All **35,712 native exported full-window probabilities match the original
recorded predictions exactly**. All nine neural originals (including attention)
replay 53,568 predictions within 1.192093e-7. Streaming/window logit differences
on 32 genuine windows per native model are at most 4.768372e-7, below the frozen
1e-5 guard. All six fresh Torch-only processes pass; no Python model/research
module is imported in those processes. A corrupted export is rejected before
response loading. Replay after archive extraction is recorded separately.

## What changed mathematically

The implementation copies all original learned fast/slow matrices, gates and
readout weights. It supplies a complete incremental kernel-native forecast
API and a typed compiled graph. The fixed five-timescale summaries have an
associative affine segment representation, with identity, normalized impulse
derivatives and streaming equivalence checked. This algebra is separate from
the nonlinear native recurrence: no full-TAC associative scan or new SSM is
claimed, and no custom scan kernel was implemented.

Kernel-native state uses 296 bytes/entity versus 256 for RMS-only, plus a
separate 128-byte prior FIFO in either case and caller bookkeeping. The kernel
model also has 66,588 nontrainable buffer bytes, including its FIT-only item
popularity and precomputed coefficients; RMS-only has 8 buffer bytes. Learned
parameter counts remain 135,056 and 135,050 respectively. Constant state size
does not imply constant training memory or equivalence to resetting sliding
128-event windows on longer streams.

## Accuracy conclusion remains unchanged

ME025's five-summary head failed its registered predictive improvement gate.
This execution result does not overturn that failure, promote the kernel head,
or establish architectural superiority over Transformers. The original tiny
MLP reference lacks checkpoint/probability artifacts and is not independently
replayed. Existing repeated development evidence does not qualify independent
generalization, useful causal actions, commercial reliability or LLM-scale
capability. Reserved users/future outcomes remain closed; new fits = 0.

Six new mathematical/falsifier tests plus the original suite pass: 906 tests.
The active-head causal test detects a deliberately unmasked attention mutant;
it does not demonstrate an existing leak in the original masked model.
See MATHEMATICS.md, REPRODUCE.md and ARTIFACT.md for equations and records.
