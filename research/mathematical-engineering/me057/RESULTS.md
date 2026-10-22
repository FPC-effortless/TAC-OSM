# ME-057 status

## Prefit result

The original checksum-locked 41-pair compiled audit passed every prospective
resource gate. No model was fitted during profiling.

| Quantity | Native TAC | GRU | Relative gap |
|---|---:|---:|---:|
| Trainable parameters | 38,268 | 38,295 | 0.071% |
| Profiled full-update FLOPs | 42,640,106 | 42,784,699 | 0.338% |
| Median optimizer CPU | 0.038346 s | 0.036971 s | paired median 2.17% |

The paired bootstrap 95% interval for TAC's compiled CPU overhead was 1.14% to
3.09%. ME-056's corresponding median was 8.01%.

## Execution amendment before any fit

GitHub Actions run 39858169855 verified all 228 FIT episodes, then PyTorch
2.14.1 Inductor aborted inside a generated recurrent transition kernel before
the first optimizer update. The failure was reproduced locally as an invalid
free and segmentation fault. The architecture, source, schedules, seeds,
objectives and parameter/FLOP budgets remain frozen; transition execution is
changed to stable eager PyTorch. CPU runtime is now reported as an efficiency
measurement under the portfolio-aligned master plan, rather than forced equal.
The eager preflight receipt below supersedes the compiled timing values.

| Eager execution quantity | Native TAC | GRU | Relative result |
|---|---:|---:|---:|
| Trainable parameters | 38,268 | 38,295 | 0.071% gap |
| Profiled full-update FLOPs | 42,640,106 | 42,784,699 | 0.338% gap |
| Median optimizer CPU | 0.096884 s | 0.063668 s | TAC 51.96% slower |

The paired bootstrap 95% interval for eager TAC overhead is 51.64% to 52.73%.
This is a material efficiency limitation and is retained as such; it does not
invalidate equal parameter, data, update and profiled-FLOP exposure.

## Completed authentic-data result

All ten 600-update checkpoints froze before DEV or TEST trajectories opened.
Actions run 39870099690 then evaluated the heldout cohorts but failed on its
last report write because NumPy produced an `int64` seed-win count. The archive
contains the frozen checkpoints. The exact run was reproduced locally from the
same commit, source hashes, schedules and seeds; its checkpoint hashes were
verified before rescoring. Rescoring used no refit.

Original-unit 16-step physical MAE, lower is better:

| Cohort | Native TAC | GRU | Persistence |
|---|---:|---:|---:|
| DEV, 76 episodes | 0.600738 | 0.587502 | not a registered comparison |
| TEST, 80 episodes | 0.663284 | 0.658526 | 0.672681 |

TAC improved over persistence by 1.40%, but was 0.72% worse than GRU and won
only one of five paired seeds. The episode-clustered 95% interval for TAC minus
GRU was +0.000285 to +0.009266 MAE, excluding zero in GRU's favor. The
registered accuracy gate therefore failed.

ME-057 is a bounded improvement over the earlier ME-052 failure because both
trained recurrent models now beat persistence on this task. It does not
qualify TAC predictive superiority, decision intelligence, policy value or
autonomous control. Eager TAC also used about 56% more optimizer CPU across the
completed local fits. The branch remains draft and unpromoted.

Aggregate scores and all ten local checkpoint hashes are preserved in
`research/me057-fused-anchor-completed-results.json`. The GitHub Actions
artifact is `12465724861`; its Azure-hosted bytes were unavailable to this
workspace, so the committed local hashes are not represented as an independent
byte-for-byte audit of that Actions archive.
