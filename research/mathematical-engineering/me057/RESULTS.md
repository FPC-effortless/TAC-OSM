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

## Training status

The architecture and five-seed, 600-update protocol are frozen.  Real-data
training is authorized by the passed numerical parity preflight but has not yet produced an
accuracy result in the committed repository.  Until the workflow finishes,
there is no improved-intelligence, predictive-superiority or control claim.
