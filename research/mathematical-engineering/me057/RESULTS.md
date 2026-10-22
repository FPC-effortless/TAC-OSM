# ME-057 status

## Prefit result

The final checksum-locked 41-pair audit passed every prospective resource
gate.  No model was fitted during profiling.

| Quantity | Native TAC | GRU | Relative gap |
|---|---:|---:|---:|
| Trainable parameters | 38,268 | 38,295 | 0.071% |
| Profiled full-update FLOPs | 42,472,433 | 42,546,619 | 0.174% |
| Median optimizer CPU | 0.036939 s | 0.035860 s | paired median 2.91% |

The paired bootstrap 95% interval for TAC's CPU overhead is 2.13% to 4.33%,
inside the registered 5% ceiling.  ME-056's corresponding median was 8.01%.

## Training status

The architecture and five-seed, 600-update protocol are frozen.  Real-data
training is authorized by the passed preflight but has not yet produced an
accuracy result in the committed repository.  Until the workflow finishes,
there is no improved-intelligence, predictive-superiority or control claim.
