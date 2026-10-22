# ME-057 status

## Prefit result

The final checksum-locked 41-pair audit passed every prospective resource
gate.  No model was fitted during profiling.

| Quantity | Native TAC | GRU | Relative gap |
|---|---:|---:|---:|
| Trainable parameters | 38,268 | 38,295 | 0.071% |
| Profiled full-update FLOPs | 42,640,106 | 42,784,699 | 0.338% |
| Median optimizer CPU | 0.038346 s | 0.036971 s | paired median 2.17% |

The paired bootstrap 95% interval for TAC's CPU overhead is 1.14% to 3.09%,
inside the registered 5% ceiling.  ME-056's corresponding median was 8.01%.

## Training status

The architecture and five-seed, 600-update protocol are frozen.  Real-data
training is authorized by the passed preflight but has not yet produced an
accuracy result in the committed repository.  Until the workflow finishes,
there is no improved-intelligence, predictive-superiority or control claim.
