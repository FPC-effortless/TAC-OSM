# ME-060 completed development and frozen transfer

All ten original 600-update models trained successfully. The full supported-
operator resource preflight and final estimate audit passed; true all-operator
FLOPs remain uncertified. Every checkpoint hash and 207,340 standardized
absolute errors replayed exactly. No old TEST trajectory was opened.

| Physical MAE | Native TAC | Matched GRU | Persistence | Servo |
|---|---:|---:|---:|---:|
| Source DEV, 76 episodes | .845300 | .843966 | .702952 | 1.105522 |
| Different task, 80 episodes | .373251 | .381716 | .351410 | .527826 |

On source DEV, TAC won one of five seeds; TAC-minus-GRU episode bootstrap
95% interval [-.000363,.002985]. On the preregistered clothes-hanging task,
TAC won three of five seeds, with interval [-.014216,-.002475]. Transfer used
the source normalization and frozen weights, with no target updates.

Both capability gates failed because persistence was better. The transfer
TAC-GRU difference is conditional evidence about these fitted models, not a
claim of useful control or causal action value. No result from the previously
opened source TEST is reused as prospective evidence.

CAL-based 90% whole-episode conformal intervals attained average DEV whole-
episode coverage of 91.84% for TAC and 89.74% for GRU. Mean physical interval
widths were 23.89 and 21.45 respectively: coverage alone does not demonstrate
useful uncertainty or decision value. The observational exchangeability
assumption remains unverified.

Mean optimizer CPU: TAC 16.509 seconds, GRU 6.043 seconds per fitted model.
This efficiency cost is retained. There were zero outcome-based tuning trials
inside this registration; a subsequent corrected design is a separate study.

The next integrated correction targets two concrete weaknesses: substantial
default dependence on a weak servo prior, and a normalized-error objective
whose weighting differs from the physical-unit evaluation. A successor should
start at exact persistence, learn its physical corrections from real errors,
and condition prior reliance on verified innovation statistics. Do not edit
these frozen models or erase their negative result.

Source implementation: commit ca259e59c832cbc7096d29d343493ef34915d4b2.
Registered transfer: commit 80b32227, before accessing any target trajectory.
The summaries, resource estimates, protocols and replay scripts are committed;
original weights/optimizer states and detailed predictions stay in the local
research artifact archive, per repository artifact policy.
