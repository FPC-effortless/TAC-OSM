# DEV-036: bounded feedback learning pilot

**Result:** Bounding the entity-local MTSK writer's fixed-responsibility correction avoided the original FP32 norm failure under this paired FP64-clipping recipe and sharply reduced physical cost relative to the otherwise identical unbounded writer. It did **not** solve cancellation tracking, and three archived reference architectures performed better overall on the new cohort. This is a one-initialization learning pilot, not an architecture ranking or a working TAC-OSM model.

The [frozen protocol](../research/dev036/protocol.json) and implementation were committed at `b871c98` before full training or evaluation. Both new arms began with byte-identical trainable tensors, 6,662 parameters and768 FP32 recurrent scalars. They received the same 216 updates, batch2 H512 public-event streams, AdamW learning rate0.002 and FP64 gradient-norm clipping. The bounded arm differs only in the writer: key norm capped at1 and effective correction `beta = alpha*sigmoid(beta_logit)`. For fixed routing responsibility, each rank3 row's correction operator has eigenvalues between0 and alpha. This is not a bound on the full state-dependent recurrence or long training products.

A new `dev036-v1-holdout` development namespace supplied576 paired-world cases at H512, including384 mature primary cases and192 mature cancellation cases. None selected the recipe. The three reference checkpoints were frozen DEV-034 models, evaluated on exactly these new cases. The generator, supplied entity IDs and task rules remain known; this is not external generalization.

| Model | Mature primary | Cancellation | Selected failures | Unnecessary services | Inspections | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbounded local multi, FP64 clip | 74.48% | 92.19% | 1,043 | 390 | 517 | 5,950.7 |
| Bounded local multi, FP64 clip | 81.25% | 62.50% | 0 | 192 | 386 | 2,610.6 |
| Archived local tied MTSK | 89.84% | 79.69% | 14 | 73 | 17 | 2,496.7 |
| Archived masked-local TAC | 95.31% | 90.62% | 9 | 79 | 117 | 2,497.7 |
| Archived learned schema41 | 99.48% | 98.96% | 33 | 26 | 0 | 2,505.0 |

Total costs sum all eight entities across576 worlds. Service-everything costs4,608 with zero failures and2,228 unnecessary services. A privileged action oracle that sees each world's untreated future would service2,380 entities with cost2,380; it is an evaluator reference, never a model input. The bounded model's zero failures involved192 unnecessary services and cost230.6 above that oracle. Its lower physical cost than the unbounded writer is real within this simulator, but it does not exceed the simpler learned references.

The role split shows the tradeoff: the bounded model was100% accurate on mature retention and acquisition cases,75% on pending cancellations and50% on reissued cancellations. The unbounded control was63.54%,50%,100% and84.38%, respectively. Stability and cancellation competence did not move together. A direct action teacher may reduce over-servicing, but it cannot by itself prove the lost cancellation information is in memory.

The unbounded arm finished216 updates under FP64 clipping, although its sampled gradient norm reached `2.00e23`; the bounded arm's maximum *reported at nine checkpoints* was2.79. These are sparse training-curve measurements, not continuous maxima. The original DEV-034 FP32-clipping failure remains a separate failed trial. DEV-035's one-step Jacobian finding motivated this intervention; this pilot still does not isolate which part of the full backward graph drove the original failure.

All five checkpoints and all576 shared evaluation worlds per arm passed separate-process exact evaluator replay. The audit independently reconstructed conditions, targets, inspection receipts and physical outcomes. The prediction replay shares model/evaluator code and is not an independent implementation or fresh-seed replication. All1,171 repository tests passed locally. Exploratory paired-world bootstrap intervals are in `DEV-036-AUDITED-SUMMARY.json`; they quantify episode variation conditional on these checkpoints, not initialization uncertainty.

The next experiments need two separate answers. A learned action policy trained from simulator-verified actions should be tested for economic utility on frozen memory, with equal data and simple calibrated-threshold controls. Memory experiments should target the repeated issue/cancel sequence where this bounded writer still guesses. Action policy success must not be relabeled memory success. The complete PLM–CDL–CASM–OSM loop and PNDS conformance gates remain open, as detailed in [the research path](RESEARCH-PATH-TO-FULL-TAC-OSM.md).
