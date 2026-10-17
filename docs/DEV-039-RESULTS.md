# DEV-039: all-role predictive-head repair on frozen memory

DEV-038 showed that reissued-cancellation labels were linearly decodable from the bounded model's existing readout on a small chosen task. DEV-039 tested an actual replacement predictor trained across **all four** maintenance roles while holding the event encoder, entity-local MTSK memory, query encoder and read adapter fixed. The refitted original MLP raised mature primary accuracy from81.25% to92.19%, cancellation accuracy from62.50% to84.38%, and lowered no-inspection total physical cost from2,692 to2,434 on a new576-world cohort. It did not fully solve reissued cancellation or meet complete TAC-OSM capability gates.

| Frozen-memory prediction head | Mature primary | Cancellation | Reissued cancellation | All-entity accuracy | Failures | Unnecessary services | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original trained MLP | 81.25% | 62.50% | 50.00% | 95.12% | 0 | 368 | 2,692 |
| Refitted same MLP | **92.19%** | **84.38%** | **68.75%** | **99.02%** | 22 | **44** | **2,434** |
| Newly fitted linear head | 92.19% | 84.38% | 68.75% | 99.02% | 21 | 65 | 2,452 |

The new head uses more selective interventions: it made2,346 services versus2,692 for the original and incurred22 failures. The privileged service oracle for this cohort costs2,324 with zero failures. Thus the refit improves the selected economic objective but still leaves a110-unit gap to the simulator oracle. All policies used the same fixed0.25 service rule and made no inspections. The DEV-036 schema41/masked-local controls were measured on a **different** cohort; those numbers are not substituted into this same-experiment table.

The role split matters. Retention and acquisition stayed at100%, pending cancellation improved from75% to100%, and reissued cancellation improved only from50% to68.75%. The DEV-038 100% probe was trained specifically on the reissued role; it was not a fair prediction of all-role head performance. The new96-world reissued result shows how much of that signal a jointly trained predictor recovered under this limited recipe.

The source and protocol were frozen at `b913b7b` before generating the `dev039-v1-train` and `dev039-v1-eval` cohorts. Each contains576 H512 worlds spanning three delays, four roles and three deadlines. Head features were exactly the model's existing 44-dimensional readout/query input; equality with the original model head was checked. The MLP began at the original head's trained weights, while the linear control began at its own fixed seed. Each received500 full-batch head-only updates. There was no evaluation-based tuning or memory retraining.

A separate process replayed both frozen feature cohorts, reproduced the two head fits exactly, compared every stored prediction and independently recomputed physical outcomes for all576 evaluation worlds. The held-out comparison is still conditional on one trained memory initialization, a known synthetic generator and an adaptively selected hypothesis. Exploratory paired-episode bootstrap intervals are in `DEV-039-AUDITED-SUMMARY.json`; they do not estimate initialization reliability. Raw evidence and archive hash are in `DEV-039-EVIDENCE-MANIFEST.json`.

The next full-model step should register and test a role-balanced optimization or temporal event-order channel on the residual reissued-cancellation failures, with the same multi-role and cost evaluation. The refitted head also provides a stronger base for re-testing action-level teacher supervision and the connected PLM–CDL–CASM–OSM agent. Full-system training, strong baseline tuning, fresh-seed confirmation and learned entity discovery remain open.
