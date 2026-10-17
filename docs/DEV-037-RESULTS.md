# DEV-037: simulator-taught action policy on frozen memory

The action experiment tested whether simulator-verified decisions or active case selection improve physical utility after DEV-036's bounded memory was frozen. The stronger simple control won: a single decision threshold chosen on training worlds cost2,502 on the fresh576-world set, while the learned action policies cost2,547 with randomly selected cases and2,535 with actively selected cases. Active selection modestly helped within the learned-policy pair; it did not beat threshold calibration. No full TAC-OSM capability is established.

The protocol and source were frozen at `4032759` before the full run. The existing DEV-036 bounded-memory checkpoint was held fixed. The policy received only its three future-query probabilities per entity. The simulator teacher chose service when its actual counterfactual cost was lower than waiting; hidden outcomes were revealed only for selected training cases. The two shared 81-parameter policy heads began with identical weights and trained300 full-batch updates on288 worlds each. Selection preserved four paired repetitions within each delay/role/deadline stratum. Active selection ranked uncertainty around the original service threshold using frozen predictions; random selection used a registered seed. A separate one-parameter threshold control was fitted on the random training worlds. No evaluation world selected a recipe.

| No-inspection policy | Failures | Unnecessary services | Services | Total cost | Teacher action agreement | Mature cancellation target agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed threshold0.25 | 0 | 388 | 2,730 | 2,730 | 91.58% | 50.00% |
| Tuned threshold0.75 | 19 | 103 | 2,426 | **2,502** | 97.35% | 68.23% |
| Random-case teacher head | 13 | 166 | 2,495 | 2,547 | 96.12% | 63.54% |
| Active-case teacher head | 13 | 154 | 2,483 | 2,535 | 96.38% | 64.58% |
| Privileged simulator oracle | 0 | 0 | 2,342 | 2,342 | 100% | 100% |

All costs sum eight entities over576 worlds; no policy inspected. The oracle can see the hidden counterfactual state and is an unattainable evaluator upper bound under the model's input restrictions. The active policy saved12 cost units versus the random policy. An exploratory bootstrap over288 paired scenario groups gave a cost-difference interval of approximately[-22,-4] for active minus random, conditional on this single frozen checkpoint. The tuned threshold still beat active by33 cost units. That small within-model active gain does not establish broad acquisition superiority or a useful reason to deploy the larger policy.

The learned policy improved action accuracy over the original fixed threshold but still failed to repair reissued-cancellation understanding. Because memory weights were frozen, this test cannot show that action supervision improves storage or retrieval. It applies the user's decision-level and simulation ideas in a bounded finite-action setting; it is not a reproduction of ActionOPD's exact objective or its language-agent benchmarks.

The separate-process audit passed checkpoint hashes, condition and split checks, the complete selected-case census, frozen prediction replay, policy-action replay and independently reconstructed physical costs for all576 evaluation worlds. Prediction replay shares model code and the teacher rule, so it is not an independent implementation. The new three contract tests passed. Full repository integrity CI remains separate from the scientific run. Raw local evidence and hashes are recorded in `DEV-037-EVIDENCE-MANIFEST.json`.

The next mechanism target is repeated event order: DEV-036's bounded writer scored50% on mature reissued cancellations. A controlled temporal-channel experiment should keep the action rule and memory budget fixed while testing whether order-sensitive public event encoding improves that role at H128/H512. Then action-level loss can be propagated through a trained memory pathway, compared against threshold calibration and tuned schema references. The trained full PLM–CDL–CASM–OSM system remains a separate required test.
