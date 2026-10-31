# LANGCAP105-EVAL01 frozen DEV results

All 18 original final checkpoints scored; zero optimizer updates. This is public same-project DEV, not independent/sealed TEST.

| Scale | Model | DEV NLL | DEV bits/byte | DEV byte accuracy | Final-model seen-FIT NLL | Initial DEV NLL | Learning-transfer seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | tac_associative_stack | 1.741724 | 2.512776 | 54.056% | 1.629568 | 5.535716 | 3/3 |
| small | causal_transformer | 2.275448 | 3.282777 | 39.066% | 2.192945 | 5.710351 | 3/3 |
| small | stacked_gru | 1.727076 | 2.491644 | 55.491% | 1.634320 | 5.533460 | 3/3 |
| medium | tac_associative_stack | 1.617619 | 2.333731 | 57.327% | 1.503027 | 5.532631 | 3/3 |
| medium | causal_transformer | 2.082963 | 3.005081 | 43.740% | 1.973925 | 5.769703 | 3/3 |
| medium | stacked_gru | 1.346757 | 1.942959 | 63.603% | 1.235382 | 5.551685 | 3/3 |

Means across all three seeds; NLL denominators and window IDs are identical across models. Per-seed, per-file/group scores and descriptive paired uncertainty are retained in JSON.

A lower DEV NLL records better prediction by these frozen checkpoints under this recipe. Training full-FLOPs/tuning parity and independent qualification are incomplete; no architecture is promoted or declared generally superior.

The fixed final-model FIT reference differs from the changing-model training-tail losses. The FIT/DEV gap can reflect both memorization and differences between source groups; it does not uniquely identify overfitting.

Throughput is teacher-forced CPU batch scoring, not autoregressive decode speed. No long-memory, executable coding, reasoning or agent capability was tested.
