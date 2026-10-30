# LANGMEM088 original result recovered read-only

All original fits, first-new-document evaluations and aggregate in run40989434082
had already completed before the Lacuna work. We did not rerun a model or read
publisher text. Original7artifact ZIPs,3newcheckpointSHA and3baseline640identities
were verified. The unchanged original aggregator reproduced its originalJSON
exactly; per-seed metrics were independently recomputed from saved rows.

These are the original new-Rust-document results, NOT the Lacuna FIT cohorts.
All6models had100,521parameters and matched640update schedules. There are3,584
sharedsourcepositions, repeated underthree seeds perarm; no iid-token claim.

| Arm | NLL | Accuracy | Mean slow share | ΔNLL removing older slow content |
| --- | --- | --- | --- | --- |
| baseline | 2.775049 | 30.50% | 9.99% | -0.002692 |
| slow_bias_plus1 | 2.785954 | 29.90% | 10.94% | -0.005895 |

Bias+1 worsens meanNLL by0.010905. Seed1702 is a favorable exception: NLL improves
by0.003162 and its old-slow removal delta is+0.000559. Seeds1701/1703 worsen;
their older slow content is harmful onaverage. The average slow share rises
only0.957percentagepoints, with heterogeneous seed changes. Do not claim every
seed fails or that bias increases the final slow share in every model.

Originalsource3431da87d29c1871f05fccb0656d957408a181ca; originalActions:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/40989434082
Originalaggregateartifact12948817884, ZIP SHA256
`050ea73b72ab7568008e56112947989561fec04558dc07031e40880777c5d8cc`.
All original FIT/evaluation/aggregate ZIPs are in replay-bundles/LANGMEM088 with
originalmetadata. These3additional biased checkpoints were NOT part of the
nine-checkpoint086/087 cold-replay certification. The archive preservation
and metric reaggregation here do not assert a new cold-model replay.

No newtraining, inference, GPU, DEV score, TEST or merge. Original bias findings
remain an internal same-publisher pilot, not factual-retrieval qualification.
