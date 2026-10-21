# ME044 complete native multiscale forecast architecture

[Draft PR309](https://github.com/FPC-effortless/TAC-OSM/pull/309) ·
[All original results](RESULTS.md) · [Mathematics](MATHEMATICS.md) ·
[Original80 models and evidence](ARTIFACT.md).

The architecture preserves the original strong context predictor while training
a separate bounded, low-rank forecast innovation from native fast/slow and
associative memory. Matched context/GRU systems use the same horizon bases and
information. The primary objective is a stronger complete predictive model;
mechanistic attribution remains deferred.

[Locked pre-fit protocol](PROTOCOL.md) precedes80 real fits on ETTh1/Tetouan
historical development, including practical old-model references. Native oil
forecasting improves over the original context/ME043B references, but the
matched competitors remain strong and both combined acceptance gates fail.
An [optional execution addendum](EXECUTION-ADDENDUM.md) separately registers
and verifies1.414x exact full-inference speedup, without refitting or changing
the original predictions. This is not a training-speed or accuracy claim.

All source, original weights/optimizer/RNG, negative outcomes, actual forecasts,
input provenance and numerical/restoration receipts remain reviewable. No fresh
test population opened; no architecture superiority, causal decisions, persistent
streaming, world-model qualification or commercial promotion is established.
