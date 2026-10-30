# LANGFAST095 original checkpoint independent replay — 2026-10-30

This **non-training, non-code metadata receipt** requests a fresh source-control
check of original original-source evidence after the archive job committed the
first benchmark. The original stored benchmark at
`research/mathematical-engineering/langfast095/BENCHMARK_RESULTS.json`
was produced by [Actions 41018282173](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41018282173).

First-run measured CPU original median 1.67286071875 ms per byte; optimized
median 1.63132639583 ms, 2.4828321002% reduction. All original byte-level
logits and all nine native state tensors were BITWISE_EQUAL on all three archived
original TAC seeds (1701, 1702, 1703) for 256 genuine bytes each; source owner,
lesion, finite-state checks passed. Original 100,521 learned parameters and
1,920 persistent-state bytes were identical. Additional exact nonpersistent
identity buffers use 320 bytes of constant per-model memory.

This is a **single hosted CPU, one-thread** result with 5 paired repeats/seed,
not a mobile, energy, GPU or cross-architecture training-compute result. The
relative timing reduction must be retested independently; no benefit should be
claimed as statistically general or as full native decoding breakthrough.
Original published LANGHEAD094 failure of tied-head generalization preserved.

Independent replay MUST cold-check all original checkpoint SHA, Rust Book source
blob, trained weights and every observed logit/state. No optimizer steps or
protected TEST; preserve any failed jobs; no merge or promotion.
