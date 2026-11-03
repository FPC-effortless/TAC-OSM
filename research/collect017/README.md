# COLLECT017: post-null attention learnability control

Additional baseline after the first COLLECT016 full-BPTT models failed local as
well as distant source-copy queries. Three ~100K native attention models receive
the exact paired 512-update source/target schedule. This is a budget learnability
check, not the original preregistered architecture comparison or a winner claim.

The 1536-byte causal KV window includes all registered contexts/generations;
storage grows with observed tokens. Fast/slow/rotary/SSA are retained. Compatible
initial tensors and FFN slices are shared with each COLLECT016 initializer.
No phase router exists, so its balancing term is absent. All remaining learned
parameters must receive real gradients before fit. Byte-copy data and scoring
reuse COLLECT016 with phase-reset explicitly inapplicable.

All existing model/data/checkpoint and public FIT/DEV restrictions remain. No GPU,
TEST, promotion or merge. Full numerical reference and gradient fixtures use the
actual native attention graph; partial Lean/CausalSequenceGraph formulations are
inherited through the source-pinned PNDS companion. Full formal release is OPEN.
