# ME-061 reliability-conditioned native world model

This is a separately preregistered successor to ME060. Preserve ME060's
failures. Both TAC and GRU receive one new integrated design, the same data and
optimizer budget, and zero within-study outcome-based tuning trials. Across
the overall development program this is design two, not an independent
confirmation of a design selected without prior DEV information.

Reuse ME060's exact 208 FIT /20 CAL /76 historical DEV episode partitions,
64-transition history, 16-step targets, five paired seeds 61001–61005, 600
AdamW updates, batch eight and final-checkpoint-only scoring. Models share
the same real source features, action prefixes, causal statistics and loss.
No old TEST trajectory is loaded. Calibration and DEV open only after all ten
checkpoints freeze. Source dataset/version/licence and censoring rules remain
those in ME060's PROTOCOL. The generic parity schema's TEST role is explicitly
the historical source DEV population. CPU time is diagnostic only.

## Integrated correction

1. Keep bounded native fast/slow/8x8 associative state and direct horizon reads.
2. Add 14 causal velocity statistics and 14 servo-innovation magnitudes,
   identical for both arms. These are observed summary features, not additional
   training data or claimed latent discoveries. They never see a future target.
3. Begin exactly at persistence with a zero-initialized residual output and
   zero signed physical gain. Positive/negative physical gain can train from
   the first step; reliability attenuates it per physical coordinate.
4. Optimize original physical-unit MAE with a small FIT-step-scaled proper
   Laplace log score. This aligns the primary objective with the scored metric.

No individual component attribution will be claimed. This tests an integrated
architecture/training recipe, using established residual initialization,
innovation observers and probabilistic regression. The registered development
accuracy gate and uncertainty/calibration method are identical to ME060.
True full all-operator/hardware FLOPs remain uncertified; the matched supported-
operator estimate is repeated before training and on final real batches.

Inherited comparator state accounting correction: ME060's GRU allocation also
includes zero slow/memory placeholder tensors. ME061 reports all allocated
state tensor bytes (720 GRU,624 TAC), rather than the 176-byte active GRU vector
alone. This does not change parameter/data/compute comparisons.

A separate target-task protocol, frozen before source DEV scores are inspected,
will select a different task excluding clean_stains and hang_clothes. Use all
ten source models without target tuning, normalization fitting or calibration.
This is different-task transfer, not proof of a different site/operator, causal
policy value or real control. Retain every score even if a gate fails.
