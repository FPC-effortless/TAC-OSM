# Preserved pre-fit source handling failure

First launch from5937696 stopped in role subset construction before creating any
model, optimizer, checkpoint or forecast. Source snapshot includes scalar
eligible_target_count/sampling_stride fields alongside row data; the helper tried
to subscript those scalar fields. The stack trace is preserved in
prefit-source-error.log and original empty-fit receipts in
results/me031-prefit-source-error locally. No model fit or outer score occurred.

Correction preserves scalars and slices only row tensors/lists. A regression test
covers the actual mixed schema. Models, roles, seeds, exposure, hyperparameters,
thresholds and source boundaries are unchanged. Corrected source is pushed before
any fit. This is a software preflight correction, not a retried failed training
fit or tuning opportunity. All future reports retain this failure.
