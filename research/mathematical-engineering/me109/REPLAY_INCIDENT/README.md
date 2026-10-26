# Diagnostic-adapter cross-host replay failure; original scores unchanged

Original model/graph/proposal/forecast/score/summary replay PASSED in CI40283071768.
The subsequently added no-fit observatory adapter FAILED its strict reconstructed
FIT input digest equality, before new probe results. No model training or score
overwrite occurred. Source-identified original FIT tensors are now exported only
if their full ten-field hashes match the previously committed POSTHOC trace; the
one-origin prefix must match the original anti-cheat finding digest as well.

This export is original data preservation, not a new fit or different sample.
The additional host diagnostic reports39 fieldwise hashes, dtypes, shapes and
numeric differences. No cause (libm, reduction or other) is presumed. Original
checkpoint, input, tool-output and predictive failure receipts are immutable.
