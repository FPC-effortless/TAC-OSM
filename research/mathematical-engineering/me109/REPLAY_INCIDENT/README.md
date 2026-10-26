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

CI40284177624 fieldwise evidence: normalized batch0 values differ at most
1.430511474609375e-6, current normalized reconstruction labels at most
1.1920928955078125e-7. Other eight fields were byte-identical. Both changed fields
PASS the pre-existing combined relative1e-5/absolute1e-6 numerical contract; the
first maximum is NOT below absolute1e-6. The exact primitive cause remains OPEN.
Original model and score replay passed before this diagnostic digest rejection.

Replay V2 does not weaken original checkpoint, batch tensor or prediction gates.
It authenticates the original frozen FIT arrays against previously committed
input digests and uses those EXACT tensors for model/gradient replay. Separately,
all39 freshly rebuilt fields must satisfy the existing float tolerance and exact
shape/dtype/Boolean/integer identity. Actual clean/poison calls still execute the
real producer and require exact within-run invariance; fresh-vs-frozen actual
model forecasts must satisfy the unchanged numeric contract. Both producer hashes
and all differences are disclosed in PREPROCESS_REPLAY_V2.json/current CI logs.
Only comparison copies omit equality of that explicitly separately validated
producer digest; no stored original input/output/score identity is overwritten.
This versioned diagnostic repair does not explain the predictive failure or
make the overall INCONCLUSIVE anti-cheat ledger PASS.
