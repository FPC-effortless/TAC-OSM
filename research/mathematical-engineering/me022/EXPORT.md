# Equivalent deployment export

Supplementary engineering check, declared before scores: a normalized current
duration input can be folded into the existing linear readout after training.
If `d_norm=(d-mu)/sigma`, replace `w_d` by `w_d/sigma` and `b` by
`b-w_d*mu/sigma`, keeping every recurrent weight unchanged. The original native
forward then accepts raw duration without normalization buffers or extra
inference arithmetic. This is algebraic equivalence, not another learned arm.

The recorded-only audit exports all 12 models as TorchScript and reloads them with
only Torch runtime. Require maximum probability difference <=1 e-6 across every
stored genuine development event. Float32 folding may change rounding. No
scientific predictive improvement follows from this equivalence check.

**Recorded numerical failure:** direct float32 folding exceeded the fixed guard
on combined seed 14003 (maximum probability difference1.1324882507324219 e-6).
Constructing folded coefficients in float64 before casting back to float32 did
not repair this rounding limit. The1 e-6 threshold is unchanged. Preserve the
attempts but do not qualify the fold as a universal deployment optimization.
The accepted standalone export keeps the exact original two-scalar normalization
inside its serialized graph and must match original predictions within 1 e-7.
No weights are refit and original predictive scores remain unchanged.
