# Equivalent deployment export

Supplementary engineering check, declared before scores: a normalized current
duration input can be folded into the existing linear readout after training.
If `d_norm=(d-mu)/sigma`, replace `w_d` by `w_d/sigma` and `b` by
`b-w_d*mu/sigma`, keeping every recurrent weight unchanged. The original native
forward then accepts raw duration without normalization buffers or extra
inference arithmetic. This is algebraic equivalence, not another learned arm.

The recorded-only audit exports all12models as TorchScript and reloads them with
only Torch runtime. Require maximum probability difference <=1e-6 across every
stored genuine development event. Float32 folding may change rounding. No
scientific predictive improvement follows from this equivalence check.
