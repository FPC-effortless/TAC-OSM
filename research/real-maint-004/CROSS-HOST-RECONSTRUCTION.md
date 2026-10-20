# Cross-host transform audit correction

The first remote replay at36bc8a6 failed input reconstruction before checkpoint
prediction replay:516of30,794repair-feature entries differed by at most
4.440892098500626e-16 (one float64 representable step) in NumPy log1p. The full
failure log is preserved. The exact CSV bytes, schema, split, labels, stored
prepared tensors and model weights were not changed.

The initial auditor required numerical equality at zero tolerance for every
recomputed feature. That was unnecessarily stronger than the registered1e-8
prediction replay tolerance for a platform-dependent transcendental transform.
The auditor now separately enforces exact categorical/unknown/missing fields
and at most ONE float64 step on the numeric log1p columns. A focused falsifier
rejects two-step changes and ANY change even one step in discrete fields. The
measured maximum is recorded in `input_reconstruction`, not hidden in prediction
drift. SHA256 still fixes the actual archived inputs used for checkpoint replay.

This corrects an audit requirement; it is not a model or scientific-gate repair.
The frozen fitting source, data, scores,24fitted models, validation rejection,
unopened later test cohort and original prediction tolerance are unchanged.
The standalone standard-library transform continues to be checked against the
original forecast outputs at the original tolerance. No claim of cross-CPU
bit-identical log1p input reconstruction is made.
