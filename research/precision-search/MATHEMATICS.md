# ME088 — precision-aware native fine tuning

All five ME087 final originals are fixed-index starting points. Every one of
99731parameters remains trainable; no best-seed selection, inference ensemble
or frozen encoder. Each new AdamW run starts with new optimizer/RNG,2048updates,
B24,T32,lr0.0003. Total lineage4096+2048=6144updates; old optimizer/RNG/weights
remain preserved. This is weight-only fine tuning, not exact optimizer resume.
The original sequential-execution no-fit manifest is retained; the final prefit
version fixes at most two independent one-thread worker processes. Workers
write disjoint original files; only the parent assembles the full5model index.

Keep the complete native time-aware encoder, fast/slow/8x8 memory, parallel
filters, noise-bounded raw-value readout, coherent mixture and implicit median.
Separate **predictive innovation** units from quantized observed-noise units:

    predictive_unit=max(FIT_quiet_noise,0.01FIT_delta_scale)
    quiet_scale=predictive_unit*h*(0.01+softplus(b0)).

Every learned scale remains active. This is not an observation-dependent oracle
regime or a hard cap on total forecast uncertainty. It prevents exact recorded
holds from making the quiet predictive scale's initial unit arbitrarily tiny.

The unconstrained nonlinear movement correction gains a known-horizon trust
factor1-0.9exp(-(h-1)); it is0.1at one frame, approximately0.955at four, near1at
sixteen. Observed velocity and FIT servo terms remain available. This explicit
short-step prior is motivated by exposed-development false movements and must
earn its value in the unchanged whole-model gates.

In addition to the frozen previous proper-score/implicit-median objective:

*0.5coordinate routing BCE predicts whether the *future observed change* exceeds
FIT observed-noise*sqrt(h). These are real FIT labels; no future outcomes enter
the inference graph. This auxiliary target does not prove latent physical regime
identifiability or calibrated intervention probabilities.
*8.0one-step factual-quiet median error is an explicit quiet-preservation
multiobjective. It uses training-only future-group labels and has no inference
branch by true movement. It is not a claim of universally optimal unweighted
posterior calibration; original proper NLL, coverage and interval-score gates
remain compulsory.

All original ME081 gates and stronger ME082 requirements remain fixed. Additional
99.2857%paired CI describes these7declared attempts, not arbitrary historical
adaptation. Same exposed40episode cohort, no protected tissue-box/CAL/TEST. All
five new originals must be Git-frozen before either history score. Same whole-
forward float64 reference and original rtol1e-5/atol1e-6. No extra attempts under
this registration, automatic merge/deploy, architecture-winner or customer claim.
