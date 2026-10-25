# ME102 — reconstruction/read-coupled current belief and hard age floors

A complete native TAC-Temporal model, not a replacement by Mamba or a seventh
model family. It implements a bounded subset of draft PR396: direct current-state
supervision, explicit source-aware uncertainty and stronger memory read/write
coupling. Multi-head SSD transitions, a full Transformer cache, CED and action
planning remain proposals, not implemented capabilities in this panel.

## Input-to-output graph

The unchanged source-aware encoder and sum/mass/evidence prefix schedules feed
native fast/slow maps and the associative8x8 affine scan. All15 models have the
same graph,98,495 active trainable/deployed values and the same incoming records.
In addition to ME101's conditional belief emission, a new102→12current head reads:
13post-admission source means,13masses,62evidence means,6source ages and8matrix-read
coordinates. The shared normalized write-key projection of the masked current
encoder queries the inclusive final matrix. This is a causal postscan read: it
cannot alter that scan's input-only gates or stored observed-world state. It gives
current reconstruction a direct supervised path through memory and address
weights; it does NOT force useful memory or prove memory necessity.

Let a be the observed-or-retained standardized-log current anchor. The head gives
q_mean=a+2*tanh(u), q_variance=.0025+softplus(s)^2+ME101_latent_variance.
For missing current channels, future location becomes ME101_location+j*(q_mean-a).
Modeled future variance is sigma_future²+j²*q_variance*missing, using the declared
conditional affine diagonal emission; j is not a full-network Jacobian.

A nonlearned FIT-only rate is the mean squared24hour difference in the SAME
standardized log1p coordinates, using actually observed endpoint pairs. The rate
is bounded below by1e-4 only for numerical positivity; no log-offset transform is
substituted. Floor=missing*recipe_multiplier*rate*age_hours/24. Final variance is
MAX(modeled_variance,floor). Thus trained variance paths cannot cancel below this
floor. It is an empirical prior under a linear-age assumption, NOT a safety bound
or90%coverage guarantee. Actual raw quantiles/NLL come from the same censored
lognormal distribution as before. Source age remains capped256h; no unlimited-age
uncertainty-growth guarantee. There is still no identified latent-state posterior.

## Authentic-label augmentation, not a synthetic primary benchmark

A FIT-only audit found243 of371 eligible current PM2.5 gap origins had no pollutant
currently observed, and284 had at most one. Blindfolding one variable only could
leave unrealistically easy proxies. Final training therefore leaves50%of origins
unchanged; among remaining eligible origins,65%hide all six pollutant channels
and35%hide one actually observed channel for4/8/16/32/64steps, capped to context.
Only authentic recorded current values become reconstruction labels. Naturally
missing current values NEVER acquire invented targets. Original all-current-missing
origins are retained unmodified. Hidden-channel ages use strict pre-block observed
history, never a concealed current reading or future response. Numeric inputs
are zero-masked, not forward-filled. Future six-hour labels remain authentic.

## Three integrated recipes, same work graph

- light_floor: reconstruction weight.3, floor multiplier.25, log interval weight.1.
- strong_reconstruction: reconstruction1, floor1, log interval.1.
- physical_score: reconstruction.3, floor1, raw normalized interval weight.1.

All calculate NLL, logMSE, reconstruction NLL, proper log intervals and proper raw
intervals on EVERY update. Only fixed scalar weights/floor buffers differ; no
extra parameters, data, tuning or dummy computational padding. Active current
heads influence future means/variances even without their auxiliary loss. Every
recipe has five identical seed initializations and precomputed observation orders,
matched1536updates/batch16/8-64-256context schedule and identical stochastic masks.
Raw interval loss is normalized by FIT-only raw pollutant standard deviation.

Torch CPU profiler measures dense forward/backward FLOP estimates, and the exact
ATen operation/shape/count signatures match at all context lengths. Unsupported
pointwise/optimizer FLOPs are not completely quantified; do not label these full
hardware FLOP counts. The same tensor graph, parameter layout and optimizer are
used throughout; postfit profiles and actual augmented-input digests must match
before a recipe-comparison interpretation. CPU time is efficiency, not parity.
No comparison with outside GRU/Transformer/Mamba or independent superiority is made.

## State, verification and open obligations

No additional persistent neural tensors:608FP32bytes,24integer bytes,104external
age bytes, plus Python/runtime metadata. Fixed global normalization/floor buffers
are counted separately from dynamic state. Each deployed model is98,495learned
parameters, not an ensemble of fifteen models hidden behind a100k claim.

PNDS proves max variance dominates both branches and positive future-noise
premises imply positive variance, plus observed/read-only properties under their
stated conditions. Independent Python reconstructs the complete chronological
base and new memory read/head/loss for FP32/FP64 state/value/gradient comparison.
Full NN/IEEE, learned calibration, useful retrieval, safe control, embedded latency
and commercial readiness remain OPEN. No percentage readiness/IQ score or timeline
is inferred from core-state bytes, scan span or kernel proofs.
