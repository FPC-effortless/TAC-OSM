# ME090 — joint future logged-reference and physical forecasting

Retain the native fast/slow/8x8 state, time-decayed parallel filters, raw-value
quiet precision and coordinate-local routing. A shared horizon latent now also
predicts future `action.qpos` references. These are model predictions under
recorded behavior, not true future inputs, selected actions, torque or identified
interventions. Source publisher field definitions and authority filtering are
preserved separately.

A64→42head reads the same shared latent already computed for physical forecasting:
14reference proposal values,14reference-change logits,14lead coefficients.

    predicted_delta =4 sigmoid(change_logit) tanh(proposal)
    predicted_reference=known_lag1_reference+predicted_delta
    proposed_servo=clip(FITgain*(known_error+sigmoid(lead)*predicted_delta)).

The lead starts nearzero, allowing the model to learn whether predicted later
references help physical forecasting. Proposed servo and total physical moving
means retain the previous bounds. The local evidence router receives predicted
reference delta/change/lead, not observed future commands. The resulting graph
is acyclic: shared state→reference forecast→physical parameters→CDF/quantiles.
It is joint multi-target training, not a full joint stochastic action trajectory.

All104886parameters are trainable:99731fixed-index ME087 source parameters and
5155new routing/reference parameters. Five final old originals initialize base
weights; exactly seven new tensor keys may be absent. No best-seed selection,
ensemble or frozen encoder.2048new updates,B24,T32,lr0.0003,new optimizer/RNG;
base4096+2048=6144total, new weights2048only. At most2independent seed workers.

Future reference targets are action[origin+h-1], whereas inference receives
only action[origin-1] and earlier. Controls are normalized into the same physical
coordinate frame. Existing intervention masks apply to both target streams.
New target vectors double the supervised vector streams, not independent data.
Future-reference Huber error is measured in asinh delta space and reference-
change BCE uses actual future recorded changes as FIT-only labels. The previous
physical objective remains with a1xquiet precision penalty. No inference-time
future label branch exists; a loader mutation test verifies that future command
label changes leave observed inputs identical.

Every original physical gate and stronger ME082 guard remains unchanged.
Additional99.4444%CI describes these9declared attempts, not independent adaptive
confirmation. Reference forecasts are additionally scored against last-known-
reference persistence, descriptively; no skill is inferred merely from a head.
All five originals/optimizer/RNG/receipts are Git-frozen before REPORT. Original
whole-forward float64 reference and rtol1e-5/atol1e-6 remain. Protected tissue-box/
CAL/TESTstay closed. No causal policy, autonomous decisions, architecture-winner,
production, merge or customer-value claim.
