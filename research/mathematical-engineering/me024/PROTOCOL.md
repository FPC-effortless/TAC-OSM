# ME024 — registered native dynamics/readout factorial

Freeze this protocol, executable source and contract before fitting. New L2
question: does slow-to-fast context help native causal predictive learning,
with split RMS conditioning controlled separately? ME023 did not test either
mechanism. L1 gates check original identity, full coupled derivatives, RMS
derivatives/bounds, causality, streaming, serialization and active gradients.
MATHEMATICS.md supplies the equations and interpretation limits.

Source is the original publisher KuaiRec archive, Zenodo18164998, CC BY4.0;
Gao et al., CIKM2022, DOI10.1145/3511808.3557220. Archive MD5
`261550d472c48eff4990fb13c0e5bcf7`, SHA256
`c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`.
Restore only the authenticated already-exposed ME022 snapshot. Check all12ME023
and12ME022parent checkpoint hashes before response access, then exact snapshot
SHA/canonical tensor, user and time fingerprints. No new source download.
Exactly10,776FIT windows before Aug1,2020 and5,952historical DEV windows Aug1–13,
user_id%16=0. Original missing/invalid exclusions,128genuine causal observations
and source-only sample order are unchanged. All other cohorts and outcomes from
Aug14onward remain closed. No independent-validation claim is possible here.

Fixed2×2cells: plain; RMS-only; coupling-only; coupling+RMS. All135,050trainables,
same seeded original native weights, no learned interaction head, same real
window draws and same4supervised responses per window. Coupling alpha0/.25,
RMS epsilon.01, no other architecture knobs. All use ME023's causal prefix
weights1/6,1/6,1/6,1/2, depths32/64/96/128, FIT-only duration standardization,
300AdamW updates, batch128, body LR .001/head LR .006, weight decay .0001,
global clip1 and180process-CPU-second ceiling. Three seeds14001–3, rotate arm
order. No pilots, tuning, early stop, best checkpoint, extra seeds or refits.
Preserve every attempt, negative and partial state. Final step300only.

Primary: Brier(coupled_rms)−Brier(rms). Candidate gate: all3paired differences
negative, mean<=−.0005; coupled_rms feature-mean replacement increases Brier by
>=.0002 each seed; mean ECE(coupled_rms)<=mean ECE(rms)+.002. No model is promoted
or reserved test opened on a development pass. Factorial main effects and
interaction are secondary; do not rename them as the primary if it fails.

Post-fit feature replacement uses the average **post-feature-map** FIT-first1024
state, avoiding a Jensen/intercept shift from RMS(raw-state-mean). Frozen
feedback-off rollout shares trained weights and removes alpha only; it is an
OOD diagnostic, not a retrained comparator. Both preserve action/duration/prior32.
Record logits/probabilities, Brier/logloss/AUC/ECE and every seed's outcomes.

Register full-Jacobian diagnostics on first8FIT windows/model with float64
weight copies, depths8/32/64/96/128. Use true chronological64×64products and
singular spectra, never products of singular norms. Autograd all-entry equality
is established before real probing. Neither retention nor local sensitivity
is an architecture success gate or causal explanation of accuracy.

Record steps0/1/25/50/100/150/200/250/300, actual CPU/wall, gradients, peak process
RSS (not isolated inference memory), batch128latency, sampler RNG, optimizer,
weights, source freeze, input fingerprints and standalone/streaming replay.
All4cells have identical primary supervision/exposure and capacity. Actual
full FLOP/energy parity remains UNESTABLISHED without a complete audit. Equal
ceilings or similar optimizer CPU cannot prove fully matched superiority.

Earlier ME023 scores are frozen historical references, not newly fitted rivals.
No SSM/LLM-scale capability, causal policy utility,128vs32memory success,
symbolic CASM, commercial qualification, merge or deployment follows.
