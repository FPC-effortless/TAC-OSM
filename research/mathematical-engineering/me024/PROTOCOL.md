# ME024 — registered native dynamics/readout factorial

Freeze this protocol, executable source and contract before fitting. New L2
question: does slow-to-fast context help native causal predictive learning,
with split RMS conditioning controlled separately? ME023 did not test either
mechanism. L1 gates check original identity, full coupled derivatives, RMS
derivatives/bounds, causality, streaming, serialization and active gradients.
MATHEMATICS.md supplies the equations and interpretation limits.

Source is the original publisher KuaiRec archive, Zenodo 18164998, CC BY 4.0;
Gao et al., CIKM 2022, DOI 10.1145/3511808.3557220. Archive MD5
`261550d472c48eff4990fb13c0e5bcf7`, SHA256
`c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`.
Restore only the authenticated already-exposed ME022 snapshot. Check all 12 ME023
and 12 ME022 parent checkpoint hashes before response access, then exact snapshot
SHA/canonical tensor, user and time fingerprints. No new source download.
Exactly 10,776 FIT windows before August 1, 2020 and 5,952 historical DEV windows August 1–13,
user_id%16=0. Original missing/invalid exclusions,128 genuine causal observations
and source-only sample order are unchanged. All other cohorts and outcomes from
August 14 onward remain closed. No independent-validation claim is possible here.

Fixed 2×2 cells: plain; RMS-only; coupling-only; coupling+RMS. All 135,050 trainables,
same seeded original native weights, no learned interaction head, same real
window draws and same 4 supervised responses per window. Coupling alpha 0/.25,
RMS epsilon .01, no other architecture knobs. All use ME023's causal prefix
weights 1/6,1/6,1/6,1/2, depths 32/64/96/128, FIT-only duration standardization,
300 AdamW updates, batch 128, body LR .001/head LR .006, weight decay .0001,
global clip 1 and 180 process-CPU-second ceiling. Three seeds 14001–3, rotate arm
order. No pilots, tuning, early stop, best checkpoint, extra seeds or refits.
Preserve every attempt, negative and partial state. Final step 300 only.

Primary: Brier(coupled_rms)−Brier(rms). Candidate gate: all 3 paired differences
negative, mean<=−.0005; coupled_rms feature-mean replacement increases Brier by
>=.0002 each seed; mean ECE(coupled_rms)<=mean ECE(rms)+.002. No model is promoted
or reserved test opened on a development pass. Factorial main effects and
interaction are secondary; do not rename them as the primary if it fails.

Post-fit feature replacement uses the average **post-feature-map** FIT-first 1024
state, avoiding a Jensen/intercept shift from RMS(raw-state-mean). Frozen
feedback-off rollout shares trained weights and removes alpha only; it is an
OOD diagnostic, not a retrained comparator. Both preserve action/duration/prior32.
Record logits/probabilities, Brier/logloss/AUC/ECE and every seed's outcomes.

Register full-Jacobian diagnostics on first 8 FIT windows/model with float64
weight copies, depths 8/32/64/96/128. Use true chronological 64×64 products and
singular spectra, never products of singular norms. Autograd all-entry equality
is established before real probing. Neither retention nor local sensitivity
is an architecture success gate or causal explanation of accuracy.

Record steps 0/1/25/50/100/150/200/250/300, actual CPU/wall, gradients, peak process
RSS (not isolated inference memory), batch-128 latency, sampler RNG, optimizer,
weights, source freeze, input fingerprints and standalone/streaming replay.
All 4 cells have identical primary supervision/exposure and capacity. Actual
full FLOP/energy parity remains UNESTABLISHED without a complete audit. Equal
ceilings or similar optimizer CPU cannot prove fully matched superiority.

Earlier ME023 scores are frozen historical references, not newly fitted rivals.
No SSM/LLM-scale capability, causal policy utility, 128 vs 32 memory success,
symbolic CASM, commercial qualification, merge or deployment follows.
