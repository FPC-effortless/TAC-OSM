# LATFUSE005 — source-authentic event memory vs exact trained alpha/beta forgetting

**Date: 2026-10-27.** Two independently registered, genuine-source, ZERO-FIT investigations on all eight original LATFUSE004 trained UCI501 checkpoints. CI source-verified. **No trained model modification, no new optimizer, no promoted retrieval/causal/intelligence capability, and no 2015 DEV/2016–17 protected numerical source access.**

## Source population and evidence

Original UCI501 Beijing Multi-Site Air Quality, source archive SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. Original authentic July 2014 FIT station-hour data, 12 source-authorized monitored physical sites, at most two earliest chronologically source-measured PM2.5 >+1.25 FIT-normalized each station with >=24h separation, matched same-source same-hour-of-day source-measured ordinary PM2.5 within [-0.25,+0.25] original normalized. **24 genuine event/control PAIRS, 48 observed anchors**, deterministic solely from original training-only normalizer and actual source readings, not trained losses, future outcome levels or previous open development scores. Original 24 hours before each anchor and source-true continuation of16/64/128/512 recorded hourly observations. All target labels at *t+L+6* were fetched only after both original-model forecasts; all from original FIT period. This is **FIT-era diagnostic**, not independent heldout generalization or event recall qualification.

Original trained seeds40401,40402 × arms4h-hard,4h-soft,2h-hard,2h-soft. Each original SHA checked against saved 2026-10-27 LATFUSE004 FIT receipts both before/after diagnostics. Original latent recurrent state **fast80+slow80+associative256+16×80 owner slots =1696 FP32 values+16 bool flags =6800 bytes per stream**. No source-derived training, extra model cache or checkpoint mutation.

### Verified immutable diagnostic runs

1. [Original real event and past-key/forecast utility study #40441308798](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40441308798) PASS; source pre-registration REGISTERED_PROBE.json, original FIT/source/SHA enforcement, 8 original trained model checkpoints, original 24 matched physical event/control pairs, 4 lengths, derived Git-owned results RESULTS.json. No optimizer.
2. [Original learned kernel alpha/beta affine transport study #40441619526](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40441619526) PASS; separately preregistered REGISTERED_ERASURE_DECOMPOSITION.json, recovered SAME source cohort hash and SAME checkpoint SHA, independently rebuilt actual FP32 gate/key/route-dependent right-affine transfer after16/64/128/512 original recorded hourly updates. Actual headwise matrix difference and independent constructed P agreement within **9.14×10^-7 maximum absolute difference** on all tested stream/history checkpoints. Derived result ERASURE_OPERATOR_RESULTS.json Git-owned. This is not a full IEEE/AD/probability-law Lean refinement.

## Result A: old matrix and explicit original WRITE-key influence is strongly attenuated

Two-seed/four-arm average using 24 original matched pairs:

| Real observed delay | Earlier matrix difference norm still present | Earlier source-anchor WRITE-key trace remaining | Mean absolute 13-channel forecast mean shift after ZEROING earlier matrix |
|---:|---:|---:|---:|
| 16h | **53.97%** | **46.52%** | 0.005137 |
| 64h | **27.87%** | **14.62%** | 0.002899 |
| 128h | **18.43%** | **7.51%** | 0.001654 |
| 512h | **4.03%** | **1.15%** | **0.000255** |

"Matrix remaining": norm of the difference between two legitimate suffix continuations from states identical except original earlier associative matrix zeroed at July source anchor, divided by norm original anchor matrix. A learned old-event key-direction trace (not the model's future forecast query) was used for the analogous projected difference. These are source-conditioned read-only numerical diagnostics, NOT probability, factual decoded PM2.5, event-specific deletion or source-independent memory capacity claims.

The source-trained forecast reads through CALENDAR+current belief, NOT a learned past-event key query. The key analysis artificially inserts the ORIGINAL write key as a probe: it detects surviving associative direction, *not* end-to-end retrieval accuracy. Zeroing the entire 24h prefix matrix removes information from ALL preceding hours, not just the designated PM2.5 spike. Fast/slow and source-entity state are held intact. Under genuine source +6h measured labels, original vs erase branch masked 13-channel MAE difference (erased minus unchanged, positive means original memory was helpful) for high event anchors averaged **+0.000647 (16h), +0.001049 (64h), -0.000226 (128h), +0.000017 (512h)**. Source-matched ordinary cases similarly negligible. No robust learned long-delay historical-event forecasting utility was established. The event anchors naturally include physical outliers: model PM2.5 MAE on 16h source-high cohort averaged ~70.95 original physical units versus legal last-observation persistence ~35.09. This is an underperformance diagnostic, not proof of general PM2.5 model failure or causal memory attribution.

## Result B: directional delta OVERWRITING is a major actual source-trained forgetting mechanism

Original fitted four- or two-head matrix right-affine update for each selected observed source head h:

M_(t+1)^h = M_t^h P_t^h + Q_t^h,
P_t^h = alpha_t^h (I-beta_t^h k_t^h(k_t^h)^T),
Q_t^h = beta_t^h v_t (k_t^h)^T.

Here k_h is original trained normalized key, alpha_h is original fitted sigmoid retention for source-authorized writer, otherwise1; beta_h is original fitted sigmoid write strength multiplied by actual learned head-route soft probability for eligible source writes, otherwise0. **Soft routing** retains/updates all heads with nonzero eligibility while **hard routing** leaves unselected heads exactly unchanged. For fixed authorized source inputs, all other state is bitwise identical under the prefix-associative-only deletion (checked), and gates do not depend on old associative contents. Thus the difference satisfies:

Delta M_(t+L)^h = Delta M_t^h * Π_(i=1..L) P_(t+i)^h.

Independent actual-source FP32 reconstruction verified this identity to maximal absolute numerical error <9.2e-7 across all original fitted heads/histories. These local input-fixed operators compose associatively but do not prove full nonlinear neural-model parallel execution.

To diagnose attenuation, reconstruct under **identical original trained source-conditioned keys/gates**:
- *alpha-only*: mathematically omit beta projection (beta set0 in derived transport, no new full model inference/training);
- *beta-only*: mathematically set alpha1 in derived transport while retaining original key-directed projection;
- actual: both factors, verified against real trained continuation.

| Delay, same source pairs | alpha ONLY historical difference survival | beta key-directed overwrite ONLY survival | actual combined survival |
|---:|---:|---:|---:|
| 16h | 95.65% | 56.17% | 53.97% |
| 64h | 83.81% | 32.66% | 27.87% |
| 128h | 70.92% | 25.38% | 18.43% |
| 512h | 28.83% | 12.29% | 4.03% |

Actual mean fitted per-step effective alpha over these source histories~0.99752, effective routed beta~0.0531. **Direction-specific replacement is already severe by 16h**, more so than broad alpha-only attenuation. By512h BOTH scalar forgetting and key-overwrite contribute to a small remaining old prefix matrix difference. These retention figures are NOT additive effects on performance and do not imply preserving ALL events is wise: old pollutant spikes may be irrelevant or harmful. Keys and subspace heads/collisions interact with beta, and beta-induced erasure targets currently visited directions. Smaller beta can hurt adaptation to new real states.

Per-head/arm numerical survival after512h, average two fitted seeds:
- 4h-hard: actual5.39%; alpha-only42.68%; beta-only9.17%.
- 4h-soft: actual2.53%; alpha-only28.14%; beta-only8.98%.
- 2h-hard: actual6.41%; alpha-only32.54%; beta-only16.25%.
- 2h-soft: actual1.78%; alpha-only11.96%; beta-only14.77%.

Soft uniform writes can lead to more cumulative global retention attenuation, and hard untouched heads can carry longer old matrix differences without necessarily improving forecasting. Wider8D keys can reduce near-cosine collision fraction while still removing earlier associations through repeated source writes in overlapping directions.

## Decision: next model improvement must be INFORMATION-USE aware, not only key geometry

1. **Primary next falsifier:** valid historical-event-specific READ query/target alignment, source-verifiable which old physical event should be remembered or allowed to expire, and whether memory affects genuine downstream forecasting. A future model needs a learned historical-event query/readout trained on real recorded history, with independently heldout physically grounded event factual retrieval targets. Current calendar-conditioned forecast query is not trained to answer “what happened at t−512?”; write-key probes alone cannot force that capability.
2. **Next architectural hypothesis:** source/event relevance-conditioned admission and selective delta overwrite, not global retention inflation. The fitted beta projection is mathematically responsible for much of lost source matrix difference. Compare source relevance, local directional beta, consolidation across heads, and obsolete-information deletion; avoid adding a Transformer token cache, recurrent cells, hidden parameters or cheating source future labels. A fixed beta schedule or stronger decorrelation alone may improve a vanity storage metric but not forecast accuracy.
3. Run genuine source matched-erase ablations isolating head routing gradient availability from decay-law change; soft-vs-hard changes BOTH active writes AND scalar retention eligibility. Strict parallel/constant-memory right-affine proof and 6800byte budget must be preserved.
4. Separately diagnose real misspecification of Gaussian pollutant and rainfall uncertainty: rainfall zero-inflation, extreme observed spikes, channel-level miscoverage and temporal gaps. No reason alpha/beta fixes calibrate Gaussian NLL.
5. Any newly trained comparison under user policy must preregister *before optimizer*, <=1% real active parameter AND independently measured FULL train FLOP parity, identical authentic source FIT/splits/original station-hour sample IDs, training budget, optimizer, true source/target provenance, two+ seeds, immutable checkpoints and independent clean replay. All sources originally protected2016–17 remain closed; already-exposed2015 DEV isn't an independent promotion gate. Diagnostics' tiny source FIT forecast changes do not qualify memory advantage.

**Status:** source-grounded long-delay trace and exact learned memory-erasure mechanism localized, no model fixed/trained here; LATFUSE005 draft remains UNMERGED and next model proposal is a falsifiable hypothesis, not a demonstrated accuracy improvement.
