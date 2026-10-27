# LATFUSE011 — disconfirming premature memory-address root-cause attribution

**2026-10-27 — original-source no-fit 24-pair forensic PASS.** [CI #40451917226](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40451917226) PASS; first attempt [#40451827494](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40451827494) failed required registration scientific_purpose before reading source, fixed without changing scientific source data/selection/thresholds, failure preserved. All original 4 LATFUSE008 trained 99,390-weight checkpoint SHAs unchanged; UCI501 physical source original ZIP SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8, source past-only original FIT normalizer, full original 24 same-site/hour 2014 May–June high PM2.5>=150 vs ordinary<=35 ug/m3 physical pairs. No new fit or optimizer, no 2015 exposed/2016–17 protected original numeric parsing. Do not claim independent model generalization; this FIT-era cohort is already exposed.

## Falsifier 1 — Sign inversion and confidence, not merely random scores

| Lag | Assoc seed80701 true event-above-control | Assoc seed80702 | trivial current PM2.5 |
|---:|---:|---:|---:|
| 16h | 6/24=25.0%, exact two-sided sign p=.02266 | 5/24=20.83%, p=.00661 | 60.42% |
| 64h | 7/24=29.17%, p=.06391 | 6/24=25.0%, p=.02266 | 56.25% |
| 128h | 7/24=29.17%, p=.06391 | 6/24=25.0%, p=.02266 | 50.0% |
| 512h | 6/24=25.0%, p=.02266 | 5/24=20.83%, p=.00661 | 45.83% |

For 6/24, Wilson95% source-pair rank interval approximately [12.0%,44.9%]; 5/24 [9.2%,40.5%], 7/24 [14.9%,49.2%]. At64h source PM2.5 real high-event group predicted historic high-event probability LOWER than ordinary (.4227 versus .5900 in seed80701; .4814 versus .5247 seed80702). Original fast/slow-only query control reached 62.5%/50.0% at64h. Opposite class-polarity scoring is only a diagnostic; inverting output does NOT rehabilitate its Brier, prospective supervision or scientific evaluation. Source site and hour of day were matched, but ordinary anchors could occur different dates; date-gap/source historical confounds are recorded, and multiple lags and seeds share the same 24 physical pairs. Sign-test p values UNADJUSTED and not independent replication; do not claim formal population inversion after posthoc multiple testing.

Current/last actual source-observed PM2.5 is a practical nuisance ranking; original source missingness handled using the last previous source physical value. At64h current reading rank56.25%; not similarly inverted. Other source/date clues could still confound historic query.

## Falsifier 2 — Correct random eight-dimensional cosine baseline

For uniform independent random vectors u,v on sphere S^(d−1), E|u·v|=Γ(d/2)/(sqrt(π)Γ((d+1)/2)), exact for d=8: **0.29102618165**; asymptotic sqrt(2/(πd))=0.28209479 (approximate only). The original query key and true write key are independently normalized per8D head. At H64 actual two-original-seed mean read-route weighted |cos| **0.30543**, matched WRONG-event same-site/hour original write key **0.29270**, and exact geometric isotropic null **0.29103**. This is weak source-specific angular separation; it is NOT a statistical proof that keys were sampled isotropically, that a particular error is exclusively addressing, or that learning cannot align keys with more training.

## Falsifier 4 — The whole memory is used, but true past event contribution is negligible

Original frozen model's native event query was issued first, before consulting the source anchor label, with only legally known historic clock and its current learned state. Original event-step matrix update bypass versus identical physical original observation suffix; complete historic read ablation implemented by replacing the query-time associative memory with zeros. The latter also removes *all* other historic facts, not a source intervention.

| Real lag | Mean abs predicted event probability response when ALL historic associative READ removed | Mean abs response when only exact event-step MATRIX delta bypassed | Original specific-event propagated read signal norm relative other write-content read norm |
|---:|---:|---:|---:|
| 16h | 0.10288 | 0.000911 | 2.094% |
| 64h | 0.11538 | 0.000417 | 0.829% |
| 128h | 0.13678 | 0.000274 | 0.548% |
| 512h | 0.11694 | 0.0000438 | 0.0687% |

The previous "memory is sitting unused" analogy is FALSE if taken literally for the full associative path: the historical event head *does* use the associative matrix and changes predictions substantially when that read is removed. But it does NOT extract meaningful amount from the particular original PM2.5 event written at that recorded step. This supports content discrimination/overwrite/query/readout/training insufficiency as a JOINT issue, not a proved unique root cause.

The exact source-conditioned per-head right-affine event-step delta transport formula Delta M_after=Delta M_event Π_i alpha_i(I−beta_i k_i k_iᵀ) matched the original actual model memory difference to max **1.02e-6** absolute FP32 on tested source length16/64/128/512 (conditional input-state, not a whole-model IEEE proof). Event signal read is the old event matrix-step influence transported to query then projected through ACTUAL source-trained historic query, head route and original source owner mask. The rest of associative read is the post-lesion whole matrix read, not mathematically guaranteed independent noise. SNR is a norm ratio heuristic, not a statistical signal-to-noise distribution or factual recall proof.

## Causal limitation and required next action

No new learned module, architecture or training here. The original source anchor label is NOT passed into the historic model query. This can rule out very simple total-reader disconnection; it cannot decide between an unlearned event writer, collisions, irrelevant input encoding, lack of training budget (only32 original updates), learned read-head attenuation or poor objective.

**Next source-safe sequence:**
1. Separate *synthetic-only* same exact model trainability diagnostic at16/64/128/512 real-length delays. Its status must never be confused with original physical source scientific performance; trained synthetic checkpoints require full optimizer/RNG/clean replay. [Separate synthetic-only draft PR #468](https://github.com/FPC-effortless/TAC-OSM/pull/468), [workflow #40452486371](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40452486371).
2. Check authentic-source learnability by preregistered **B: retrieval alignment objective** first, not new age-aware model A first. Compare true source original historic event truth read/ordinary controls, training-only source recalled content reconstruction, and source current/fast/slow same-parameter baseline. A later time-addressed write+read mechanism must have real known query age/time at inference and no hidden historical oracle.
3. No original neural source fit until <=1% actual whole training compute and parameters matched, genuine primary source true outcomes and frozen data/splits/optimizer/source order, >=2 independent trained seeds, protected evaluation access reviews, independent checkpoint/optimizer/source hash/CI replay. Original source data and earlier negative learned checkpoints unchanged.
4. Maintain exact original1696 FP32+16 flag 6800byte recurrent state and native source-time parallel scan mathematical contract; full IEEE/AD/probability-law proof remains OPEN.

**Scientific decision:** event write grad connectivity passes, full read is utilized, specific real recorded event is overwhelmed/poorly selected. Both age-awareness and key alignment remain CONJECTURES, not qualified architectural fixes.
