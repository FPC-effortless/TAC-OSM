# LATFUSE001 — unified latent representation, attention alternatives and belief-memory engineering

**2026-10-26 | status: UNTRAINED / NO-FIT ENGINEERING REFERENCE, not scientifically qualified.**
**Model source pin:** TAC-OSM commit 068c3c81bef766eb97660ba5f01891275460287f
**TAC implementation:** src/tac_osm/latent_fusion_001.py
**PNDS companion:** PNDS/LatentFusion001.lean and foundation/v1/latent_fusion_001_math.py
**Ancestry:** TAC research/master-research-plan@56fa11e8939e2bf7aa36eac5af664d016820583c; native PredictiveStateAgent plus historical E2E005, ME110, ME111 as INSPIRATION, NOT already integrated trained descendants.
**No replacements of source checkpoints, failed qualification gates, six approved model families, or official variant source registers.**

## Problem and falsifiable claim

Separate five functions previously concentrated in attention: contextual embedding, retention/forgetting, content-addressable retrieval, source-entity association, and short/long temporal state. Test whether a legal bounded-state fused model can improve *authentic held-out prediction or decision cost* at matched data/capacity/training budget, relative to GRU/SSM/attention controls. Prior-art mechanisms are not claimed novel.

This is a **research kernel and provisional graph**, NOT a qualified 100k-param candidate:
- Depth-enhanced masked/time-aware embeddings: channels + availability mask + elapsed time + learned decay of last legally observed value (GRU-D-inspired); no future observation accepted through the API.
- Signed diagonal affine fast state: a_t=tanh(Au_t+b), fast_t=a_t⊙fast_(t−1)+tanh(Bu_t+c). Conditional on precomputed u_t, prefix pairs compose associatively. Nonlinear encoder and decoder do not invalidate this subgraph's scan legality. Entire graph, including recurrent associative and entity writes, is NOT parallelized by the current scan.
- Slow state: slow_t=.99 slow_(t−1)+.01 tanh(W_s u_t). Fast/slow separation is *nominal rates*, not statistical independence.
- Rectangular key/value associative memory: alpha_t M_(t−1)+beta_t (v_t−alpha_t M_(t−1) k_t) k_t^T, k normalized. This is still a linear-attention-family / fast-weight mechanism, NOT full softmax attention equivalence. Shared alpha erases unaddressed contents; no perfect key isolation guarantee.
- Source-typed entity memory: registered index only; learned gate applies a convex coordinate update to one slot and updates a Boolean occupied mask. Full slot bank stays fixed-size; overflow is an explicit error, not an invisible modulo hash. Source authentication is delegated to caller and remains UNVERIFIED.
- Read-only candidate-conditioned predictive head: (mean, positive raw-Softplus scale) of observed future quantities. This is **not** a calibrated probability law or a do(action) counterfactual policy. No outcome-feedback API or CED decoder loop has yet been integrated.

Shapes for the current source: observation [B,D_o], mask [B,D_o] bool, elapsed [B,D_o]>=0, last_seen [B,D_o], source_slot [B] int, actions [B,C,D_a].
u [B,H], fast/slow [B,H] each, M [B,D_v,D_k], entity_bank [B,N,H], active [B,N] bool; forecasts [B,C,D_o] mean and scale. Recurrent state count per entity = 2H + D_v D_k + NH **floating values**, plus N booleans. State is constant in history length, **not** constant across entity-count increases. The candidate constructor is not constrained to 152 floats and *must not be fitted* under the binding 95k–105k rule without an actual matching preflight.

## Evidence by source — proposed interventions, not assumed defects

| Source | Observed design fact | Plausible risk | Planned falsifier |
|---|---|---|---|
| Native predictive-state prototype | tanh over one linear 16D observation layer; 32D fast/32D slow/8x8 associative (at source) | shallow representation; retrieval crosstalk; fast/slow bleed | fixed-budget residual embed, legal key-pair interference matrix, temporal state lesions |
| integrated_e2e_005.py | text GRU, 2D Conv, 1D Conv; concatenated features then learned 40D fusion; 16 entity slots | modality variance imbalance, pooling topology loss, fixed-slot overflow | per-modality gradient norms, source modality dropout, topography-preserving tokens versus pooled, occupancy curves |
| ME110 | station + mask + elapsed time present; historical uncertainty gates fail | missing-time belief misrepresentation and statistical miscalibration | missingness-stratified coverage/CRPS, historical last-value decay, independent prior calibration |
| ME111 draft | native recurrent/parallel graph with conditional positive uncertainty-scale treatment; registered five fits not implied complete | tight state capacity, unqualified uncertainty, incomplete mathematical refinement | state ablation and measured memory survival on unchanged original source; preserve 2016–17 TEST seal |
| Proposed ME111 compact affine scan | associative scan of affine pieces may accelerate causal temporal dynamics | linear-in-state expressivity ceiling; nonlinear recurrent parts not scan-equivalent | same-parameter signed-spectrum, multi-layer scan/nonlinear sandwich and serial-parallel AD parity |

Corrections: tanh embedding is nonlinear though shallow; a learned fusion projection is not “equal modality weights”; exact memory capacity is not a generic eight-fact theorem (independent key rank <=8 constrains arbitrary exact linear key/value maps under conditioning assumptions); uncalibrated interval failure does NOT prove latent Euclidean distance invalid.

## LOCAL engineering falsifiers and evidence (not scientific advancement)

Local Python 3.13 / PyTorch no-fit suite (constructed tensors) ran 12 tests total: 8 model/memory/scan invariants, 4 independently implemented PNDS scalar-reference comparisons. All 12 passed. The first run had one too-strict bit-equality check on a fully addressed floating slot; repaired to numerical tolerance while retaining exact untouched-slot checks. Tests exercise scan serial/prefix forward+gradient parity, associative update, authorized-slot isolation, NaN masking, nonnegative time, read-only candidate permutation, finite backpropagation, and reference correspondence. Local Lean compiler unavailable: **Lean compilation and theorem axiom audit OPEN until PNDS CI**. Cross-repo source checkout and actual trained model forward/backward/parity OPEN. Real-data training/evaluation, counterfactual validity and numerical stability over long timelines OPEN.

## Required study protocol before any training

1. Freeze source SHA, input-time boundary, source eligibility, license, FIT/DEV/TEST chronology and physical entity IDs. Reuse genuine UCI501 station observations for coherent uncertainty studies and authorized real LeRobot trajectories for sensorimotor prediction only when exact source/clock/action legality is established. Do not fabricate actions or latent entity labels. If no real candidate-action outcomes/propensities exist, restrict to observed-action forecasts.
2. Define and preregister objective. Predict future real observed outcome at H=1/4/16 and long-history H128/H512 where actually available; diagnostic slot-key survival and interference are secondary (constructed fixtures never establish intelligence).
3. Compare three matched trained arms: (A) state-only selective affine/SSM, (B) same backbone + source-authenticated slots and gated delta, (C) (B) with constrained local window attention. Include matched GRU or Mamba-family strong control where feasible plus practical persistence reference. For comparison claims, **each neural arm 95k–105k active trainable/deployed parameters and <=1% pairwise gap; identical sample IDs, order, preprocessing, optimizer exposure/tuning/evaluation and independently verified training compute**. CPU time is a measured outcome, not a fairness equality constraint. Never pad with unused parameters.
4. Use five fixed seeds unless preregistration documents an alternative. Freeze read-only validation decision rule before opening protected TEST; do not optimize on observed TEST failures. Run mandatory scripts/validate_research_parity.py preflight with an independently verified, source-specific FAIR_COMPARISON.json. This file is **not created now** because genuine source receipts, architecture-matched parameter counts, compute/FLOPs and runner links do not exist. **STOP: no fits.**
5. Log real per-feature missingness, entity overflow rate, key Gram spectrum, retrieval confusion, active effective rank, multi-rate state sensitivity, latent collisions, gradient conflict, architecture-specific ResearchProbeAdapter, reward-outcome independence, candidate invariance, counterfactual identifiability evidence, score calibration, per-subgroup errors, real decision cost and resource usage.
6. Run independent checkpoints replay, source partition and poison checks, entire valid authorizations from acquisition through state update, heldout real-score gates and PNDS exact-real + finite-precision / full AD / probability refinement. No new qualified result unless every actual relevant gate passes.

**Explicit blockers:** source-authorized dataset adapters; real-data preregistration and parity receipts; 100k-parameter engineering; genuine outcome logs; independent leak/reward instrumentation; benchmark runs; Mathlib Lean CI; full source↔Python↔Lean refinement; historical seventeen-version registry validator is hardcoded and cannot honestly be extended without a coordinated schema migration and synced main/master counterpart.

## Prior art (primary sources)
- Yang et al., Gated Delta Networks, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html
- Hatamizadeh et al., Gated DeltaNet-2, NVIDIA 2026: https://research.nvidia.com/index.php/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- Che et al., GRU-D, Scientific Reports 2018: https://www.nature.com/articles/s41598-018-24271-9
- Grazzi et al., Negative Eigenvalues in LRNNs, 2024: https://arxiv.org/abs/2411.12537
- Tumma et al., Preconditioned DeltaNet, ICML 2026: https://proceedings.mlr.press/v306/tumma26a.html

## Promotion truth table

| Layer | Status |
|---|---|
| Local PyTorch constructed operator tests | PASS (8) |
| Local independent PNDS scalar comparisons | PASS (4) |
| PNDS standalone Python CI | PENDING |
| PNDS Lean build / axiom audit | OPEN |
| Source-pinned full neural+AD+FP32/FP64/probability proof | OPEN |
| Six-family real source adapters and research-probe registrations | OPEN |
| Matched 100k-capacity/compute real FIT protocol | NOT REGISTERED: STOP |
| Heldout real intelligence, uncertainty and decision qualification | NOT RUN |
| Causal counterfactual and real-world safety | NOT ESTABLISHED |
