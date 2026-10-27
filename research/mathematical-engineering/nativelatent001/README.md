# NATIVELATENT001–003 — original trained native TAC-OSM latent-memory diagnosis

**October 27, 2026.** Three no-fit/source-authentic experiments, five ME111 original trained checkpoints, real UCI501 historical FIT data. **No protected scoring, no new fits, no changed checkpoint, no architectural superiority or causal world-model claim.** This document is diagnostic evidence and a falsifiable repair plan, not a qualified improvement.

## Original source, scope and verified receipts

- Native original graph: \`src/tac_osm/variants/entity_sensor_world.py\`, inherited \`residual_role_sensor_world.py\` and \`conditional_belief_sensor_world.py\`, gate wrapper \`retention_controlled_world.py\`, right-affine associative schedule \`parallel_affine_memory.py\`. Native source base \`research/me111-conditional-belief-learning-2026-10-26\` SHA \`0dce638b6564f254860a8c2d61df035058391b84\`.
- Original checkpoint family: \`research/replay-bundles/ME111/conditional_belief/seed-111001..111005/final.pt\`; original SHA for every seed was verified against \`FIT_INDEX.json\` before inspection and checked again after. Original models have **104,663 FP32 learned weights each** and follow prior 4,096-update ME110 parents plus 1,536 additional ME111 updates. No new optimizer created or run.
- Original archive: UCI501 Beijing Multi-Site Air Quality, CC BY 4.0, original SHA256 \`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8\`. Source loader phase \`fit\` parses only records before 2015. Fixed station-hour 12000 for each of the 12 source-authenticated station streams; lookback lengths H16/64/256/512. **No future-target extraction or labeled prediction scoring** was performed. Original 2015 exposed REPORT and protected 2016–2017 numeric rows were neither read nor evaluated.
- CI receipts: [NATIVELATENT001 run 40423464942](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40423464942) **SUCCESS**, [NATIVELATENT002 run 40424029031](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40424029031) **SUCCESS**, [NATIVELATENT003 run 40424538120](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40424538120) **SUCCESS**. Initial 001 run 40423284133 failed before producing a result because it instantiated native source under \`torch.inference_mode()\`, which does not expose the \`Parameter._version\` guard used by the original model; changed to \`torch.no_grad()\` (still no fit) and preserved the failed workflow log. No historical checkpoint/schema/source modified.
- Immutable derived outputs: [RESULTS.json](RESULTS.json), [LOCALIZATION.json](LOCALIZATION.json), [TRANSFER.json](TRANSFER.json). All were committed to an isolated draft research branch. These are post-hoc FIT-only operator probes on overlapping station-time records; **five checkpoint seeds are not five independent real-world cohorts**.

## Source-accurate latent-state decomposition

**Correct earlier misconception:** Native ME110/ME111 carry **13 source weighted-sum scalars +13 source mass scalars +62 temporal evidence-sum scalars + an 8×8 associative matrix (64) =152 FP32 scalar values per stream (608 bytes)**, plus owner/hour/count/parameter-version metadata (outside the 152 floating scalars). The 48D \`tanh(Linear(49,48))\` representation and intermediate fast/slow feature maps are **transient within each source forward**; they are NOT separate persisted 32D fast and slow states from the old prototype. Native persistent state is not the proposed H80 LATFUSE002 1,696-float state. This matters for memory-size and fast/slow attribution.

Known genuine input fields: standardized original 13 sensors including original categorical wind sin/cos, original masks, per-source observation age, source clock encodings, registered station owner and hour. Future outcomes were never passed in as predictors.

## Result 1 — physical sensor differences survive the encoder far better than memory addressing

Per-seed original 2014 FIT source with 64 consecutive recent observations per station, source-authenticated stations and hour. A **source-distinct** pair requires at least three jointly observed sensor coordinates and observed normalized sensor root-mean-square difference >=1.0. There were **5,502 eligible distinct source pairs per seed** (pairs correlated by site/time, not independent people or measurements).

| Derived key geometry, all five original checkpoints | Five-seed mean |
|---|---:|
| Mean absolute cosine among these source-distinct 48D encoder representations | **0.58820** |
| Mean absolute cosine among their source-distinct 8D memory-write keys | **0.66871** |
| Source-distinct pairs with abs cosine >0.9 at encoder | **0.0327%** |
| Source-distinct pairs with abs cosine >0.9 at memory-write key | **9.6510%** |
| Increment in mean abs similarity from encoder to key projection | **+0.08051** |

Per-seed source-distinct near-collision fraction at key output: **9.47%,14.94%,3.89%,1.34%,18.61%**. Key similarity across all 64-hour recent pairs is ~0.77–0.86 depending seed. Actual original 8×8 matrix effective rank is generally 7–8 (SVD relative cutoff 0.1%), so nonzero/high matrix rank does **not** ensure unaliased content-specific retrieval.

**Localization:** At this operational sensor-distinctness threshold, additional representation collision arises between the 48D encoder and 8D address. This does *not* prove physical information is lost from the total network: address keys might intentionally collapse predictively equivalent observations, and direct target-identity/retrieval labels are absent. Still, modifying the 48D encoder alone is not the first well-supported remedy for key aliasing; the learned addressing/read scheme merits direct attention.

## Result 2 — preceding associative information influences short-term forecasts but dissipates fast

Real checkpoint state prefix is run with legitimate source observations. Only its earlier **associative matrix** is then replaced by zeros; other state fields, legal suffix observations and original weights are untouched. Report readout means of abs changes to native standardized-log-location forecasts. This is an **out-of-distribution state-lesion diagnostic**, not a retrieved-fact accuracy score.

| Real observed suffix length | Five-seed mean absolute log-location difference |
|---:|---:|
| 1 update | **0.08878184** |
| 4 | **0.04667608** |
| 16 | **0.01272911** |
| 64 | **0.00096162** |
| 128 | **0.00003533** |

The associative memory is therefore **not an unused/disconnected prediction pathway**. It has measurable immediate effect, which becomes tiny after a long source continuation. This does not by itself identify erase-vs-write-vs-read cause, because the output includes many other signals.

For context, erasing **previous evidence summaries** before the final64-step suffix changes native log-location predictions by mean **0.07800756**; erasing previous associative memory changes them by **0.00096162** under the same setup. The two groups have different dimensions/roles, so this is a local sensitivity comparison, not a normalized importance estimate or a claim the evidence mechanism is universally better.

## Result 3 — source-native affine transport verifies substantial forgetting occurs *inside memory itself*

Native memory update (the actual trained source uses a matrix-right affine kernel, not LATFUSE delta):

\[
 M_{t+1}=M_tP_t+Q_t,\quad
 P_t = \alpha_t(I-\beta_t k_tk_t^\top),\quad
 Q_t=\beta_tv_tk_t^\top.
\]

Here \(\alpha_t\) is native controlled retention including the \((1-\beta_t/4)\) factor, \(\beta_t\) is native controlled write strength and \(k_t,v_t\) come from actual original learned source projections. Holding **all other original state and source inputs** fixed, the difference of two memories with the same future source sequence is:

\[
 \Delta M_{t+h} =\Delta M_t P_tP_{t+1}\cdots P_{t+h-1}.
\]

This identity was independently reconstructed with each native model's actual forward hooks and checked in FP32 on all five original checkpoints against independent memory-zero suffix outcomes; **maximum observed absolute numeric discrepancy ~6.15e−8** on the tested streams. This verifies a scoped source/tensor identity in Python; it is **not** a full IEEE/Lean/AD proof or claim about latent representation stability outside the observed source.

| Source-controlled suffix | Mean native effective α | Mean write β | Mean retained original-memory difference norm ratio | Mean transport spectral norm |
|---:|---:|---:|---:|---:|
| 1 update | 0.94880 | 0.17977 | **0.83975** | 0.94880 |
| 16 | 0.94658 | 0.18304 | **0.12075** | 0.41801 |
| 64 | 0.94580 | 0.18771 | **0.009937** | 0.03077 |
| 128 | 0.94621 | 0.18684 | **0.0003893** | 0.001194 |

**Diagnosis grounded by two independent observables:** The native trained \(\alpha\) is around0.946 in real FIT source context. The actual 8×8 transition multiplies away the influence of a pre-existing matrix state over tens/hundreds of writes; its measured matrix-difference transfer is only ~0.99% after64 steps and ~0.039% after128. Since the **memory state difference itself** decays, the weak far-lag forecast lesion cannot be explained solely by a downstream decoder failing to read perfectly preserved old matrix contents. This proves neither all events must be retained nor real delayed event retrieval failure without ground-truth queries, but it pinpoints the dominant local mechanism.

## What is proven vs still open

**Verified read-only source facts:** fixed native 152-FP32 carried state; actual original model hash/shape/input/time checks; source-distinct learned address-collision geometry; native matrix carries nontrivial short-lag influence; actual original affine transport and numeric decaying matrix differences; immutable negative ME111 complete joint qualification 0/5. Original ME111 positive width/interval score and failed gap coverage remain unchanged. The whole model is not generally reducible to this one affine prefix without source gate/input contracts.

**OPEN:** Did the collided keys conflate *decision-relevant* historical observations rather than intentionally compress predictable redundancy? This study has no source-grounded event retrieval truth per key. Does long-lag forgetting harm a particular real future outcome? Need paired authentic labels and readout-safe experimental splits. Do malformed masks/timestamp semantics create nuisance leakage? Prior scoped audits pass, full independent end-to-end access proof remains INCONCLUSIVE. Native conditional interval calibration still fails independently of this diagnosis (ME111 gap PM2.5 coverage ~86.4%, gap NO2 ~71.2%, full gate FAIL0/5). Neither memory fix nor source-derived gate identity by itself repairs that uncertainty.

**Do not conflate** the LATFUSE002 H80 candidate's learned retained alpha ~0.998 on separate original real FIT samples with source native ME111 alpha ~0.946. Memory dimensions, training exposure, output distributions and evaluation cohorts are not matched; these are descriptive diagnostics only, not comparative model superiority.

## Proposed smallest controlled fixes, not implemented or trained here

1. **Address-level fix before encoder rewrites:** Preserve 48D latent encoder as starting point; investigate key projection normalization/rank, source-label-conditioned contrastive/key decorrelation regularization and retrieval-aware evaluation. Critically, arbitrary key uniformity is not the objective: prioritize correct forecast or source-verifiable delayed information use. Detect over-regularization that separates physically equivalent observations.
2. **Retention-level fix without more recurrent floats:** Alter native effective \(\alpha,\beta\) constraints or conditional gate targets to maintain important source evidence longer; inspect tradeoff between obsolete-information removal and old-event retrieval. Maintain the **same 64-element associative bank** and 152-FP32 scalar carried state, while retaining parallel legal \(M@P+Q\) schedules if possible. A changed gate may improve memory span but degrade needed updates; this is a falsifiable hypothesis.
3. **Controlled rectangular addressing under the same bytes:** Compare original8×8 versus alternative 4-value×16-key (still64 matrix floats), only if actual source/readout contracts can be adapted and parameter gaps match; higher address dimension trades away content dimension and is not a free improvement.
4. **Reader/use and latent bottleneck probes:** Fixed-source-key injection, previous-read query perturbations, key-vs-value disentanglement, per-source feature sensitivity and meaningful delayed facts on genuine historical data. Keep source authorization explicit and avoid post-hoc synthetic evidence being called real-world capability.
5. **Separate uncertainty mechanism:** ME111 properly sharpens intervals but fails on outage conditional calibration. Analyze natural missingness, per-channel tails, recent vs long-history error, out-of-station transfer and uncertainty-score instability as a distinct prediction-law project; neither key decorrelation nor alpha tuning has validated this.

### Promotion gate for any NEW training

Current original experiment **performed no training**. Any prospective fix requires source-original freeze, verified real observation provenance, original source-authorized labels/time boundaries, 95k–105k real active/deployed parameters per trained model, <=1% parameter gap across trained comparators, equal data/splits/optimizer/tuning/compute budget, prespecified seed count and heldout/source-level case independence, source-enforced anti-cheat evidence, original checkpoint preservation and clean checkout replay before any promotion. Do not open protected 2016–17 numeric TEST or permit retroactive threshold changes. Create a new registration: do **not** refit historical ME111, change original checkpoints or merge exploratory drafts.

## Research decision

The original native architecture needs **two specifically separable memory interventions** — more discriminative *source-justified addressing* and slower *source-justified retention* — while keeping its 608-byte FP32 carried state and legal prefix algebra if that is the user objective. The evidence justifies targeted prospective ablation design, not wholesale adoption of the much larger LATFUSE002, guaranteed recovery of content-based retrieval, or claims of commercial-level real-world intelligence.

Follow-up issue #444 tracks native-only next actions; LATFUSE002 key-collision issue #447 is retained as a separate experimental family.
