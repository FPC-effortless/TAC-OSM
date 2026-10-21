# ME029 — Genuine chronological state fails preregistered incremental-information gates

**Status:** ORIGINAL PREREGISTERED REAL-DATA FITS AND SHA-VERIFIED CHECKPOINT REPLAY COMPLETED; outcome = **negative**, no new architecture promoted. Results from [GitHub Actions run 39533936088](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39533936088), both jobs success; **920 repository tests passed plus 12 subtests**. Original nine-model weight/optimizer/RNG/probability artifact [#12316358946](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39533936088/artifacts/12316358946), 7,629,003 bytes. GitHub-provided and independently computed archive SHA256: **40c953657bd2163b27967bf01072691855eb3adc463b53d0d7474ed1d5dc0226**. Original archive inner \`results/me029/results.json\` SHA256: **494dded6d15d4d793ec7ce8119f915cffbd05f620e3b4a897b583b0ae30a66f2**. All 9 models (3 arms × 3 seeds) fit at exactly original 300 steps and all **53,568 authentic historical DEV forecasts** replayed exactly against SHA-pinned saved original predictions.

## Source and scope

Official publisher KuaiRec original ZIP SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c. Exactly **10,776 genuine FIT** histories (original user%16==0 before Aug01 2020), **5,952 already-exposed historical DEV** examples (Aug01–Aug14), 441 observed real DEV users; no synthetic benchmark outcomes or newly opened future / user4 / user5 cohorts. Four authentic causal next-event targets per full real history (cutoffs 32,64,96,128), weights 1/6,1/6,1/6,1/2. Same original next offered item/duration, old response-rate prior and 26 causal feature channels in all three arms.

## Fixed-300-update 3-seed mean DEV scores (lower is better)

| Arm | Full trainable params | Nonembedding trainables | Mean Brier | Mean ECE10 | Optimizer CPU sec / fit (approx) |
|---|---:|---:|---:|---:|---:|
| Chronological native ME028 nonlinear TAC | 135,377 | 7,377 | **0.164198272** | **0.027659302** | 29.42 |
| SAME native TAC, blocks of 32 sorted by item ID before recurrence | 135,377 | 7,377 | **0.163810937** | 0.028373479 | 29.93 |
| Genuine order-free pooled nonlinear 64-D head | 135,913 | 7,913 | 0.164478885 | 0.034671441 | 3.43 |

**Important:** Chronological native exactly reproduced the prior ME028 historical DEV score 0.164198272; this is a fresh fit on the same FIT/DEV source, not a new independent dataset. Sorting is deterministic within fully observed 32-event blocks, with attached event feedback/timing/duration moved together. This is a block-boundary causal order treatment, NOT physically valid mid-block streaming; original per-event timing metadata can still encode temporal traces, and sorting itself can create a different inductive bias. The pooled control treats history as a set and cannot recover event order (but has a nonlinear projected-mean body). Original state equations are unchanged in both native arms.

### Registered primary: chronological native versus order-free state

The mean chronological − order-free Brier contrast is **−0.0002806130** (chronological somewhat better on average, far smaller than registered −0.0005 threshold). Fixed-seed paired differences:
- seed14001: **+0.001702202** — chronological native **WORSE**
- seed14002: **−0.001776957** — chronological native better
- seed14003: **−0.000767084** — chronological native better.

Primary gate **FAILS** because not all three seeds improve and mean gain is too small. User-ID clustered paired 1,000-bootstrap 95% CI on exact original historical 441 users: **[−0.001107810, +0.000676942]**, includes zero. This is conditional on three frozen seeds, and can NOT correct repeated DEV usage or compute mismatch.

### Mandatory second gate: chronological native versus block-sorted same native model

Chronological − block-sorted mean Brier **+0.0003873350** (chronological **WORSE**). Per seed:
- seed14001: +0.000372094
- seed14002: +0.000419559
- seed14003: +0.000370352.

Block sorting wins **ALL THREE seeds** on this reused development split. Chronological order advantage gate **FAILS**. Paired user-clustered 95% CI **[−0.000237691, +0.001035665]**, includes zero. Hence no strong inferential evidence of population ordering harm, and especially NOT proof that chronological history is intrinsically useless.

### Calibration and state substitution

Mean chrono ECE10 0.027659 vs order-free ECE10 0.034671; the chrono-vs-pool calibration guard PASSES (difference **−0.00701214**). Block sorting ECE10 0.028373, so modest mean calibration tradeoff while Brier improves. The **global promotion rule fails** because both independent accuracy requirements failed.

Fixed trained-state FIT-mean substitution per seed (change in real historical DEV Brier, positive means varied state helps):
- chrono seed14001 **−0.001040704** (replacing the learned state HELPED the model);
- chrono seed14002 **+0.000459793**;
- chrono seed14003 **+0.000268087**.
- block-sorted native: **−0.00111564, +0.00097554, +0.00050501**.
- order-free pool: **−0.00014074, +0.00085466, −0.00033300**.

None of these observational mean-state interventions is an identified causal state-value effect or a conditional information estimate. They further caution against presenting native hidden state as consistently useful.

## Dense registered learning curves (no best-checkpoint cherry-picking)

| Fixed update count | Chronological TAC mean Brier | Block-sorted TAC | Order-free pooled nonlinear |
|---:|---:|---:|---:|
| 25 | 0.18694658 | 0.18655269 | 0.18558512 |
| 50 | 0.16655109 | 0.16637262 | 0.16660631 |
| 150 | 0.16365411 | 0.16422426 | 0.16366110 |
| **300 (only registered primary)** | **0.16419827** | **0.16381094** | **0.16447889** |

The order-free model uses about **8.6× less optimizer CPU** than native TAC at the same 300 step/target exposure on this specific GitHub runner, but its calibration is worse. That is an engineering observation, not full FLOP/data/parameter-compute architectural superiority; it uses a different order-invariant inductive bias and around 7.3% more *nonembedding* parameters even though global active parameter gap <0.4%.

## What is learned, and what NOT to claim

ME029's non-promoted results strongly challenge the assumption that the present native recurrence extracts crucial extra forecasting signal from original event order beyond the common nonlinear observable readout. However, the 32-block sorting manipulation alters the data presentation in a way that also interacts with attached time-gap metadata; it does NOT prove strict permutation invariance or chronological failure in all real tasks. The order-free control is near-equal trainable count and identical genuine source/data exposure but actual CPU time and FLOPs differ; scientifically strict fully equal-compute winner remains UNPROVEN. This was a repeatedly inspected, historically exposed development period, NOT an independent first-exposure replication.

**Do not promote** the nonlinear ME028 variant, sorted native or pooled model based on ME029. Do NOT add memory gates, hierarchical NCM codes, Sep-Latent anchor tokens or iterative latent System-2 computations in response to this result: such complexity is unearned. The native path and all previous negative PRs remain intact, unmerged. A productive next step is to build a genuinely source-only FIT-user crossfit conditional residual test controlling identically optimized nonlinear readout, and preregister a NEW truly unseen real cohort after model design is frozen. Compute-match architectural comparisons by measured actual compute and item/content feature access; take advantage of efficient inference only if real predictive utility is present.

## Reproduction and audit

Original workflow [#39533936088](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39533936088): source-based protocol \`research/mathematical-engineering/me029/PROTOCOL.md\`, models \`src/tac_osm/me029_order_state.py\`, driver \`scripts/me029_order_state_research.py\`, tests \`tests/test_me029_order_state.py\`. Download [original nine trained models and exact predictions](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39533936088/artifacts/12316358946), verify original ZIP SHA above, inspect original JSON and per-checkpoint SHA digests, then run \`python -m scripts.me029_order_state_research --out results/me029 --replay\` with the authenticated official publisher dataset (or compare recorded per-event probability arrays). Never claim new outcome access. No code weights or selection rules were changed after observing these results.
