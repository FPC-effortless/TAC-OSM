## Owner-directed LANGRECOMP100 continuation — 2026-10-31

The owner's “try it” applies activation recomputation to the existing trained
10M models. Allow only the registered LANGRECOMP100 paired16-update FIT-only
continuations, exactly10,000,000 active/deployed parameters, three fixed seeds
and two execution policies. One documented measurement-correction replay is
allowed (at most192 optimizer steps across two attempts from original parents).
See its SCALE_AUTHORIZATION.md. All original-source,
data, optimizer/replay, compute and no-promotion gates remain. Default100K and
all historical099 evidence remain unchanged.

## Owner-directed LANGTRAIN099 scale exception — 2026-10-30

The owner's explicit request to test a trained10,000,000-parameter model
supersedes the100K limit ONLY for LANGTRAIN099 under
`research/mathematical-engineering/langtrain099/SCALE_AUTHORIZATION.md`.
Require exactly10,000,000 active/deployed learned parameters per declared seed;
all source, FIT/DEV, checkpoint/replay, compute and scientific gates remain.
This is internal research, with no protected TEST, promotion or architecture
superiority authority. The default100K gate and all historical results remain
unchanged for every other study.

## Execution-level amendment — operator instruction, 2026-10-27

Ordinary exploratory research on genuine, publicly licensed, authorized FIT/DEV
records may run as **internal_research** without independent cryptographic
signers. Require recorded publisher/license and hashes, frozen splits, causal
correctness, replayable original checkpoints, honest metrics and all existing
100k/parameter/data/training-budget comparison gates. Findings must be labeled
**internally measured, not independently qualified**. This supersedes earlier
blanket signer requirements for ordinary internal research only.

**independent_qualification** retains external verification for protected
evaluations, reward-integrity certification, safety claims and checkpoint
promotion. No internal flag authorizes protected TEST, certification or
promotion. Preserve failures and historical work; never fabricate signatures.
See `research/RESEARCH-EXECUTION-LEVELS-2026-10-27.md`.

# TAC-OSM research policy: real-world evidence is the promotion gate

**Effective:** 2026-10-19. **Status:** standing rule for every NEW research
registration, model-selection decision, architecture claim, and research handoff.
This changes prospective evaluation rules; historical experiments, their locked
protocols, failed runs, and evidence remain intact.

## Non-negotiable requirements

1. **Real-world, independently sourced observations are the default training
   and the REQUIRED primary evaluation.** New capability experiments must target
   externally collected records from actual operational equipment, actions,
   people, institutions, or physical systems. Identify the exact source, version,
   licence, dates, entity identity, measurements, event definitions, missingness,
   collection method, and file/checksum provenance before fitting.
2. **No synthetic/simulated benchmark can establish TAC-OSM capability,
   reliability, architectural superiority, world-model performance, or
   commercial readiness.** Generated event streams, scripted worlds, handcrafted
   labels, simulated sensor physics, benchmark oracles and injected rewards may
   only be used as clearly labelled unit/integration tests, negative controls or
   mechanism diagnostics. Do not spend a new standalone research registration
   optimizing synthetic task accuracy. A passing synthetic suite has *zero*
   weight in primary scientific or product promotion.
3. **Physical laboratory measurements are not field deployment.** A real
   instrument with an experimentally injected fault is empirical physical data,
   but any performance claim must say it was measured on controlled apparatus,
   not on naturally occurring operational failures.
4. **Evaluate the exact information available at prediction/decision time.**
   Preserve time order, source entity boundaries, censoring, operational
   missingness, label availability and shift. No future leakage, post-action
   fields, oracle belief, answer-bearing identifiers, hindsight-only labels,
   fabricated alternative outcomes, or arbitrary train/test reassignment.
5. **Separate prediction, decisions and causality.** Predicting a repair label
   or an observed treatment outcome is *not* evidence that a proposed repair
   prevents failure. Estimates of real-world intervention benefit require
   identifiable comparisons (randomized policy/propensities, justified
   observational identification, or prospective authorized measurement).
   Publisher-provided evaluation-cost matrices are **expert-defined proxy
   costs**, not measured savings or actual inspection/repair invoices.
6. **Pre-register a falsifiable real-data protocol BEFORE any held-out labels
   are opened:** hypothesis, approved cohort/exclusions, measurement windows,
   entity/time/site split, access budget, preprocessing fit only on train,
   missingness/censoring handling, baselines, metrics, thresholds, seed count,
   risk stratification and decision rule. Keep test frozen; all attempted runs,
   failures and negative results remain in the ledger.
7. **Compete against the simplest good solution, not only TAC ablations.**
   Depending on the task, include last-observation/persistence, useful
   regression/AR or survival/hazard baselines, calibrated trees, GRU/SSM,
   attention if relevant, and do-nothing/always-act policies wherever actions
   truly exist. Match information, horizon, training budget and realistic
   end-to-end compute; measure memory, latency, calibration and total cost.
8. **Report the limits explicitly.** Real-world *offline prediction* is not
   real-world *policy utility*, causal counterfactual ability or autonomy.
   Do not claim commercial qualification without observed inspection/action
   logs, actual prices and independently verified post-action physical outcomes,
   plus out-of-time/site evaluation. The missing fields are blockers, never
   blanks to fill with synthetic proxies.
9. **Existing synthetic studies remain reproducible historical diagnostics.**
   Freeze their findings and preserve their hashes/checkpoints, but do not
   extend synthetic-only pilots or promote them to primary evidence. In
   particular, the SEQ-TRANSFER-006 synthetic pilot on PR #232 is engineering
   evidence of a narrow, supervised ability only; its reserved synthetic TEST
   cannot satisfy this policy or be treated as a world-model milestone.

10. **Compulsory equal-resource architecture comparisons.** Any new experiment
    that ranks TAC-OSM against GRU, SSM, Transformer/attention, MLP, memory
    ablations or another *trained architecture* MUST satisfy the matched
    architecture-comparison contract below for its primary superiority claim.
    Equal real-world data, trainable model parameters and training budget are
    experimental requirements, not optional caveats. Non-matching runs are
    exploratory and cannot establish architectural superiority.

## Compulsory prospective model-size budget — approximately 100,000 parameters (effective 2026-10-23)

**Non-negotiable rule 12 — every NEW trained neural model, architecture variant, ablation fitted from scratch, fine-tuned model, individual seed and trained comparator MUST target approximately 100,000 parameters.** This requirement applies to native TAC-OSM, Concept-CED, GRU, RSSM, SSM, recurrent/attention controls, event models, physical models and any other learned model developed in the TAC-OSM research program. The training environment, benchmark domain or source dataset does not create an exception.

- **Exact target:** 100,000 trainable parameters **per independent trained model**, with an inclusive ±5% tolerance: **95,000–105,000**. This is an absolute model-size requirement, not merely a condition that compared architectures have similar sizes. Count encoder, recurrent/fast/slow/associative state projections, shared experts, concept components, action encoders, all decoder passes, routing, forecasts, uncertainty and decision heads, trainable embeddings and other scored-pathway learned parameters. Count shared weights only once; repeated decoder execution does not multiply parameter count. Independently count **all deployed learned weights**, including any frozen pretrained backbone: the same 95k–105k scale applies. Report untrained deterministic priors and runtime memory separately.
- **Fair matches remain stricter:** for two or more trained alternatives, every arm must individually lie within **95k–105k**, AND trainable and deployed counts must match **within 1% between arms**, with the original real-world data, split, optimizer steps and training/FLOP budget fair-comparison requirements unchanged. A matched pair of 40k or 125k networks **fails** the new absolute rule even if their sizes match each other perfectly. A 95k-versus-105k pair **fails** the existing 1% parity rule.
- **Fail closed before any optimizer step:** record exact live-model `sum(p.numel() for p in model.parameters() if p.requires_grad)` and `sum(p.numel() for p in model.parameters())` from the **actual instantiated implementation**. Record these in the preregistered `FAIR_COMPARISON.json` (single trained models use `study_type: single_model_training`). Check again against the actual fitted model and checkpoint in the postfit `--phase final` evidence. New noncomparative diagnostics must declare `no_new_model_fits: true`; a purported diagnostic must not train around this rule. The machine validator `scripts/validate_research_parity.py` enforces the prospective 100k count alongside the existing 1% fair-match gate; standalone implementations may call `--model-parameters COUNT --deployed-parameters COUNT` before fitting.
- **No gaming:** do not add unused/dummy weights, frozen weight padding, artificial width inflation, external hidden learned models, selectively uncounted heads or a materially weaker comparator just to hit the size. Make every counted trained component useful and trainable on identical eligible data; count all model-owned pretrained/frozen learned weights within the deployed total.
- **Classical controls:** pure persistence, deterministic physics/servo rules and non-neural analytical filters with no learned neural parameter tensor remain baseline methods, not ~100k-parameter neural training candidates. Keep them as compulsory external baselines instead of artificially padding them.
- **Historical preservation and prospective scope:** ME001–ME072 and DEV001–DEV061 are frozen legacy registrations and retain their original parameter counts, scores and checkpoints; do not retrain them, manufacture larger models or retrospectively relabel results. The first future ME-series registered model training is ME073 and later (DEV062 and later); new unrecognized study IDs also follow the 100k rule. **A new fit, resumed new experimental training, newly trained seed or new model configuration—even when forked from legacy code—falls under this rule.** Renaming a new experiment to an old legacy ID is prohibited. The software's historical-study compatibility path is only for replaying unchanged original results.
- **Exceptions:** there is no implied exception for convenience, device runtime, unavailable compute or an attractive preliminary result. If a fundamental scientific reason requires a different model scale, obtain an explicit updated owner research instruction and formally amend this rule **before** beginning that registered study; document the deviation as a separate model-scale research program, never present it as satisfying this 100k contract.
- **No CPU wall-time parity:** the existing rule that model CPU seconds do NOT have to match remains intact. Same original real source, fair optimizer budget and comparable measured training compute remain compulsory; the new 100k target does not replace them.

**Promotion gate:** missing counts, 94,999 or 105,001 trainable/deployed learned parameters, uncounted learned modules, an unmatched comparator, or an absent pre/post-fit receipt is a **training authorization FAIL** for prospective 100k studies. Do not spend real model-training budget until the absolute and relative parameter gates pass on the real implementation.

## Compulsory architecture-comparison contract (hard gate)

**Applies prospectively to ALL trained-architecture comparisons and ablations**
including TAC-OSM vs GRU, SSM, attention, other recurrent models, and
retrieval/planning variants. Preregister the complete arm matrix before any
held-out result is inspected. A table of uncontrolled model sizes, data or
budgets is an *exploratory systems comparison*, NOT a valid head-to-head
architecture win.

1. **Equal trainable parameter counts.** Count every trained component used
   in the scored pathway (encoders, state/memory, adapters, projectors, readout,
   predictors, decision heads and controllers). Match exactly when feasible;
   otherwise set the same preregistered target and demonstrate an absolute
   relative parameter gap of **at most 1%** for *every* trained arm before
   training. A comparison above that tolerance is **INELIGIBLE** for a
   parameter-matched superiority claim. Record total deployed parameters,
   non-trainable memory/state bits and actual activation/KV/state storage
   separately; do not hide capacity in external learned weights or caches.
2. **Equal real data and exposure.** Use the identical original real-world
   source/version, eligible entities, feature channels, available-at-decision
   timestamps, labels, training cohort, train/validation/test split, causal
   preprocessing, prediction horizon, observation history/context, sample
   order or deterministic shuffle schedule, and missingness/censoring rules.
   Match examples, tokens/readouts, sequence lengths, batches and *total data
   presentations* per seed. Do not use extra pretraining, distillation,
   synthetic augmentation, privileged states, extra modalities or lookahead
   for one arm unless every arm receives the equivalent allowed source and
   exposure. Any such study needs its own separately labeled non-parity lane.
3. **Equal training resource budgets.** Pre-register and enforce the SAME
   optimizer-step/exposure ceilings AND the SAME maximum charged training
   compute for all arms. Report actual FLOPs/MACs or audited profiler
   estimates, GPU/CPU model and count, accelerator-seconds or CPU-seconds,
   per-arm update count, examples seen, peak memory and failures. Use the
   identical compute measurement convention, environment and seed matrix.
   End-to-end costs include representation building and all learned
   components. **Unused budget is not evidence of equivalent spend.**
   Architectures that cannot meet both the matched-exposure and matched-compute
   gates must NOT be claimed as having won a fully matched comparison; run
   separately identified *compute-matched* and *exposure-matched* analyses and
   disclose that the three-way parity gate was unmet.
4. **Equal selection and tuning privileges.** Apply identical validation-only
   selection rules, hyperparameter-search trial counts, tuning FLOP/time
   allowance, early-stopping opportunities, seed counts, checkpoint policy,
   retraining rules and access to held-out information. Report all failed
   runs and tuning attempts. Hold test labels sealed until the registration's
   one permitted final evaluation; never tune only TAC-OSM or a favored rival.
5. **Equal evaluation contract.** Score on exactly the same held-out entities,
   periods, recorded actions (when applicable), targets, metrics, threshold
   rules, deployment constraints and statistical aggregation, including
   per-seed distribution and confidence intervals. Report predictive
   accuracy/calibration, decision cost *when genuinely observed*, and
   inference memory/latency/energy as separate dimensions. Equal parameter
   count does NOT imply equal compute or deployed memory.
6. **Baselines with inherently unequal parameters.** Persistence, prevalence,
   fixed physics and do-nothing/always-act baselines **MUST still be reported**
   as practical reality checks, but explicitly label them *unmatched
   reference baselines* rather than padding them with useless trainable
   parameters. They can establish whether a complex model is useful at all;
   they cannot be evidence of a parameter-matched architecture comparison.
   Likewise unmatchable pretrained or foundation models belong in a separate
   clearly labeled transfer/deployment comparison, never the parity lane.
7. **Mandatory audit and claim gate.** For EACH arm publish a machine-readable
   comparison manifest: source checksum; fixed IDs/splits; input contract;
   effective history and data presentations; trainable/total parameter count;
   training and tuning budgets **allocated and actually consumed**; hardware;
   seed; selection rule; checkpoints; inference resources; and evaluated
   outcomes. State PASS/FAIL for parameter, data/exposure, compute,
   tuning/selection and test parity before discussing performance. **Any FAIL
   means NO ARCHITECTURAL SUPERIORITY CLAIM**, regardless of the winning score.
   Never retrospectively loosen parity thresholds after results are known.

Preserve all historical studies with their original controls and accurately
label those without full parity as historical diagnostic or unmatched evidence.
This contract does not authorize synthetic primary evaluation and does not
replace the stronger real-world outcome/causality requirements above.

## First primary real-world dataset: SCANIA Component X

- **Original operational data:** heavy-duty truck fleet with 33,641
  anonymized vehicles, irregular longitudinal readouts, specifications, and
  workshop records of Component X repair/replacement.
- **Source and fixed version:** Swedish National Data Service / Researchdata,
  version 3 (2025-04-09), DOI https://doi.org/10.5878/bnh5-ka77 ;
  https://researchdata.se/en/catalogue/dataset/2024-34/3
  (11 files, about 1.54 GiB; CC BY 4.0). Document all file hashes after
  actually acquiring them; a catalogue entry is NOT download verification.
- **Publisher split:** train 23,550 trucks / 1,122,452 readouts; validation
  5,046 / 196,227; test 5,045 / 198,140. Retain disjoint vehicle IDs and
  the published validation/test partition. Train has 2,272 observed repairs
  and 21,278 right-censored/nonrepair study endpoints. Validation/test labels
  classify lead time to a recorded repair in five classes, with severe
  imbalance; a no-repair observation is not guaranteed permanent health.
- **Task A (primary):** prospective *as-of-readout* predictions of
  time-to-observed-repair / failure-associated replacement at registered
  lead-time windows, censoring-aware risk, uncertainty/calibration, and
  per-class recall/precision. Never use later readouts or target labels
  for the prediction feature vector.
- **Task B (model-specific):** evaluate whether recurrent/entity-local
  memory improves *the same* true observed-risk prediction against
  last-readout and equal-information time-series baselines; report skill
  by history depth, staleness, missingness and unseen vehicles.
  This dataset does not directly label reversals/TOGGLE semantics or
  two-entity relationships: do not infer those capabilities from it.
- **Decision evaluation:** report the SCANIA challenge's **expert-specified
  asymmetric misclassification cost matrix** as offline decision *proxy
  cost*, always next to raw class metrics, PR-AUC, calibrated loss and
  false-alert/missed-alert counts. Do not call its scores actual fleet
  savings. No actionable proof of repair efficacy, counterfactual
  interventions, or independently verified post-repair function is in
  the released training contract.
- **Important limitations:** published values are anonymized; operational
  counters and rates, absolute dates, readout rates and repair frequencies
  may be perturbed. A random vehicle-held-out split is NOT an out-of-calendar-time
  generalization test. Independently sourced operational data must corroborate
  findings before field robustness claims.
- **Baseline shortlist:** calibrated historical prevalence/constant-risk,
  last-readout logistic or LightGBM/CatBoost, simple survival model,
  GRU/temporal baseline, then TAC-OSM. Choose details by pre-registration.
- **Status:** dataset and schema verified from publisher descriptions;
  no completed local raw-data download, model fit or performance claim
  is established by this policy document.

Publisher: Kharazian et al., *Scientific Data* 12, 493 (2025),
https://doi.org/10.1038/s41597-025-04802-6

## Independent corroboration / action-outcome path

- **Backblaze Drive Stats**:
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data .
  Actual data-centre SMART snapshots and drive-failure indicators across
  calendar years. Use untouched forward-time quarters for independent
  operational failure-risk generalization. Hard-drive removal is not proof
  of a counterfactual prevented failure; there are no measured repair actions
  or inspection charges in this dataset. Source attribution/reuse restrictions
  apply.
- **MetroPT-3 (UCI 791)**:
  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset .
  Actual train-compressor readings, four operator-reported failure windows;
  too few independent faults to support broad commercial reliability.
  Sensor prediction/diagnostic external test only. Preserve known ambiguity
  in recorded maintenance times. CC BY 4.0.
- **FHWA LTPP / Malawi waterpoint follow-up**:
  see research/commercial/DATA-SOURCES.md for already audited real maintenance,
  reported costs, measured physical states and their exact incompleteness.
  Potential action/outcome studies; neither qualifies as causal success
  without source-specific timelines, independent verification and confounding
  analysis. Never splice unrelated datasets to fabricate joint actions/costs.
- **Real logged-decision benchmark (separate domain, not maintenance):**
  Open Bandit Dataset, https://huggingface.co/datasets/zozonext/open-bandit ,
  supplies recorded choices, logging propensities and observed click reward
  on a retail platform. Eligible for a *separately registered* offline
  policy-evaluation test with correct support/positivity checks, not for
  long-horizon maintenance or physical repair claims. CC BY 4.0.

## Execution order and acceptance record

1. Source and checksum-verify SCANIA version 3 without using TEST outcomes
   for selection; audit schema, censoring, target construction, leakage and
   irrecoverable perturbations. Preserve existing supplier splits.
2. Register the readout-as-of risk prediction experiment, its real-data
   baseline matrix, calibrated performance and asymmetric *proxy* cost
   interpretation, then fit/train only after registration.
3. Independently verify on held-out trucks; separately test forward-time
   drift/failure forecasting with Backblaze calendar-based splits. Preserve
   every failed fit and all negative findings.
4. Pursue real repair/action-cost/verified physical outcomes as a distinct
   field utility gate. No simulator or arbitrary cost schedule can close it.

**Claim ledger:** engineering diagnostic / real operational prediction /
observational offline decision proxy / identified action effect /
independently verified field cost utility are different evidence categories.
Only the last category qualifies commercial claims, after its own preregistered
reliability and out-of-site/time gates.


## Enforced research reset — 2026-10-22 (prospective; read before any new experiment)

The existing equal-resource clauses above were **not adequately enforced** in
ME049–ME051. Merely disclosing unmatched parameters or CPU time does **not**
satisfy the compulsory instruction. For all FUTURE ME/DEV experiments, require
an immutable prefit `FAIR_COMPARISON.json` validated by
`python scripts/validate_research_parity.py --manifest ... --phase preflight`
and an independent postfit actual-resource receipt validated with
`--phase final`. The new `.github/workflows/research-scientific-gate.yml`
blocks changes to future experiments lacking the registration and both hooks.
If source, parameter (within 1%), training data/order, optimizer exposure,
matched training-compute allocation and measured FLOP accounting,
tuning, or exact evaluation conditions cannot be matched, STOP that head-to-head
experiment rather than
run unmatched architectures with a comparison footnote. No scientific win can
be claimed from an unverified self-reported manifest; reproduce measurements
from original job logs/weights. Existing unmatched studies are preserved as
historical diagnostics only, without score edits or retroactive parity.

Every architecture proposal must also link actual published prior work,
identify the established mechanism it REUSES, specify a still-unanswered TAC
capability gap, and prefer one integrated capability study over serial
low-information minor architecture variations. Do not run any additional
ME052 architecture training until a fully matched preflight and authentic
outcome/decision evidence contract exist. See
[Capability-first research reset](CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md).

## User correction — 2026-10-22: CPU speed is NOT a matched-budget requirement

**The compulsory matched dimensions are (1) model parameters, (2) real
training data and exposure, and (3) training/tuning budgets and evaluation
conditions.** Previously a 5%-matching CPU-runtime condition was incorrectly
added by the assistant and used to block training. **That requirement was not
requested and is rescinded prospectively.** It is NOT valid to label a
comparison unfair solely because TAC-OSM and a GRU take different amounts of
wall-clock time or process CPU time to execute the same prescribed study.

- **Parameters:** trained pathway counts within the existing 1% threshold,
  without dummy/unusable parameters.
- **Data:** exact source, splits, input features, labels, temporal horizons,
  presentations/order and independent test records.
- **Training budget:** same prescribed optimizer updates, batch sizes,
  examples/tokens/history exposure, permitted training/tuning compute budget,
  stopping/selection privileges and tuning trials, with original usage receipts.
  FLOPs may quantify and audit compute allocation and consumption; CPU runtime
  is not interchangeable with allocated training compute.
- **Runtime:** record CPU/GPU elapsed/process time, peak memory, energy and
  throughput as informative **efficiency outcomes only**. They can differ by
  any amount without triggering a parity FAIL, provided the actual specified
  parameter/data/training-budget obligations pass.

The machine preflight and final validator must **not** require equal CPU
runtime, a common CPU-time maximum, or matching profiled CPU seconds. Where
the existing profiler cannot accurately account for training FLOPs, report
that uncertainty and do not pretend partial FLOPs are fully measured. This
does not give permission to vary real data, optimizer exposure or the
common allocated compute budget to favor a preferred architecture.

The earlier 56.7% TAC-versus-GRU CPU-runtime difference is an **efficiency
measurement, not a fairness violation**. The original measured values remain
unaltered; only their scientific interpretation is corrected.


## Language-specific scale authorization — owner request 2026-10-29 (LANGLAYER035)

The owner explicitly requested training/benchmarking the previously registered ~500K
layered language TAC architecture after selecting a 500K language research track.
For study **LANGLAYER035 only**, allow live 475,000–525,000 active and total
learned model parameters with explicit `parameter_budget_contract:
owner-2026-10-28-500k`, exactly as already scoped for LANGTRAIN025.
This does **not** change the global prospective 100k rule for other studies,
relax the cross-model 1% relative size matching requirement, or retrospectively
change any historical result. This authorizes only bounded CPU internal FIT/DEV
training on the previously audited genuine publisher corpus and never permits
protected TEST, independent certification, reward-based promotion, GPU quota,
a merge, unregistered model configurations, or external expenses.

The pilot is an **exposure-matched exploratory three-arm study**, not a
training-FLOP-matched scientific superiority comparison. It may report
individual actual trained scores as **exposed-DEV engineering measurements**;
all superiority or independent generalization claims remain ineligible. Require
source integrity, frozen sampling, causal boundaries, actual parameter counts,
finite gradient checks and checkpoint+optimizer+RNG cold replay per seed.
PNDS source-level unit correspondence is inherited from #62/#63; full
whole-model Lean/IEEE/AD mathematical certification is still OPEN and must be
completed before architecture qualification, not silently treated as PASS.

## Narrow owner-directed follow-on language FFN depth control — 2026-10-29

Study **LANGLAYER037** is explicitly included in the author's 500K-language
research scale after the owner instructed to proceed from LANGLAYER036's
frozen recurrent/concept-gradient and ablation findings. Only this named
prospective two/three-layer **tokenwise FFN-only** depth-control experiment
is authorized at 475K–525K genuine learned coordinates on the identical
original CPython FIT data as LANGLAYER035. The policy validator now permits
this study ID with the exact `owner-2026-10-28-500k` contract. Its general
100K constraint remains unchanged for all other studies.

The two controls are 499,661 and 499,684 actual active/deployed learned
parameters; each matches its corresponding original recurrent-depth arm
(499,725/499,934) to much better than 1%. Five seeds 1701–1705, exactly
320 optimizer updates, B8, 128 original-byte next-targets per window, same
source-owned sample schedule and optimizer/clip, zero GPU. Original prior
15 checkpoints are immutable and only reused as *historical same-protocol
references*, never silently refit or rewritten. DEV is previously exposed;
TEST is not accessed. Historical comparison training FLOPs remain unmatched;
performance differences are engineering observations, not certified proof of
architectural superiority. Failed runs, entire seed cohort, replay evidence,
source integrity, lack of external eligibility and complete negative evidence
must remain in the research ledger.

## Narrow 500K language-model fit exception — LANGLAYER039, 2026-10-29

Owner explicitly requested training and benchmarking the already built
LANGLAYER038 recurrence-only upper TAC model. Study LANGLAYER039 only
may train at 475,000–525,000 actual trainable AND deployed learned
parameters under the existing owner 500K language program contract.
The prospective global 95K–105K rule stays unchanged for other studies;
old LANGLAYER035/037 exceptions and original runs remain frozen.

This is five internal-research CPU fits at 499,901 actual parameters on
original verified publisher CPython 3.13.0 FIT bytes: five fixed seeds,
320 updates, B8, 128 next-target bytes/sequence, original deterministic
window source order, disjoint previously exposed DEV for descriptive
metrics. No hyperparameter tuning, GPU, protected TEST or external signer
is required for ordinary internal source-verified FIT/DEV training.
Mathematics is scoped PNDS #64 only; complete composition and IEEE/AD
are openly UNPROVED. Training FLOPs are NOT matched to earlier baselines
and no architecture superiority/independent qualification may be
claimed from this run. Mandatory checkpoint hashes, model+optimizer+RNG,
source-order receipts, numerical cold replay, all five seeds, complete
negative evidence and all training failures must be retained. No promotion,
merge, release or TEST access follows automatically.

## Narrow owner-directed LANGLAYER040 factorial research — 2026-10-29

The owner directed a genuine-source ~500K upper graph × state-width factorial after LANGLAYER039. Only LANGLAYER040's two missing arms at 499623 and 499848 active/deployed learned parameters are newly authorized under the 500K language scale. The global 100K policy remains. Original five-seed 035/039 checkpoints are frozen; the new arms use identical 14 publisher CPython FIT document owners, seeds 1701–1705, 320 AdamW updates, batch8 and 128 next-byte targets per example. Four previously exposed DEV docs only. Protected TEST/GPU/production/independent certification remain closed. All 10 real trained checkpoints, optimizer/RNG/source hash and cold replay must be complete before aggregation; no failures are dropped. Decoder width varies to conserve the parameter budget; this factorial measures allocation, not pure fixed-decoder causal changes. Training FLOPs are NOT matched. Complete composed Lean, IEEE and formal AD remain open.

## Owner-directed CODEVALUE003 longer500K language run —2026-10-30

The owner explicitly requested a longer CODEVALUE002 follow-up on Kaggle dual
T4s and supplied credentials for that purpose. For **CODEVALUE003 only**, retain
the exact existing499623p TAC,500165p GRU and499782p simple selective SSM, three
seeds1701–1703, identical original CPython sampling,10240 AdamW updates atB8x128,
no tuning. This is the separate owner500K language track, never compliance with
the global100K contract. The named validator scope allows these nine original
model fits as internal, exposure-matched engineering measurements, with every
arm count within1%, checkpoint replay and final actual paired sample receipts.

Training FLOPs are explicitly NOT matched; incomplete supported-operator
profiles cannot qualify architecture superiority. This study must not claim
architectural superiority, independent qualification, general English ability,
commercial viability, reward certification, checkpoint promotion or protected
TEST access. Exact unchanged graph inheritance leaves composed PNDS/IEEE/AD
certification OPEN. All other studies keep their existing size/resource rules.

The current owner request authorizes one private offline dual-T4 allocation,
with at most240 aggregate GPU-minutes and no automatic retries. Direct Kaggle
API submission using the owner-supplied credential is in scope for this named
run; the protected Actions workflow is not being weakened or repurposed. Pin
source bytes/licenses and cohort before fit; seal all45 original checkpoints
before first held-out scoring. Existing historical evidence remains unchanged.
