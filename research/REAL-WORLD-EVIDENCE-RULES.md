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
7. **Use credible contemporary neural architectures as the PRIMARY scientific competitors.**
   A capability-focused study must compare TAC-OSM against a properly trained,
   faithfully implemented Transformer/attention architecture and a competent
   recurrent/SSM competitor wherever relevant to the declared task. Report
   exact depth, width, parameter count, tokenizer, training budget, context
   access and version; a tiny one-layer attention control must never be
   described as the representative modern Transformer standard. Preserve
   simpler current-byte/window, persistence, regression, calibrated-tree and
   always-act controls as **diagnostics and practical lower bounds**, to catch
   shortcut tasks and overcomplexity. They do not determine the headline
   architecture ranking or justify inventing tasks that the simple controls
   cannot solve. When compute/model scales differ, call the capable reference
   unmatched and do not claim a parameter-matched superiority result.
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

## Model-scale policy v2 — prospective amendment approved 2026-10-31

**Supersedes only the unconditional 100,000-parameter ceiling for newly declared v2 experiments.** The user's explicit instruction to change the model-size rule permits controlled larger-model studies. All earlier committed experiments remain under their *original* scale and evaluation contracts; the amendment never retroactively changes counts, scores, failures, or eligibility. The original 2026-10-23 rule remains enforced on historical/unamended prospective manifests for reproducibility.

### New rule 12: preregister the scientific scale, not an arbitrary universal ceiling

- Any genuinely **new** training registration may choose a scale justified by its research question: e.g., ~100K for mechanism checks, ~1M / 10M for capability-learning and scaling studies, ~50M / 100M / larger for modern sequence-model research. These are examples, **not exact tiers or automatic training authorizations**. No arbitrary upper parameter limit is imposed by the scale policy.
- The frozen `FAIR_COMPARISON.json` must specify `model_scale_policy: "declared-target-v2"` and `parameter_scale` containing **integer** `target_trainable_parameters`, `target_deployed_parameters` (>= trainable), `relative_tolerance: 0.05`, `registered_before_first_fit: true`, and a meaningful `rationale`. Exact declared targets are mandatory before optimization. Missing v2 declaration retains the earlier 100K legacy gate for existing manifests; no implicit exception or retroactive relaxation.
- Every trained arm must fall within ±5% of its **declared absolute trainable and deployed** targets. For architecture superiority claims, separately maintain the existing tighter **≤1% trainable/deployed parity between arms**, identical authentic source/splits, training exposures, optimizer schedule, tuning budget and measured training FLOPs. The absolute band does **not** replace the cross-arm bound.
- All learned parameters count: embeddings, attention, recurrence, associative state encoders, routers, heads, learned experts and **pretrained/frozen weights deployed by the evaluated model**, including backbone weights. Do not pad with dummy parameters, silently exclude pretrained backbones, or compare a 100K model to a 100M model and call them parameter-matched.
- There is no requirement to equalize **CPU wall-clock** time between architectures. Report end-to-end time, prefill/decode throughput, memory/KV/state, actual training FLOPs, optimizer updates, data tokens, hardware, and energy if independently measured. Equal-compute and equal-data learning curves are separate mandatory analyses for comparative claims.
- Contemporary large pretrained Transformer or Mamba-family systems may be **unmatched capability references** when clearly labeled, even when sizes differ. They cannot establish matched superiority of TAC and must not be mislabeled as ~100K controls.
- Do not train merely because this scale amendment exists. A registered authentic-data cohort, source/target hashes, licensing/split/evidence review, **measured prefit model counts**, sufficiently specified optimizer and compute budget, signed independent-source/reward evidence and valid authorized training workflow remain mandatory. No benchmark result, exploratory DEV score or model-size selection may enter prefit target selection.
- Classical, stateless, windowed and other inexpensive controls remain compulsory diagnostic or practical baselines, but are not substitutes for a credible, properly trained contemporary architecture in primary capability comparisons.

### Historical continuity and scope

The former 2026-10-23 **100K v1** policy applied to all new trained neural models at that time, with trainable/deployed size 95K–105K and cross-arm parity ≤1%. Preserve this history. Existing v1 studies and their locked runs—including LANGMATH101, ME073+ or DEV062+ registered before this policy change—**retain** their original 100K check, losses, experimental design and independent-attestation requirements. ME001–ME072 and DEV001–DEV061 retain the original pre-100K grandfathering rule. Editing a historical model manifest to `declared-target-v2` is forbidden; start a **new experiment ID** for any larger fresh fit, fine-tuning, additional seed, or changed source/architecture. In particular, PR #679 and its 135/138 original fits plus three post-hoc recoveries are immutable.

The validator `scripts/validate_research_parity.py` now recognizes explicit v2 targets in addition to legacy 100K registrations. A standalone `--model-parameters` check is **count-only**, never a signed training authorization; `--target-trainable-parameters` and `--target-deployed-parameters` may be supplied for numerical validation. The signed research-integrity gate, prospective source audit, final measured checkpoint parity, and independent protected evaluation requirements remain binding.

**Scientific evidence and promotion gate remain unchanged:** Real authentic data are required for primary capability claims. Synthetic tasks are wiring and shortcut diagnostics only. No claim of superior intelligence, commercial readiness, or reliability is authorized merely by scaling a model.

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
