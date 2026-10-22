# TAC-OSM capability-first research reset — 2026-10-22

Status: governance and architecture-design correction, **not** a trained capability
improvement. Preserves all ME001–ME051 experiments and original scores/weights.
This supersedes **new experiments** proposed by the dated
[architecture plan](ARCHITECTURE-RESEARCH-PLAN.md); it does not erase it.
[Real-world evidence policy](REAL-WORLD-EVIDENCE-RULES.md) remains binding.

## WHY THE PREVIOUS RESEARCH STRATEGY IS BLOCKED

The users' compulsory parity rule was already written, but actual training
continued with different trainable parameter counts and wildly different CPU
consumption. That is a process failure. A footnote about mismatches is NOT a
substitute for a pretraining hard gate. The earlier sequence of small experiments
has also retested generic questions (can a recurrent module forecast? can an
auxiliary old-history reconstruction head fit?) without closing TAC-OSM's key
intelligence question. Do not schedule more retrieval-head, gate-bias, pairwise
order, multiscale microvariation or inference-speed studies until they are
necessary to address a capability-blocking failure of an integrated model.

Examples of nonqualifying historical **architecture comparisons**, retained as
engineering or practical-reference evidence only (no retroactive parity):
- ME049/PR315: context, GRU, native and action-injected native trained with
  different active parameter counts and CPU costs; eight model fits have genuine
  data but are not a fully controlled four-way architecture comparison.
- ME050/PR316: native chronological-vs-reversed is a *partially matched*
  within-native ablation; GRU/context comparator sizes and optimizer CPU differ.
  Full cross-architecture parity was not reached.
- ME051/PR317: two native variants have identical 46,180 active parameters;
  GRU 52,362 and context 41,866, and CPU differs by multiples. No validated
  TAC-vs-GRU architecture-superiority comparison, even though original human
  data, checkpoints and prediction replay are valuable.

## DON'T REDISCOVER EXISTING MACHINE LEARNING

Read prior art FIRST, cite mechanism/method and identify the unresolved
capability gap. Established observations are NOT TAC discoveries:

| Existing work | Established idea to REUSE, not rediscover | What is still unanswered for TAC |
| --- | --- | --- |
| [R2D2 (ICLR 2019)](https://willdabney.com/publication/r2d2/) | Recurrent replay, burn-in and recurrent-state staleness controls | Does TAC learn useful entity state over real delayed operational events and losses? |
| [DreamerV3 (Nature 2025)](https://doi.org/10.1038/s41586-025-08744-2) | Joint world dynamics, actor/critic imagined rollouts and robust prediction training | Can TAC's native selective relevance state help beyond a budget-matched standard recurrent world model? |
| [TD-MPC2 (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html) | Task-conditioned latent dynamics and value-oriented predictive control already exist | Does TAC's counterfactual relevance routing retain *decision-changing* rather than merely predictive facts? |
| [Titans (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) | Persistent/long-term neural memory and online memory training are established | Does TAC learn better persistent entity-state retention, without changing TAC into a Titans architecture? |
| [Mamba-2 / SSD (ICML 2024)](https://proceedings.mlr.press/v235/dao24a.html) | Carefully derived recurrent-state structure and scan/duality engineering | Apply analogous mathematical rigor to TAC's own bounded update equations; DO NOT relabel or convert TAC as an SSM |
| [Saito & Joachims (ICML 2022)](https://proceedings.mlr.press/v162/saito22a.html) | Offline policy evaluation, propensity support and variance in large action spaces | Which TAC decisions show reliably improved REAL rewards under logged supported interventions? |

The new work must cite additional relevant 2025/2026 work as needed and explain
precisely what is already solved. A result known from these papers is not a
novel TAC mechanism. Novelty claims require a falsifiable distinction and an
adequate contemporary matched rival, not novelty-by-component-combination.

## One INTEGRATED CAPABILITY PROGRAM, not dozens of tiny experiments

**Research question:** Can one fully learned, numerically stable native
TAC-OSM predictive-state agent maintain task-relevant state across authentic
long event histories, predict the real consequences of available actions,
update beliefs on verification, and take better cost-sensitive decisions than
a **fairly trained, equally resourced recurrent world model**?

Build a coherent native integrated model as a single architecture objective,
using previous TAC math/CASM/CDL lessons without treating their names as
proof of ability:

1. **Belief state z** updated by real observation and executed action; use the
   tested TAC fast/slow/associative recurrence as starting point. Retain entity
   identity and timestamps without injecting future labels; keep state across
   actual events, not fake feature-coordinate time. Event addressability is
   learned; avoid hand-coded oracle identity state or scripted policy fixes.
2. **Action-conditional dynamics** trained on authentic state transitions
   `p(z_(t+h), observed_outcome | z_t, real_action_t, h)` for multiple real
   horizons. Do not claim a causal do(a) model without randomized assignment,
   controlled policy data or justified identification.
3. **Decision-oriented representation** learns future outcomes and a trained
   value/cost prediction from actual observed cost/reward; no fabricated service
   costs, proxy action wins or trivial always-act policies. Learn policy choice
   from these values; do not substitute a scripted global planner.
4. **Verification-informed correction**, learning a bounded residual when
   later real evidence contradicts the prediction. The correction must
   preserve independent verified facts; measure destructive forgetting.
5. **Reasoning/goal refinement** may be learned as latent updates, but must
   produce improved external decisions or calibrated forecasts. A repeated
   neural pass is NOT evidence of System 2 thinking.

This is an engineering DESIGN DOCUMENT, not authorization to claim the modules
are implemented in the newest checkpoint. Avoid prescribing extra modules
before the integrated objective and actual data interface are fixed.

### Mathematical engineering, not architecture conversion

Let observed authentic trajectory be
`D={(o_t,a_t,r_t,o_(t+1),verified_outcome_t,entity_t)}`.
Train with an audited loss whose terms each have **real measured targets**:

`L = L_multihorizon_real_future + λ L_real_action_outcome
   + μ L_real_decision_value + ν L_real_verification + ξ L_state_stability.`

Learning weights/preprocessing are selected only on DEV with the same
predeclared optimization budget for every arm. No term may introduce
future-label leakage or synthetic surrogate world states. Additional
retention penalties need a demonstrable increase in actual future utility,
not just old-event reconstruction.

### Dataset acceptance BEFORE model building

Two separate, honest data lanes are necessary; do not invent a single
dataset that magically provides everything.

- **Longitudinal operational state lane:** prioritize the already researched
  authentic SCANIA Component X maintenance/readout dataset, with observed
  entity/time history and real repairs, and independently check that each
  claimed action/outcome/cost is actually available. It can support state
  tracking and forecasting. If causal logged propensities are missing, it
  CANNOT qualify counterfactual policy improvement.
- **Real logged action/reward lane:** evaluate whether authentic Open Bandit
  Dataset's actions/rewards/propensities permit meaningful supported policy
  learning and valid off-policy estimates. Contextual-bandit records support
  one-step policy assessment, not persistent multi-step entity memory.

The two lanes have independent source hashes, splits and scorecards. Mixing
them under a single "general intelligent agent" aggregate is forbidden. If
neither offers genuinely linked sequential transitions with verified decision
costs, identify an alternative *real* logged dataset before proceeding.
Avoid MovieLens and Criteo feature recurrence as substitutes for controlled
decision intelligence; their previous limitations are already established.

### The compulsory pre-flight and resource gate

Every proposed ME/DEV architecture experiment shall have a source-controlled
`research/mathematical-engineering/<id>/FAIR_COMPARISON.json` BEFORE its
first model fit. Run:
`python scripts/validate_research_parity.py --manifest PATH --phase preflight`.
The guard requires established literature and a genuine unresolved TAC
question, real-source SHA, identical sample and evaluation ID checksums,
indistinguishable feature/target contracts, >=2 matching seeds, full active
capacity within 1%, matched update/exposure/tuning and FLOP ceilings, and\nprefit profiler-measured FLOPs within 1% and CPU-seconds within 5%\n(on the same authenticated real-input batch shape). An equal CPU/FLOP cap is\nNOT enough to pass. Missing measured evidence or unequal conditions:
**STOP. NO TRAINING.**

After original training, pass `--phase final` with actual measured receipts,
training FLOPs, CPU-seconds, original sample-order replay and SHA-frozen
checkpoints before claiming comparison eligibility. Actual training FLOPs
difference >1% or measured CPU-seconds difference >5% FAILS. CPU has 5%
measurement tolerance only due to runner jitter and must be reported in full;
5% is not evidence of equal latency. All runs must be on identical hardware.
The script is a fail-closed gate for machine-readable claims; it does NOT
independently prove a JSON manifest is truthful. Original runner and
independent log/weights verification remain necessary.

Reference-only persistence and rules engines may be shown as **unmatched
practical baselines**, never as fair neural-architecture comparisons.
Any inability to match real inputs, trainable capacity AND training resources
means no comparative architecture experiment proceeds. Do not change the
thresholds after results are known.

### Resource and scientific qualification milestones

- **Data qualification:** Authentic original outcomes, entity-linked
  chronological history, real action/cost evidence, no leakage. Before train.
- **Comparator qualification:** At least two seed-matched models with
  preregistered active counts, same examples/order/sequence/token budget,
  same FLOP and CPU maximum, same validation selection access and sealed
  holdout. Before train.
- **Scientific capability qualification:** Independent heldout future
  prediction and downstream cost-sensitive decision improvements over the
  qualified matched baseline across seeds and domains; confidence intervals
  and user/entity clustered uncertainty. Report all negative results.
- **Transfer qualification:** Separate new operational site/time/domain
  never used in training/tuning; no unsupported inference from fixed FIT.
- **Deployment qualification:** Separate legal source rights, reliability,
  online intervention safety and commercial permission; this campaign does
  NOT claim such qualification.

## Explicit stop conditions

Stop rather than fabricate, relax or retrofit if source legality, logged
action/outcome support, equal actual FLOPs, same original example order,
state persistence, resource instrument or independent test sealing fails.
There is no micro-study fallback to rescue a favorite TAC model. Go back to
data qualification or mathematical architecture design, **not** trial-and-error
on an already-opened TEST set.

## Immediate authorized work in this branch

This branch implements process enforcement and the design/literature map.
**No additional ME052 model experiments are started** until an integrated
capability protocol can show real observations/actions/costs and pass machine
pre-flight parity. Existing study weights and final results stay unchanged.
