# DEV068 — learn state, action consequences and control from public outcomes

Status: preregistration; no fitting results. Supersedes no earlier evidence.

## Question and observable outcome

Can a fully neural controller learn useful state and action consequences from
actual paid inspection receipts, factual fees and final verification, without
analytical posterior/transition/action teacher targets? This is a capability
experiment in a synthetic world, not proof of CASM novelty or commercial repair
performance. REAL-MAINT-003 remains a separate measured physical forecasting
component. Real repair qualification remains unmet.

## Information and structural priors

The black-box adapter exposes public 46-field event vectors, two episode-random
16-dimensional entity identifiers, shuffled five-action choices, sensor error,
inspection price, public fees and final verification. No hidden bits, private
parents, simulator transitions or known-law action targets enter the primary
learner. Typed fields, exact identity cosine binding, two entities, five actions,
12 paired-measurement/graph classes, 21 future/terminal-verification classes,
64-event FIFO and a six-type supervised admission gate are explicit supplied
structure. Neural CASM is a dense mixture of three MLP operators; not discovered
symbolic programs, sparse hardware execution or C5 indexing.

An independently sampled uniform random behavior policy supplies factual action
transitions. Extra pairs of paid exact inspections obtain training state labels
before each main action and after nonterminal actions. Their action/receipt/event
traces and fees are retained. These instrumentation records NEVER enter the
student's shadow history; policy budgets count only its main actions. This is
training instrumentation, not free information available to an evaluated agent.
The synthetic physics assumes static state between interventions and IID repair
randomness; additional probes advance simulator RNG/action-clock counters but
neither alter physical state nor change that transition law. Real inspections
may violate these assumptions. Public relative event-gap inputs omit those
instrumentation clock gaps. Main action candidates still shuffle normally.

Only final verification may expose fault count and the terminal physical penalty.
Intermediate costs contain observations/inspection/repair fees, never differences
in provisional hidden-fault penalties. Terminal learning cost adds a fixed 20-unit
noncompletion penalty, including physically healthy budget exhaustion; this is
reported separately from actual physical fees, not a claimed real monetary price.
Final verifier overhead is unpriced by the inherited simulator and disclosed.

## Arms and boundaries

1. `tac_outcome`: shared entity recurrence, CDL-style bank attention, neural CASM.
2. `gru_outcome`: conventional sequence GRU, matched parameter budget.
3. `no_casm_outcome`: same entity/attention path, retrained matched MLP refinement.
4. `tac_teacher_belief`: explicit teacher reference; only its encoder uses exact
   known-law PUBLIC posterior targets. Its other heads still use factual outcomes.

Arm 4 cannot open evaluation or qualify a public-outcome candidate. Its posterior
file is separately named and no primary loss consumes it. Graph labels come only
from public relation records. All arms use identical main/probe data, seeds,
update budgets and policy objectives; parameter counts within 5% of TAC.

## Three stages, then frozen selection

Encoder: 8,192 AdamW updates, batch64, proper public measurement CE (teacher arm
proper posterior KL), float32 raw / float64 EMA .995, clip1, weight decay .0001.
World: freeze encoder; 4,096 float64 updates batch128, proper 21-class CE plus
MSE of immediate physical cost divided by40; chosen factual action only.
Policy: freeze encoder AND world; 8,192 float64 updates batch256, offline Double
DQN with EMA .995, undiscounted finite-horizon continuation, Bellman MSE, CQL
coefficient .01 (`logsumexp(-Q)+Q_observed`). World/policy LR .001. No reward
weighting of probability targets. No policy gradients into probability heads.

Q receives ONLY learned future probabilities, learned cost, public candidate
kind/binding, prices and remaining action budget. No encoder latent, actor vector,
current belief or analytical planner bypass. Action selection minimizes learned
Q using inherited epsilon1e-6 deterministic index tie selection; no stop threshold,
retry rule or forced STOP at the horizon. External verification stays independent.

Pilot seed96800 at encoder LRs .001/.003; both run all four arms. Select each arm's
LR by validation public-measurement NLL, ties smaller LR. Candidate entry requires
measurement NLL at least .02 below a Laplace-smoothed TRAIN-only public-graph-
conditioned empirical prior, plus >=95% completion and zero final faults in EACH
H32/H128 32-world validation cell. No analytical posterior-tail pilot selection.
Among eligible outcome arms at their selected LRs, choose minimum mean validation
physical cost, ties lexical arm name. Teacher-only success cannot open tests.
No eligible outcome arm => VALIDATION_REJECTED; all tests remain closed.

If entry passes, execute the complete four-arm ×20 fresh training initialization
matrix, including poor arms; no cherry-picking. Before opening tests the selected
primary must pass the same validation screen in at least18/20 fresh fits. Otherwise
close VALIDATION_REJECTED. No post-test recipe tuning. Incomplete/capped/failed
fits cannot qualify. Preserve all fits/curves, caps, pilot failures and selection.

## Data, resources and tests

32,768 minimum training transitions (finish final episode), 512 minimum validation
transitions. Fresh procedural namespaces6681001train/6681002validation,
6681003primarytest/6681004secondary; initialization96801–96820. H32/H128 training;
sensor errors .1/.2 and inspection fees .05/.1/.2 balanced by episode index.
Maximum10 main actions. Two local CPU workers; no paid compute. Per-group CPU cap
1,800 seconds, total fitting/preparation cap24,000 CPU seconds. Scheduling retains
budget for a full next pair of groups; incomplete matrices cannot promote.
Smoke uses +10,000,000 namespace offset, tiny data/updates; excluded from evidence.

Primary test: all20×4 arms, 64 shared fresh worlds each H32/H128/H512/H1024,
error .1 / inspection .1. Secondary32worlds each H128/H1024, error.2/price.4.
Known-law posterior is allowed ONLY as a diagnostic evaluator for outcome arms;
report median/p95/p99 KL and exact-posterior action flips. It is not primary
training supervision. Candidate promotion requires >=18 individual seeds with
>=95% completion in EACH primary history and p95KL<=.05/p99KL<=.1, no additional
final faults vs either matched control in any seed/stratum, and simultaneous
cost noninferiority upper bounds <=+.02 vs GRU and no-CASM (paired seed averages,
one-sided Bonferroni95% across two contrasts, fixed 10,000 paired-seed bootstrap
resamples RNG968003). Secondary reliability regressions also block promotion.
Report all individual seeds, fault exposure/interventions, physical cost and
separately charged acquisition. No superiority claim from saturation/parity.

## Integrity and release

Commit protocol + actual fitting source before actual data collection/fitting.
Do not change frozen files to repair a failed screen. Preserve old DEV058–067 and
REAL002/003 artifacts, including unopened MetroPT July B. New experiments require
new registrations. Replay every saved final checkpoint against retained public
inputs, independently verify corpus accounting/hashes and teacher-file separation,
and package a Torch-only inference artifact if a complete model exists. Native
Veritas outcome replay stays independent and must not be called real-world
validation. The broader maintenance goal stays ACTIVE until actual outcome/cost
data, external generalization, reliability and economic gates pass.
