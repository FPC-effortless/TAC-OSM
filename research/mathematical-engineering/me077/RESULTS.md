# ME077 — selective native physical prediction: mixed development result

Five original models trained on authentic BodenAI RW-RL-HIL observations.
**MIXED_TRADEOFF: meaningful improvement over ME076, full registered gain gate
FAILED.** This is exposed historical development, not independent generalization,
control intelligence, architectural superiority or a commercial qualification.

| Same original real query cohort | Queries/seed | ME076 MAE | ME077 MAE | Persistence MAE | FIT servo MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All next physical states | 960 | 0.080158 | 0.066476 | 0.068668 | 0.097457 |
| Actual movement >=0.025 mean joint units | 84 | 0.748786 | 0.741981 | 0.743241 | 0.704517 |
| Near-stationary | 876 | 0.016043 | 0.001701 | 0.003983 | 0.039245 |

Means across five original model seeds; units are original recorded joint-state
units. These are exactly the same40 episodes/960 query origins and targets as
ME076, verified row by row.4800 model forecasts are **960 distinct genuine
observations**, clustered into40 episodes; repeated seeds are not new real cases.
The old and new seed IDs differ, and are paired by predefined roster order for
reference reporting; confidence uses five-model averages per factual query.

Overall error decreased **17.07% vsME076**, **3.19% vspersistence**. Stationary
error decreased89.39% vsME076 and57.28% vspersistence. Each of five seeds and
each of five task means improved overall vsboth references. Moving-state error
improved only0.91% vsME076 and0.17% vspersistence; the servo reference remains
better on moving queries. Thus the main observed gain is suppression of
unnecessary corrections, **not effective anticipation of substantial motion**.

Paired episode-cluster5,000-resample percentile95% interval for new-minus-old
MAE: **[-0.014158,-0.013212]**. New-minus-persistence: **[-0.002446,-0.001945]**.
Conditional on these fitted models and this already exposed population; no
independent training/site/operator uncertainty or multiplicity-adjusted claim.
Original counts, seeds, task scores and bootstrap are in[CAPABILITY_RESULTS.json](CAPABILITY_RESULTS.json).

## Registered decision

- Every seed >=5%all-case gain vsold AND persistence: **FAIL** (persistence gain<5%).
- Five-seed-mean moving-state >=5%gain vspersistence: **FAIL**.
- No task >10%regression vspersistence: PASS.
- Episode-cluster CI upper(new-persistence)<0: PASS.

Do not relax these thresholds or launch hidden refits. No protected tissue-box,
source CAL or new TEST was opened. Preserve ME076's original negative result.

## Integrated architecture and training

**102,503 active/deployed learned parameters**, bounded 96D native fast recurrence.
Adds already-observed velocity context; learns a stationary/moving Laplace
mixture and an auxiliary factual motion-occurrence probability. Point forecast
is the numerical mixture median, and5/95% intervals and proper NLL come from the
same mixture. The auxiliary global motion event is separately supervised; its
probability is not claimed identical to the per-coordinate mixture assignment.
No SSM replacement, policy optimization, action authorization, slow/associative
memory advantage or new mathematical invention is claimed in this System1 lane.

Native original120 FIT / 40 exposed REPORT episode split, FIT-only preprocessing,
known-at-time lag-1 actions, eight observed frames; future intervention only
filters evaluation eligibility, never a predictive input. Censoring limits
claims to actually observed autonomous transitions. Uniform eligible origins
within each round-robin source task, **no target-based motion oversampling**.
Five seeds77001–77005,1,024 AdamW updates×24 targets =24,576 supervised transition
presentations per seed (not necessarily unique), final checkpoint only, no
sweeps, restarts or REPORT selection. ME076 had160×12=1,920 presentations/seed.
Different training exposure and parameter count: **whole-system development,
not a fair architecture or sample-efficiency comparison**. CPU is report-only;
FIT receipts' CPU excludes source download/preprocessing/sampling construction.

Mean marginal90% interval coverage93.96%, width0.30094 recorded units, proper
interval score0.96283. These are **per-coordinate** quantities, not simultaneous
14D coverage or whole-episode reliability. Motion-event Brier0.07921 overall,
0.79581 on the84 moving queries: the global classifier is still weak on actual
motion. Do not call uncertainty/action selection calibrated or trustworthy from
coverage alone. No hard maximum interval or clipping of observed outcomes.

## Original evidence and replay

Preregistration/source/implementation freeze **1396564b** preceded all updates.
All five original weights+optimizer+Torch/NumPy/Python RNG checkpoints and live
receipts frozen at **1b358218** BEFORE any ME077 REPORT scoring. Original raw
forecasts and scores frozen at **3cabf74c**. Parameter preflight/final gate PASS;
actual whole-model receipt counts and1,024 completed updates verified for each.

Independent fresh GitHub clone restored all five original models, reloaded the
hash-verified source, and reproduced **every raw forecast/score** within frozen
rtol1e-5/atol1e-6, no refitting. Original[replay receipt](../../replay-bundles/ME077/INDEPENDENT_REPLAY.json)
and[model index](../../replay-bundles/ME077/FIT_INDEX.json). This verifies
reproducibility, not new-source predictive skill. Separate [GitHub replay run39992505986](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39992505986)
**PASS** at original-score commit3cabf74c, including restored source, every original
raw forecast and score, preflight and final live-model receipts. Normal CI at
that same commit also passed. [External receipt](../../replay-bundles/ME077/GITHUB_INDEPENDENT_REPLAY.json).
These observations do not promise identical execution on every platform.

**1,142 regression tests PASS** in this branch, including four new numerical tests
and eight imported capability-claim tests;182 supported Python files pass the
training-budget static audit. Count differs from DIAGKEY004 because this branch
stacks on ME075/076; no historical tests or artifacts removed. Original ME076
models/protocols/scores have zero diff. Two inherited malformed historical-fit
workflow triggers were corrected to manual opt-in only; original runs/checkpoints
unchanged and no historical training launched by this study.

## Next capability objective

Retain the full trained candidate as a developmental System1 improvement. The
next substantial physical model must improve **actual moving-state forecasts**
while keeping these stationary gains, with authentic multistep targets and a
causally valid known-action contract where available. Do not spend the protected
final task, add an attribution campaign, claim real-world intelligence from
synthetic recall, or sell an action/safety policy from these offline forecasts.
External pilot/operator value remains UNKNOWN/NOT_MEASURED.
