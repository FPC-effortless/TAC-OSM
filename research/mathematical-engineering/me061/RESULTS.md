# ME-061 completed native world-model development

Twenty authentic-data models were trained in ME060 and ME061 combined, with
all failed results retained. ME061 is the second integrated development recipe,
not a design chosen without historical DEV access. No old source TEST
trajectory was read. All ten ME061 models froze before CAL/DEV scoring and
before the 80 registered tea-pouring transfer trajectories opened.

## Factual physical prediction

Original-unit physical MAE, averaged over five fitted seeds and episodes:

| Evaluation | Native TAC | Matched GRU | Persistence | Servo-only |
|---|---:|---:|---:|---:|
| Source DEV,76 episodes | .603894 | .602600 | .702952 | 1.105522 |
| Frozen tea-pouring transfer,80 episodes | .493206 | .509535 | .455992 | 1.421244 |

ME061 TAC improves 14.09% over source DEV persistence. Its mean is 28.56%
below the preceding ME060 TAC recipe's .845300 on the same DEV cohort. That
cross-study improvement changes capacity, initialization, features, loss and
seeds together; it is system-level engineering progress, not an isolated
mechanism or matched cross-study architecture effect.

Within ME061's matched comparison, source DEV TAC is .21% worse than GRU,
winning one of five seeds. The episode-clustered TAC-minus-GRU 95% interval
is [.000083,.002394]. The registered source development gate fails.

On the different task, frozen TAC is 3.20% better than frozen GRU and wins all
five paired seeds. The corresponding interval is [-.021299,-.012083].
However, TAC is 8.16% worse than target persistence, so the registered transfer
gate also fails. No target normalization, calibration, fit or checkpoint
selection was allowed. The task uses the same robot interface; independent
site/operator provenance is unknown. Intervals condition on the fitted models
and are not adjusted across this multi-study development program.

## Forecast uncertainty

Both models learned positive Laplace scales. The same 20 whole FIT episodes
were reserved for 90% whole-episode split-conformal calibration. On source DEV:

| Quantity,mean over seeds | Native TAC | GRU |
|---|---:|---:|
| Whole-episode coverage | 85.79% | 86.05% |
| Marginal channel/horizon coverage | 99.73% | 99.70% |
| Mean physical interval width | 11.1160 | 11.3626 |

Whole-episode coverage misses nominal 90%, and widths remain large relative to
point errors. These are implemented and measured uncertainty mechanisms, not
qualified reliable abstention or information-value intelligence. The source
observational exchangeability condition is not established. Review thresholds
must be caller-supplied in physical units; no autonomous action is authorized.

## Resource and replay evidence

- TAC 33,548 versus GRU 33,802 trained/deployed parameters: .751% gap.
- Supported-operator update estimates 51,215,499 versus 50,851,865: .710% gap.
- Final repeated estimates total 30,729,299,400 versus 30,511,119,000 per fit.
- Mean optimizer CPU 18.810 seconds TAC versus 7.905 seconds GRU.
- Allocated recurrent tensors 624 bytes TAC versus 720 bytes GRU, excluding
  activations, parameter buffers and optimizer state.
- All ten original checkpoint hashes verified in a separate replay process.
- 207,340 source CAL/DEV standardized absolute errors replayed exactly.
- Full frozen target-task result dictionaries replayed exactly without refits.
- Numerical preflight and final manifest validators pass; these are common
  profiler estimates, not independently certified complete hardware FLOPs.

Models trained from source commit 9d7b93e9 with Python 3.12.14, PyTorch 2.14.1+cpu,
NumPy 2.3.5, PyArrow 21.0.0 and one PyTorch CPU thread. Full optimizer states,
weights, traces, result arrays and provenance are preserved in the research
artifact archive; raw publisher trajectories are excluded.

## Capability assessment and next construction target

Native TAC is now a trained action-conditioned probabilistic trajectory
predictor with bounded persistent state and causal innovation statistics. It
shows material useful source forecasting and a repeatable neural-comparator
transfer advantage, while failing the strongest practical-reference transfer
gate. Neither causal policy value, action selection, useful deliberation,
continual weight adaptation nor commercial readiness was evaluated.

Retain ME061 as a development candidate. The next integrated capability target
is learning when a residual forecast is more reliable than persistence under
task shift, using authentic future losses and calibrated predictive risk.
Require unseen-task utility over the strongest prior; a better neural rival
score alone is insufficient. Continue architecture/training construction;
use detailed mechanism diagnostics only for a specific blocking failure.
