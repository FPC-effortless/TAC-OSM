# DEV-042: causal event-order transfer with identical last public events

**Status: completed and independently audited development transfer diagnostic. Not confirmatory.** See [GitHub Actions run 39005327719](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39005327719) and its [complete four-file evidence archive](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39005327719/artifacts/12091065793). Source was frozen before evaluation; no checkpoint optimization, refitting or new threshold selection occurred.

## Prospective structural test

Each paired world has H512 public events, identical final public observations for **every** entity, identical tick and query, and a future service due at the same time. The critical historical difference is whether scheduling happens *before* a generation-changing reset (service cancelled) or *after* it (service pending). Before the deadline both targets have the same future; at and after it the wait outcomes are opposite. The original DEV-041 trained models were evaluated on this new adaptive synthetic domain at 3 delays x 3 deadlines x 16 paired repetitions = **288 worlds, 96 mature pairs/192 mature worlds, 96 pending worlds**, per checkpoint. Only public events and queries were passed to models.

The exact public-history reconstruction reached 100% mature wait accuracy. A classifier restricted to the identical final event/query of each pair has exactly 50% maximum mature target accuracy on balanced pairs: it cannot distinguish opposite futures.

## Frozen deployed head classification and physical policy

| Model | Mature target wait accuracy (all three seeds) | Cost/entity per seed | Failures per seed | Unnecessary services per seed |
| --- | ---: | --- | --- | --- |
| Trained bounded MTSK | 50% | 0.563368 / 0.563368 / 0.563368 | 0 / 0 / 0 | 96 / 96 / 96 |
| Initial random writer frozen, rest trained | 50% | 0.803819 / 0.843750 / 0.850694 | 72 / 64 / 76 | 434 / 550 / 530 |
| Memoryless | 50% | 1.000000 / 1.000000 / 1.000000 | 0 / 0 / 0 | 1102 / 1102 / 1102 |
| Last-event table trained on old DEV-041 calibration | 50% | 0.563368 / 0.563368 / 0.563368 | 0 / 0 / 0 | 96 / 96 / 96 |
| Exact public-history reference | 100% | 0.521701, same cohort | 0 | 0 |

The trained recurrent controller's zero failures arise from servicing both members of each critical paired scenario: **96 unnecessary interventions**; it cannot be promoted as a successful selective controller merely on physical safety. Frozen writer and memoryless controls also fail hard target classification. Physical outcomes were independently reconstructed by replaying actual simulator interventions.

## Post-hoc prediction-level inspection: historical signal survives but is saturated

Inspecting the complete probabilities *after* the registered 50% classification result reveals that both recurrent models consistently assign higher failure probability to the pair member whose queued service was cancelled. For **all 96 mature pairs in each of the three seeds**, the pairwise score order agrees with the true wait-label order.

| Seed | Trained: mean P(fail, cancelled) − P(fail, pending) | Frozen: same contrast | Trained hard predictions that differ within pair | Frozen hard predictions that differ |
| ---: | ---: | ---: | ---: | ---: |
| 4101 | +0.000089882 | +0.008264848 | 0 / 96 | 0 / 96 |
| 4102 | +0.001228269 | +0.010514515 | 0 / 96 | 0 / 96 |
| 4103 | +0.000080680 | +0.007450674 | 0 / 96 | 0 / 96 |

For trained MTSK, the mean probabilities of target failure (cancelled versus pending) were approximately 0.99997/0.99988, 0.99980/0.99857 and 0.99997/0.99988 across the seeds. It did **not** turn the faint but correctly ordered signal into discriminating hard predictions. This is stronger evidence of a **readout/representation scale and transfer-calibration limitation**, not evidence that historical information vanished. The frozen random writer also has correct directional ranking and therefore shows that this ordering signal **cannot be attributed to learning the writer** from these data alone.

This pairwise inspection was not a preregistered primary and is explicitly post-hoc. Predictive ordering alone does not prove that a simple calibrated threshold (trained on permitted fresh development cases) will fix the policy or that the model learned a causal law. Saturation, within-pair score separation, and new-domain calibration must be evaluated prospectively.

## Integrity and provenance

* Three previously archived DEV-041 checkpoints per seed were reused unmodified (source `4ba2ea65cd3d4f3961277d80c49d7fb641f556fc`, parent [PR #197](https://github.com/FPC-effortless/TAC-OSM/pull/197)). No training in DEV-042.
* The standard-library exact public-history generator reconstruction, identical-last-event pairing, 100% reference solvability and service-counterfactual costs all passed checks.
* **1,203 regression tests passed**, with separate frozen-model CPU inference jobs and independent physical replay audit passing.
* Four-file complete archive: three detailed per-seed reports and aggregate `analysis/analysis.json`; **534,334 bytes**, ZIP SHA-256 `487ad57ee7247081bae9695102530da5c5c050385328ac6fa4d52ec1fff0c5dc`. The archived bytes were downloaded and tested with ZIP integrity and SHA-256 checks.

## Research decision

Do **not** declare trained MTSK a successful causal world-state predictor. It learns useful writer dynamics on the original task, but this transfer test exposes saturated readout decisions. Do **not** infer complete erasure or absent temporal encoding from chance argmax performance: the existing scores preserve consistent but tiny directional signals; a frozen random writer also preserves ordering.

The next controlled study should freeze both recurrent writers and compare *new-domain calibrated thresholds* against a prospectively supervised **head-only refit** on new disjoint training/calibration/evaluation cohorts, with correct physical costs and both same-cohort reissued cases and causal-order stress cases. Compare a lightweight linear head, frozen random reservoir, trained MTSK, explicit 41-scalar schema, and last-event baseline. Require both improved hard discrimination and cost/failure gates before any claim. Do not train or choose thresholds on the inspected DEV-042 cohort.
