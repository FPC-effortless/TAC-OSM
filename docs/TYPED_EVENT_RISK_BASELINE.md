# Typed Event Risk — classical model baseline (product experiment)

**Status:** Importable runnable Python baseline for a future source-authorized typed binary event-risk specialist model. Not TAC-OSM, not trained on real data in this PR, and not mobile/cloud deployed. The unit tests are only constructed fixtures.

Why first: binary risk classification from time-stamped numeric observations supports an immediate product hypothesis, easy baseline comparison, typed probability and clear model/decision authority. The selected **buyer and lawful real source remain NOT VERIFIED**. This can be rejected if a simpler heuristic or market alternative proves better.

## Commands (user-supplied legitimate data only)

Run with Python 3.11+ and standard library; no model server or network calls:
- `python specialist_models/typed_event_risk.py fit --fit-cases FIT.jsonl --source SOURCE.json --out model.json`
- `python specialist_models/typed_event_risk.py evaluate --model model.json --dev-cases DEV.jsonl`
- `python specialist_models/typed_event_risk.py infer --model model.json --observation request.json`
- `python -m unittest discover -s tests -p 'test_typed_event_risk.py' -v`

Source metadata must declare a source URL (independently verify it later), dataset license and steward, `fit_cutoff` (UTC ISO), list of 1..32 named finite numeric features, and `status: source-claimed-not-independently-audited`. A record contains `observed_at` and exactly those feature objects, each with numeric `value` and `available_at`. A FIT/DEV record additionally has binary `outcome` and `outcome_at`.

**FIT rejects** observed data or outcomes unavailable before the frozen FIT cutoff, future-available features, missing values, mixed time zones and single-class learning. This does not prove source rights, actual provenance, fraud-resistance, entity independence, fit/dev cohort independence or legal access. Training is deterministic 300-round full-batch L2 logistic regression, fit-only z-normalization and bounded standardized features. The serialized model records the actual FIT file SHA256, metadata assertions, preprocessing and learned coefficients; never serialize private FIT examples or claim proof of legal licensing based solely on declarations.

**DEV** requires outcomes at/after the FIT source cutoff; it strips labels before calling inference, reports Brier and log loss, and identifies its prevalence reference as **nonprospective** (it uses the evaluated DEV label prevalence, so it is not a genuine prespecified competitor). Before scientific/product comparisons, add frozen fit-only majority, simple rule, calibrated regression and appropriate tree/GRU controls on the *same genuine source* under applicable fair comparison rules.

**Inference** accepts only predecision observations (never label fields), outputs machine-native binary probability and advisory recommendation. The score is **NOT independently calibrated**, not verified to be accurate or safe, and does not authorize external actions. An app or Woyengi owns final authority.

## Promotion and business gates

1. Identify a customer job and legally usable publisher-origin data, rights and exact visible-at-time fields, entity/site identities, verified outcome times and a frozen FIT/DEV/untouched TEST split. Prevent cross-entity, causal and dataset contamination; confirm from real records.
2. Reproduce strong baseline performance (point accuracy, Brier, log loss, rare-risk costs, calibration, group error and out-of-time transfer). Lock paired thresholds and evaluation budgets before opening TEST.
3. Select **fast cloud** or **on-device** based on real requirements and measured end-to-end p50/p95 latency, throughput, infra-cost per 1000 useful decisions, RAM/energy/offline operation, model packaging and license. This repository script is neither a serving stack nor a mobile export.
4. Independent Veritas security/outcome checks and actual developer feedback; document failure honestly. A product architecture may use a different size under explicit authorization; prospective TAC neural 95k–105k rule remains binding separately.

Links: [#450](https://github.com/FPC-effortless/TAC-OSM/issues/450) product owner, [master plan](../research/MASTER_RESEARCH_PLAN.md), [Edge #440](https://github.com/FPC-effortless/TAC-OSM/issues/440).
