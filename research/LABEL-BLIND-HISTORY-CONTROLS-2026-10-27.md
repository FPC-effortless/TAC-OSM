# Historical-memory control investigation

Status: proposed engineering increment; no new model fits or source scoring.
Tracked in Issue #472. Capability question: does frozen associative memory
retain source-specific historical content beyond current/nonassociative state?

## Verified starting evidence

[Draft PR #476](https://github.com/FPC-effortless/TAC-OSM/pull/476), inspected at
`0d9c9fbc8c5ed94762c9da129379158d36125e7f`, reports a failed H64 reconstruction
gate in both seeds. Its `input_features` obtains `wrong_key` by flipping each
high/ordinary matched pair. Because pair construction uses opposite PM2.5
classes, that comparator is label-correlated. Its advantage cannot establish
that an arbitrary incorrect historical address outperforms the true address.
This inspection does not independently reproduce the reported source scores.

## Implemented control boundary

`tac_osm.historical_read_controls.select_intervening_writes` accepts only actual
writer identities, target identity, query hour, frozen seed and control count.
It selects without replacement from the same original owner, strictly after
the target and strictly before the query. SHA256 ordering makes selection
independent of input enumeration. It records all eligible/selected hours and a
receipt digest. Duplicate identities, missing target or insufficient controls
block rather than substitute the opposite-label pair or a random vector.

The source runner must construct this identity ledger from actual writes in
the exact model trace, including missingness and routing; ordinary hourly rows
are not proof a write occurred. The helper does not authenticate that ledger.
Historical keys remain **privileged diagnostic inputs**, never native query
inputs. Time-only selection removes direct label-driven key selection; it
does not establish statistical independence of sensor values, eliminate age
effects, repair the label-selected original cohort or create an unseen holdout.

The existing `lab` phone diagnostic now exercises constructed negative controls
for label/score poisoning, owner/future boundary contamination, row ordering,
missing candidates and replay identity. Its PASS remains fixture evidence only.
Run the existing reviewed planner before checks:

```sh
python scripts/phone_lab.py plan research/phone_queue/phone-lab-diagnostics.json
PYTHONPATH=src python -m pytest -q tests/test_historical_read_controls.py
```

The proposal routes bounded no-fit checks to GitHub CPU with zero GPU minutes.
Local unit verification checks source code; no model evaluation occurs here.

## Next source experiment — not yet authorized or executed

Before using this selector to score any source/checkpoint, preregister a separate
phone_queue evaluation proposal, run its planner, obtain source/evaluator
authorization and independent evidence, and freeze the actual trace/source,
checkpoint identities, seed set, candidate count and source-time cohorts.
Do not overwrite LATFUSE013's original registration or results.

1. Capture actual same-owner write keys with their source hours from frozen
   replay. Resolve selected hours back into that trace; fail on missing keys.
2. Compare true-key read, several identity-selected intervening-key reads,
   isotropic keys, native query and source/nonassociative controls on exactly
   the same query-time state. Report candidate ages and all negative results.
3. Preregister source-continuous content metrics with centering estimated on
   authorized training records only, within-window retrieval controls and
   source/entity clustered uncertainty. Never choose centering, key count or
   thresholds from the already exposed May–June outcomes. Do not claim those
   FIT-era records are independent model holdout.
4. Preserve model parameters/state/checkpoints and sealed TEST. A neural change
   needs the full 100k, matched-budget, reward-signature and PNDS gates; this
   selector changes no computational graph and closes none of those gates.

Missing audited source trace and independent evaluation authorization are
BLOCKED for model-level execution. There is no trained capability result here.
