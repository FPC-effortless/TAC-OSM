# HISTORY-QUERY016: query-diverse original-source memory trial

**Both seeds FAIL the preregistered timestamp-specific memory-usefulness gate.**
Internally measured on genuine exposed FIT data; **not independently qualified**. No protected TEST, GPU, checkpoint promotion or merge.

The eight-query objective did not establish useful historical-time discrimination in the original associative memory. Correct-clock predictions are worse than both collapsed clocks and circular wrong clocks; all four station-bootstrap gain intervals are strictly negative. Associative centered skill is negative for both seeds. Removing the associative matrix at H64 improves error in both seeds. This is a failure of this fixed recipe, not a proof that the architecture cannot learn memory.

Preserve the partial success: seed26016 beats the equally trained fastslow baseline by13.35% (positive station95% gain interval). Seed26017 gains1.41%, with an interval crossing zero. Overall reconstruction advantage in one seed does not establish timestamp-specific associative-memory utility.

## Original data and prospective design

UCI501 Beijing multi-site air quality, DOI10.24432/C5RK5G, publisher UCI, CC-BY-4.0 (source/license receipt inherited from history_audit014). Original archive SHA256d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. No synthetic performance data or unrelated outer archive members. Original pre2015 FIT loader, preMarch2014 normalization, and causal last/age caches verified.

Unchanged LATFUSE018 original inference code fd201207cb76b30470f64e1fc1baecf9de44560d, 99,390 trainable/deployed parameters and6800B nominal original state per arm. Two fresh seeds26016/26017; 64 AdamW updates x16 windows,88 source events/window,8 historical clocks/window. Same source, training order, losses and full exposure/budgets for both arms; no tuning. Both past and forecast outputs exist before their target values/masks enter loss. Historical head receives source owner and legal past clocks, no target-derived key.

Preregistration944c8a04 preceded source profiling; all64-batch profile amendment3d7946a3 preceded every fit; frozen plan/profile/manifest committed48159ec7 before nonzero updates. Source-only1137 train windows (March1–April18),141 primary July1–August15 exclusive windows,117 secondary May1–June10 windows. All are exposed FIT, not virgin holdout. Primary1106 observed/22 missing target slots; secondary915 observed/21 missing. Predict all eight clocks, score genuine observed slots only; window-weight errors and station-block bootstrap,4000 draws/seed15015. No pooling of seeds.

## Primary measurements

Errors are normalized log1p PM25 units, lower is better. Centered skill0 denotes no within-window discrimination; negative is worse.

| Seed | Model | All-query MAE | Collapsed-clock MAE | Wrong-clock MAE | H64 MAE | Zero-read H64 MAE | Centered skill |
|---|---|---:|---:|---:|---:|---:|---:|
|26016|assoc_historic_query|0.777443|0.756884|0.771255|0.714020|0.563669|-0.146826|
|26016|fast_slow_historic_query|0.897179|0.888672|0.899615|0.793221|0.793221|-0.083129|
|26017|assoc_historic_query|0.716237|0.691399|0.705316|0.611118|0.561873|-0.240601|
|26017|fast_slow_historic_query|0.726476|0.720703|0.725842|0.588273|0.588273|-0.041279|

A positive gain means associative error is lower than that control. Error-gain intervals are in normalized log-PM25 units, preserving whole source stations and all their windows.

| Seed | Primary comparison | Relative gain | Station bootstrap95 mean error-gain interval | Gate |
|---|---|---:|---|---|
|26016|circular_wrong_query_MAE|-0.80%|[-0.011871,-0.000604]|FAIL|
|26016|collapsed_query_MAE|-2.72%|[-0.025825,-0.015702]|FAIL|
|26016|fastslow_all_query_MAE|13.35%|[0.094438,0.146968]|PASS|
|26016|zero_read_H64|-26.67%|[-0.193227,-0.104322]|FAIL|
|26017|circular_wrong_query_MAE|-1.55%|[-0.014543,-0.007145]|FAIL|
|26017|collapsed_query_MAE|-3.59%|[-0.028723,-0.020867]|FAIL|
|26017|fastslow_all_query_MAE|1.41%|[-0.007836,0.031217]|FAIL|
|26017|zero_read_H64|-8.76%|[-0.090986,-0.008092]|FAIL|

## Secondary replication

| Seed | Model | All-query MAE | Collapsed-clock MAE | Centered skill |
|---|---|---:|---:|---:|
|26016|assoc_historic_query|0.690171|0.666096|-0.056542|
|26016|fast_slow_historic_query|0.871104|0.864787|-0.000003|
|26017|assoc_historic_query|0.676719|0.650789|-0.090969|
|26017|fast_slow_historic_query|0.682034|0.678824|0.017557|

The secondary cohort does not rescue either associative specificity failure. No secondary threshold was substituted for the primary gate.

## Integrity and durable receipts

- Full source prefit run40493315659 inventories every scheduled forward/backward/clip/AdamW batch at lr0 with unchanged weights. Repeat40493408718 reproduces plan, profile, proof and comparison manifest byte-for-byte. Common/differing complete ATen mathematical accounting bounds gap0.006609621%, below1%, under the registered multiply-add2 accounting convention. Unknown differing math blocks. This is not exact hardware FLOPs, equal elapsed CPU time, or energy parity.
- Training/replay run40494577797 at48159ec7: all256 nonzero optimizer steps match the frozen per-batch fingerprints. Final resource gate PASS before assessment; each seed/arm retains actual source/order/steps, parameters, optimizer43 tensors atstep64, RNG, losses and exact operator fingerprints. Instrumented per-arm CPU elapsed112.44/111.75/111.18/110.79 seconds; instrumentation is included and these are not uninstrumented efficiency benchmarks.
- All four checkpoints and full per-case outputs/targets/masks preserved in new bundle HISTORY-QUERY016, commit e0aaf943eb704334bb8ee414c965152fb2129901. Historical LATFUSE018 and older checkpoints/results untouched. Fresh-process source/checkpoint inference replay PASS exactly. RESULTS SHA25688e2e765e5f6c20cfdbee9d1a2897c96972f85074ef3b94700032b545208b262.
- Fit/replay artifact12730737461 SHA2561e2747a500883f7b8c0f1e1b8b4f934ef3d834ec13857e69b1dfbf0799e81b2f. Durable checkpoint commit is retained beyond artifact expiry.
- PNDS draft48, formulation4d6c1cb8: Lean build/axiom inspection and Python loss equations/tests pass. Constructed original loss/reference agreement within1e-6 and absent-target loss gradients zero. Full original inference/write/read graph IEEE/AD/source equivalence remains OPEN; no mathematical or scientific completion inferred.
-1222 local engineering tests PASS, including unknown differing math, mismatched budget/recipe, actual batch-fingerprint tampering and genuine original graph constructed-input checks. These engineering tests are not performance evidence.

## Limits and next capability blocker

Two seeds, one short fixed training budget and twelve regionally dependent stations do not identify architectural impossibility. Comparing with LATFUSE018 uses different seeds/target exposures, so no causal estimate of the objective change is claimed. Zero-read is an OOD intervention, and target centering is scoring-only. Source-reloaded future-poison replay remains INCONCLUSIVE/not implemented; prefix reconstruction and feature-only prediction boundary are the implemented causal checks. Full whole-model mathematical equivalence and independent qualification remain OPEN.

The unresolved empirical blocker is source-time-specific content retention/addressing: this training recipe does not make the correct past clock useful beyond label-blind controls. No unregistered hyperparameter search, larger fit or GPU escalation was used to hide this failure. Future changes require a new source-safe preregistration and matched budget; do not promote these checkpoints.

## Replay

With the original source archive and a clean worktree at fd201207cb76b30470f64e1fc1baecf9de44560d:

```sh
python scripts/validate_research_parity.py --manifest research/replay-bundles/HISTORY-QUERY016/FINAL_COMPARISON.json --phase final
python scripts/train_history_query_diverse.py --mode replay --implementation-root /path/to/clean-original-content-code --source /path/to/PRSA2017-original.zip --output /tmp/history-query016-replay
```

See draft TAC PR505, PNDS PR48 and Issue472. No merge has occurred.
