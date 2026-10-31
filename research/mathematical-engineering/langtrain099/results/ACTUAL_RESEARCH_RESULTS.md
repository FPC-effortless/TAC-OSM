# LANGTRAIN099 actual research results

**Three exact10M models trained and tested. Language learning supported; registered lexical-memory and ARC-transfer gates FAIL.** Internally measured, not independently qualified.

## Original execution

Training/evaluation source: `068ed9bd5bd68c19b0fd98dcff9a42197c1e4d21`.
[Original Actions run41153068368](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41153068368) completed all three FIT jobs and held-out evaluation successfully. Model code remained unchanged after the original zero-update profiles. No failed or partial neural fits, hyperparameter trials, best-checkpoint selection or replacement checkpoints occurred. The earlier source-preparation and metadata workflow history is preserved. No GPU, protected TEST, promotion or merge.

All three actual trained checkpoints contain **10,000,000 learned parameters in17 parameter tensors**, each with exactly2,048 original AdamW update records and full finite moment/RNG archives. Read-only inspection independently confirmed counts, changed initial/final weights and complete optimizer-step exposure. Every original FIT cold replay had maximum absolute full-model/state difference **0.0**. All six initial/trained episode and language predictions replayed bitwise in fresh processes before target scoring. Source and final exact-size gates passed.

| Seed | Code DEV NLL | Documentation DEV NLL | Positive lexical reuse | Uniform code sample | ARC exact | ARC valid format |
|---|---:|---:|---:|---:|---:|---:|
|9901|1.752635|2.120728|0/32|4/16|0/8|1/8|
|9902|1.756016|2.108233|0/32|3/16|0/8|1/8|
|9903|1.757385|2.112642|0/32|4/16|0/8|0/8|

Uniform code completions are a separate sampling group, not evidence of positive-case persistent lexical reuse. Negative lexical cases also scored0/16 per seed. Both registered exact-capability gates failed for every seed.

## Authentic held-out language prediction

The model trained on197 authenticated licensed original code/documentation files containing3,219,298 bytes from nine publishers. Complete held-out source files were excluded before sampling, including all four old lexical DEV files previously present in the broad pilot corpus. Language DEV comprises22 complete source files with176 fixed windows/45,056 target bytes per checkpoint:32 code windows and144 documentation windows. Documentation includes code examples and is not pure prose. Reused episodic FIT contains128 lexical and24 ARC examples; episodic DEV contains64 lexical examples and8 ARC tasks.

| Held-out modality | Original initialization NLL | Trained NLL | Trained next-byte accuracy | FIT-only bigram reference NLL |
|---|---:|---:|---:|---:|
|Code|5.545483|1.755345|53.487%|2.461696|
|Documentation with code examples|5.547254|2.113867|42.873%|2.829125|

Values average the three fixed seeds; NLL is nats per byte. Initial-minus-trained mean NLL improved in every seed. Source-file cluster bootstrap95% intervals for the gain are[3.698181,3.923624] on the four code files and[3.362388,3.499851] on18 documentation files. These are internal public-DEV estimates from small source populations. The count baseline is an unmatched reference; no GRU/SSM/Transformer10M controls were trained, so this is not architectural or scaling superiority evidence. Results cannot be directly ranked against older experiments using different datasets, objectives or budgets.

## Persistent-state attribution and transfer

On32 positive lexical cases, trained first-byte NLL is3.283519 under carry. Reset-minus-carry NLL is+0.072518 with95% interval[-0.033947,+0.178982], so the reset advantage is not established. Donor-minus-carry is+0.152857 with interval[+0.040302,+0.238424], an exploratory probability-level content effect. Both exact carry-minus-reset and carry-minus-donor advantages are zero. Neither first-byte result establishes useful exact reuse.

ARC carried first-byte NLL improves against reset by+0.661373 with interval[+0.019685,+1.294243], but the first output byte is strongly influenced by grid syntax. No correct grids were generated. Only2/24 carried outputs were valid grids. The ARC donor comparison is INCONCLUSIVE over the complete cohort because five tasks lack a legal donor; the available three-task subset has interval[-0.069680,+0.349599]. Binding-shuffle and exact reset comparisons provide no exact-solution advantage. The experiment cannot cleanly separate abstract-rule inference from output-format failure and does not establish general intelligence.

The past-only Witten–Bell/raw-history reference got8/32 positive lexical completions,2/16 negative completions,5/16 uniform completions and1/8 ARC solutions. These are unmatched classical references, not trained neural competitors. Their nonzero exact outcomes underscore the unresolved gap in learned acquisition/use. The oracle-information-at-query diagnostic was not run; acquisition, retention, addressing, readout and decoding have not been causally localized by this study.

## Actual training and efficiency

Fixed FP32 AdamW recipe:2,048 updates/seed, batch4,1,536 full language steps at128 bytes and512 balanced future-answer/prefix episodic steps at272 slots. Each seed observed786,432 language targets plus its recorded genuine episodic targets. This bounded pilot sampled less than one full broad-language corpus epoch; it is not complete language pretraining.

| Seed | Actual supervised bytes including episodic prefixes | Future episodic answer targets | Padded slots | Training/checkpoint/replay seconds | Peak CPU RSS GiB |
|---|---:|---:|---:|---:|---:|
|9901|1,051,041|26,209|292,447|782.172|0.742|
|9902|1,045,995|25,673|297,493|736.203|0.753|
|9903|1,046,679|25,796|296,809|624.886|0.745|

Sum of inner fit/checkpoint/replay wall times:35.721 runner-minutes across three concurrently scheduled jobs. Package setup and orchestration are separately visible in the Actions timestamps. CPUs were AMD EPYC9V74/7763, two Torch threads, Torch2.5.1+cpu. All fit budgets passed and no GPU minutes were used. Hardware/time receipts are measurements, not matched-compute competitor qualification.

Per-stream persistent tensors contain2,776 FP32 values =11,104 bytes; learned weights occupy40,000,000 bytes. Across72 DEV streams the acquired tensors occupy799,488 bytes, while their original acquisition observations total only4,953 bytes (64–155 bytes per case). Fixed learned state therefore has **no demonstrated memory-compression advantage at these short histories**. Longer-history sufficiency/use must be measured, not inferred from constant state width.

The two scans have O(T) state work and ideal O(log T) span at fixed geometry. Dense learned projections/readout require9,921,852 multiply-accumulates per observed byte before elementwise/scan costs. Approximately96.67% of learned parameters are in the width-dependent feed-forward terms; the study does not isolate the marginal value of memory width versus decoder capacity. About21.8–22.1% of dense training slots were padding. Length-grouped episodic computation is a concrete future engineering opportunity, subject to full forward/gradient correspondence and a fresh registration. No code or training schedule was changed after DEV became visible. Saved batched inference costs are not a p50/p95 single-stream latency benchmark, and no globally optimal or sub1ms/byte claim is made.

## Mathematics, diagnostics and limitations

TAC constructed checks:51 passed, covering exact size, independent full NumPy graph, FP64 scan/sequential/chunk correspondence, parameter gradients and finite differences, causality, native-state/owner boundaries and default100K-policy preservation. PNDS scoped CI run41152809360 passed22 exact-real theorem axiom checks and6 independent NumPy tests for source0f628b876bd6ed81b496771a2625dbc54c84932e. The ordinary PNDS Lean CI also passed in run41154036739 after draft registry checks were correctly pinned. TAC shared reciprocal math CI passed in run41154193394. The earlier shared CI failures were stale main/master registry comparisons; exact equality remained required and was not bypassed.

The full persistent-state invariant, bounded actual bank/read behavior and affine scan algebra are proved under exact-real assumptions. Whole-model IEEE/automatic-differentiation refinement remains **OPEN**. These scoped proofs do not certify learned capability, global efficiency optimality or the complete intelligence theory. The whole TAC project test suite was not run; source-relevant scoped checks passed.

DEV is public and previously exposed in LANGACQUIRE097. Broad FIT may contain lexical names labeled novel under the earlier restricted corpus; inherited labels are not a new-vocabulary guarantee. Complete file/hash separation does not prove semantic deduplication. ARC is a small human-authored diagnostic. Internally separated actors/evaluators are not independently attested writer-ACL or reward-integrity certification. No protected TEST was accessed and no external qualification or checkpoint promotion is claimed.

## Reproduction and original evidence

Original archives, SHA256 values, checkpoint hashes and local verified copies are indexed in ORIGINAL_ARTIFACTS.json. Every original is retained in GitHub Actions (90day expiry) and the local verified artifact directory. Original full per-step receipts, optimizer/RNG archives, native states, sealed logits/greedy generations, language predictions and fresh replays remain in those archives. Small source-controlled files here are summaries/indexes and do not replace originals.

Check out training/evaluation source068ed9bd5bd68c19b0fd98dcff9a42197c1e4d21 for source-authenticated model replay. Extract each artifact into a folder named after its artifact name. The optional recalculate_archives.py performs read-only archive/score arithmetic, with no neural model forward or optimizer:

```sh
python research/mathematical-engineering/langtrain099/results/recalculate_archives.py   --artifacts /path/to/extracted-artifacts   --judge /path/to/authenticated/eval-package/DEV_JUDGE.json   --output /path/to/read-only-review.json
```

The negative LANGACQUIRE097 findings and all prior math variants remain unchanged. The supported outcome is real language learning in three trained exact10M models, alongside failed exact lexical-memory and ARC-transfer gates. Future capability work should first isolate information availability/readout versus memory acquisition/use under a fresh preregistration; this study does not justify an intelligence or competitive architecture claim.
