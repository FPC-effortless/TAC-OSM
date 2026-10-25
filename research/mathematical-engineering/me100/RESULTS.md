# ME100 — trained entity sensor world model; full capability gate FAIL

Five original 97,235-parameter TAC-Temporal models completed1,536 guarded updates each. They predict six pollutant concentrations six hours ahead from authentic UCI501 station measurements, observation masks, causal staleness, calendars and supplied station identities. Inputs are masked, never numerically forward-filled. Each batch uses actual FIT observations with contexts8/64/256; this is an integrated single-model capability pilot, not a GRU/Transformer head-to-head study or PR394 Phase2 qualification.

All five original models, optimizer/RNG states, exact sampled histories, FIT statistics and guarded live-parameter receipts were frozen in Git commit `1d721082` before the2015 reporting period was opened. Pre-registration/code commit `46af2f1c`. FIT2013–2014; numerical2016–2017 outcomes remain unscored. No gate or training budget changed after scoring.

## Real2015 PM2.5 results

Raw source concentration units, MAE lower is better. Means below average the five fitted models; the classical refs are deterministic. Each model forecasts the same143 genuine missing-current sensor queries, spanning21 global calendar-week clusters. Ordinary daily evaluation contains4,288 eligible noon queries, spanning53 week clusters.

| Access condition / reference | Natural-gap MAE | Ordinary daily MAE |
|---|---:|---:|
| Last observation within256h |56.076923|34.515625|
| FIT-only ridge (lambda100) |43.915070|32.767415|
| Trained model, recent8h |46.073106|35.477731|
| Same trained model, full256h |38.986917|35.689034|

Longer access lowers gap PM2.5 MAE15.38% versus recent8,30.48% versus LOCF and11.22% versus ridge in the five-seed mean. All five seeds satisfy the registered5% long-versus-short gate. This is within-model evidence about this access condition, not attribution to the associative matrix or a matched architectural win.

The global calendar-week clustered95% interval for long-minus-short MAE is **[-15.369833,+1.868555]**, including zero. Only21 calendar-week clusters have eligible gap targets;143queries and715seed-query forecasts are not143/715independent trials. The interval conditions on the five fitted models, does not include retraining uncertainty and is not multiplicity adjusted. No reliable population-level advantage is established.

## Gate ledger

| Seed | Long gap MAE | Short gap MAE | History | Practical reference | General guard | Gap coverage>=85% | Full |
|---|---:|---:|---|---|---|---|---|
|100001|43.326|46.375|PASS|FAIL|FAIL|FAIL|FAIL|
|100002|44.032|48.889|PASS|FAIL|PASS|FAIL|FAIL|
|100003|31.930|40.624|PASS|PASS|PASS|FAIL|FAIL|
|100004|40.426|44.904|PASS|PASS|PASS|FAIL|FAIL|
|100005|35.222|49.574|PASS|PASS|PASS|FAIL|FAIL|

Long-context nominal90% PM2.5 interval coverage averages **67.69%** on gap queries, versus90.81% on ordinary daily queries. Full gate **0/5 PASS**. The coherent likelihood fixes distributional consistency, not conditional calibration. Daily mean PM2.5 error remains3.40% above LOCF and8.92% above ridge. On gap queries, long access also worsens CO/O3 mean errors and six-channel density NLL relative to short access. These regressions are preserved, not hidden behind the PM2.5 result. Exact six-channel NLL, MAE, coverage, proper interval scores and original forecasts are in `research/replay-bundles/ME100/RESULTS.json` and `FORECASTS.npz`.

## Mathematical and execution verification

967 local tests pass. Independent serial full-network evaluation agrees with parallel execution in FP32/FP64 for output distributions, all memory values and all25 parameter-tensor gradients. This also passes on all five original fitted models using real FIT queries; largest observed state/output discrepancy5.722046e-6 in FP32 and7.105428e-15 in FP64, within the preregistered numerical tests. Real fitted full-versus-chunk predictions agree. This is tolerance-based verification, not bit-exactness or full-network Lean proof.

PNDS source-trace bounds, missing-observation identity and associative composition/action compile. Existing native matrix-memory bounds are inherited. The full neural network's IEEE formal refinement remains open. Exactly six active model families remain; PNDS is the foundation, not an extra model.

Fixed primary state is608FP32bytes/sample, plus24integer bytes for count/owner/clock and104bytes external staleness bookkeeping. Model weights, Python/version provenance, intermediate arrays and runtime overhead are additional. Mean measured per-model training CPU approximately57.4seconds; no matched FLOP/runtime superiority claim. All25 parameter tensors have recorded finite nonzero gradients at update2.

## Interpretation and next architecture question

This candidate can retain useful measured sensor context through authentic missing observations, and its full-history point forecasts are promising on the small gap cohort. The run does **not** demonstrate an overall capability win: severe gap undercoverage and daily-reference failures remain. It is not safe for uncertainty-based operational decisions.

The next integrated design should model uncertainty over unobserved current physical state jointly with future change, using source age and retained evidence mass, rather than treating a single retained source anchor as known. A coherent latent-regime mixture or a learned state-conditioned covariance envelope is a candidate, not a proven solution. It should be preregistered and trained on FIT, preserve these originals and end this protocol here. Do not add CED/geometric depth or repeat this run againstREPORT to manufacture a passing result.2016–2017 remains reserved for later fully matched qualification.

Natural missingness is not verified delayed arrival, dynamic entity discovery, source authority, intervention learning or decision intelligence. This public-source temporal pilot is not independent physical-site transfer, live deployment, commercial validation or a master-plan completion claim. Both research PRs remain draft and unmerged.

Independent clean-checkout original-score replay and original fitted-graph audit PASS. GitHub original-model replay run40173945086 also PASS. PNDS build40173558472 and dedicated mathematical/source steps in40173558621 PASS (the job endpoint initially retained a stale in-progress status despite completed successful steps and successful run). Dedicated score/graph replay performs no new training. Restoration receipt and SHA256SUMS preserve exact original payloads.
