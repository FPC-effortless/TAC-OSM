# GRADCOVERAGE055 — complete declared gradient audit

Internally measured, not independently qualified. [Draft590](https://github.com/FPC-effortless/TAC-OSM/pull/590), [CPU40832487934 PASS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40832487934), execution source2ac0f6a22ab01cf2fdeb0e83d6c972fe651116ca. [Preregister757c7fa5](https://github.com/FPC-effortless/TAC-OSM/commit/757c7fa5faeed168f97b0a27dbdb22c660d972f0) preceded all checks. Saved router decision: GitHub CPU10min, no GPU/fit; scientific preflight PASS_DIAGNOSTIC_ONLY.

## Concrete gate failure and change

Original053 numerical comparator accepts two matching missing-gradient lists. A common omission leaves matching partial dictionaries, so equality alone is insufficient to establish complete trained-parameter coverage. The new audit captures the LIVE model's unique trainable names/identities/shapes/dtypes/devices and complete state hashes BEFORE forward; each result must independently cover that declaration before the historical comparator runs. No original053/054 code, weights, source records or findings were rewritten. This fixes a prospective verification weakness; it does not assert those original experiments actually omitted gradients.

## Actual original-source result

Same read-only trained LANGLAYER040 graph-free48D seed1701/FIT320 checkpoint SHA256 `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`; original optimizer/RNG/sample order verified. Genuine PSF CPython3.13 FIT owner-local first seed1701 source windows, B1x32 and B2x64; true next-byte labels only enter the loss after prediction. Source publisher/license/hash and cohort records inherited intact. Two PyTorch CPU threads, FP32 PyTorch2.10.0+cpu; no updates.

| Actual source workload | Policies checked | Declared gradient tensors/elements per pass | Maximum logits/gradient difference |
|---|---|---:|---:|
|B1x32|Native, upper-only, full nonreentrant checkpoint|55 / 499623|0 / 0|
|B2x64|Native, upper-only, full nonreentrant checkpoint|55 / 499623|0 / 0|

All six complete coverage checks passed. Actual RNG and frozen model/buffer hashes were unchanged. Each declared tensor had at least one nonzero element in these workloads; this does not establish that every scalar is useful or that its derivative is mathematically correct. Finite zero gradients are explicitly allowed by the generic coverage gate.

**Omission falsifier:** remove the first144-element gradient from a COPIED actual result on each pass. Historical comparator accepts both equally partial results with499479 elements. Strict independent live-contract coverage rejects all6/6. These manipulations are constructed evidence controls, not a model capability measurement or modification of the original evidence.

## Checks and preserved artifacts

42 focused software tests PASS; full local suite1433PASS/4SKIP,1437 collected, historical baseline freeze327 unchanged. Compiler/diff checks PASS. Source-original H64/PR483 and memory050/051 negatives are preserved.

[Original artifact12879213032](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40832487934/artifacts/12879213032), ZIP SHA256 `336f880e6925f6589a78a63d1b3c203348c73d461c97491af32aba7ef7871129`, downloaded and verified. The full original ZIP is preserved losslessly in `research/evidence/gradcoverage055/ORIGINAL_ARTIFACT.zip.b64` beyond Actions retention; decode with standard base64, then verify the ZIP SHA. Extracted RESULTS.json, preflight, compute plan, original CI log and per-file SHA256.json are committed alongside it. No checkpoint binary is duplicated. Each actual gradient tensor has a named shape/count/nonzero/hash record; the archive contains gradient hashes, not trained gradient arrays.

## Claim limits and next action

This closes a source-specific gradient coverage loophole, not a new language skill, mathematical AD proof, trained architecture, 100k-scale exception, reward/safety certification or promotion. The original499623p weights are historical no-fit replay only. Every NEW trained/deployed candidate still requires95k–105k, <=1% matched-arm gap and original data/exposure/train/tune gates. No optimizer step, GPU, DEV/TEST, mobile RAM/energy or quality/latency experiment occurred. PNDS66 scoped Lean/Python remains inherited; complete source/AD/IEEE bridge OPEN. Numerical gradient presence/equality cannot upgrade it.

Next: use this independent coverage contract before optimizing recomputation boundaries or fitting an eligible100k model. Useful distant-history retrieval still needs to overcome the established key-eviction negative with equal-information controls; not demonstrated here. Explicit dispatch can replay the frozen audit; report-only commits do not automatically repeat it. Draft review remains pending; do not merge or promote automatically.
