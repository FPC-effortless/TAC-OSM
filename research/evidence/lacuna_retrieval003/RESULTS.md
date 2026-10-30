# LACUNA-RETRIEVAL003 — stronger historical recovery, no passing prediction gate

Internally measured, not independently qualified. Original frozen640-update
LANGMEM087 checkpoints, three seeds1701/1702/1703, two newly registered CPython
documents (fractions/weakref). All integrity checks passed on the first diagnostic
run. No new fit, GPU, protected TEST, checkpoint promotion or merge.

The four-condition comparison is now fully specified: fixed cosine attention
maps a query to raw history, the byte-code readout is shared, decoder reads are
matched to native amplitude, direct current-write contribution is preserved,
and wrong-address/value-shuffle controls are included. Both storage banks have
256B payloads. This is a conditional one-clock diagnostic, not autonomous
replacement-model evaluation or proof of equal whole-model memory efficiency.

## Historical byte recovery

| Age bin | Distinct source/query/bin cells | A native query/matrix | B supplied key/matrix | C native query/raw | D supplied key/raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8–31 | 28 | 3.57% | 5.95% | 1.19% | 34.52% |
| 32–127 | 97 | 2.75% | 5.15% | 5.50% | 21.31% |
| 128–239 | 101 | 0.99% | 6.27% | 3.30% | 21.45% |

Top1 literal **past-byte** accuracy under the registered, fixed, untrained cosine
readout, not language next-byte accuracy.226 source/query/bin cells repeat
across3checkpoints =678measurements;154distinct current query positions and
218distinct historical positions. All480 possible source query clocks enter
availability accounting. Bins can reuse the same current target. No iid seed or
token significance is asserted. Source continuation matches the future byte in
60.71%,61.86%,67.33% of selected cells respectively; agreement never selects them.

D's wrong-address raw control obtains22.62%,17.87%,18.48%; D's value-shuffle
control7.14%,9.62%,6.60%. D therefore has descriptive address/value sensitivity
under this operator, with no prospective recovery-superiority significance test.
The isolated original encoded value is byte-decodable in100% of these cells;
that evaluator reference is not a read from the accumulated native matrix.

Supplying a past key improves matrix readout versus the native query. However,
the matrix wrong-key control has **better past-byte NLL than B at every age**:
B4.6903/4.2588/4.2204 vs wrong4.5864/4.1884/4.2130. Thus B's improvement versus A
alone does not establish correct-address-specific native matrix recall.

## Original decoder: next-byte loss

Positive values below mean original NLL minus intervention NLL, i.e. improvement.
Each intervention keeps the original native read magnitude and current write.
Patched next states are discarded; original source trajectory always advances.

| Age bin | B supplied key/matrix | C native query/raw | D supplied key/raw | Registered diagnostic utility signal |
| --- | ---: | ---: | ---: | --- |
| 8–31 | +0.006171 | +0.003948 | +0.004923 | None |
| 32–127 | −0.008817 | −0.008254 | −0.009198 | None |
| 128–239 | −0.000302 | −0.001485 | −0.000979 | None |

Every bin exceeds the registered20-cell minimum, but no arm meets all utility
requirements. Favorable short-age averages are preserved: B fails seed1702
(−0.000326) and weakref(−0.002446); C fails seed1703 and weakref and is worse than
raw-wrong selection(+0.004539). D is favorable in all3seeds at short ages, but
fails weakref(−0.001380). D's short-age advantage over raw-wrong is only0.000383;
B's over B-shuffle is0.000771. These controls substantially narrow interpretation.

All subgroup means,10conditions' NLL/top1 and availability records remain in
original RESULTS.json. Better past-byte recovery did not produce a consistent
next-byte NLL benefit in the unchanged decoder. Intervention outputs may be
outside its training distribution; no result uniquely localizes decoder failure.

## Geometry and interference telemetry

| Age bin | Matrix effective rank (of8) | Other-old-write / target-read norm | Native-query target rank (of243) | Supplied-key target attention |
| --- | ---: | ---: | ---: | ---: |
| 8–31 | 3.46 | 53.01 | 135.48 | 4.16% |
| 32–127 | 3.30 | 69.64 | 127.20 | 5.25% |
| 128–239 | 3.23 | 107.40 | 129.76 | 5.17% |

Native-query target attention is0.210%,0.372%,0.251%. A supplied key is shared by
7.82,4.06,3.59 retained records on average, including repeated contexts that can
have different continuations. About20–21 different-context keys also have cosine
>.9 with it. These are operator-specific geometry observations, not proof that
rank, collision, interference or any single mechanism dominates language loss.
The norm ratio concerns projected conditional contributions, not information
bits or a general signal-to-noise ratio. Fixed attention temperature8 was not tuned.

## Memory, compute and verification

- Native bank8x8float32=256B; raw ring256B=8B clock+248rawbytes. Reconstructed
  keys/values are workspace, not persistent per-record side indexes.
- Native state1920B tensor+8B logicalclock. Actual evaluator additionally holds
  the256B ring (2184B combined logical payload),786432B decomposition/key/value
  workspace, transient tensors and report objects; peak process RSS372609024B.
  Shared fixed byte-code buffer8192B and original learned weights are separate.
  Native upper/query state comes from the original trajectory. No full-budget
  autonomous architecture or inference-speed advantage is established.
- Diagnostic process CPU100.9253s, two CPU threads. Each of10arm/control reads
  and native forward is charged; raw attention scans243 reconstructed records.
  The measured stream contains24576native observed-byte steps and6780main
  counterfactual decoder calls. The source code performs3420raw attention
  calls including controls and the6future-poison checks (5per preparation).
  This includes evaluator diagnostics, not an isolated production latency measure.
- Replay verified9original086/087checkpoints. New replay anchors are not original
  job outputs; historical replay and086RNG limitations remain as in parent637.
- Transport maxerror1.34e-7; reconstructed key2.69e-7/value8.95e-8; native A
  identity-logit3.82e-6; source slow_zero0; future poison7.63e-6. All registered
  tolerances passed. Actual-source local read VJP/FD maxerror0.000174<0.002.
  Independent attention reference maxerror3.89e-16. Frozen weight hashes unchanged.
-26targeted source/probe/replay tests passed in Actions. Local full suite:
  1626passed,4skipped,1existing profiler warning; test collection1630.
- PNDS partial Lean/Python formulation89 passed scoped build/compiled-output,
  no-proof-hole and axiom checks. Initial failed proof build41006980829 is
  preserved; corrected proof run41007194178 passed. Full neural IEEE/AD and
  independent qualification remain OPEN.

## Immutable evidence and disposition

Registration f18085cd4d6e1c5cb86644c61393a49575cd8d3e preceded source acquisition.
Diagnostic code3ca07f661e0468e838ce76359187590320df0bab,
execution a74627fc1d8d1eaf209ff6eb5be828fdbf326768.
[Original Actions run41007483485](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41007483485),
artifact12956147521; original ZIP SHA256
f20430a26bc5a72b1d1cd6f56fd7d99d086306e9560780c46a1e5b7d3dd80a76.
Saved-report verification authenticates original ZIP, source cohort/target/wrong
exclusions and recomputes summary/gates without any model execution.

The fresh document cohort is now exposed DEV. Do not call subsequent scoring on
it a first unseen evaluation. Evidence supports examining both address geometry
and accumulated-value interference; it does not select a unique architectural
fix. No condition passes the new diagnostic utility gate, and original001/002
failed gates remain unchanged. Keep the neural sweep stopped. Longer-context
credit assignment, trained readout adaptation and independent full-budget
architecture comparisons require separate prospective protocols.
