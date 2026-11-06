# CSL-125 R4 schema-aware serving continuation — 2026-11-06

Parent: PR #716 at 8cbf5a465f9f1d8fb2f054543828517456976752.
This is a serving/data-format change for the existing R4 model, not R5 and not a new learned architecture. No parent source, checkpoint or negative receipt is overwritten. Learned weights remain 123,873,263 at the default 32K vocabulary; the user hard cap is 125,000,000. The universal 100K restriction was superseded for new declared-target-v2 studies by merged PR #680 (8a28843550236cea9695552670d33491b2b95854); stale ancestor documentation is not revived as a size veto.

## Goal

Close the known syntax-only decoder gap and avoid unnecessary output projection during byte-level tool inference. Reuse XGrammar 0.2.8 rather than invent another grammar engine. This dependency is Apache-2.0 and must remain identified as upstream work. Primary references:
- https://xgrammar.mlc.ai/docs/latest/api/python/grammar.html
- https://xgrammar.mlc.ai/docs/latest/api/python/grammar_matcher.html
- https://xgrammar.mlc.ai/docs/latest/api/python/tokenizer_info.html
- https://pypi.org/project/xgrammar/0.2.8/

XGrammar's any_order mode relaxes required-key and uniqueness guarantees. The adapter deliberately uses fixed order, with name BEFORE arguments so a chosen tool determines its argument schema. Output-only supervised targets are changed to the same ordering; R4's old alphabetic serialization is preserved but must not be mixed into this grammar-specific target path.

## Predictions and falsifiers registered before hosted execution

1. Every byte of canonical legal examples must survive the grammar mask; completed outputs must independently validate under jsonschema Draft 2020-12 AND the R4 contract.
2. Invented names, missing/duplicate/extra keys, wrong types, enum violations, integer bounds, extra root keys and more than four calls must not be decodable. Invalid examples may be rejected before completion; never counted as correct no-call.
3. Independent empty-list output remains available. The grammar does not select a correct tool for the user or supply argument values.
4. Last-state candidate projection must agree with parent final-position byte logits (atol/rtol 3e-5 in reduced unit geometry; 3e-4 full-size CPU screen). Same weights, same input and same active depth; per-layer memory states must be identical in same-chunk tests. Streaming/chunk parity and identity ownership are checked separately.
5. Mutating an original tool schema after compilation must not change a live compiled request. Every request gets an isolated matcher; no shared conversation state.
6. No added trainable weights, model fitting, tool execution, persistent verified write or calibrated confidence. Bound truncation returns incomplete, never an invented fallback call.

## Efficiency experiment

Frozen seed 61; original R4 versus same-weight restricted serving path; independent 1M/10M/default125M configurations; CPU FP32, one thread, batch1, T64; two warmups and seven alternating-order samples. The task is prefill plus next-byte scores. Original path emits all T*V logits and auxiliary heads; candidate emits only final features and at most 256 byte logits. Report raw samples and output errors, not only means. This screen excludes grammar compilation/masking, tool execution, full autoregressive decoding, GPU/mobile runtime, memory high-water measurements and quality. It cannot establish a whole-agent speedup or intelligence-per-compute improvement.

The runtime also chunks long schema prompts (default128 bytes) to avoid one huge prefill activation/logit allocation. No BPE is fitted. Byte IDs0..255 are a fixed research codec, not interchangeable with existing or future BPE checkpoints. Full model storage still includes every embedding row and every unused auxiliary head; restricted projection is not compression.

## Training readiness, not additional bureaucracy

Inherited scan audit #714: 11/84 normalized-read gradient checks still fail; unchanged by this serving work. Original failures are preserved. A precise next numerical experiment should isolate scan precision versus scalar-objective precision and quantify parameter-gradient impact rather than treating CI success as a numerical repair.

Proposed publisher source: Salesforce/xlam-function-calling-60k (APIGen), publisher card https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k. It is synthetic data, not real user outcomes. Card specifies CC-BY-4.0 and gated acceptance. No gate was bypassed or terms accepted; no corpus downloaded in this increment. A dataset license/card alone does not satisfy this repository's source/outcome attestation. Published dataset examples, independent evaluation and a training set must not be conflated.

The supervised path is executable, but training still needs actual authorized corpus bytes and fingerprint, FIT/DEV source-separated split (prefer tool-family-disjoint diagnostic split), FIT-only tokenizer, prospective declared-target-v2 FAIR_COMPARISON, matched competent Transformer/SSM controls and verified compute/checkpoint/replay evidence. No private signatures or trusted checks can be fabricated by the training agent. Prepared API/code is not a trained checkpoint.

## Compute and boundaries

Run scripts/phone_lab.py plan on research/phone_queue/csl125_r4_schema_runtime_20261106.json before tests. Bounded GitHub CPU/no-fit only; zero GPU quota and zero optimizer steps. Keep draft and unmerged. No new background automation. A green software workflow does not close #714's numerical failures or prove tool-call accuracy. Unsupported schemas fail closed through the inherited schema checker; final independent validation remains mandatory even with token masking.
