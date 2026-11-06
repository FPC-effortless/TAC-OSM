# CSL-125-R4 — schema-grounded tool-calling model (DRAFT, NO FIT)

Date: 2026-11-06. Branch: research/csl125-r4-toolcalling-2026-11-06. Parent: draft R3 PR #715, stacked on #714/#713/#712/#711/#710. No existing model source is overwritten. The **11/84 R2 scan-normalization gradient failures** from #714 remain failures and must be independently resolved before fit/promotion.

## Problem and capability contract

The owner wants a Needle 3-class compact model whose PRIMARY skill is calling tools, not unconstrained chat. [Needle 3 official release](https://www.cactuscompute.com/needle) and [repository](https://github.com/cactus-compute/needle) describe 2–20 deployable subnetwork depths, tool choice from supplied schemas, typed arguments, ordered multi-calls, off-topic empty calls, constrained JSON and confidence calibration. Its published deployment and quantization benchmarks are **not reproduced or matched** here.

CSL125R4 currently implements the *tool-call proposal interface*, not the learned skill:
- Condition on caller-provided tool definitions and instruction; learn via next-byte tool-response supervised loss at every valid rung.
- Emit a bounded JSON envelope of function_calls, possibly empty, with up to four ordered calls. Dynamic names are generated through the tied language head; no fixed tool-ID class list required.
- Strict runtime validation against a safe JSON-schema subset (required fields, closed objects, bounded arrays, enums, numeric/string bounds); duplicates and unknown names fail closed. No functions invoked inside this module.
- Reference JSON-prefix decoder forces envelope beginning and masks illegal **JSON syntax** bytes, but currently **does not compile the entire tool-specific schema into each decoding step**. The post-parser rejects schema violations afterward, which is less capable than Needle 3's published byte-level schema grammar.
- Separate host review classifies read/write/high-impact calls against explicit allowed-name set and confirmation flags. Even calls marked candidate_only have **no execution authority**; an external authenticated host must perform ACL, idempotency/reconciliation, user-confirmation and evidence checks.
- Tool observations can be rendered as bounded structured data in a subsequent request. Learned tool-result-conditioned planning and error correction remain **NOT PROVEN**.

## Architecture, training and parameter accounting

R4 is a **zero-added-parameter** subclass of R3:
1. Instruction + tool schemas → canonical JSON prompt.
2. Existing embedding → 20-layer compositional role-state/conv/FFN backbone, local GQA at 4/8/14/20.
3. Existing tied embedding/output projection → JSON byte stream, supervised at 2/8/20 exits.
4. Reference byte grammar masks syntax-incompatible tokens; typed validator rejects invalid tool names/arguments.
5. Proposals only → external host review → independently authorized execution if available → result receipt.
6. R3's uncalibrated error/revision heads remain counted but **do not grant confidence/permission** or make the trained tool calls correct.

Default R4 parameters: **123,873,263**, including tied 32,768×768 embedding/output, all blocks and all specialist heads. Hard maximum **125,000,000** (headroom **1,126,737**). This is NOT an 8–29MB deployment claim. For ideal 2-bit *weight-only* packing, lower bound is **30,968,316 bytes (~29.53 MiB)** before scales, metadata, kernels, tokenizer and persistent/KV state. R4 has **no fitted 32K byte-fallback BPE**. The reference adapter maps UTF-8 bytes to IDs 0..255 only: a distinct, token-inefficient research codec whose future BPE agreement needs separate tests.

Training API CSL125R4.tool_sft_loss() consumes token IDs and an output-only next-token Boolean suffix mask. Labels never enter same-timestep predictions. It rejects packed-document mask islands; independent documents need reset/partition. Optional R3 revised logits are trainable through a separate weighted loss, default off. This API allocates NO optimizer and performs NO updates without independently approved training.

Reference propose_tools() uses actual untrained model logits with syntax-constrained byte decoding. It can return incomplete or invalid outputs. No scripted answer generator or oracle is reported as a neural capability.

## Central experiment

**Hypothesis:** persistent role-state and schema-conditioned byte generation might attain comparable tool-call exactness per total parameter and compute against competent compact Transformer/SSM baselines. Zero-shot transfer to unseen tool catalogues and learned refusal are empirical questions, not implied by schema conditioning alone.

Only fit after source rights, train/development split integrity, tokenizer FIT-only provenance, reward/gold ownership, leakage/ACL review, scale cap/optimizer/compute ledger and FAIR_COMPARISON registration are satisfied. Pre-register at least 5 paired seeds, fixed token and tuning budgets, held-out tests, checkpoint/RNG receipts and failure thresholds.

### Experimental arms (first 1M/10M; 125M only after evidence gates)

- R4 full tool-call SFT at early/middle/full exits.
- Parent R3 unchanged and a same-parameter no-persistent-state ablation.
- Competently trained GQA Transformer and DeltaNet/SSM at matched parameters, same tokenizer/data/updates; disclose any compute inequity.
- Tool schema removal/reordering/counterfactual name-description mismatch, no-tool negatives, distractor arguments, held-out tool names, multi-call sequence, observation conditioning and malicious text injection.
- R3 output-correction disabled/constant/shuffled/full/top-K as separate controlled arm, not assumed beneficial.
- Downstream effect authorization is a separate measurement from JSON or AST correctness.

### Metrics and falsifiers

Tool capability: exact ordered tool-name sequence, typed AST argument exact-match, required/extra field errors, no-tool abstention precision/recall, false-call rate, invented tools, multi-turn result provenance, copy fidelity, invalid/truncated generation rate and confidence/ECE only after independent calibration. BFCL v4 single-turn and mobile-action tests are future qualified comparisons, not results.

Efficiency: per-exit parameters, schema/input/output length in *bytes and tokens*, prefill/decode token/s, p50/p95, peak RAM/VRAM, all-in runtime and tokenizer + KV/state + compressed weights + quantization metadata. Include grammar, validation, review and result-reconciliation overhead. Physical CPU/GPU/mobile and real quantization remain unmeasured.

Fail claim if: inherited R2 gradient failure affects training, unseen tool generalization or abstention collapses, multi-call order fails, typed argument errors persist, schema shuffling does not hurt, competent controls dominate at fixed all-in cost, or quantized deployment degrades quality.

## Implementation status

- src/tac_osm/csl125_tool_protocol.py — typed catalogue, strict parsing, source-masked examples, bounded observations.
- src/tac_osm/csl125_tool_grammar.py — **JSON syntax-only** ASCII-prefix reference, not full JSON-Schema grammar.
- src/tac_osm/csl125_tool_model.py — no-new-parameters R4, output-only supervised loss, bounded model-conditioned inference.
- src/tac_osm/csl125_tool_review.py — review-only host policy; **no executor**.
- tests/test_csl125_tool_*.py — engineering/causal/security/gradient tests, not trained capability evidence.

NO optimizer updates, actual learned tool-call accuracy, calibration, signed dataset, mobile export, CQ2 weights, arbitrary JSON-Schema compile, external effects, real-world benchmark or Needle 3 parity. Keep draft; no auto-merge, auto-fit, or device superiority claims.
