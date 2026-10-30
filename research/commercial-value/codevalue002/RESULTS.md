# CODEVALUE002 — 500K code-domain prediction and commercial-utility results

**2026-10-30 decision: UTILITY_GATE_FAIL.** Authentic source-trained code prediction was measured on real independent publisher Python source, with original source/checkpoints replayed. The trained native TAC beats the particular trained GRU and simple SSM controls, but fails to beat a cheaper FIT-only trigram and does not justify a paid code-completion product.

## Source and CI evidence

- [Original nine source-trained jobs, 41023534058](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41023534058): 9/9 actual training jobs passed. Three seeds per arm: 499,623-param graph-free layered TAC upper48; 500,165-param GRU2; 499,782-param simple selective diagonal SSM (not Mamba-3). Identical genuine 14-file PSF CPython code FIT, 320 AdamW updates, batch8 x128 next-byte targets = 327,680 authentic target exposures per model. Equal original per-seed sampling; actual training FLOPs not matched. No GPU, protected TEST or paid compute.
- [Git-owned original full checkpoints and seal](https://github.com/FPC-effortless/TAC-OSM/blob/research/codevalue002-500k-code-2026-10-30/research/replay-bundles/CODEVALUE002/SEAL.json) verify all nine model, optimizer, RNG, source hashes and schedules before external data read.
- [First DEV evaluation CI 41023905851](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41023905851) read actual heldout source and computed predictions but **FAILED ONLY REPORT SERIALIZATION**, using a PosixPath object as the JSON output. No score file was written. [Reporter-only correction commit 4d676dc](https://github.com/FPC-effortless/TAC-OSM/commit/4d676dc744d955b49cd7da1a34589ee4ee887059) changed that single final JSON serialization line, without retraining, selecting new source or changing scores.
- [Successful original same-source recovery, 41024166723](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41024166723) passed. [Original JSON result](https://github.com/FPC-effortless/TAC-OSM/blob/research/codevalue002-500k-code-2026-10-30/research/commercial-value/codevalue002/RESULTS.json) preserves all seeds and publisher outcomes. This is the same source opened in the prior failed reporter, NOT a second independently blind evaluation.

## Real code results

Holdout: 2 original source files each from the Requests, HTTPX and Click Python publishers, no cross-owner concatenation, 4 reset 128-byte windows/file at registered offsets, exactly 3,072 original next-byte targets/model. All three neural seeds evaluate the SAME source labels. Exactly 48 precommitted completion offers per model, two fixed prefix lengths per window, fixed 2/4/6-byte greedy candidate policies.

| Predictor | Learned params | Mean NLL down | Top1 up | Exact2 per 48 | Exact4 primary per 48 | Exact6 per 48 |
|---|---:|---:|---:|---:|---:|---:|
| FIT-only P3 trigram | 0 | **2.325021** | **48.63%** | **17** | **5** | 0 |
| Original graph-free TAC | 499,623 | 2.484746 | 41.22% | 9.0 | 0.667 | 0 |
| GRU2 | 500,165 | 2.604304 | 39.34% | 7.667 | 0.333 | 0 |
| Simple selective SSM | 499,782 | 3.011593 | 27.31% | 5.0 | 0 | 0 |

Neural suggestion counts are seed means, NOT fractional actual suggestions. Native TAC original seeds exact-four: 0/48, 0/48 and 2/48; the two successful positions were one Click and one Requests. The P3 exact-four count was 5/48: one Click, two HTTPX and two Requests.

The preregistered rule requires native TAC mean exact-four optimistic saved keys to exceed both zero and P3, with at least two publisher groups and two seeds favorable. **It failed: 0/3 group wins and 0/3 seed wins.** Under the theoretical perfectly-accept-correct and reject-incorrect oracle, exact four-byte completion saves at most three keystrokes, yielding 15 keys saved for P3 and a mean two keys for TAC across 48 offers. This is NOT measured developer time, actual acceptance or revenue. All six-byte exact completions failed across every arm.

## Compute, memory and limitations

- Neural original state_dict tensor bytes: TAC 1,998,492; GRU 2,000,660; SSM 1,999,128. This is not total deployed RAM or energy.
- FIT-only P3 logical packed count-store estimate is 333,011 bytes, which omits Python dictionary allocator/RSS overhead; no equal-memory cost winner claim.
- Hosted CPU full-reset-window median forward approximately 3.0 ms TAC, 4.1 ms GRU, 8.0 ms SSM; no comparable P3 full inference benchmark and no physical Android benchmark. Do not transfer these into an optimized decoding or mobile superiority claim.
- TAC beats the particular other neural controls in NLL under matched parameters/data/updates, but unmatched total training FLOPs, untuned baselines and non-state-of-art SSM prevent strict superiority.
- No actual user, invoice, customer pricing, support costs, contract or paid pilot. No commercialization or profitability result proved.

## Decision

Do NOT promote native 500K TAC-alone Python autocomplete from these results. Keep original failed 100K AUTOCOMPLETE001 and this negative experiment separate, transparent and unmerged. The inexpensive classical baseline is stronger for this exact job. The commercial follow-on should target a real, source-grounded event prediction or locally repeated-session task in which bounded persistent memory *actually saves a buyer time or money*, with strong classical and neural controls, physical Android qualification and verified paid customer unit economics. Do not tune against these now-exposed source labels and call it independent validation.
