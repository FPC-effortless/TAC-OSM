# CODEVALUE003: longer500K code-language training on dual T4s

Status: all9 fits and45 checkpoints completed; all392 score files recovered.
The remote kernel ended in ERROR during final JSON summary serialization;
local recovery fixed only NumPy win-count integer conversion, without retraining
or producing new predictions. See [RESULTS.md](RESULTS.md), [RECOVERY.json](RECOVERY.json)
and the unchanged remote [RUNTIME.json](RUNTIME.json). Utility screen: FAIL.
Original CODEVALUE002 evidence remains unchanged.

The owner requested longer training and better language/usefulness tests, then
provided a Kaggle credential on2026-10-30. This named follow-up uses exactly the
original499623p TAC,500165p GRU and499782p simple selective SSM, three paired
seeds, original PSF CPython code, B8x128 and10240 AdamW updates/model. That is
10,485,760 next-byte training labels per model,32 times the old exposure. All
three models restart from scratch, retain a constant learning rate and have no
tuning. Keeping the corpus/architecture fixed isolates longer optimization;
overfitting to the small repeated corpus is a possible negative outcome.

## Runtime and original evidence

Two physically observed, distinct-UUID T4s are mandatory, with real CUDA probes.
GPU0 trains seeds1701/1703; GPU1 trains1702; each performs all three arms
sequentially. The final seed tail is imbalanced. The coordinator enforces a7140
second end-to-end deadline, below the120 minute allocation/240 GPU-minute ceiling.
No accelerator fallback, automatic retry or automatic extra allocation exists.

Save complete model, AdamW, Python/NumPy/torch CPU/CUDA RNG and sampler state at
updates320/1280/2560/5120/10240:45 original checkpoints. Hash the actual sampled
owners/offsets/bytes, verify the replayed schedule for every arm/seed, and cold-load
the model on the same device to compare predictions. Restore RNG after the cold
constructor so auditing does not change training. The final gate verifies all
nine actual parameter counts and paired sample receipts. Seal all45 checkpoints
before opening any DEV bytes for scoring. Failure/timeout preserves original
completed checkpoints and logs and cannot yield a complete-cohort success claim.

`SOURCE_RECEIPT.json` pins30 actual publisher files with original commit, blob,
size and SHA256:14 FIT,6 exposedDEV and10 transferDEV. Nine original license texts
are archived in the private dataset. Exact duplication and a sampled64-byte
shingle overlap screen passed. Historical exposure across all prior archives is
INCONCLUSIVE; semantic copying is not ruled out. TransferDEV is internal DEV,
not protected or access-separated blind TEST. Publisher identities were frozen
before content checks. No held-out scores were used to choose the protocol.

## Evaluation

For every model checkpoint, score all exposed Requests/HTTPX/Click targets from
byte64 onward. Score final models on ten additional files from Django, FastAPI,
Rich, Pydantic and pytest, once after sealing. Each target is scored once with
identical windows/masks:64-byte burn-in plus up to64 scored targets. Partial
final blocks count only genuine targets. Reset state between windows/files.
The final10240 endpoint is primary; checkpoint curves are descriptive, never a
best-transfer-checkpoint selection.

Offer512 deterministic, unique suggestions per file:5120 on transferDEV plus
3072 on exposedDEV. Autoregressively generate32 bytes from a rolling128-byte
observed/generated prefix; gold suffixes are never fed back. Retain generated
bytes and score exact2/4/8/16/32 and longest correct prefix, NLL, bits/byte,
top1/top5, Brier score and ECE10. Compare no suggestion, original FIT-only P3
(tau4), fixed interpolated Witten–Bell order6 and the two neural controls.
Classical models read every FIT byte; neural sampled exposure is disclosed.

Record synchronized batched timing separately from genuine batch1 p50/p95
suggestion latency (16 measured offers after8 warmups, firstDEV file/model).
Report process RSS, GPU allocated/reserved peaks, tensor/checkpoint size and
logical classical count-store estimates. T4 latency is not Android performance.

The exact4 utility screen requires mean TAC gain over BOTH count models, lower
paired95% publisher/file hierarchical bootstrap bound above zero, wins in>=4/5
publishers and>=2/3 seeds, and positive net keys under a four-key wrong-suggestion
cost. Bootstrap uncertainty is conditional on the three observed seed runs.
Publish each seed/file/group and preserve every failed gate. Oracle/rejection
key costs are simulations, not observed developer productivity.

## Gates, scope and reproducibility

The owner-scoped500K exception is limited to this exact study and matrix; global
100K rules remain unchanged. `FAIR_COMPARISON.json` passes internal preflight and
forbids architectural-superiority claims. Supported prefit operator FLOPs are
recorded on actual B8x128 FIT input, but complete operator/backward coverage and
matched training FLOPs are NOT established. The study is exposure-matched
internal engineering evidence, not a compute-matched architecture comparison.
The original graph is unchanged; whole-model composed PNDS/IEEE/AD certification
remains OPEN. No protected TEST, independent certification or checkpoint
promotion is authorized. Direct API submission is specifically authorized by
the current owner request/credential; protected GitHub workflow rules are not
weakened. This is a single bounded submission, not background ongoing research.

Local validation includes inherited-model count/causality/gradient checks,
scoring denominator and autoregressive future-poison tests, paired sampler
compatibility, fail-closed size/scope/final-receipt tests and architecture-agnostic
probe tests. Constructed tests provide zero trained-capability evidence.

```sh
python -m pytest -q research/commercial-value/codevalue003/test_run.py tests/test_research_parity_gate.py tests/test_research_probe_agnostic.py
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/codevalue003/FAIR_COMPARISON.json --phase preflight
python scripts/phone_lab.py plan research/phone_queue/codevalue003-long-code-t4x2.json
```

`prepare_sources.py` verifies/archives immutable publisher bytes and licenses.
`package_run.py` creates a credential-free whitelisted source/data ZIP pinned to
an exact clean Git commit, private dataset metadata and an archive-hash-checked
offline bootstrap. All original model weights, receipts, logs, seal, full per-file
scores and RESULTS.json remain in Kaggle outputs for download/archival.

A utility-screen pass would motivate a separately registered randomized human
pilot. General English modeling, executable code correctness, actual developer
time/acceptance, physical device performance, willingness to pay and fully loaded
training/serving costs are not measured here. Commercial viability remains open
regardless of a lower next-byte loss.
