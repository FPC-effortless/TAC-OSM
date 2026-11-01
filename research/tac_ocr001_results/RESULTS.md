# TAC-OCR-001 — measured engineering results, 2026-11-01

The deliverable is a usable streaming **pretrained Tesseract OCR pipeline**
and three **untrained custom grounded decoders**. These are separate execution
paths. No custom OCR quality, trained architecture win, or Unlimited OCR 2
parity has been measured. No optimizer updates or protected TEST access occurred.

## Actual constructions and CPU execution

Four layers, decoder width 128, UTF-8 byte vocabulary 259, 128 visual references.
Every candidate shares the same vision bridge; active decoder parameters are
within 0.04%. The actual frozen publisher encoder after dropping its unused
pooler has **21,812,352** parameters. Its immutable revision is
`04e994ab854b0089d4929f48c2b4dbe2ce78a340`.

| Candidate | Trainable decoder | Total deployed with frozen encoder | Eager median ms/byte | Carried tensors | Shared visual K/V |
| --- | ---: | ---: | ---: | ---: | ---: |
| TAC | 1,309,376 | 23,121,728 | 2.680 | 295,936 B | 131,072 B |
| Cached RoPE Transformer | 1,309,312 | 23,121,664 | 1.643 | 520,192 B | 131,072 B |
| Residual GRU | 1,309,824 | 23,122,176 | 0.992 | 2,048 B | 131,072 B |

The optional frozen TAC step measured **1.497 ms/byte**, a **1.79x** execution
speedup over its same-weight eager path in this profile. The 12-step FP32
logit difference was zero. Compilation took **1.315 s** and is excluded from
steady latency. Fixed-geometry FP64 tests additionally cover state/logit
correspondence, changed visual regions and a large dynamic clock.

Scope: CPU, one PyTorch thread, random decoder weights, cached constructed
features, three timed rounds after one warm-up, 64 steps per round. Vision
encoding is excluded. Timings are observations on a shared managed host;
earlier eager profiles varied. These are not trained OCR throughput results,
fair trained-compute comparisons or GPU measurements. TorchScript's trace
specialization/deprecation warnings are preserved in `CPU_PROFILE.json`.

The real pinned TrOCR encoder produced `[1,576,384]` patch features from an
upstream image crop, which passed through all three decoder graphs with
finite `[1,3,259]` outputs. `ENCODER_INTEGRATION.json` records that connection,
not learned transcription. The original integration receipt is preserved:
it included an unused newly initialized pooler, subsequently removed from
execution and deployed counts.

## Long-input software stress

`REPEATED_PAGE_SMOKE.json` measures 1, 10, 50 and 100 PDF pages containing
copies of **one** upstream Tesseract test image, plus a 50-page one-worker
reference. This checks scheduling, rendering, output completeness and
throughput. It cannot establish genuine long-document recognition or
cross-page understanding. All page records were present and repeated-image
transcripts were identical. Outputs were written to disk.

| Pages | Page workers | End-to-end seconds | Pages/second |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 0.285 | 3.51 |
| 10 | 2 | 1.286 | 7.77 |
| 50 | 2 | 6.065 | 8.24 |
| 100 | 2 | 12.094 | 8.27 |
| 50 | 1 | 11.702 | 4.27 |

Main-process high-water RSS is reported separately. The runtime did **not**
expose OCR child processes in the sampler; total process-tree memory remains
**INCONCLUSIVE**, and `sampled_peak_process_tree_rss_bytes` is null. Do not
interpret main-process memory as total OCR memory. No GPU was available.

The initial profiler incorrectly used virtualized process IDs with host
`/proc` paths. Its original receipt is retained in
`SMOKE_INITIAL_INVALID_MEMORY.json`; its memory figures are INVALID. The
final sampler resolves the current process correctly and refuses to invent
a total when children are unobservable.

## Checks and remaining qualification

- **1,211 existing repository tests passed**.
- **34 new OCR tests passed**; the focused command ran 56 checks including
  22 already-existing shared probe tests. Counts must not be added twice.
- The PNDS companion's independent reference and new source-pin mutation
  fixtures passed; the historical 17-variant validator passed.
- Local Lean invocation failed with `error: failed to locate application`
  in the managed runtime. A dedicated explicit CI root/axiom check is provided;
  compilation is not claimed from local execution. Full-model proof remains
  OPEN regardless of that scoped theorem's build result.
- **No real OCR training/evaluation is qualified.** Original real-data
  acquisition/authentication, fully measured parity/tuning budgets, independent
  signed prefit evidence and the approved private dual-T4 training workload
  remain required. `PROTOCOL.json` is a no-fit plan, not a passing training gate.

Practical limits: no validated table/formula/handwriting recognition or complex
reading-order guarantee; no learned relevance/SSA selector; no whole-document
image-to-trained-TAC transcript; no production readiness claim. The first
research milestone is genuine image-grounded transcription under a frozen
encoder, followed by a separate authentic long-document test.
