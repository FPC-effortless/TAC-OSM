# SENSE-TOK-007 — first bounded CPU training pilot

One original 101,008-parameter S005 model completed **128 genuine AdamW updates**
in 161.04 seconds (seed 1, FP32 CPU, 131,072 original-byte targets).
Source and recipe were frozen before the first optimizer update at
`c7d8f369b6ae6e895d2b454c6c632a2e4ff78637`.

The owner explicitly approved this one noncomparative exploratory fit, deferring
independent attestation, full formal verification and comparative qualification.
The ordinary launchers and qualification requirements remain unchanged. No
additional seed, retry, GPU, sealed TEST, checkpoint promotion or superiority
claim is authorized by that exception.

## Actual endpoint results

| Exposed DEV condition | Bits per byte |
| --- | ---: |
| Initialized adaptive | 8.037543511 |
| Trained adaptive | 5.461519271 |
| Trained weights, forced BYTE (posthoc) | 5.461053599 |
| Trained weights, forced TOKEN (posthoc) | 5.857633534 |

The model learned to reduce next-byte prediction loss on these already exposed
DEV prefixes. This is not an independent generalization or capability result:
there is one seed, no trained comparator and no sealed evaluation. Forced modes
are same-checkpoint, posthoc interventions, not independently trained baselines.

**Useful adaptive switching was not demonstrated.** TOKEN occupies 98.50%
of DEV decisions (1507/1530), while forced BYTE is marginally better by
0.000465672 BPB. The controller
mostly selected TOKEN, rather than establishing a useful conditional tradeoff.
The raw local-byte readout remains active in both pathways.

A posthoc fixed-key code-value intervention on the ast.py prefix raises BPB
from 5.216061789 to 5.239382826 with shifted values, or 5.242472916 with zero
values. Joint row permutation leaves logits exactly unchanged. Code values
have measurable influence on that prefix; useful semantic tokens or a general
codebook advantage do not follow from this one possible out-of-distribution
intervention. No endpoint or learning-rate tuning followed these results.

## Original source and exposure

FIT consists of CPython v3.13.5 fractions.py and statistics.py: 101,852 original
bytes, whole-file SHA256 and publisher Git blobs checked before the run.
397 complete 256-byte windows were eligible; 220 trailing bytes were excluded
and recorded. Sampling with replacement produced 512 training sequences.
Existing six exposed DEV prefixes total 1,536 scored bytes, evaluated only
before and after training. Exact FIT/DEV 256-byte duplicates were excluded
(none found). Near-duplicate/boilerplate overlap and common-repository domain
limitations remain; no independent holdout claim is made.

## Replay and integrity actually checked

Five original checkpoints at updates 0/32/64/96/128 retain weights, AdamW,
RNG, step and source-order state. All 11 parameter tensors changed; all 11
AdamW state records at the final checkpoint report step 128. Two fresh Python
processes reproduced initial and final logits bitwise, and recomputed exact
endpoint BPB. They verified all original checkpoint hashes and each journal
sample's source bytes/order. Future-prefix perturbation preserved preceding
predictions on all six exposed DEV prefixes. No extra optimizer updates were
used for audits; trained continuation/resume is not claimed.

The full original weights, optimizer states, journal, source receipts,
prediction tensors, audit outputs and report are preserved in the evidence ZIP.
Independent evaluator/ACL qualification, complete PNDS model/IEEE/AD proof,
complete adaptive probes and full comparative compute accounting remain OPEN.
Local source suite: 1,311 passed. Qualified capability and architecture superiority:
**NOT ESTABLISHED**.

## Reproduction

Training entrypoint: `scripts/run_sense_tok_007_cpu_pilot.py`; read-only replay:
`scripts/audit_sense_tok_007_cpu_pilot.py`. Original run output must not already
exist, preventing accidental repeat fits. Re-training is outside the exhausted
one-fit approval; checkpoint replay is read-only and performs zero optimizer
updates. The evidence archive contains original outputs and source digests.
