# Integrated memory development successor

Status: **development implementation; not preregistration or a confirmatory result**.

This lane implements the concrete mechanism gaps found in the 10 October 2026
architecture review. It depends on Integration-002 PR #158's noisy-address
benchmark and preserves its negative pilot. Frozen FULL-LEARNED-SWILA-001,
PERSISTENCE-002, ADDRESS-003 and INTEGRATION-001 contracts/results are untouched.
Do not merge or cite this branch as positive science based on passing CI.

## Architecture

`visible write -> finite learned delta memory -> learned expert read routing -> selected generic operators -> answer -> public post-action observation/evidence -> learned verifier -> gated memory transition -> later query`

The 768-scalar recurrent state per episode has shape `[4 experts, 8 values, 24 keys]`.
There is no retained address/value history available for query-time replay.
Diagnostic write-event traces grow with history and are preserved for auditing;
fixed recurrent state is not a claim of fixed total profiling memory. The first query uses the existing 12D
noisy-address synthetic benchmark. The model learns its answer transformation,
including the current bit, instead of executing a fixed XOR answer rule. XOR
defines evaluator labels and the classical baseline only.

Key-side erase and value-side write gates are distinct and channel-wise. This
is a small independent **gated delta-rule adaptation**, inspired by recent
memory work; it is not a reproduction of Gated DeltaNet-2 or SwiLA. It cannot
inherit their benchmark results, throughput or generalization claims.

Sources: https://arxiv.org/abs/2605.22791 and https://arxiv.org/abs/2609.39034.

## Conditional computation

Top-1 operator selection uses a selected-branch straight-through surrogate:
hard normalized weights in the forward pass, softmax derivatives in backward.
Unlike renormalized `p/p`, this provides a nonzero task-gradient signal. It is a
biased estimator because unselected operators are not executed. The code and
tests require nonzero, finite task gradients at the selector, not merely a
non-None gradient tensor.

Training keeps a soft halt gate. In evaluation, each halted sample is removed
before operator dispatch and later depths. Forward hooks independently verify
that dispatch traces equal actual executed operator/sample calls. No fractional
halt activity is presented as saved work. The pilot does not supervise optimal
stopping, so the existence of early-exit mechanics is not a claim that training
discovers efficient depth allocation.

## Verified carry test

After the first task, a new public observation writes an address/value with
continuous sensor reliability evidence independent of query correctness. A
later noisy query sees the address and current bit, never the payload or oracle
target. Paired partners have identical pre-commit memory, query and reliability
evidence and opposite write values/targets. Reset and no-commit controls have an
exact 50% ceiling. The returned state is actually used by the later decision.
Its loss trains the write pathway; that gradient is measured separately from
verifier supervision.

The verifier is a synthetic **sensor-validity** learner. It is not a verifier
of arbitrary reasoning, task success, or a proof of self-correction. Evaluation
also supplies negative reliability evidence. A rejected write gates the entire
memory transition, including decay, and preserves the prior state exactly.
False accepted writes and rejection errors remain measurable outcomes rather
than grounds for silently discarding a negative pilot.

The follow-up has a short delay and no new intervening writes. Long-term
consolidation, repeated task transfer and preservation through adversarial
post-commit writes remain future experiments.

## Frozen full PLM compatibility lane

`development/full_plm_reference.py` is a byte-for-byte copy of #143's source at
`8792cbd37701b2d33c7c86db1b0514b6e33dd480`, SHA-256
`c307667fc05590fd4e0729b1e748e8eeff3152cc52410e82c254b3ca103c931c`.
It preserves the reference for comparison, including the bugs being diagnosed;
it is not the recommended runtime class. `DevelopmentPLM` replaces CASM and
the commit kernel and exposes `observe`, `decide`, and `commit` with explicit
`StreamingState(memory, responsibility)`. Each observe consumes only one fresh
text/image/audio observation. Responsibility carry survives streaming chunks.
The following observation and decision receive the returned commit state, and
a mechanism test verifies gradient flow from that decision to the write
projection. The old runners and experiments are not changed.

The streaming `decide` method tests the action/memory/compute mechanism. It does
not invoke the reference OSM planner. This lane is tested separately and is not
the model trained in the noisy-address pilot. It is not a measured full PLM
multimodal capability result.

## Development schedule and evidence

The default runner uses three training seeds (43,59,71), 180 steps/seed,
batch size 8, AdamW, learning rate .003 and gradient clipping at 1.0. Evaluation
seeds are 131,149,167; H=32/128/512, D=1/16, hard fraction=0/1 and noise=.08/.25.
Two opposite-past pairs per cell give 144 paired rows and 288 initial-task
episodes plus 288 follow-up episodes. There is no checkpoint selection, tuning
from evaluation outcomes or positive threshold attached to this pilot.

Evaluation measures learned, reset, swapped-history, no-intervening-write,
dense Euclidean and nonlearned fixed-cache controls. It preserves targets,
per-episode predictions, regenerated episode fingerprints, actual write-event
and dispatch traces, all loss steps, component gradient diagnostics, checkpoint
hashes, source hashes, environment and per-seed summaries. Learned fixed-state
answer accuracy is not mislabeled as target recall; target recall is available
only for the dense Euclidean comparator.

`validate_integrated_memory_dev.py` reloads every checkpoint and replays every
scheduled pair. It requires exact predictions, data identities and operation
traces, rejects source/checkpoint mismatch and overlapping/duplicate evaluation
episodes, and recomputes aggregate/per-seed metrics. This is independently
executed checkpoint replay using the same model/generator code, not an external
scientific audit. Recorded training-gradient diagnostics are not independently
retrained by the validator.

These evaluation seeds use the development generator family that informed the
design. They are not an untouched confirmatory corruption/key family. A fresh
prospective scientific contract is required before confirmatory experiments.

## Cost and comparison boundaries

History ingestion remains O(H+D). Queries read a fixed number of memory experts.
The state-element and component-call traces are not FLOPs, hardware efficiency,
or a fair accuracy/cost ratio against dense retrieval. Fixed state bounds storage
and imposes interference; it does not preserve unbounded history losslessly.
Gated DeltaNet-2, faithful SwiLA, GRU and attention/hybrid baselines under matched
parameter/state/training budgets and seed-level confidence intervals are still
required before claiming modern architectural competitiveness.

## Run

```bash
python scripts/run_integrated_memory_dev.py
python scripts/validate_integrated_memory_dev.py artifacts/INTEGRATED-MEMORY-DEV-003
```

`--smoke` uses two steps/one seed and is explicitly labeled as a software check.
The workflow runs mechanism tests before measurement and preserves checkpoints
and evidence even when replay fails. Remain draft until evidence is reviewed.
