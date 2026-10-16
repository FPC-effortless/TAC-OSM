# DEV-030: Kaggle execution preflight for DEV-029

The user selected Kaggle with available GPU quota. DEV-030 is an engineering
preflight for the existing research program, not a new architecture experiment.
DEV-029's protocol and recorded outcomes remain unchanged. The six-arm tuning
grid is 72 trials, before twenty fresh confirmation initializations per arm;
the approximately 14 CPU-hour estimate covered tuning only, not confirmation
or the complete system. DEV-028 remains initialization attribution, not a
full-system capability test.

## Scientific interpretation remains bounded

Schema41's 100% and masked-local TAC's 75% cancellation scores were observations
from one initialization and one 216-update recipe. They do not prove that
unpartitioned models cannot represent cancellation, that entity structure is
the sole cause, or that the masked-local writer is a proven redesign. The
classical table is explicitly entity-indexed and task-informed; its small head
is learned. It is not a non-neural competing model. Its cancellation failure
under this recipe cannot be attributed to lost state when its sufficient-state
construction passes the public causal checks.

## Frozen preflight

[Protocol](../research/dev030/protocol.json): all six arms, B2, H512, FP32,
two warmup updates, five timed updates and three repetitions. Timing includes
public fixture generation, tensor encoding, before/after receipt prediction,
backpropagation and AdamW. CUDA synchronizes timing boundaries. Only the
`dev030-engineering-fixture` namespace is used; no scientific validation or
test is generated.

Compare same-initialization CPU/CUDA logits (rtol1e-5/atol1e-6), exact initial
categorical predictions and service choices, finite gradients, and one-update
logits (rtol1e-4/atol1e-5). These are preflight criteria, not retrospective
relaxations of old replay tolerances. Long training trajectories are not
claimed identical across backends. Failed checks retain evidence and block a
GPU recommendation. Smoke timings cannot choose a research backend.

The GPU is an engineering candidate only when every arm passes parity and the
sum of per-arm median end-to-end update times is at least 20% lower than CPU.
The recommendation selects compute, not scientific outcomes or architectures.
Small recurrent states, batch two and Python loops can make kernel-launch and
transfer overhead dominate. A GPU allocation is not evidence of a speedup.
No AMP, TF32, compilation, larger batches, task changes or training-update
changes are introduced.

`DeviceBoundary` moves public tensors at encode/update/decode boundaries while
preserving state-dictionary names. The classical table's deterministic public
updates remain on CPU; its learned head uses the selected device. This hybrid
placement is explicit. The resumable tuning runner records device, torch/CUDA
versions, GPU, precision settings, CPU/GPU RNG and optimizer states, and rejects
resume under a different recorded runtime. Defaults remain CPU.

## Private Kaggle package

`package_kaggle_preflight_dev030.py` builds a private GPU notebook, API metadata
and checksum manifest. The notebook fetches a full pinned public git SHA,
checks its checkout, runs only the preflight, and retains partial JSON/logs
if checks fail. It uses Kaggle's installed CUDA-enabled torch and records that
actual version; it does not assume the old CPU version is installed or certify
an identical software environment. A later study must freeze the observed
runtime and retain it as a separate cohort.

No credential is embedded or transmitted into the notebook. Launch credentials
are supplied only to Kaggle's authenticated CLI. No paid Hugging Face job is
started. Native Kaggle GPU/quota access and API authentication are separate
checks. Quota is verified by API, not inferred from the user's selection.

Generate from a committed, pushed source:

```sh
python scripts/package_kaggle_preflight_dev030.py \
  --source FULL_40_CHARACTER_COMMIT_SHA \
  --username YOUR_KAGGLE_USERNAME --output /workspace/dev030-kaggle-package
kaggle kernels push -p /workspace/dev030-kaggle-package
```

Without `--username`, the notebook is importable manually, but the metadata
contains an explicit username placeholder and must not be submitted unchanged.
The notebook requires GPU and Internet enabled; metadata requests both and
sets `is_private=true`. Accelerator type is observed from torch rather than
assumed. Preserve a notebook version with its outputs; returned logs and JSON
are under `/kaggle/working/dev030-evidence`.

Kaggle source documentation: [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[API authentication](https://www.kaggle.com/docs/api).

## What is not completed by this preflight

No architecture advantage, full-system capability, discovered entity structure,
72-trial tuning result, twenty-seed confirmation, or unseen-identity/capacity
result follows from successful preflight. Validation-only selection still
requires complete tuning coverage and independent auditing, a separate committed
recipe manifest, and only then a separately frozen untouched-test evaluation.
CASM, CDL, OSM, verification/repair and multi-timescale integration remain on
the connected integration track; a compute move does not replace them.
