# ME025 original checkpoints and ME026 native runtime artifact

Archive: `/workspace/me026-original-checkpoints-and-runtime.tar.gz`

SHA-256: `d075fd80e2ce6caeeca2ba3fccf1a01b3cd9da41c767fd4eb0f94e23a9051c01`

Size: 10,836,446 bytes; 30 files.

The untouched original GitHub ZIP has SHA-256
`3bdcd318441e5bc7cf5c8b8338a5785d3ee32b690e7cd8a03d3fe56e1c0ec7ae`.
It contains all nine original neural weights/optimizer/RNG states, original
probabilities, traces and execution log. `runtime/` contains six derived
standalone exports, native probabilities, every timing repeat, replay records,
failed/cancelled workflow metadata and source/test provenance. Raw publisher
observations are excluded. The unmatched MLP weights were never supplied and
are not invented.

`artifact-index.json` records every other payload file's SHA-256 and size.
After extraction, verify these hashes before using any model. REPRODUCE.md
describes the authenticated ME022 parent cache needed for recorded replay.
Local artifact restoration is not independent scientific validation.
