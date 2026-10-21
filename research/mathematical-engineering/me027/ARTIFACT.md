# ME027 complete fit and resource record

Archive: `/workspace/me027-checkpoints-and-resource-records.tar.gz`

SHA-256: `1e5d5c4fa499c02d159320b1888052ab63db9cd78cd5a4ef997ef99c8194f3a2`

Size: 77,033,077 bytes; 228 files.

All twelve fits, 72 registered budget weights/optimizer/RNG checkpoints,
twelve primary standalone exports, original probability files, latest partial
checkpoints, nine state-mean diagnostics, traces, CPU charges, inference timing
samples, source/test metadata and profiler subtotals are preserved. All failed
gates and unfavorable budget points remain. Raw observations are excluded.

`artifact-index.json` records every other payload file's SHA-256 and size.
Replay additionally requires the authenticated original ME022 cache as detailed
in REPRODUCE.md. Verify archive and payload hashes before using any checkpoint.
Local artifact replay is not independent scientific validation.
