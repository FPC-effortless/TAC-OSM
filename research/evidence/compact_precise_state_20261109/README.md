# Hosted precise-state integration evidence

[Source11ad36d2](https://github.com/FPC-effortless/TAC-OSM/commit/11ad36d2d1591769b887b263ecb6c2d639910756) /
[CPU run42579214583](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42579214583) /
[artifact13656801152](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42579214583/artifacts/13656801152).

Original HOSTED.zip is preserved losslessly, SHA256
`968afc332c6465afa3733ec3594af859b246f412ed6742479035ed02cfe4766e`.
It includes all six source/config/storage/precision/output rows, all timing
samples, 73-test JUnit and original CPU plan. `verify_saved.py` checks the exact
archive/results/coverage/test counts and within-host weight identity with no
Torch/model/corpus/checkpoint execution. The original unsupported-version failure
transcript is separately preserved without normalizing its whitespace.

**PASS:** all73selected hosted tests, all6full-size untrained R4/FP64 rows,
bitwise copied-state and next-token all-head equality at same execution
partitions; FP64 memory and state class preserved, all weights FP32/unchanged.
Exit20 backing: T64 88,334,336→1,826,816bytes; T128 92,659,712→2,213,888bytes.
Hosted copy medians0.634/0.659ms; initial local0.331/0.360ms. Raw repetitions
remain separate; no timing superiority claim or pooling across hosts.

**Identity caveat:** the local/hosted untrained weight hashes **DIFFER**, although
each before/after copy pair has identical weights. Torch versions are local
2.14.1+cpu versus hosted2.10.0+cpu; a unique cause was not investigated or inferred.
Same seeded source is not an exact checkpoint identity guarantee. Treat these
as separate internal within-host engineering replications, **not an independent
identical-checkpoint replay**. Same-checkpoint cross-host replay: **NOT RUN**.
Both initialization hashes and the mismatch are preserved in receipts/local
results; the stored checker will not silently turn this negative into PASS.

No fitted model/data score, optimizer, GPU, TEST, reward/safety certification or
promotion. State backing bytes exclude weights, allocator/autograd/runtime;
not total/device/isolated peak RAM. Copy adds time/transient allocation; old
aliases/attached graphs can retain storage. No automatic detach, precision-mode
installation or state repair. Original checkpoint/negative histories untouched.

PNDS scalar scope and partial NumPy identity reference unchanged; full-source
Lean/IEEE/AD/allocation refinement OPEN, fresh Lean NOT RUN. Actual device target,
operators, numerical agreement, physicalp50/p95/RAM/energy/airplane-mode NOT RUN.
No offline-inference exception or remote fallback. Current100k/matched<=1% gates
remain; the123.87M random-weight object is not a fit/deployment exemption.

Evidence-only closure freezes full resource execution to the measured source
or explicit dispatch. Ordinary tests remain automatic. No new full-model
measurement is implied when that step is skipped on closure/future source.
