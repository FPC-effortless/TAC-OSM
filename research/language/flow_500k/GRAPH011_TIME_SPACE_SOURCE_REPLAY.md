# GRAPH011 — Recomputing Source State Instead of Saving Every Intermediate State

**Date: 2026-10-28. Status: engineering research diagnostic, draft-only; no trained architecture promotion or independent language-quality gain.**

## Supplied attachment and verified scientific basis

Read the user's transcript **"A Computer Scientist Broke a 50-Year Barrier by Rethinking How Memory Works #migoroedu.txt"** (a documentary-style discussion of the Ryan Williams time/space result; do not redistribute the full text). Its central points are:

- Time and space are different resources. Computations can be **re-derived** rather than retaining every intermediate result.
- Cook–Mertz catalytic/tree evaluation shares otherwise occupied registers; Williams uses an implicit computation-tree reduction to produce a remarkable asymptotic space simulation.
- The transformation can be **dramatically slower** and does not require that every useful intermediate value be resident at once.
- It does not imply that a small lossy recurrent state is a lossless database of arbitrary old facts.

Verified against primary source: R. Ryan Williams, **"Simulating Time With Square-Root Space"**, STOC 2025, DOI [10.1145/3717823.3718225](https://doi.org/10.1145/3717823.3718225) and [MIT published archive](https://dspace.mit.edu/entities/publication/e86e8aa5-c6b0-43d9-9b28-fb9cf9fe9c42). The published statement is a deterministic **multitape Turing-machine** time-to-space simulation with `TIME[t(n)] ⊆ SPACE[O(sqrt(t(n) log t(n)))]` (the abstract informally writes O(sqrt(t) log t)). It uses implicitly specified tree evaluation and can require an impractical time penalty. It is **NOT** an efficient general random-access/GPU memory compression primitive; the attachment itself notes scope limitations.

**Important distinction:** GRAPH011 is NOT Williams' universal simulator, Cook–Mertz catalytic register algorithm, or any claim of P vs PSPACE resolution. The practical specialization implemented here is ordinary **deterministic sparse checkpointing with source replay**, which predates and is mathematically much weaker than those results.

## What has been implemented in TAC-OSM

`source_checkpoint_replay_v7.py` wraps a frozen deterministic streaming `torch.nn.Module` with the existing GRAPH009 source cache. Given the **original ASCII-token source**, a caller-supplied owner, an exact SHA256 source hash and fixed model weights:

1. Recompute state once during indexing, retaining a **complete snapshot every B source tokens**, not at every intermediate step.
2. For a next-character query at observed position t, use the latest retained checkpoint strictly preceding t, replay the genuine source bytes from that point, and emit the same model next-byte distribution within an explicit FP tolerance.
3. Verify owner/source hash, model tensor digest, absolute state count, probability normalization and nonempty valid prefix. Refuse changed source or changed model, missing initial state and forged owner.
4. Record resident serialized checkpoint bytes (including all Python external count queues), **original retained source bytes**, model parameter bytes, replayed token count and actual Python/CPU time separately. It intentionally **does not** report lower true RAM usage merely because the number of checkpoints is smaller.

`test_source_checkpoint_replay_v7.py` exercises both sparse and dense replay on the **authentic checked-in original GRAPH009 Python cache implementation** (`bayesian_cache_scan_v5.py`); it does not use fabricated language evaluation examples or protected TEST. `bench_source_checkpoint_replay_v7.py` measures B=1/8/32/128 and 9 registered query positions on the first 128 genuine source characters; a SHA256 digest, checkpoint count, total serialized bytes, replay token count and query latency receipts are emitted as CI artifacts.

### Algebra / expected scaling, *not* theorem borrowed from Williams

Let T be source events, B the checkpoint interval, S the true serialized state bytes and D the retained source bytes. Ignoring per-checkpoint serialization headers and nonconstant cache occupancy:

- Dense snapshots: `~T*S + D + weights` storage, about **1** replayed source step per query.
- Sparse snapshots: `~ceil(T/B)*S + D + weights`, up to **B** replayed source steps per query.
- Checkpoint-free replay: minimal snapshot storage, but up to **T** source steps per query.

If D dominates storage, sparse recurrent checkpoints may barely improve the overall memory footprint. Large external source caches can make each saved snapshot much bigger than the nominal 520 floating recurrent values. Queries and activation/temporary scratch add costs too. This is why the actual byte/timing receipts and memory/accounting are compulsory.

`PNDS/LanguageReplay011.lean` proves deterministic `foldl` composition, replay after any source split, a one-token transition lemma, checkpoint ordinal strictly preceding t and bounded replay length <=B, explicit non-removal of source bytes, source-owner/hash equality guards and an **old-state erasure** counterexample. These are exact structural identities, not bit-for-bit FP32/PyTorch or the Williams 2025 theorem.

## Relationship to GRAPH009 and GRAPH010

GRAPH009's count-based source cache remains the baseline for useful retrieval. Source replay can reconstruct original recurrent state when an older checkpoint was discarded; **it cannot reconstruct source bytes that have not been retained somewhere**. It is neither a new semantic query nor a substitute for original records.

**Mandatory negative evidence from GRAPH010** (same branch history, [GRAPH010_REAL_FIT_NEGATIVE_RESULTS.md](GRAPH010_REAL_FIT_NEGATIVE_RESULTS.md)): five genuine Vim FIT-only seeds, a 2,816-parameter approximate query, originally frozen .6 fusion. Trained query NLL **3.657214**, untrained query **3.761400**, but exact three-character count cache alone **3.474112**. Therefore learned approximate retrieval improved relative to the untrained matcher yet **FAILED the preregistered exact-cache NLL gate in 0/5 seeds**. Validation-weight exploration favored .1 (~3.429566), but this is post-hoc and not independent confirmation. GRAPH011 source-replay work **does not reverse this result or promote the learned query.** Any revised fusion must be frozen in a separate protocol and tested on genuinely new real source.

## What needs to pass before production

- Cold CI numerical parity of sparse vs dense on original real source code, with deterministic frozen model and declared FP32 tolerance.
- Actual serialized state+source+parameter bytes, requested query density, mean and tail CPU/GPU time, and re-materialization accounting; benchmark the same input/cost on a similarly cache-equipped SSM.
- A real system must own an authenticated, lawful, immutable source archive with stable owner scope and documented retention/deletion rules. Source-label strings are **not authorization**; stored SHA256 alone does not establish access rights.
- Never assume a universal time-space theorem establishes a practical architecture cost win. Full parameter-matched training, real external language TEST, semantic mismatch, GPU energy and complete machine-checkable IEEE/AD/PyTorch refinement remain OPEN.

**Research decision:** preserve GRAPH009 and the negative GRAPH010 result; evaluate optional recomputation against full resident-state retention for latency and memory, not text accuracy. Keep it in the existing System-1/Language research family, not a seventh TAC variant.
