# LANGSCALE001 execution decision — concrete proposal

**The owner explicitly authorized the reviewed FIT-only exception on 2 November 2026. Independent scientific qualification and DEV/TEST access remain excluded. The execution record is EXPLORATORY_AUTHORIZATION.json.**

The qualified-study route remains blocked by missing independent evaluation split/access evidence, full compute qualification and the signed prefit attestation. Those are external evidence requirements. This actor cannot honestly issue its own independent signature or label actor-accessible source as a sealed independent TEST.

## Authorized exploratory FIT-only execution

`EXPLORATORY_AUTHORIZATION.json` now records the explicit owner grant. The workflow verifies that record and its exact source/protocol hashes before launching the matrix. Denial behavior is retained and tested with separate unit fixtures.

Concrete work proposed:

- 30 fresh FIT runs: original TAC, head-to-state TAC, memory-scaled TAC, original-geometry GRU and Transformer; 1M and 10M; seeds 10301/10302/10303.
- Each run uses the same authenticated original CPython FIT corpus and matching seeded order; 2,560 AdamW updates, batch four, 128-byte reset chunks and unchanged original hyperparameters.
- GitHub Actions CPU workers, two threads, at most three concurrent jobs. This is an explicit alternative compute proposal, not a silent fallback from the qualified dual-T4 plan. Historical CPU worker observations support a 35-minute estimate per fit; each job has a 45-minute platform ceiling and checkpoints before a 40-minute soft stop. A timeout is incomplete, never scored as a completed fit.
- All five full weight/AdamW/RNG milestones at 256/512/1024/2048/2560 are retained separately. Raw checkpoint payload estimate is 9,897,542,640 bytes (about 9.9 GB) for the campaign, plus small metadata. Proposed Actions retention is 90 days. Worst-case platform allocation is 1,350 runner-minutes; these are proposal ceilings, not consumed quota or price estimates.
- An interrupted checkpoint can be resumed only with a supplied matching SHA256, same source/geometry/authorization/protocol and complete loss/optimizer/RNG/order position. Outputs never overwrite originals.
- A read-only final cohort auditor checks all 30 actual fit receipts, every retained milestone digest, live parameter count, finite weights/moments and optimizer update counts, plus bounded trained-graph/reference equality. It fails on missing or incomplete fits.
- Original LANGCAP105 DEV and every TEST remain closed. No original checkpoint is resumed or replaced. The resulting FIT learning curves/weights are exploratory evidence only; they cannot qualify architecture superiority, independent generalization or closing the original DEV gap.

The pending authorization is a human work-authorization record, not an independent scientific attestation or authentication system. Source JSON cannot itself grant authority: the explicit owner instruction must exist before the authorized record is changed. The record pins the exact protocol and runner/model source bytes. The qualified preflight validator and its failures remain unchanged.

## Compute accounting result

`step_census.py` traces the real forward + byte-CE + backward + gradient-finiteness paths on constructed input, then executes first/steady AdamW mechanics on disposable copied tensors. Model weights and all original checkpoints remain unchanged. This executed 20 disposable tensor optimizer updates across ten configurations, **zero real-data model updates**. It is a unit mechanics trace, not language training.

The official Torch FLOP counter has no formula for ordinary CE/AdamW packets and reports zero for those stages. Every packet and stage is retained in `evidence/RECIPE_CENSUS_LOCAL.json.gz`. Formula-covered Transformer totals differ materially from the earlier profiler totals; neither counter alone establishes full hardware or full-recipe compute parity. The full-FLOP registration fields therefore remain unfilled. No negative evidence is replaced with an invented number.

## What is actually verified before approval

The default-deny guard, changed protocol/source rejection, missing authorization record rejection, and DEV boundary rejection pass unit tests without reading the corpus or constructing a model. The owner workflow's pending plan must pass CI with all FIT/audit jobs skipped. The full optimizer loop and live resume/audit path are prepared but have **not** been exercised on real FIT data; such execution requires the explicit exception or the independent qualified route.

This proposal does not remove, weaken or self-sign the scientific gate. [AGENTS.md](../../AGENTS.md) states “Missing receipts or failed gates block fitting.” The owner may explicitly grant an exploratory FIT-only exception while leaving scientific qualification blocked, as a separate decision from accepting a scientific claim.
