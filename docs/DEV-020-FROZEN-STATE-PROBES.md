# DEV-020 — Where does native TAC lose critical future-state distinctions?

**Status: fixed post-hoc diagnostic; execution and audit completed.**
See [audited results](DEV-020-FROZEN-STATE-PROBES-RESULTS.md). The recipe and
interpretation rules below describe the pre-results protocol; TAC diagnostic
recovery gates did not pass.
Stacked on draft DEV-019 (#175). Do not merge as a successful world-model capability. DEV-017 and DEV-018 original checkpoints, raw rows, endpoints and prior failed gates remain immutable.

## Motivation / frozen prior evidence

- [DEV-019](https://github.com/FPC-effortless/TAC-OSM/pull/175) identified 0/768 opposite-future H512 paired-class disagreements in the DEV-017 TAC variants, without identifying storage versus readout failure.
- [DEV-018 original complete run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971), source `6b319dee36291021bc486c51cd40136af4b08d4e`, trained/audited 60 checkpoints. Its primary results: schema41 96.88% (short) / 90.28% (long), tac_joint 50% / 50%, GRU/delta/memoryless/small full-history transformer 50% / 50%.
- **New question:** Does paired causal information survive in the frozen TAC recurrent state and query-conditioned read features when native prediction fails, or do the pairs become indistinguishable earlier?

## Strict boundary / experimental units

**Evaluation:** exact original DEV-018 H512 pre-sensing balanced pending/reissued-cancellation pairs, at/after due, delays 32/128/512, four repetitions per cell = 96 episodes / 48 opposite-future pairs per seed. Frozen seeds 43, 59, 71; short and long checkpoint regimes. No history from the final evaluation enters diagnostic probe fitting.

**Probe training:** freshly generated disjoint public histories `dev020-probe-train`, 12 repetitions per cell = 288 episodes / 144 balanced pairs per seed. Separate deterministic domain namespace and different seeds. The original trained model weights are reloaded from the exact DEV-018 `DEV-018-RAW-{regime}-{seed}-NOT-CONFIRMATORY` artifact and SHA-256 checked against the audited report. No original module weight is updated. No generation, future bit or oracle target enters the original `encode` or `decode` runtime. Held-out world outcomes are used only as evaluation truth and (on the training split only) to train diagnostic probes.

**Arms:** memoryless, dense GRU768, schema41, tac_joint, tac_encoder. All use original existing DEV-018 model objects, checkpoints and exact target wait-future query. This is neither a complete CASM inference/verification comparison nor the proof of all TAC writer variants.

## Three measurement levels

1. **Stored recurrent state:** exact 12 current features / 41 structured scalars / 768 dense scalars. Compute mean L2 distances between opposite-outcome paired histories at prefixes 32, 128, 512 from the *same original H512 histories*, never regenerated new H32 episodes.
2. **Read interface:** intercept the exact input given to the inherited final head (32 memory-read features + 12 encoded public query features). Compute paired distance. Interception is measurement-only, not a replacement reader.
3. **Original logits and decisions:** log mean absolute paired positive-probability gap, percent opposite-pair class disagreements and exact held-out accuracy. On H512, every original output must match the frozen DEV-018 archived probability to 2e-5, and the primary accuracy must match the archived summary exactly, or invalidate the study.

The prefixes use historical public observations only. The future query tick/horizon is adjusted to keep the **same absolute original forecast endpoint**. Earlier prefix diagnostic scores are not DEV-018's registered primary and cannot replace it.

## Independent supervised probes (diagnostic only)

Fit *separate* predetermined single-layer binary logistic probes to:
- target-entity-conditioned recurrent state plus original public query; and
- exact target wait-query read-head input.

Structured state uses only the public indexed entity slot (five dimensions), global scalar and public query. For dense states, use the 768-scalar state gated by each one-hot public target entity, plus public query. **These probe parameter budgets are not architecture-matched**; they are within-model accessibility diagnostics, not fair cross-model comparisons. Deterministically train 160 AdamW updates, learning rate .015, weight decay .05, standardized by *training split statistics only*. Evaluate frozen readouts on the original held-out DEV-018 paired histories. Pre-specified shuffled training-label probes act as a negative control. The memoryless arm must remain unable to distinguish the two members when their last public observations and queried entity are identical.

Only one test evaluation occurs; no hyperparameter sweep or checkpoint selection from held-out results. A probe trained with label supervision does **not** demonstrate that the native architecture has learned to use that information for decisions.

## Decision interpretation (pre-results)

- Original predictor at chance, but held-out state probe **>=75%**, >=10 pp above shuffled control, **and** a failed original reader: **positive evidence of accessible retained past under the specified supervised probe**, indicating reader/interface/optimization limitation.
- State probe succeeds, read probe fails: prioritize **query/read-key alignment** (not proof of no readout representation under other capacities).
- Both state and read probes succeed, original output fails: prioritize final head / objective / optimization alignment.
- Trained probes fail or shuffle is similar: **unresolved**, not proof that the memory erased all information. Increase diverse histories, probe capacity and controls before causal attribution.
- If prefix pairs show large latent differences but no probe succeeds, those distances can encode irrelevant observations; do not equate L2 separation with task information.
- For native TAC, no memory-attribution success, reliable world-state capability, commercial readiness or efficiency claim follows automatically. Original DEV-018 overall TAC success stays false.

## Integrity, replication and publication

Frozen artifact [original run 38802569971](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971) (audited archive SHA256 `e297a432e1b2eec708731adfb5547cf3908bbbfb910ad443df225b5624001e31`). Raw checkpoint hashes must agree with the per-seed `report.json`. Reconstruct exactly the registered 96 DEV-018 test rows, checking their public fingerprints, target labels, original probabilities and summary accuracy. Publish JSON outcomes by arm, seed, regime and prefix as workflow artifacts, then commit a separate read-only analysis after checking run results.

Failure, skipped training, missing artifacts, malformed checkpoints, mismatched frozen predictions, non-balanced pairs or train/test fingerprint overlap must not be relabeled as scientific success.

### Execution
```bash
python scripts/run_dev020_state_probes.py --seed 43 --regime short --smoke --output artifacts/DEV-020-smoke
python scripts/run_dev020_state_probes.py --seed 43 --regime short --frozen path/to/DEV-018-RAW-short-43 --output artifacts/DEV-020-short-43
```

The primary evidence remains DEV-018; DEV-020 is a diagnosis of its failure modes.
