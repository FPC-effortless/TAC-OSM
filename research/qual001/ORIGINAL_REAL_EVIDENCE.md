# QUAL001 — original-source headroom circuit-breaker

## Status
**Completed new read-only audit of original ESWAM012 predictions.**
No new neural training, simulation, source labels, split changes, untouched TEST,
or architecture qualification. All historical checkpoints and raw predictions
are preserved unchanged in the stacked ESWAM012 branch.

## Frozen evidence
- Original source: SHA-pinned publisher INTERACTION traffic, **one**
  intersection `TC_BGR_Intersection_VA`. Original FIT recordings 000–002;
  **previously exposed** DEV recording 003.
- Original ESWAM012 native-256, native-8, native-unscoped, GRU-256 and
  causal-attention-256 forecasts, fixed seeds 86001/86002.
- Exactly **266 paired real actor-future positions per horizon per seed**,
  **37 actor tracks**; horizons +10 and +30 recorded frames.
- Five arms have exactly matched source recording/frame/actor/horizon keys,
  publisher future coordinates, current coordinates, physical prior and
  original actor identities; no future features are fed into new training.
- Independent paired actor-cluster bootstrap uses 5,000 draws and reproducible
  xorshift32, resampling whole original actor tracks (overlapping time points
  are not independent); same actors appear in two fit seeds, which must NOT
  be treated as 74 independent clusters.

## Original real-data findings (L2 error, meters; smaller better)

| Frozen original fit | H+10 | H+30 |
|---|---:|---:|
| Native 256 frames, 2-seed mean | 0.247357 | 1.862645 |
| Native 8 frames, 2-seed mean | **0.236878** | **1.797508** |
| GRU 256, 2-seed mean | **0.229117** | **1.739160** |
| Causal attention 256, 2-seed mean | 0.232278 | 1.770261 |

Paired native-256 **minus native-8** (positive = long-history worse):

| Seed and horizon | Source delta (m) | Original actor-cluster bootstrap 95% CI (m) |
|---|---:|---|
| 86001, H+10 | +0.009058 | [+0.002247, +0.016177] |
| 86002, H+10 | +0.011899 | [+0.005496, +0.019029] |
| 86001, H+30 | +0.050079 | [+0.022008, +0.081879] |
| 86002, H+30 | +0.080196 | [+0.050036, +0.112814] |

These values were recomputed independently from each original stored source-case
L2 error, using 5,000 paired original-track bootstrap resamples. The confidence intervals were independently recomputed from the
original rows using the Python runner's reproducible xorshift32 actor-cluster
bootstrap and verified to match the reported endpoints (shown rounded).
The two seeds repeat the same 37 actor tracks and are not independent sites.

Both low-speed (<0.2m/s) and moving (>=0.2m/s) source regimes also favor
native-8 over native-256 at both horizons in both seeds.

**Verdict:** original ESWAM012 does NOT establish useful long native memory.
Its 256-window candidate descriptively loses to its matched-parameter 8-window
ablation. Long-frame inputs alone are not evidence for persistent memory.
The earlier original ESWAM012 study lacked certified full training-FLOP
parity and guarded optimizer proof; no native-vs-GRU/Transformer superiority
test is qualified here. Original same-site DEV is not an untouched site.

## Reproduction

```bash
python scripts/qual001_eswam012_original_headroom.py --self-test
python scripts/qual001_eswam012_original_headroom.py --verify-frozen-means --output /tmp/qual001_observed.json
```

The executable fails closed on mismatched original query IDs, labels, source
physical priors, non-finite or wrongly scored original forecasts and altered
case/cluster counts. It uses no external data downloader or fit runner.

## Next scientific gates (NOT completed by this audit)

1. Find **genuinely untouched** publisher-original sites and freeze provenance
   before any exposure; prohibit adaptive repeated TEST access.
2. Train **both** 8- and 256-frame causal attention under matched conditions
   as a dataset-headroom diagnostic, alongside a source-history necessity test.
   Transformer failure is insufficient evidence that the task has no long signal.
3. Train prospective native/GRU/attention/gated-physical comparators with
   matched active 95k–105k parameters, complete observed data, *measured full
   training compute*, update exposure and original source splits.
4. Independently audit original checkpoints, precision contracts and
   source-cluster confidence intervals before opening exactly one untouched
   location. Evaluate stationary-vs-moving Pareto and memory advantage.
5. Keep nonauthoritative forecasts separate from action decision claims; this
   observational traffic source has no causal do(action) decision reward.

Do not call this evidence a new scientific win, and do not relax old gates.
