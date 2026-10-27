# PAR-MEM-012 — frozen historical read attribution on genuine source data

**Research verdict: original source-native trained associative read carries some useful
historical-prediction signal at 16/64 events, but correct historical event identity
is NOT reliably recovered; the next-hour forecast head barely uses this new
read. The 128-event correct read is worse than zeroing it.**

Status: COMPLETED frozen-inference diagnostic (no new training), original first
result Git frozen, all ten models and full physical source query/prediction arrays
reproduced from independent clean checkout.

## Original sources / complete replay assets

- [Registered protocol](PAR-MEM-012-PROTOCOL.json)
- [Complete per-seed authentic source numerical results](PAR-MEM-012-RESULTS.json)
- [All source/anchor/lag and interventions' original tensors](PAR-MEM-012-ORIGINAL-PREDICTIONS.pt)
- [Unmodified genuine source trained model originals](../replay-bundles/PAR-MEM-011/TRAIN_RECEIPT.json)
- [Original preregistered previous source clock plan](PAR-MEM-011-FROZEN-PLAN.json)
- [Original initial model predictions, committed before this test](../replay-bundles/PAR-MEM-011/DEV_RESULTS.json)
- [Original source-verified independent no-fit audit and clean-checkout replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40537090307)

Original source: true public publisher UCI PRSA2017 Beijing physical air sensor
records, 12 physical station identity streams; publisher original ZIP locked at
SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
Source adapter only reads original 2013/2014; 2015–17 numerical data
untouched. Experiment applies to **previously examined genuine May–June 2014
development**, never independent virgin TEST. Exact original 5 seeds × 2
trained arms from PAR-MEM-011, 98,326 original genuinely learned parameters
and 410 FP32 source-stream values per checkpoint, 129 observed events per
case; no training, optimizer calls, altered source cases or model weights in
this PAR-MEM-012 experiment.

## Original frozen intervention

For each identical physically observed station and original anchor, retain
the entire original 129-event temporal input, real observed PM2.5, source
owner and original hours, source-admitted writes, time-aware backbone,
two 16×8 persistent memory heads, all saved model weights and readout
decoder. At inference, independently recompute from the exact SAME frozen
pre-current-write matrix a 16D read using:
1. **Correct** source-observed past event's write/read ordinal.
2. **Adjacent wrong** next previously observed event, with original
   historical target intentionally unchanged.
3. **Farther wrong** observed event approximately halfway to current,
   with original target unchanged.
4. **Zero associative read** (replace the 16D read with zeros at BOTH
   original historical-content head and next-hour predictive delta).

This disables only *retrieval at inference*. It does **NOT** erase
the actual 410-value state throughout all previous writes, and is
NOT a retrained memoryless comparator. All original native predictions
from corresponding correct- or wrong-trained PAR-MEM-011 checkpoint had
to match source-saved original arrays before scoring. All ten original
checkpoint SHA256 identities and real source clock schedules were verified,
all 360 same-source DEV queries per checkpoint independently replayed.

## Real original historical-value results

Five-seed means from **correct-address-trained** frozen models, 120 original
observed source queries per lag/seed; normalized historical PM2.5 MAE,
smaller is better.

| Actually observed historical lag | Correct read | Adjacent wrong read | Farther wrong read | ZERO read | Current PM2.5 copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 events | **0.604188** | 0.604806 | 0.614987 | 0.615561 | 0.734462 |
| 64 events | 0.508405 | **0.506993** | 0.514784 | 0.512366 | 0.573954 |
| 128 events | 0.574039 | 0.572815 | **0.568114** | 0.571683 | 0.665352 |

A paired seeded original-read result:
- Correct read versus ZERO improves average old-value MAE by **1.8477%**
  at 16 events, source-seed wins **5/5**.
- Correct read versus ZERO improves average old-value MAE by **0.7731%**
  at 64 events, source-seed wins **5/5**.
- Correct read versus ZERO is **0.4121% WORSE** at 128 events,
  source-seed wins only **2/5**.
- Correct versus NEXT-WRONG is a far smaller distinction at 16 events
  (0.604188 vs 0.604806). Wrong-address is **better** at 64/128 and
  far-wrong unexpectedly yields the lowest MAE at 128. Thus **no
  qualified event-specific retrieval** or long-range memory benefit.
- The five same-seed checkpoints trained with an intentionally wrong
  adjacent-event historical address produced nearly the same effect
  patterns: zero-read MAE (16/64/128) 0.615558/0.512367/0.571683,
  versus their original wrong-event read
  0.604963/0.507507/0.573246. The training signal does not discriminate
  the exact correct event robustly.

## Read activity versus forecasting impact — key additional finding

Across the five correct-trained checkpoints:
- Mean original authentic source write/read **key cosine at adjacent
  observed event indices ≈ 0.8673** (for all three lags). Nearby source
  address vectors strongly resemble each other; this source-known
  Fourier address was designed for write/read identity, NOT guaranteed
  collision-free indexing.
- Correct historical 16D read has mean L2 norm 0.191 (lag16),
  0.203 (lag64), 0.181 (lag128). Mean read L2 separation from
  adjacent WRONG observed event is 0.0363/0.0486/0.0376.
- Source historical-content head predictions change by an average
  absolute normalized magnitude 0.0318/0.0333/0.0289 when the read is
  zeroed at 16/64/128 events. This is **meaningful model dependence on
  the read vector**, not zero gradients or a disconnected memory head.
- But the same frozen model's physically observed **next-hour forecast**
  predictions change by only approximately
  `2.0e-7`, `2.2e-7`, `1.9e-7` normalized PM2.5
  when the aligned read is removed. The updated predictive memory
  residual path is effectively negligible on these source cases.
  Forecast differences at this FP32 level should not be promoted as
  operational improvement or useful world prediction.

**What is and isn't established:** causal frozen-computation attribution
indicates the 16D associative read affects the trained historical-value
decoder and can improve errors in the first two lag bands *relative to
its own zero-read lesion*. However, correcting the event address yields
no reliable advantage over a neighboring or farther wrong event,
especially for lag 64 and 128. This does not prove TAC retrieves the
requested earlier fact. Neither zero-read lesion nor trained
wrong-event model is equivalent to an independently trained equal-budget
stateless/GRU/SSM control. The original observed source outcome can be
strongly predicted from atmospheric seasonality, site and other
measurements even with irrelevant keys.

## Decision: stop replacing keys in isolation

The master plan already had aligned key/value memory (DIAGKEY004),
owner-aware source admission (ME100) and soft-routed explicit source
content memory (LATFUSE004/018). The reintroduced mechanism is
present and active, but correct historical event-specific recovery
remains unresolved. Next capability experiment should therefore
target content discrimination **and incremental predictive value**,
not merely narrower addressing collision rate or greater key entropy.

Do not turn these repeatedly examined 2014 development anchors
into a benchmark to optimize endlessly. Select an actual independent,
legally obtained unexposed real longitudinal data cohort and preregister
all source history/identity/available-time eligibility before touching
its outcomes. Then train original five-seed full matched budget learned
stateless, read-blind, temporal fast/slow, GRU/SSM controls (95–105k
active params and ≤1% pairwise gap), with whole train compute parity
including complete forward/backward/optimizer; CPU wall time not forced.
Preserve exact model, source order, optimizer/RNG, source-owned state,
immutable frozen checkpoints and independent replay. As of this
experiment, **none** of those strong new trained comparator
qualification gates is silently claimed passed.

No old branch merged, no original source model or checkpoint rewritten,
no held-out 2015–2017 physical data or physical actuation, no seventh
TAC family and no complete PNDS FP32/AD/probability formal proof.
