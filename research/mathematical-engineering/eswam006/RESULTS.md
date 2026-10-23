# ESWAM006 — original real 256-history traffic forecasting baseline results

**October 23, 2026. Authentic real future observations, no neural optimizer fits.** Factual source-qualified physical prior outcome: naive historical mean velocity over 256 actual frames is substantially WORSE than using the latest source measured velocity or the last eight real velocities. This is NOT neural memory intelligence proof or a native architecture-superiority result.

Original publisher INTERACTION TC traffic source original SHA256 9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec. Original camera scene TC_BGR_Intersection_VA original member vehicle_tracks_003.csv, which was predeclared as source DEVELOPMENT before future-position scoring, distinct from reserved FIT recording files000/001/002. All four original files are different camera recordings from ONE physical intersection site, not a cross-location unseen TEST. The earlier source-only audit looked at track lengths but no future-position prediction errors. Original source x/y in meters and vx/vy in meters per second; history exactly 256 consecutive native recorded frames with strictly increasing source time. All target future x/y positions originate from actually observed original publisher rows without padding/interpolation/generation.

## Original genuine future-position scores

At each eligible real actor/time origin, the four formulas receive the SAME actual historical 256-frame window and known previous timestamp gap. Forecast future x/y at 10 and30 original frames; no future timestamp, actual future velocity or oracle future intervention is supplied. Fixed source stride10 time steps; zero hyperparameter fitting on source outcome. The score is Euclidean physical position error in meters, lower is better.

| Factual fixed predictor | Future10-frame L2 error m | Future30-frame L2 error m |
| --- | ---: | ---: |
| Persistence (no change) | 1.065919 | 2.478515 |
| Last actual measured velocity | **0.339054** | **1.741285** |
| Mean of last 8 actually observed velocities | 0.482956 | 1.924956 |
| Mean of last 256 actually observed velocities | **1.071671 (worse)** | **2.627482 (worse)** |

Full original future10: **876** genuine query origins from **45** distinct source-local actor tracks. Full original future30: **788** genuine query origins from **41** source-local actor tracks. Total actual scored cases 1664 but clustered and overlapping within actors, NOT 1664 independently collected environments. The original source contains 937 actor IDs in DEV recording; only actual >=256-history and future-available tracks are eligible.

Paired historical 256-minus-8 fixed velocity-mean error differences, actor-track clustered 2000 resamples:
- Future 10 frames: **+0.588715 meters** worse, 95% actor-bootstrap interval **[+0.497559,+0.706493]**.
- Future 30 frames: **+0.702525 meters** worse, 95% interval **[+0.561504,+0.854950]**.
The paired intervals reflect one single physical source recording, not independent site generalization. Instantaneous source velocity is strongest of all four controls on both horizons.

## Research interpretation

The publisher raw source is genuinely suitable for natural long histories and multiagent observations (ESWAM005 verified 157 uninterrupted 256+ actor tracks and 29,921 real simultaneous >=4-agent timestamp samples across four original recordings). Yet **longer history alone is not equivalent to better intelligence**: a naive equal-weight accumulation of stale real velocities systematically harmed simple physical future prediction. This provides a falsifiable reason to test entity-aware selective retention, temporal relevance, time-aware retrieval and conflict-preserving evidence roles rather than simply expanding TAC-OSM memory or averaging all recorded past information. It does not prove that a learned long memory is better; the native ES-WAM core, GRU and Transformer have not been trained in ESWAM006.

This first physical prior test predicts one observed original actor's future while other actors are present in the original source. It does not score a learned interacting multiagent relationship representation, causal action choice, decision cost, verified intervention, safe deployment or task completion. Data have observed velocities, not real commanded robot actuators; no do(action) claim is valid. Source remains previously inspected DEVELOPMENT at one camera site.

## Compulsory next scientific test (not yet conducted)

Register real original recording FIT000/001/002 vs original scene DEV003 actor-disjoint exposure; decide an independent external intersection location for future true transfer testing. Construct model inputs from actual multi-agent co-timed actor histories with stable original source IDs and no invented missing records; use identical available real 8/128/256 observations and the same genuine future x/y labels. Then train exact parameter/equal exposure/equal optimization-compute native ES-WAM exchange-memory, GRU and causal attention controls, preserving best fixed source last-velocity prediction reference, identity conflict and temporal memory ablations and true covariance/correctness of forecasts. No synthetic distractors as real success claims. Do not silently assume four real actors were co-visible through all 256 frames: qualification must validate that separately.

Original source result, identities and per-case errors were committed to Git; no model checkpoint exists for this no-fit comparison. Preregistered protocol and unchanged physical formulas are preserved. Research draft/unmerged.

**Business/intelligence status:** genuinely observed long-history prior test COMPLETE, naive 256-step mean FAILS strong actual short-history predictor; learned TAC state/memory skill UNKNOWN, independent generalization/real policy/ROI NOT MEASURED.
