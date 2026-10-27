# LATFUSE006 — genuine historical-event decodability in frozen TAC memory

**2026-10-27. Original physical-source sidecar diagnostic PASS. Native TAC-OSM event recall NOT established.** [GitHub Actions original-source run #40442709108](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40442709108) SUCCESS. The prospective original rules are in REGISTERED_REAL_EVENT_DECODER.json and full eight-model/four-delay/four-decoder outcomes are preserved in RESULTS.json. All original latent-fusion parent checkpoints stay unchanged and unmerged.

## Source and evaluation design

Original UCI501 Beijing multi-site air-quality source, 12 real physical stations, archive SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. Actual past PM2.5 sensor observation high (train-FIT z>=+1.25) versus source-clock-matched ordinary (abs train-FIT z<=0.25) at identical source station and hour-of-day; source-observed, not synthetic. Train an EXTERNAL ridge sidecar on 96 event/control pairs from original 2014-Mar–Apr18 and evaluate frozen sidecar on the previously analyzed 24 event/control pairs of original July2014 FIT. Source control pairs/labels are selected independently of model outcomes; horizon L=16,64,128,512 source-real hours. Each original fitted LATFUSE004 seed 40401/40402 × four 4/2-head soft/hard arm checkpoint SHA verified before and after. No original TAC optimizer, no pretrained-model new parameters, no original code/weights/state mutation. No 2015 numeric DEV, protected2016–2017 original data, future weather or unobserved source target enters decoder queries.

**Important limits:** The original frozen models had already been fitted on FIT data including July2014, so July is temporally held out ONLY from the external decoder, NOT independent model evaluation. The external ridge probes train additional parameters on real source labels; they are NOT the pretrained model's native factual retrieval behavior and are not deployable within the original 100k parameter/fixed-state limit without integration and fresh budgeting.

## Four honest feature controls

All include same current physical sensor readings/masks/age/last, source station ID, and calendar at query time t+L (the actual historic event value/label and original write-key are never supplied).

1. Local nuisance only (current source readings, masks, source clock).
2. Local plus original model fast80, slow80 and authorized entity source-slot80 — no associative memory.
3. Above plus the model's **actual calendar/belief-conditioned 16D associative forecast readout**, not an oracle historical-event-specific key.
4. Above plus the **whole frozen 256-value associative matrix**, an external linear decodability upper bound, not the original model's usual query, and not a resource-matched deployable readout.

External ridge fixed lambda1, fit on source genuine March–April paired labels only, per-feature train-only centering/scaling, feature dimensional norm scaling and no July score-driven tuning. Report binary balanced accuracy, Brier score, and same-source same-hour true-event versus ordinary paired-probability ranking. Chance event ranking=0.5; constant 50% prior Brier=0.25.

## Original July 2014 genuine source-event results averaged over all eight frozen models

| Delay | Source feature | Matched event rank (higher) | Brier (lower) |
|---:|---|---:|---:|
| 16h | current nuisance | 29.17% | 0.3744 |
| 16h | nuisance+nonassociative fast/slow/entity | **96.35%** | **0.1893** |
| 16h | same+actual learned associative read | 94.79% | 0.1978 |
| 16h | same+whole associative matrix (external) | 88.54% | 0.1844 |
| 64h | current nuisance | 16.67% | 0.3993 |
| 64h | nuisance+nonassoc | 35.42% | 0.4300 |
| 64h | same+actual learned read | 40.89% | 0.4594 |
| 64h | same+whole matrix (external) | **73.70%** | **0.2782** |
| 128h | current nuisance | 25.00% | 0.4000 |
| 128h | nuisance+nonassoc | 53.12% | 0.4488 |
| 128h | same+actual learned read | 59.38% | 0.4123 |
| 128h | same+whole matrix (external) | 63.02% | 0.4118 |
| 512h | current nuisance | 31.25% | 0.4559 |
| 512h | nuisance+nonassoc | 26.30% | 0.4867 |
| 512h | same+actual learned read | 32.81% | 0.4920 |
| 512h | same+whole matrix (external) | 29.69% | 0.4673 |

**Observations:** Original fast/slow/entity state supports a linearly learnable 16h historic PM2.5 distinction. The original learned associative read makes Brier slightly worse at16h. At64h, the EXTERNAL full-matrix decoder distinguishes authentic paired events much better than the trained model's actual associative read (73.7% versus40.9% rank); this is a **potential retrieval/readout bottleneck**, not proven historic factual retrieval or reliable probability: even the whole-matrix Brier0.2782 loses to a constant neutral0.25. At128h the actual read marginally helps pair ranking over nonassoc, but all decoder probabilities are badly calibrated; at512h all models/readout methods perform near/below chance. Two seeds, 24 matched evaluation pairs and FIT overlap severely restrict generalization claims. No architecture superiority demonstrated.

## Resulting next architectural research question

A historical query with only legally known **requested lag/time and station** could train model memory to answer real recorded t−L high-event labels or regression targets. Current pretrained forecast query is future-known calendar plus present belief; it is **not trained for old event retrieval**. To test learning this correctly, introduce a source-target-supervised (historical time-conditioned) readout in a new model while keeping the original 256 associative matrix cells and 1696 FP32+16 bool=6800 per-stream storage. Readout first; relevance-conditioned beta overwrite second as a separate matched architectural variable. Under compulsory user rules, any trained architecture comparison needs <1% live parameter and **full forward+AD+optimizer FLOP parity**, same original recorded source & station/hour sampler order, independently verified true target eligibility, seeds>=2, original checkpoint hash+clean replay, no original protected numeric data access, genuine forecast proper scoring and memoryless/source-only trained control. The current external sidecar is intentionally NOT counted as a successful TAC model. No new neural fit is authorized under the finished LATFUSE004 registration.

**Status:** real external sidecar event-readout diagnostic complete; moderate information in whole stored matrix at64h, learned current reader fails to surface it consistently. Historic factual source-aligned model retrieval, relevance-gated retention and calibrated forecast predictions remain open hypotheses.
