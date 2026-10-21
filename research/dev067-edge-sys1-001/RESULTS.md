# DEV067-EDGE-SYS1-001 — frozen trained checkpoint vs proposed Edge Bidding Brain

**Research-only software & inference feasibility gate, October 21, 2026. No trained ad-bidding result.** The DEV067 original frozen controller is a learned maintenance-action scorer, not a pretrained ad-bidding policy.

## Original model provenance
- Original SHA-verified GitHub Actions checkpoint package: [DEV067 original run 39271150621](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39271150621), artifact **12194964127** `dev067-restored-checkpoints-0`. Demo inputs, expected logits and 3 original model paths are bundled.
- TAC trained `tac_frozen_log`: **23,446 params**, checkpoint SHA256 `10ccaf9892982b0340c1d1e0f71137d08ce6f4d1962e87e346f725166d6bd302`; 4/4 original recorded requests replayed scores and choices exactly within 1e-8. Original demonstration: **synthetic maintenance cases**, not ad auctions.
- Trained `gru_frozen_log`: 23,287 params, 3/3 original demo requests reproduced.
- Trained `no_casm_frozen_log`: 23,478 params, 5/5 original demo requests reproduced.
- **No training or checkpoint edits were performed in this test.**

## CPU inference timing (measured in local model environment)

One PyTorch CPU thread, `torch 2.10.0+cpu`, NumPy 2.3.5, 160 warm repetitions per point; timings are model forward only, with already assembled public event/choice tensors, no API/network/JSON/store/auction execution. To exercise the full 64-record compute path, populate its public FIFO with the **first 64 original archived DEV067 demo events**, marking all present. This is a *software timing fixture*, not the result of trained admission or a long-lived campaign memory.

| Trained model | 64-event p50 ms | p95 ms | p99 ms | <5ms calls | 64-event p99 under 100ms |
|---|---:|---:|---:|---:|---|
| DEV067 TAC | **4.7764** | **5.8394** | **7.2753** | **115/160** | YES local |
| Matched trained GRU | **3.5395** | **4.1303** | **5.2590** | **157/160** | YES local |
| Matched no-CASM | **4.8259** | **6.1033** | **7.7510** | **102/160** | YES local |

### Stricter follow-up: identical recorded inputs across all three models

After identifying that the first sweep used each export's own original saved demonstration tensors, I ran a new 160-call CPU sweep for each model with **exactly the same TAC original demo keys, 64 typed event vectors, five candidates and price vector**. This is the preferred input-matched control; the original table above is preserved as historical first measurements.

| Trained model | 64-event p50 ms | p95 ms | p99 ms | <5ms calls |
|---|---:|---:|---:|---:|
| DEV067 TAC | **5.164** | **7.822** | **11.199** | **63/160** |
| Trained GRU | **3.374** | **3.919** | **4.406** | **159/160** |
| Trained no-CASM | **4.450** | **5.838** | **7.427** | **141/160** |

All 480 matched-input calls still finish under 100ms locally; this is not a live exchange deadline. The two sweeps' timing variation demonstrates p99 instability of the software microbenchmark and reinforces that sub-5ms claims are not production-qualified. **A narrow <5ms p99 service guarantee fails for TAC on the matched-input sweep; the trained GRU is faster here.** GitHub CI runs the input-matched audit separately on its own runner; those timing values must not be confused with these local metrics.

This is a single software environment and a small microbenchmark, not a p99 service SLO under real traffic. The original model **fails the strict 5ms p95/p99 objective**, and the simpler trained GRU is faster. Full local original-adapter ingest+decide medians are 4.737ms TAC, 3.416ms GRU, 4.741ms no-CASM; these also exclude feature assembly, network, database, concurrency, ad rules, authorization, and exchange response deadlines. No claim of microseconds or throughput under multi-campaign load.

## Direct software gates

1. **Frozen learned policy execution:** PASS in its original maintenance domain (4/4 original TAC demo replay). Outputs are logits for **five maintenance actions**, predicted repair outcomes/costs; **no bid-price, throttle or pass semantics** are trained or produced. Original demos remain synthetic.
2. **Last-action-input sensitivity:** MODIFY the declared three action-type coordinates of one *recorded* historical event, without changing learned weights. TAC logits change by max **0.523762**; trained GRU by **2.253374**; no-CASM by **2.034903**. This proves dependence on an action-related input, **not correct counterfactual auction responses** or last-bid-conditioning calibration.
3. **Campaign isolation:** a Python application dictionary can maintain independent model FIFO states and restore cloned states without cross-key changes. SOFTWARE PASS; the trained model has **two entity slots for one maintenance episode**, not native cross-campaign routing or a compressed persistent bid-state.
4. **Hard safety envelope:** 2,000 deterministic seeded **software fixture** trials passed bounded bid, budget-left, frequency stop and shutdown invariants. The hard caps are **external** and only as correct as the upstream spend/frequency counters; neither guaranteed minimum *future* ROAS nor shared-across-nodes atomic budget protection is implemented. These fixtures are **synthetic/unit tests only**, never real-world proof.
5. **Fast and slow persistent state:** **NOT SATISFIED** by DEV067. Its local encoder is a GRU with two entity-relative passes over a 64-event typed FIFO. It is not the 32D/32D native fast/slow dynamical state of ME035–ME038 and does not persist its GRU hidden state between separate decisions.
6. **Memory footprint:** native FIFO alone is `64*46*8 = 23,552 bytes` per original float64 episode, excluding 2x16 float64 keys and masks. Do not claim only 256 bytes of campaign storage based on newer 64D latent-state designs; this exact exported controller retains records.
7. **Zero hot-path DB reads, TTL, heartbeat, restart, async trainer:** external runtime and infrastructure responsibilities. The local harness performs no DB lookup, but does not test production memory TTL, process restarts, concurrency, or background training.
8. **Offline auction economic results:** **NOT TESTED.** DEV067 accepts 46-dimensional typed maintenance events, 2×16 public identity keys, 5×19 candidate-maintenance actions and 2 inspection/price features. Real auction features, bid, pacing/throttle, price-market feedback, click/conversion delay and causal/action-support propensities require a new trained adapter and new real logged-data evaluation.

## Real RTB data path and promotion requirements

Official iPinYou RTB auction dataset (https://contest.ipinyou.com/) contains advertiser bid, impression, click/conversion and paying price information, but is **non-commercial** and historical 2013 data. The recorded high-bid logging scheme does not by itself certify counterfactual action overlap, complete bidder competition, individual win/loss and randomized propensities, so a naive replay cannot validate ROAS lift or a 10–30% shading savings claim. Start with authenticated data provenance, chronological factual bid-win/click baselines and strict observed-action evaluation. Select original weights once before validation; score no policy advantage from unobserved counterfactual auctions.

**Next engineering requirement:** build a declared campaign-keyed native 32D/32D memory + auction event encoder + *trained* bid-price/throttle/pass head; include executed action and observed feedback as explicit inputs; keep external atomic budgets, frequency limits and kill-switch fallback. Train using actual logged auctions, match a GRU on data/splits/model size/training compute and evaluate on an untouched temporal test. Do not do risky live bids, autonomous financial spending or invoke untested checkpoint as a live bidder.

## Reproduce
[CI workflow](../../.github/workflows/dev067-edge-bidding-feasibility.yml) pulls original restored DEV067 artifacts and runs [frozen checkpoint functional and latency test](../../scripts/test_dev067_edge_bidding_feasibility.py) separately for TAC, GRU, no-CASM. Original source is not modified; archived model/checkpoint manifest check is enforced. The alternative full local research package includes original exported weights/source, individual 160-repeat latency metrics and exact software tests.

**Final verdict: PARTIAL SOFTWARE INTEGRATION FEASIBILITY, NOT A FUNCTIONING LEARNED BIDDER.** No model promotion, architecture superiority claim, confirmed live 100ms SLA, savings, ad-model transfer success, or merge.


## Independent GitHub Actions run: exact original weights, identical input, green

[Original-source frozen replay/latency workflow 39619467446](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39619467446) **PASS**; authentic original checkpoint/demos downloaded from DEV067 archive 12194964127; no training or checkpoint modifications. Result [artifact 12354685370](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39619467446/artifacts/12354685370) contains all three machine-readable measurements. GitHub Actions server differs from the local environment, so platform-specific latency is reported separately.

| Original trained model | Frozen model p50 ms | p95 ms | p99 ms | <5ms calls, 64-event context |
|---|---:|---:|---:|---:|
| TAC | **6.4423** | **6.6011** | **6.7171** | **0/160** |
| GRU | **5.0511** | **5.2195** | **5.4564** | **1/160** |
| no-CASM | **6.3245** | **6.4500** | **6.5852** | **0/160** |

All 480 GH samples complete below 100ms, excluding exchange network, traffic contention, upstream accounting, feature assembly and production side effects. Original recorded output replay PASS TAC4/4 GRU3/3 noCASM5/5. Action-field sensitivity and deterministic software envelope tests PASS. **No 5ms p99 SLO achieved on this runner.** Relative latency GRU is faster than both controller arms on matching inputs.

The original local first pass, stricter local rerun, and independently observed GitHub times differ; **never infer deterministic p99 or production latency** from these samples. An earlier intermediate attempted CI run containing a script scope error was corrected without refitting or changing test data; successful final run and retained failed attempt are traceable on the branch. The source fixture is synthetic maintenance demo data, not real ad-auction input. The only real-world outcome test remains NOT RUN and no bid policy is qualified.
