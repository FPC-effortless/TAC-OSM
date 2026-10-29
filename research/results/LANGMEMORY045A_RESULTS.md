# LANGMEMORY045A — source-grounded cheap complementary memories and selective verification

**Actually measured original publisher FIT evidence, 29 October 2026.** Draft [#575](https://github.com/FPC-effortless/TAC-OSM/pull/575), [successful original-source GitHub CPU workflow #40821789839](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40821789839), [original original-source evidence artifact #12874743374](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40821789839/artifacts/12874743374). Workflow preflight and 5 constructed causal/determinism tests passed. **No neural TAC training or NLL results, no checkpoint, GPU, protected TEST, or model promotion.**

## Exact tested causal task

The original frozen publisher CPython FIT 14 document owners, first at most 1536 original byte events each. For all 4928 eligible t positions (t≥128, t mod4=0), 3 previous bytes as real query, strictly earlier observed source continuation values as candidates, and original true byte x[t] as label only AFTER prediction. No future contamination in registered interface. *Actual source held out here is only the current outcome byte, not a new unseen document or independent DEV/TEST.* Long-history subset means 2255 positions whose exact 3-byte context occurred outside recent64, not guaranteed semantic task-relevant memory.

- A: exact matching context occurrences from last64 observed source bytes. Accurate but often abstains.
- B: 256 FNV hash buckets with majority-vote candidate (Boyer-Moore), approximate and collision susceptible.
- Cheap fusion: A if present else B.
- Budgeted V: at no more than every fifth measured query and when A/B absent/disagree, scan *all prior same-owner source positions* for exact historical context and choose majority continuation; if no match, fall back cheap.
- Expensive V whenever available: upper-cost reference, not an oracle next-label lookup.

## Actual original source results

| Arm | Covered positions / 4928 | Correct / 4928 | Accuracy among answered | Long-history coverage / 2255 | Long-case accuracy among answered |
|---|---:|---:|---:|---:|---:|
| Recent A | 668 (13.56%) | 408 (8.28%) | 61.08% | 19.78% | 60.09% |
| Sketch B | 2910 (59.05%) | 842 (17.09%) | 28.93% | 66.52% | 45.40% |
| Cheap A+B | 3078 (62.46%) | 933 (18.93%) | 30.31% | 71.35% | 45.37% |
| Selective A+B+V | 3208 (65.10%) | **1050 (21.31%)** | 32.73% | 77.12% | 48.71% |
| Exact historical V whenever available | 3724 (75.57%) | 1528 (31.01%) | 41.03% | 100% | 58.76% |

The selectively gated verifier improves **+117 correct out of 4928 (+2.37 absolute percentage points)** and +130 answered positions over cheap A+B. It triggers on 17.9586% (885) of all sampled queries; scans **737845** source positions, about 834 scanned source positions per actual V invocation. The gain cannot be treated as free. A is strong conditioned on coverage but abstains 86.4%; B broadens coverage with many collisions/incorrect continuations. This exposes meaningfully different failure modes and an engineering signal for candidate verification.

## Critical limitations and next falsifiers

- Conditional correctness is NOT equal-coverage accuracy. Compare success over *all positions* separately, then precision–coverage/latency Pareto.
- The exact V requires retaining original historical bytes beyond the nominal cheap A+B packed payload of 1344 bytes (64B ring + 256×5B sketch); Python object overhead and source storage excluded. Any learned TAC integration must include verifier lookup-index storage, cache implementation, p95 latency and energy.
- Current deterministic gate has no trained calibration, and 17.96% invocation is not contrasted with **matched-budget random/informed verification**, nor a same-cost historical hash-index comparator. Those must precede claims of selective efficiency.
- Exact 3-byte continuation majority is not semantic history understanding, nor automatically useful to the TAC logit head.
- Original FIT-only source and original documents are previously exposed for protocol development. No untouched publisher evaluation or per-owner distributional robustness assessed.
- No trained neural TAC weights, NLL, Brier, next-byte model score, held-out independent generalization, or architectural superiority established.
- Prior LANGLAYER041/042 distant-history failures remain unchanged. User's more expensive agreement model is a **research hypothesis**, not already built into the production TAC architecture.

## Decision

**Proceed to a preregistered memory selection falsifier**: equal 17.96% verification-query budget random/uncertainty/always-on-truncated controls, source history identity destruction, candidate false-positive audit, exact-history hash lookup with counted storage/latency, and owner-disjoint untouched documents. If these deliver a real quality-cost benefit, write a source-pinned PNDS Lean+Python neural memory-verifier graph and do five-seed compute/parameter-matched TAC model fitting; otherwise preserve the failure. Do not promote new checkpoints or claim improved intelligence.

The first two experimental attempts failed safely at the phone router kind and missing publisher license evidence; both were corrected without modifying model math or evaluation labels. Original failures remain visible in GitHub Actions history.
