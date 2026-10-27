# LATFUSE012 fixed elapsed 512-step synthetic write-count sweep — DIAGNOSTIC ONLY

**2026-10-27. [Successful source-FREE GitHub Actions #40455383108](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40455383108), original frozen two 99,390-param LATFUSE011 synthetic-only checkpoint SHA verified. No model fits/optimizer, no real sensor source download, no real-world performance claim.**

The previous synthetic positive-control task reported event pair ranking100% at 16/64/128/512 steps for a simple fabricated ±3 channel0 bit remembered from original step23. However both elapsed lag and number of intervening synthetic source writes increased, so the task failed to isolate capacity and forgetting.

### Registered fixed-delay experiment

At **exactly 512 intervening steps** after the same original synthesized bit, use identical original generated source-FREE paired 64 event/control test cases, nuisance observations and source clock. Restore both learned synthetic checkpoints seed11801 and11802. All 512 intervening source observation updates still update original fast/slow/entity paths normally. The ONLY intervention is copying the original associative matrix from before to after each of 512−N selected intervening write steps, with eligible N in (0,8,32,128,512). Source write eligibility times are distributed uniformly over the suffix by integer schedule. This suppresses both current delta write **and** original alpha retention decay for those steps; it is an out-of-distribution model state counterfactual, **not** a pure theoretical isolated overwrite operator and NOT an architecturally trained sparse write agent. Code verifies nonassociative state is exactly identical for each count condition; full N=512 reproduces the original saved diagnostic numbers.

| Number of intervening associative updates, elapsed 512 fixed | Seed11801 synthetic Brier ↓ | Seed11802 synthetic Brier ↓ | Both paired rank |
|---:|---:|---:|---:|
| 0 | 0.05032 | 0.04332 | 100% |
| 8 | 0.05607 | 0.04798 | 100% |
| 32 | 0.06103 | 0.05267 | 100% |
| 128 | 0.07049 | 0.07031 | 100% |
| 512 | 0.09342 | 0.11824 | 100% |

Both learned classifiers were trained under the **full N=512 observed suffix distribution**, so removing writes is an OOD intervention but reduces confidence/error on this deliberately simple past-bit task. At N=512, original true binary balanced classification: 92.2% (seed11801), 87.5% (seed11802); with N=0, both 100%, source-free Brier below0.051. This is evidence that intervening ASSOCIATIVE UPDATE COUNT (write and/or decay) changes this already-learned synthetic model's delayed-bit probabilities while time and nonassociative processing are held constant. It does **not** distinguish beta overwriting from alpha forgetting; requires separate beta-only/alpha-only mathematical transport control (LATFUSE005 had exactly that on genuine source). It does NOT prove a general capacity ceiling or a real physical long-range memory skill.

**Scientific boundary:** no original UCI501 data, no source rights/protected-row issue; synthetic-only task from deterministic fabricated channels. All archived weights/optimizer RNG and original SHA are retained; all real original learned checkpoints untouched and no neural optimizer run in this follow-up. A successful synthetic check cannot replace authentic source event fact-recall and +6h physical forecasting controls, nor can chance-nice synthetic scores qualify the device/enterprise model. No full training FLOP/resource parity inference. No original protected source opened.

**Conclusion:** a real model with fixed-size memory can learn this simple synthetic delayed event, and increasing intervening matrix updates at fixed time degrades its probability metric despite rank remaining100%. This supports exploring **source relevance-based admission to reduce unnecessary writes only AFTER legal actual historic-content query reconstruction is established**, not prematurely increasing memory capacity or claiming every long-range event can be recalled. No architecture modification or model FIT in this PR.
