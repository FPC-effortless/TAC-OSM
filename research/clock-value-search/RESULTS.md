# ME085–ME086 frozen clock/value capability results

Two separately preregistered complete-model recipes, five cold starts each; all original weights/optimizer/RNG and data-order receipts were Git-frozen before REPORT. Exposed40episode development population; no protected source was opened. Both fixed predictive gatesFAIL.

| Study | H1 MAE | Moving H1 | Quiet H1 | H4 MAE | H16 MAE | Full predictive gate |
|---|---:|---:|---:|---:|---:|---|
| ME085 | 0.05338952 | 0.38013820 | 0.02205745 | 0.15759004 | 0.68190478 | FAIL |
| ME086 | 0.05168648 | 0.37201422 | 0.02097012 | 0.15890228 | 0.68835176 | FAIL |

The second recipe clears all original numerical forecast guardrails except quiet precision and the paired episode CI; it also fails the stronger additional99%CI. Neither changing the density-loss weight nor successful strict replay makes this a full predictive pass. The source, old gates and all original arrays remain fixed.

ME087 is a new, separately prospective integrated model after the ME085 failure: all-coordinate FIT-noise-bounded quiet retrieval plus correct implicit-CDF gradient of the actual scored median. ME086 continued unchanged. All six declared attempts are retained in its ledger; no hidden restarts or test-cohort tuning. The new study is still training and has no score.

Original whole-forward float64 reference from float32 learned values retains rtol1e-5/atol1e-6. Optional compiled B2profiles are separate execution engineering and do not qualify every shape or create new learned parameters. No architecture-winner, causal action, independently held-out intelligence or customer-value claim is made.
