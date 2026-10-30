# TAOPILOT003 — independent original bitwise replay discrepancy (2026-10-30)

**First independent scoring attempt: FAIL (no DEV score accepted).**

[GitHub Actions #41011303831](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41011303831) independently checked out the five genuine Git-backed original checkpoints, restored models and optimizers, then failed the strict `torch.equal` logits gate before scoring any DEV. No original model/optimizer checkpoint was overwritten.

A new **no-DEV diagnostic** [GitHub Actions #41012324560](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41012324560) committed `research/evidence/taopilot003/seed1701/INDEPENDENT_REPLAY_DIAG.json` without modifying original bytes. It found exact original trained weight hashes and optimizer step 32 **for all five**. Four arms (AdamW/Muon/AdaMuon/pre-NS) have bitwise identical 1×16×256 logits. **NorMuon is not bitwise**: 76/4,096 logits differ, max absolute 4.76837158203125e-7, mean absolute 4.329194780439138e-9, top-1 predictions unchanged, weights exact, optimizer step 32. `torch.allclose(atol=3e-5,rtol=1e-4)` was true for all arms using the earlier repository numerical replay convention; the diagnostic is a measured report, not an invented proof.

**Status:** `INDEPENDENT_BITWISE_REPLAY=FAIL`; `INDEPENDENT_EXACT_WEIGHTS=PASS`; `INDEPENDENT_OPTIMIZER_STATE=PASS`; `INDEPENDENT_3E-5/1E-4_NUMERICAL_LOGITS=PASS`; `DEV_SCORE=NOT_YET_ACCEPTED` for this first attempt. Never relabel the first run as bitwise success.

A prospective versioned **numerical replay eligibility** extension may permit descriptive exposed-DEV evaluation with exact source/weights/optimizer and the pre-existing 3e-5 absolute/1e-4 relative bound, but it must **preserve this failure** and honestly report non-bitwise origin, independent source-replay limits, and no optimizer superiority or checkpoint promotion.
