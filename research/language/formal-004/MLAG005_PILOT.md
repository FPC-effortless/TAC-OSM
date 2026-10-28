# MLAG005 — matched context320, one-seed feasibility pilot; NOT the five-seed research result

Date 2026-10-28. Exact preregistration: `MLAG005_PREREG.md` committed **before this FIT**. **Scientific status:** incomplete, exploratory human-authored Vim9.1 original FIT/previously exposed DEV, only seed1701 SiLU pair finished. No independent TEST, no new architecture superiority or commercial claim. Never promote this on one seed.

## Source/data and objective

- Same original byte-hash-verified 30 Vim9.1 FIT + six already examined DEV chapters; all 36 chapter SHA256 checked, no content redistributed. Original model source SHA256 `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`.
- Exactly same pretrained-free source architecture, 100,680 trained parameters; 408-float streaming state. Each arm initialized with the same Torch seed 1701 and identical **sample/target SHA256**, batch8, 320-character contexts, 120 AdamW .002 updates, weight decay .01 and global norm clip1. Historical coefficient exactly 0.30 across both; neither arm has additional learned heads, and both use the same source-ordinal aligned read.
- Fresh control A: `L_char + .15 L_hist64 + .15 L_hist96`; fresh candidate B: `L_char + .10 L_hist64 + .10 L_hist128 + .10 L_hist256`. Historical reconstruction at lag96 is NOT directly supervised in B. Extra source reads mean training computation is not FLOP parity, despite matched update/data/parameter budgets.
- Four predetermined 320-character DEV windows per each of six source documents, no data contamination claimed beyond repeated use of DEV. Correct-history, zero-source, FIT smoothed unigram and +1/+8 wrong-key are evaluated, with source targets intersected appropriately for each wrong-key lag.

## Single seed1701 real DEV numerical results (nats/char; lower better)

| Metric | Fresh dual320 control | Fresh tri320 candidate |
|---|---:|---:|
| Next-character NLL | **2.596486** | 2.599011 |
| Historical lag64 NLL | **3.527422** | 3.568582 |
| Historical lag96 NLL | **3.578404** | 6.590328 |
| Historical lag128 NLL (control unsupervised) | 3.657897 | 3.660877 |
| Historical lag256 NLL (control unsupervised) | 6.582212 | **3.702762** |
| FIT-only unigram lag256 NLL | 3.639902 | **3.639902** |
| Historical memory removed lag256 NLL | 4.852030 | 4.852030 |
| Lag256 exact-character accuracy | 1.237% | 14.518% |
| Lag256 wrong-query +1 NLL | 6.940277 | 3.742552 |
| Lag256 wrong-query +8 NLL | 3.908438 | 3.772112 |

Interpretation: explicit 256-character supervision produced a large **within-seed** improvement over a newly trained longer-context control, but **did not yet outperform FIT-only unigram** (3.703 vs 3.640), made next-character NLL slightly worse (2.599 vs 2.596), and wrecked untrained lag96 predictions (6.590). A wrong-key +1 penalty of +0.040 nat is marginal on a smaller valid-target intersection; stronger source identity evidence remains open. These are preliminary one-seed diagnostic findings, not a five-seed qualification result. Baseline/control at 320 is the correct matched-context reference; the old 128-context-trained models are NOT such a reference.

## Hard incomplete-work gates

The full preregistered ten-fit (five seeds × two arms) matrix **was not completed in this research execution**, so no mean, confidence interval, research superiority or promotion can be asserted. Two binary seed1701 checkpoints, saved optimizer and RNG state, source and sample-order hashes, cold-reloaded full DEV scores, exact executed Python code, protocol and partial `RESULTS.json` are preserved in the local evidence bundle. They are not in GitHub cold CI. Re-run complete five-seed pre-registered arms with original source-authorized Vim docs before interpreting a 256 result.

The stronger previous result stands: MLAG004 showed the source model loses 96-character retrieval quality immediately beyond its 128-character training position. Here both arms were actually trained on the longer 320-character context, which addresses exposure as a confound but does NOT remove all capacity and objective tradeoffs.

**Decision**: do not advertise solved 256/512 or long-context retention, and do not prune aligned memory. Continue source-authorized multi-horizon training with a fair, performance-preserving auxiliary objective, independent whole-document TEST, strong equal-parameter SSM/GRU/attention baseline and binary GitHub cold replay. Do not silently revise MLAG005's objective to chase this reused DEV result.
