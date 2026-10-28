# GRAPH008 causal copy-mixture — preregistered no-fit extension

**Status: useful next-token engineering gain, NOT TAC architectural superiority.** A new six-document source set outside the 30 FIT, 6 DEV and previous six supplemental files was SHA-256 frozen in [GRAPH008_COPY_MIX_PRE_EVAL.json](GRAPH008_COPY_MIX_PRE_EVAL.json) before predictions. Documents: develop.txt, filetype.txt, fold.txt, gui.txt, quickfix.txt, repeat.txt. Eight equally spaced 257-byte windows per document; five original GRAPH007 and GRAPH008 next-only checkpoints, no additional weight fitting.

At time t, a source-only causal 4-character suffix-match lookup finds the most recent earlier four-character context with the same observed suffix, and reads the already-observed next character from the *old* matching context. It never examines the current target x[t+1]. A smooth, valid categorical copy distribution gives 0.95 to that known old candidate and 0.05 to the FIT-only smoothed character prior. With a hit, use
`p_mix=(1-lambda)*p_model+lambda*p_copy`; otherwise preserve `p_model`. This is a known algorithmic hybrid / n-gram cache, not learned semantic retrieval. `lambda` was selected from {0,0.1,...,0.7} on FIT-only original usr_01..08 manual windows, pooled over the five v4 seeds, before scoring the frozen six new documents. Chosen `lambda=0.5`.

| Frozen six-document next-char metric (five-seed mean) | Model alone | Identical FIT-calibrated copy mixture |
|---|---:|---:|
| GRAPH007 next-only NLL ↓ | 2.54538 | **2.36468** |
| GRAPH008 next-only NLL ↓ | 2.55062 | **2.36866** |
| GRAPH007 next-only perplexity ↓ | 12.75066 | **10.64169** |
| GRAPH008 next-only perplexity ↓ | 12.81776 | **10.68418** |

The source matcher found legal previous contexts at **16.99%** of positions, and the retrieved old next character matched the actual new target at **71.70%** of hits. A standalone FIT-unigram plus n-gram copy reference has NLL 3.26785, much worse than either neural hybrid. The same algorithmic improvement occurred in BOTH models. GRAPH008 with larger episodic state still **underperforms** GRAPH007 under the same copy access. This is a capability opportunity for the wider ecosystem, not demonstrated unique novel TAC memory superiority.

The unfitted, direct GRAPH008 *latent* query intervention previously worsened exposed DEV NLL from 2.32817 to 2.33250 across five seeds; by contrast, the properly normalized FIT-calibrated probability-level copy mixture improved fresh-new-document loss. This isolates a remaining engineering bottleneck: learned retrieval gating/query-value fusion. The shared historical-token decoder succeeds at known past ordinal 64/192/512, but the actual language decoder is not yet trained to select and use long-range source evidence.

Open gates: full source-authenticated owner isolation, learned autonomous query policy, adequate equal-cache 500K selective-SSM/GRU/attention competitors, original repository-backed LFS checkpoints, end-to-end PyTorch/IEEE/AD and blind external language datasets. No model promoted. This is genuine public Vim text but still within the same publisher/domain. Do not count FIT calibration as independent TEST.
