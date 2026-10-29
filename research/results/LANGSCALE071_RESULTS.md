# LANGSCALE071 — nine original trained TAC models x five equal-target reset-context lengths

Original read-only result: [TAC draft #610](https://github.com/FPC-effortless/TAC-OSM/pull/610), [nine original checkpoints CPU #40872684271](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40872684271), [source-verified aggregate JSON #12896243274](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40872684271/artifacts/12896243274). All nine native no-fit score jobs and the fail-closed nine-result audit passed. No new model fitting or checkpoint modification, GPU/Kaggle, protected TEST, promotion or merge.

## Strict original source and comparison identity

Original [LANGSCALE070 trained three-size nine original checkpoints #40871491842](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842), source SHA+optimizer+RNG+weights verified again, and original exposed 4096 DEV NLL replay matched exact original 070 reported NLL to <=1e-6 before each new evaluation. Four original publisher PSF CPython3.13 exposed DEV document owners, exactly 64 deterministically preregistered target positions per document = **256 authentic original next-byte labels** for EVERY model and length. Source target fingerprint is `5c267b3f6cf43ebbfd8e46a6e1e1376f2f730ec35a5db19293d4f0af6659f803`; source-input byte hashes for each L identical across all nine models. Prediction at each source t uses only same-document `raw[t-L:t]`, never label byte `raw[t]`. Each input window **resets recurrence**. Context lengths L16/32/64/128/256, where 256 is **untrained-context extrapolation** (models fitted on 128B windows), not carried state or a long persistent history.

## Three-seed NLL means on EXACT SAME original targets

| Reset input length | 250,133 params | 499,623 params | 1,000,239 params |
|---|---:|---:|---:|
| 16 bytes | 2.308840 | 2.204434 | 2.121262 |
| **32 bytes** | **2.298581** | **2.195253** | **2.105213** |
| 64 bytes | 2.310748 | 2.201724 | 2.111712 |
| 128 bytes | 2.322143 | 2.216329 | 2.122132 |
| 256 bytes | **2.520124** | **2.359908** | **2.275988** |

**Negative scaling finding:** the larger FFN model scores lower NLL at every context, but **longer input windows after 32 bytes do not help**. Context32 beats128 for all nine size/seed checkpoints; original context256 is worse than128 for all nine size/seed checkpoints. The mean penalties for 128->256 are +0.197982 nats/byte (250K), +0.143579 (500K), +0.153856 (1M), all on the **same** 256 targets. Many local intermediate 32->64/64->128 outcomes are mixed by seed; no fabricated unanimous pattern.

## Interpretation and evidence boundaries

This demonstrates *capacity scaling of the readout* does not remove a **reset-window context scaling** weakness. It does **not** prove TAC is unable to persist memories, show information loss in native 8x8 matrix, or show that a larger trained context wouldn't help; the test intentionally **resets** at every target. At L256 absolute positions 128..255 were never trained in the frozen 128B full model and the native sinusoidal/log positional features are extrapolated; a positional-source mismatch is a plausible confound, not yet a proven causal diagnosis. The 256 target cohort is already exposed original DEV and is not an independent qualification. No GRU, SSM or Transformer under matched data/parameter/compute is included. Full PNDS actual source, IEEE/AD/streaming parity and mobile readiness OPEN.

**Next registered falsifier, not posthoc acceptance:** evaluate `native` versus registered source `cap_all` positional transform in the SAME nine frozen trained models on L256 reset windows, using the SAME 256 exact target bytes and a L128 unchanged-position negative control. If capping untrained t>=128 improves NLL, it implicates positional extrapolation as a contributor; it cannot by itself prove or fix long-term information retention. Even a positive result is an exploratory adaptation on previously exposed DEV only, not a model checkpoint or generalization gain.
