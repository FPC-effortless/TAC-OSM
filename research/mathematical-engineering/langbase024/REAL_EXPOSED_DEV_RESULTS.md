# LANGBASE024 — authentic original-byte next-symbol baseline results

**2026-10-28 • System-1/language • genuine original CPython3.13.0 publisher docs • previously EXPOSED internal DEV, NOT unseen TEST.**

## Preregistered evaluation outcome

The entire protocol was frozen in commit `77224e40e77810ed4b674b40e88ad66c244aabd3`, before running this study's DEV predictions. Dedicated [GitHub Actions run 40751909582](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40751909582) completed **SUCCESS**: all six constructed source-boundary/numerical tests pass, then exact original 17-file publisher source SHA and Git blob verification plus Unicode supplement source verification, then full authentic DEV next-byte scoring. [Original numerical JSON artifact 12844705980](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40751909582/artifacts/12844705980) uploaded (ZIP SHA256 `aa8463964daee7b8e5a3d122c658ddf56cbcc611ad9150e655e963d6e749f02b`; included JSON SHA256 `967e1c62358e6fc0f3593f67a77872b15e22687371d9ff08f8ab42e9bda2d093`).

**Targets:** all **69,542** next original-byte labels in four original separately owned DEV documents; no source window selection/TEST; no label future leakage. Training fit contains 13 original frozen publisher FIT docs, 194,186 next-byte examples and **ZERO** non-ASCII next-byte FIT labels. The separately preregistered Unicode FIT supplement raises exposure to 226,071 next-byte labels, with only **62** non-ASCII FIT targets. Source exposure differs across arms.

| Original byte source and fit policy | Next-byte NLL (nats/byte) ↓ | True bits/original byte ↓ | Brier ↓ | Top1 ↑ | Top5 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 13 FIT / P0 Dirichlet-smoothed unigram | 3.33560141 | 4.81225561 | .93724885 | 18.5384% | 42.1702% |
| Original 13 FIT / P3 hierarchical context 1–3 | 1.81222495 | 2.61448795 | .60278544 | 52.7667% | 82.9110% |
| 13 FIT + registered UTF8 supplement / P0 | 3.33664651 | 4.81376338 | .93733225 | 18.5384% | 42.1702% |
| 13 FIT + registered UTF8 supplement / P3 | **1.79832479** | **2.59443426** | **.59955350** | **53.1046%** | **82.9283%** |

The Unicode FIT supplement improves the hierarchical three-byte P3 model by **0.01390016 nats per original byte** on these exposed DEV sources but *worsens* the unigram P0. Increased training exposure means **not** a matched-compute/data comparison. These are not learned TAC neural model outputs.

**Per-document P3 NLL (nats per raw byte):**

| Publisher DEV document | Source targets | Original FIT | + Unicode FIT |
| --- | ---: | ---: | ---: |
| `Doc/tutorial/controlflow.rst` | 40,759 | 1.7599961 | 1.7481588 |
| `Doc/tutorial/index.rst` | 2,385 | 1.8353088 | 1.8324468 |
| `Doc/tutorial/introduction.rst` | 18,983 | 1.8468381 | 1.8259490 |
| `Doc/tutorial/venv.rst` | 7,415 | 2.0032811 | 1.9923836 |

**Rare byte failure (critical):** only **13 non-ASCII DEV next-byte labels**, all in `controlflow.rst`. Original FIT P3 rare-byte NLL **15.45652** nats per rare target; FIT+Unicode P3 **12.16805**. This reduces loss on those 13 targets but is still very poor and does not prove robust UTF-8 or multilingual prediction. A 256-value byte codec is an *exact information-preserving interface*, not competence across all 256 bytes; only 113 distinct byte values occur in the separately admitted Unicode supplement.

## Engineering/provenance/limits

- `research/mathematical-engineering/langbase024/byte_next_reference.py` reads no weights, accepts only original SHA-and-Git-blob-verified publisher files, fits source-owner-isolated byte counts and scores raw target bytes without mapping high bytes to `?`. Actual raw bits per byte is mathematically valid here.
- Frozen concentration `tau=8`, Dirichlet pseudocount .5 and max 3 prior bytes; no DEV-driven tuning. This counts preceding **three bytes**, not token-index three words. Hierarchical exact empirical n-gram reference is established classical modeling, **not architectural innovation**.
- Six constructed numerical/software tests passed; original two FIT arms and all four genuine DEV chapters scored in GitHub with only public PSF source. The JSON receipt includes original per-document metrics and its SHA is recorded above. GitHub Artifacts have finite retention; this report keeps aggregate and per-document values in Git.
- The original Vim9.1 499,290-parameter TAC archive is independently authenticated and local second-process replay passed in TEMP023, **but original user-owned weights must not be published before owner/source IP review**. `GIT_COLD_REPLAY=BLOCKED`.
- The 256-byte, width501 **499,978-parameter TAC constructor is UNTRAINED**. This 1.79832-nat/byte classical bar is for future **same-source/same-vocabulary** comparisons, NOT for retroactive numerical comparison to original 128-symbol TAC NLL 2.53744 on a **different source and lossy codec**. No network, new model training, GPU, protected TEST, PNDS whole-model proof, customer results or parameter-equal neural superiority assertion.

**Scientific evidence:** AUTHENTIC_SOURCE_VERIFIED / INTERNAL_EXPOSED_DEV_CLASSICAL_REFERENCE / NO_NATIVE_LANGUAGE_INTELLIGENCE_GAIN / INDEPENDENT_TEST_NOT_RUN.
